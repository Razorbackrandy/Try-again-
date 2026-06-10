"""
Tests for task_router.py
Run with:  pytest tests/
"""
import sys
import os
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from task_router import (
    Task,
    TaskResult,
    TaskStatus,
    TaskRouter,
    Route,
    RetryPolicy,
    NO_RETRY,
    FAST_RETRY,
    logging_middleware,
    metrics_middleware,
    HandlerTimeoutError,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitOpenError,
    RateLimiter,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    r = TaskRouter(middleware=[])
    yield r
    r.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

class TestExactRoute:
    def test_match_returns_succeeded(self, router):
        router.add(Route.exact("ping", lambda t: "pong", retry_policy=NO_RETRY))
        result = router.dispatch(Task(type="ping"))
        assert result.status == TaskStatus.SUCCEEDED
        assert result.value == "pong"

    def test_no_match_returns_rejected(self, router):
        result = router.dispatch(Task(type="nonexistent"))
        assert result.status == TaskStatus.REJECTED

    def test_wrong_type_does_not_match(self, router):
        router.add(Route.exact("ping", lambda t: None, retry_policy=NO_RETRY))
        result = router.dispatch(Task(type="ping.extra"))
        assert result.status == TaskStatus.REJECTED


class TestPrefixRoute:
    def test_prefix_matches_subtypes(self, router):
        router.add(Route.prefix("sms.", lambda t: "sent", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="sms.otp")).status   == TaskStatus.SUCCEEDED
        assert router.dispatch(Task(type="sms.alert")).status == TaskStatus.SUCCEEDED

    def test_non_prefix_not_matched(self, router):
        router.add(Route.prefix("sms.", lambda t: "sent", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="email.send")).status == TaskStatus.REJECTED

    def test_prefix_does_not_match_unrelated_prefix(self, router):
        router.add(Route.prefix("sms", lambda t: "sent", retry_policy=NO_RETRY))
        # "smtp.connect" starts with "sm" but not "sms"
        assert router.dispatch(Task(type="smtp.connect")).status == TaskStatus.REJECTED


class TestRegexRoute:
    def test_regex_matches(self, router):
        router.add(Route.regex(r"push\.(apns|fcm)\.send", lambda t: "pushed", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="push.apns.send")).status == TaskStatus.SUCCEEDED
        assert router.dispatch(Task(type="push.fcm.send")).status  == TaskStatus.SUCCEEDED

    def test_regex_non_match(self, router):
        router.add(Route.regex(r"push\.(apns|fcm)\.send", lambda t: "pushed", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="push.web.send")).status == TaskStatus.REJECTED


class TestPredicateRoute:
    def test_predicate_match(self, router):
        router.add(Route.predicate_match(
            predicate=lambda t: t.priority > 5,
            handler=lambda t: "high",
            retry_policy=NO_RETRY,
        ))
        assert router.dispatch(Task(type="anything", priority=10)).status == TaskStatus.SUCCEEDED
        assert router.dispatch(Task(type="anything", priority=1)).status  == TaskStatus.REJECTED


class TestWildcardRoute:
    def test_catches_all(self, router):
        router.add(Route.wildcard(lambda t: "caught", retry_policy=NO_RETRY))
        for t in ["a", "b.c", "x.y.z"]:
            assert router.dispatch(Task(type=t)).status == TaskStatus.SUCCEEDED

    def test_fallback_after_specific(self, router):
        router.add(Route.exact("special", lambda t: "special", retry_policy=NO_RETRY))
        router.add(Route.wildcard(lambda t: "generic", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="special")).value == "special"
        assert router.dispatch(Task(type="other")).value   == "generic"


# ---------------------------------------------------------------------------
# First-match wins
# ---------------------------------------------------------------------------

class TestRouteOrdering:
    def test_first_match_wins(self, router):
        hits = []
        router.add(Route.exact("x", lambda t: hits.append("first")  or "first",  retry_policy=NO_RETRY))
        router.add(Route.exact("x", lambda t: hits.append("second") or "second", retry_policy=NO_RETRY))
        router.dispatch(Task(type="x"))
        assert hits == ["first"]

    def test_specific_before_wildcard(self, router):
        router.add(Route.exact("x",   lambda t: "exact",    retry_policy=NO_RETRY))
        router.add(Route.wildcard(     lambda t: "wildcard", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="x")).value    == "exact"
        assert router.dispatch(Task(type="y")).value    == "wildcard"


# ---------------------------------------------------------------------------
# Decorator registration
# ---------------------------------------------------------------------------

class TestDecorator:
    def test_route_decorator(self, router):
        @router.route("greet")
        def greet(task):
            return f"Hello, {task.payload}!"

        result = router.dispatch(Task(type="greet", payload="World"))
        assert result.status == TaskStatus.SUCCEEDED
        assert result.value  == "Hello, World!"

    def test_decorator_with_retry_policy(self, router):
        attempts = []

        @router.route("unstable", retry=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=False))
        def unstable(task):
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("transient")
            return "ok"

        result = router.dispatch(Task(type="unstable"))
        assert result.status == TaskStatus.SUCCEEDED
        assert len(attempts) == 2


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetry:
    def test_succeeds_on_third_attempt(self, router):
        calls = []

        def flaky(task):
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("temporary")
            return "ok"

        policy = RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False)
        router.add(Route.exact("flaky", flaky, retry_policy=policy))

        result = router.dispatch(Task(type="flaky"))
        assert result.status == TaskStatus.SUCCEEDED
        assert result.value  == "ok"
        assert len(calls)    == 3

    def test_dead_after_exhausted_retries(self, router):
        def always_fail(task):
            raise RuntimeError("boom")

        policy = RetryPolicy(max_attempts=2, base_delay=0.0, jitter=False)
        router.add(Route.exact("dead", always_fail, retry_policy=policy))

        result = router.dispatch(Task(type="dead"))
        assert result.status == TaskStatus.DEAD
        assert result.error  is not None

    def test_no_retry_policy_fails_immediately(self, router):
        calls = []

        def fail_once(task):
            calls.append(1)
            raise RuntimeError("fail")

        router.add(Route.exact("single", fail_once, retry_policy=NO_RETRY))
        result = router.dispatch(Task(type="single"))
        assert result.status == TaskStatus.DEAD
        assert len(calls)    == 1

    def test_non_retryable_exception_stops_immediately(self, router):
        calls = []

        def raise_value_error(task):
            calls.append(1)
            raise ValueError("not retryable")

        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.0,
            jitter=False,
            retryable_on=(RuntimeError,),   # only RuntimeError retries
        )
        router.add(Route.exact("typed", raise_value_error, retry_policy=policy))

        result = router.dispatch(Task(type="typed"))
        assert result.status == TaskStatus.FAILED
        assert len(calls)    == 1

    def test_result_duration_is_positive(self, router):
        router.add(Route.wildcard(lambda t: None, retry_policy=NO_RETRY))
        result = router.dispatch(Task(type="any"))
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Dead-letter queue
# ---------------------------------------------------------------------------

class TestDeadLetterQueue:
    def test_dead_tasks_accumulate_in_dlq(self, router):
        def always_fail(task):
            raise RuntimeError("dlq test")

        policy = RetryPolicy(max_attempts=1, base_delay=0.0, jitter=False)
        router.add(Route.exact("fail", always_fail, retry_policy=policy))

        router.dispatch(Task(type="fail"))
        router.dispatch(Task(type="fail"))

        assert len(router.dead_letter_queue) == 2

    def test_custom_dlq_callback(self):
        captured = []
        r = TaskRouter(
            middleware=[],
            dead_letter=lambda result: captured.append(result),
        )
        r.add(Route.exact(
            "fail",
            lambda t: (_ for _ in ()).throw(RuntimeError("custom dlq")),
            retry_policy=NO_RETRY,
        ))
        r.dispatch(Task(type="fail"))
        r.shutdown(wait=False)
        assert len(captured) == 1

    def test_succeeded_tasks_not_in_dlq(self, router):
        router.add(Route.wildcard(lambda t: "ok", retry_policy=NO_RETRY))
        router.dispatch(Task(type="ok"))
        assert len(router.dead_letter_queue) == 0

    def test_dlq_is_snapshot(self, router):
        def always_fail(task):
            raise RuntimeError("dlq")

        router.add(Route.exact("fail", always_fail, retry_policy=NO_RETRY))
        router.dispatch(Task(type="fail"))

        snapshot1 = router.dead_letter_queue
        router.dispatch(Task(type="fail"))
        snapshot2 = router.dead_letter_queue

        assert len(snapshot1) == 1
        assert len(snapshot2) == 2


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------

class TestPriorityQueue:
    def test_drain_processes_high_priority_first(self):
        order = []

        def record(task):
            order.append(task.priority)

        r = TaskRouter(workers=1, middleware=[])
        r.add(Route.wildcard(record, retry_policy=NO_RETRY))

        for p in [1, 5, 3, 10, 2]:
            r.enqueue(Task(type="t", priority=p))

        futures = r.drain()
        [f.result(timeout=5) for f in futures]
        r.shutdown()

        assert order == sorted(order, reverse=True), \
            f"Expected descending priority, got {order}"

    def test_enqueue_is_thread_safe(self):
        r = TaskRouter(workers=2, middleware=[])
        r.add(Route.wildcard(lambda t: None, retry_policy=NO_RETRY))

        def enqueue_batch():
            for i in range(50):
                r.enqueue(Task(type="t", priority=i))

        threads = [threading.Thread(target=enqueue_batch) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        futures = r.drain()
        [f.result(timeout=10) for f in futures]
        r.shutdown()

    def test_drain_max_tasks(self):
        r = TaskRouter(workers=1, middleware=[])
        r.add(Route.wildcard(lambda t: None, retry_policy=NO_RETRY))

        for i in range(10):
            r.enqueue(Task(type="t", priority=i))

        futures = r.drain(max_tasks=3)
        assert len(futures) == 3
        r.drain()            # drain remainder
        r.shutdown()


# ---------------------------------------------------------------------------
# Async submit
# ---------------------------------------------------------------------------

class TestSubmit:
    def test_submit_returns_future(self, router):
        router.add(Route.wildcard(lambda t: 42, retry_policy=NO_RETRY))
        fut = router.submit(Task(type="any"))
        result = fut.result(timeout=5)
        assert result.value == 42

    def test_concurrent_submissions(self, router):
        router.add(Route.wildcard(lambda t: t.type, retry_policy=NO_RETRY))
        types = [f"task.{i}" for i in range(20)]
        futures = [router.submit(Task(type=t)) for t in types]
        results = [f.result(timeout=10) for f in futures]
        assert all(r.status == TaskStatus.SUCCEEDED for r in results)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class TestMiddleware:
    def test_middleware_is_applied(self):
        log = []

        def spy(task, next_fn):
            log.append(f"before:{task.type}")
            result = next_fn(task)
            log.append(f"after:{task.type}")
            return result

        r = TaskRouter(middleware=[spy])
        r.add(Route.exact("x", lambda t: None, retry_policy=NO_RETRY))
        r.dispatch(Task(type="x"))
        r.shutdown(wait=False)

        assert log == ["before:x", "after:x"]

    def test_middleware_stack_order(self):
        order = []

        def outer_mw(task, next_fn):
            order.append("outer-in")
            result = next_fn(task)
            order.append("outer-out")
            return result

        def inner_mw(task, next_fn):
            order.append("inner-in")
            result = next_fn(task)
            order.append("inner-out")
            return result

        r = TaskRouter(middleware=[outer_mw, inner_mw])
        r.add(Route.wildcard(lambda t: None, retry_policy=NO_RETRY))
        r.dispatch(Task(type="any"))
        r.shutdown(wait=False)

        assert order == ["outer-in", "inner-in", "inner-out", "outer-out"]

    def test_metrics_middleware(self):
        counts: dict = {}
        r = TaskRouter(middleware=[metrics_middleware(counts)])
        r.add(Route.wildcard(lambda t: None, retry_policy=NO_RETRY))
        r.dispatch(Task(type="alpha"))
        r.dispatch(Task(type="alpha"))
        r.dispatch(Task(type="beta"))
        r.shutdown(wait=False)
        assert counts == {"alpha": 2, "beta": 1}

    def test_no_middleware(self):
        r = TaskRouter(middleware=[])
        r.add(Route.wildcard(lambda t: "bare", retry_policy=NO_RETRY))
        result = r.dispatch(Task(type="x"))
        r.shutdown(wait=False)
        assert result.value == "bare"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_shuts_down(self):
        with TaskRouter(middleware=[]) as r:
            r.add(Route.wildcard(lambda t: "ok", retry_policy=NO_RETRY))
            result = r.dispatch(Task(type="any"))
        assert result.status == TaskStatus.SUCCEEDED

    def test_repr(self, router):
        router.add(Route.exact("a", lambda t: None, retry_policy=NO_RETRY))
        router.add(Route.exact("b", lambda t: None, retry_policy=NO_RETRY))
        assert "routes=2" in repr(router)


# ---------------------------------------------------------------------------
# Payload passthrough
# ---------------------------------------------------------------------------

class TestScheduling:
    def test_schedule_in_fires_after_delay(self):
        fired_at: list = []

        def record(task):
            fired_at.append(time.time())

        r = TaskRouter(middleware=[], scheduler_tick=0.02)
        r.add(Route.wildcard(record, retry_policy=NO_RETRY))
        r.start()

        t0 = time.time()
        r.schedule_in(Task(type="t"), delay=0.1)

        # poll until executed or timeout
        deadline = t0 + 2.0
        while not fired_at and time.time() < deadline:
            time.sleep(0.02)

        r.shutdown(wait=False)
        assert len(fired_at) == 1
        assert fired_at[0] - t0 >= 0.09         # fired no earlier than requested
        assert fired_at[0] - t0 < 1.0           # and reasonably soon after

    def test_schedule_at_past_fires_immediately(self):
        fired: list = []
        r = TaskRouter(middleware=[], scheduler_tick=0.02)
        r.add(Route.wildcard(lambda t: fired.append(1), retry_policy=NO_RETRY))
        r.start()

        r.schedule(Task(type="t"), run_at=time.time() - 1.0)

        deadline = time.time() + 2.0
        while not fired and time.time() < deadline:
            time.sleep(0.02)

        r.shutdown(wait=False)
        assert len(fired) == 1

    def test_schedule_ordering(self):
        order: list = []
        r = TaskRouter(workers=1, middleware=[], scheduler_tick=0.02)
        r.add(Route.wildcard(lambda t: order.append(t.payload), retry_policy=NO_RETRY))
        r.start()

        # Submit out of order
        r.schedule_in(Task(type="t", payload="third"),  delay=0.30)
        r.schedule_in(Task(type="t", payload="first"),  delay=0.10)
        r.schedule_in(Task(type="t", payload="second"), delay=0.20)

        deadline = time.time() + 3.0
        while len(order) < 3 and time.time() < deadline:
            time.sleep(0.02)

        r.shutdown(wait=False)
        assert order == ["first", "second", "third"]

    def test_cancel_prevents_execution(self):
        fired: list = []
        r = TaskRouter(middleware=[], scheduler_tick=0.02)
        r.add(Route.wildcard(lambda t: fired.append(1), retry_policy=NO_RETRY))
        r.start()

        task_id = r.schedule_in(Task(type="t"), delay=0.20)
        assert r.cancel(task_id) is True

        time.sleep(0.40)
        r.shutdown(wait=False)
        assert fired == []

    def test_cancel_returns_false_for_unknown(self):
        r = TaskRouter(middleware=[])
        assert r.cancel("nonexistent-id") is False
        r.shutdown(wait=False)

    def test_context_manager_starts_scheduler(self):
        fired: list = []
        with TaskRouter(middleware=[], scheduler_tick=0.02) as r:
            r.add(Route.wildcard(lambda t: fired.append(1), retry_policy=NO_RETRY))
            r.schedule_in(Task(type="t"), delay=0.05)
            deadline = time.time() + 1.0
            while not fired and time.time() < deadline:
                time.sleep(0.02)
        assert fired == [1]

    def test_start_is_idempotent(self):
        r = TaskRouter(middleware=[])
        r.start()
        thread1 = r._scheduler_thread
        r.start()
        thread2 = r._scheduler_thread
        assert thread1 is thread2
        r.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Execution timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_slow_handler_produces_timeout_error(self, router):
        router.add(Route.exact(
            "slow", lambda t: time.sleep(5), retry_policy=NO_RETRY, timeout=0.05,
        ))
        result = router.dispatch(Task(type="slow"))
        assert result.status == TaskStatus.DEAD
        assert isinstance(result.error, HandlerTimeoutError)

    def test_timeout_is_per_attempt_not_total(self, router):
        calls = []

        def handler(task):
            calls.append(1)
            if len(calls) < 3:
                time.sleep(5)
            return "ok"

        router.add(Route.exact(
            "retry-timeout",
            handler,
            retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False),
            timeout=0.05,
        ))
        result = router.dispatch(Task(type="retry-timeout"))
        assert result.status == TaskStatus.SUCCEEDED
        assert len(calls) == 3

    def test_fast_handler_not_affected(self, router):
        router.add(Route.exact("fast", lambda t: "done", retry_policy=NO_RETRY, timeout=5.0))
        result = router.dispatch(Task(type="fast"))
        assert result.status == TaskStatus.SUCCEEDED
        assert result.value == "done"

    def test_no_timeout_unchanged_behavior(self, router):
        router.add(Route.exact("x", lambda t: 42, retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="x")).value == 42

    def test_elapsed_bounded_by_timeout(self, router):
        router.add(Route.exact(
            "slow", lambda t: time.sleep(10), retry_policy=NO_RETRY, timeout=0.1,
        ))
        t0 = time.perf_counter()
        router.dispatch(Task(type="slow"))
        assert time.perf_counter() - t0 < 1.0


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def _route(self, handler, *, threshold=3, recovery=0.15, success_req=1, **kw):
        cb = CircuitBreaker(
            failure_threshold=threshold,
            recovery_timeout=recovery,
            success_threshold=success_req,
        )
        route = Route.exact("svc", handler, retry_policy=NO_RETRY, circuit_breaker=cb, **kw)
        return route, cb

    def test_closed_passes_through(self, router):
        route, cb = self._route(lambda t: "ok")
        router.add(route)
        result = router.dispatch(Task(type="svc"))
        assert result.status == TaskStatus.SUCCEEDED
        assert cb.state == CircuitBreakerState.CLOSED

    def test_opens_after_failure_threshold(self, router):
        def fail(t): raise RuntimeError("x")
        route, cb = self._route(fail)
        router.add(route)
        for _ in range(3):
            router.dispatch(Task(type="svc"))
        assert cb.state == CircuitBreakerState.OPEN

    def test_open_state_fast_fails(self, router):
        def fail(t): raise RuntimeError("x")
        route, cb = self._route(fail)
        router.add(route)
        for _ in range(3):
            router.dispatch(Task(type="svc"))
        result = router.dispatch(Task(type="svc"))
        assert result.status == TaskStatus.REJECTED
        assert isinstance(result.error, CircuitOpenError)

    def test_transitions_to_half_open_then_closes(self, router):
        def fail(t): raise RuntimeError("x")
        route, cb = self._route(fail, recovery=0.05)
        router.add(route)
        for _ in range(3):
            router.dispatch(Task(type="svc"))
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.1)
        route.handler = lambda t: "recovered"
        result = router.dispatch(Task(type="svc"))
        assert result.status == TaskStatus.SUCCEEDED
        assert cb.state == CircuitBreakerState.CLOSED

    def test_probe_failure_reopens_circuit(self, router):
        def fail(t): raise RuntimeError("still failing")
        route, cb = self._route(fail, recovery=0.05)
        router.add(route)
        for _ in range(3):
            router.dispatch(Task(type="svc"))
        time.sleep(0.1)
        router.dispatch(Task(type="svc"))   # probe — fails
        assert cb.state == CircuitBreakerState.OPEN

    def test_success_threshold_gt_one(self, router):
        route, cb = self._route(
            lambda t: (_ for _ in ()).throw(RuntimeError("x")),
            recovery=0.05,
            success_req=2,
        )
        router.add(route)
        for _ in range(3):
            router.dispatch(Task(type="svc"))
        time.sleep(0.1)
        route.handler = lambda t: "ok"
        router.dispatch(Task(type="svc"))   # probe 1
        assert cb.state == CircuitBreakerState.HALF_OPEN
        router.dispatch(Task(type="svc"))   # probe 2
        assert cb.state == CircuitBreakerState.CLOSED

    def test_cb_thread_safety(self):
        def fail(t): raise RuntimeError("x")
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout=60.0)
        route = Route.exact("svc", fail, retry_policy=NO_RETRY, circuit_breaker=cb)
        r = TaskRouter(workers=8, middleware=[])
        r.add(route)
        futures = [r.submit(Task(type="svc")) for _ in range(50)]
        [f.result(timeout=10) for f in futures]
        r.shutdown()
        assert cb.state in (CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)

    def test_cb_none_is_disabled(self, router):
        router.add(Route.exact("x", lambda t: "ok", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="x")).status == TaskStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_within_rate_allows_burst(self, router):
        rl = RateLimiter(max_rate=20.0, max_wait=0.0)
        router.add(Route.exact("svc", lambda t: "ok", retry_policy=NO_RETRY, rate_limiter=rl))
        results = [router.dispatch(Task(type="svc")) for _ in range(20)]
        assert all(r.status == TaskStatus.SUCCEEDED for r in results)

    def test_beyond_burst_rejected_immediately(self, router):
        rl = RateLimiter(max_rate=5.0, burst=5.0, max_wait=0.0)
        router.add(Route.exact("svc", lambda t: "ok", retry_policy=NO_RETRY, rate_limiter=rl))
        results = [router.dispatch(Task(type="svc")) for _ in range(7)]
        rejected = [r for r in results if r.status == TaskStatus.REJECTED]
        assert len(rejected) == 2

    def test_max_concurrent_limits_inflight(self):
        barrier  = threading.Barrier(2)
        released = threading.Event()

        def handler(t):
            barrier.wait(timeout=3)
            released.wait(timeout=3)
            return "ok"

        rl = RateLimiter(max_concurrent=2, max_wait=0.0)
        r  = TaskRouter(workers=4, middleware=[])
        r.add(Route.exact("svc", handler, retry_policy=NO_RETRY, rate_limiter=rl))

        f1 = r.submit(Task(type="svc"))
        f2 = r.submit(Task(type="svc"))
        time.sleep(0.1)
        result3 = r.dispatch(Task(type="svc"))   # 3rd: all slots taken
        assert result3.status == TaskStatus.REJECTED
        assert isinstance(result3.error, RateLimitError)
        released.set()
        f1.result(timeout=5)
        f2.result(timeout=5)
        r.shutdown()

    def test_max_wait_allows_blocking(self, router):
        rl = RateLimiter(max_rate=10.0, burst=1.0, max_wait=1.0)
        router.add(Route.exact("svc", lambda t: "ok", retry_policy=NO_RETRY, rate_limiter=rl))
        router.dispatch(Task(type="svc"))   # drain the single token
        t0     = time.perf_counter()
        result = router.dispatch(Task(type="svc"))   # blocks ~0.1s for refill
        assert result.status == TaskStatus.SUCCEEDED
        assert time.perf_counter() - t0 >= 0.08

    def test_rate_limiter_none_is_disabled(self, router):
        router.add(Route.exact("x", lambda t: "ok", retry_policy=NO_RETRY))
        assert router.dispatch(Task(type="x")).status == TaskStatus.SUCCEEDED

    def test_rejected_result_contains_error(self, router):
        rl = RateLimiter(max_rate=1.0, burst=1.0, max_wait=0.0)
        router.add(Route.exact(
            "r", lambda t: "ok", retry_policy=NO_RETRY, rate_limiter=rl, name="r",
        ))
        router.dispatch(Task(type="r"))
        result = router.dispatch(Task(type="r"))
        assert result.status == TaskStatus.REJECTED
        assert isinstance(result.error, RateLimitError)

    def test_rate_limiter_thread_safety(self):
        # Verifies no corruption under concurrent load: all 200 tasks must
        # complete with a valid status (none dropped, no exceptions escaped).
        rl = RateLimiter(max_rate=100.0, burst=100.0, max_wait=0.0)
        r  = TaskRouter(workers=8, middleware=[])
        r.add(Route.exact("svc", lambda t: "ok", retry_policy=NO_RETRY, rate_limiter=rl))
        futures = [r.submit(Task(type="svc")) for _ in range(200)]
        results = [f.result(timeout=10) for f in futures]
        valid_statuses = {TaskStatus.SUCCEEDED, TaskStatus.REJECTED}
        assert all(res.status in valid_statuses for res in results)
        assert len(results) == 200
        # Burst is 100, so at least 100 must succeed (some extra may due to refill)
        succeeded = sum(1 for res in results if res.status == TaskStatus.SUCCEEDED)
        assert succeeded >= 100
        r.shutdown()


class TestPayload:
    def test_payload_is_accessible_in_handler(self, router):
        router.add(Route.wildcard(lambda t: t.payload["x"] * 2, retry_policy=NO_RETRY))
        result = router.dispatch(Task(type="math", payload={"x": 21}))
        assert result.value == 42

    def test_metadata_is_accessible(self, router):
        router.add(Route.wildcard(lambda t: t.metadata.get("env"), retry_policy=NO_RETRY))
        result = router.dispatch(Task(type="any", metadata={"env": "prod"}))
        assert result.value == "prod"
