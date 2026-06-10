"""
Task Routing System
===================
Routes tasks to registered handlers by type, prefix, regex, or custom predicate.

Features:
  • Priority queues            – higher-priority tasks run first
  • Scheduled execution        – run tasks at a future time or after a delay
  • Retry with backoff/jitter  – configurable per route
  • Execution timeout          – per-attempt time budget per route
  • Circuit breaker            – auto open/close on failure threshold
  • Rate limiter               – token bucket + concurrency cap per route
  • Middleware chain           – logging, metrics, auth, etc.
  • Thread-pool workers        – concurrent, bounded execution
  • Dead-letter queue          – permanently failed tasks land here
  • First-match routing        – routes checked in registration order

Quick start::

    router = TaskRouter(workers=4)

    @router.route("email.send")
    def send_email(task):
        ...
        return {"sent": True}

    router.add(Route.prefix("sms.", handle_sms))
    router.add(Route.wildcard(fallback_handler))

    result = router.dispatch(Task(type="email.send", payload={...}))
    future  = router.submit(Task(type="sms.otp",    payload={...}))
"""
from __future__ import annotations

import heapq
import logging
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    DEAD      = "dead"      # exhausted all retry attempts
    REJECTED  = "rejected"  # no route matched


@dataclass
class Task:
    """Unit of work dispatched through the router."""
    type: str
    payload: Any = None
    priority: int = 0          # higher value → processed first
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    attempt: int = field(default=0, compare=False)
    scheduled_at: Optional[float] = field(default=None, compare=False)  # Unix epoch

    # Heap ordering: negate priority so max-priority = min-heap root
    def __lt__(self, other: "Task") -> bool:
        return self.priority > other.priority

    def __le__(self, other: "Task") -> bool:
        return self.priority >= other.priority


@dataclass
class TaskResult:
    """Outcome of a single task execution (all retry attempts included)."""
    task: Task
    status: TaskStatus
    value: Any = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """
    Controls how many times a failed task is retried and how long to wait.

    ``max_attempts`` is the *total* number of attempts (1 = no retry).
    ``retryable_on`` limits retries to specific exception types.
    """
    max_attempts: int = 3
    base_delay: float = 1.0       # seconds before first retry
    max_delay: float = 30.0       # cap on delay
    backoff_factor: float = 2.0   # multiply delay by this each attempt
    jitter: bool = True           # ±50 % random jitter
    retryable_on: Tuple[Type[Exception], ...] = (Exception,)

    def delay_for(self, attempt: int) -> float:
        import random
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5
        return delay

    def should_retry(self, attempt: int, exc: Exception) -> bool:
        """Return True if we should make another attempt."""
        return (
            attempt < self.max_attempts
            and isinstance(exc, self.retryable_on)
        )


# Convenient presets
NO_RETRY   = RetryPolicy(max_attempts=1,  base_delay=0.0, jitter=False)
FAST_RETRY = RetryPolicy(max_attempts=3,  base_delay=0.1, max_delay=1.0)
SLOW_RETRY = RetryPolicy(max_attempts=5,  base_delay=2.0, max_delay=60.0)


# ---------------------------------------------------------------------------
# Resilience exceptions
# ---------------------------------------------------------------------------

class HandlerTimeoutError(Exception):
    """Raised when a handler exceeds its per-attempt ``Route.timeout`` budget.

    Counts as a failure and will trigger retries unless excluded via
    ``RetryPolicy(retryable_on=...)``."""
    def __init__(self, timeout: float) -> None:
        super().__init__(f"Handler did not complete within {timeout:.3f}s")
        self.timeout = timeout


class CircuitOpenError(Exception):
    """Raised (as ``TaskResult.error``) when a circuit breaker fast-fails a request."""
    def __init__(self, route_name: str) -> None:
        super().__init__(f"Circuit open for route {route_name!r}")
        self.route_name = route_name


class RateLimitError(Exception):
    """Raised (as ``TaskResult.error``) when a rate limiter rejects a request."""
    def __init__(self, route_name: str) -> None:
        super().__init__(f"Rate limit exceeded for route {route_name!r}")
        self.route_name = route_name


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreakerState(Enum):
    CLOSED    = "closed"     # normal — requests pass through
    OPEN      = "open"       # tripped — requests fast-fail
    HALF_OPEN = "half_open"  # recovery — one probe allowed through


@dataclass
class CircuitBreaker:
    """
    Per-route circuit breaker (thread-safe).

    States
    ------
    CLOSED    requests pass through; consecutive failures are counted.
    OPEN      fast-fail all requests; after *recovery_timeout* s → HALF_OPEN.
    HALF_OPEN one probe request allowed; success → CLOSED, failure → OPEN.

    Parameters
    ----------
    failure_threshold:
        Consecutive failures in CLOSED state that open the circuit.
    recovery_timeout:
        Seconds in OPEN state before allowing one probe (transition to HALF_OPEN).
    success_threshold:
        Consecutive probe successes in HALF_OPEN required to re-close.
    """
    failure_threshold: int   = 5
    recovery_timeout:  float = 30.0
    success_threshold: int   = 1

    _state:               CircuitBreakerState = field(
        default=CircuitBreakerState.CLOSED, init=False, repr=False
    )
    _consecutive_failures:  int   = field(default=0,    init=False, repr=False)
    _consecutive_successes: int   = field(default=0,    init=False, repr=False)
    _opened_at: Optional[float]   = field(default=None, init=False, repr=False)
    _probe_in_flight: bool        = field(default=False, init=False, repr=False)
    _lock: threading.Lock         = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def state(self) -> CircuitBreakerState:
        """Current state snapshot (may change concurrently)."""
        return self._state

    def allow_request(self) -> bool:
        """Return True if this request may proceed; False = fast-fail."""
        with self._lock:
            if self._state is CircuitBreakerState.CLOSED:
                return True
            if self._state is CircuitBreakerState.OPEN:
                if (self._opened_at is not None
                        and time.monotonic() - self._opened_at >= self.recovery_timeout):
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._consecutive_successes = 0
                    self._probe_in_flight = True
                    return True   # this caller is the probe
                return False
            # HALF_OPEN: allow only one probe at a time
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state is CircuitBreakerState.HALF_OPEN:
                self._consecutive_successes += 1
                self._probe_in_flight = False
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._consecutive_failures = 0
                    self._opened_at = None
            elif self._state is CircuitBreakerState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state is CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                self._opened_at = time.monotonic()
                self._probe_in_flight = False
                self._consecutive_successes = 0
            elif self._state is CircuitBreakerState.CLOSED:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    self._opened_at = time.monotonic()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """
    Per-route token-bucket rate limiter with optional concurrency cap (thread-safe).

    Parameters
    ----------
    max_rate:
        Maximum sustained throughput in tasks/second (token bucket). 0 = disabled.
    max_concurrent:
        Maximum tasks executing concurrently on this route. 0 = disabled.
    max_wait:
        Seconds :meth:`acquire` will block before giving up. 0 = fail immediately.
    burst:
        Token bucket capacity. Defaults to ``max_rate`` (one second's worth).
    """
    max_rate:       float = 0.0
    max_concurrent: int   = 0
    max_wait:       float = 0.0
    burst:          float = 0.0

    def __post_init__(self) -> None:
        cap = self.burst if self.burst > 0 else max(self.max_rate, 1.0)
        self._tokens:      float = cap
        self._capacity:    float = cap
        self._last_refill: float = time.monotonic()
        self._lock                        = threading.Lock()
        self._semaphore: Optional[threading.Semaphore] = (
            threading.Semaphore(self.max_concurrent) if self.max_concurrent > 0 else None
        )

    def _refill(self) -> None:
        """Refill token bucket based on elapsed time (call inside self._lock)."""
        if self.max_rate <= 0:
            return
        now = time.monotonic()
        self._tokens = min(
            self._capacity,
            self._tokens + (now - self._last_refill) * self.max_rate,
        )
        self._last_refill = now

    def acquire(self) -> bool:
        """
        Acquire a rate-limit token and a concurrency slot.

        Blocks up to ``max_wait`` seconds. Returns ``True`` on success,
        ``False`` if either limit is exceeded within the wait window.

        **Must** call :meth:`release` in a ``finally`` block on ``True``.
        """
        # 1. Token bucket
        acquired_token = False
        if self.max_rate > 0:
            deadline = time.monotonic() + self.max_wait
            while True:
                with self._lock:
                    self._refill()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        acquired_token = True
                        break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(0.005, remaining))

        # 2. Concurrency semaphore (acquired after token to avoid wasting tokens)
        if self._semaphore is not None:
            if not self._semaphore.acquire(blocking=True, timeout=self.max_wait):
                if acquired_token:
                    with self._lock:
                        self._tokens = min(self._capacity, self._tokens + 1.0)
                return False

        return True

    def release(self) -> None:
        """Release the concurrency slot. Safe to call with no semaphore configured."""
        if self._semaphore is not None:
            self._semaphore.release()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

Handler    = Callable[["Task"], Any]
Middleware = Callable[["Task", Handler], Any]


def logging_middleware(task: Task, next_fn: Handler) -> Any:
    """Built-in middleware: logs task start, finish, and duration."""
    log.info(
        "[%s] dispatch type=%s priority=%d attempt=%d",
        task.id[:8], task.type, task.priority, task.attempt,
    )
    start = time.perf_counter()
    try:
        result = next_fn(task)
        log.info(
            "[%s] succeeded  %.1fms",
            task.id[:8], (time.perf_counter() - start) * 1000,
        )
        return result
    except Exception as exc:
        log.warning(
            "[%s] failed     %.1fms  %s",
            task.id[:8], (time.perf_counter() - start) * 1000, exc,
        )
        raise


def metrics_middleware(counters: Dict[str, int]) -> Middleware:
    """
    Returns middleware that increments ``counters[task.type]`` on every call.

    Example::

        counts: Dict[str, int] = {}
        router = TaskRouter(middleware=[metrics_middleware(counts)])
    """
    def _mw(task: Task, next_fn: Handler) -> Any:
        counters[task.type] = counters.get(task.type, 0) + 1
        return next_fn(task)
    return _mw


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass
class Route:
    """Associates a match predicate with a handler and retry policy."""

    predicate: Callable[[Task], bool]
    handler: Handler
    retry_policy:    RetryPolicy              = field(default_factory=RetryPolicy)
    name:            str                      = ""
    timeout:         Optional[float]          = None  # per-attempt seconds; None = disabled
    circuit_breaker: Optional[CircuitBreaker] = None  # None = disabled
    rate_limiter:    Optional[RateLimiter]    = None  # None = disabled

    # ---- Convenience factories -------------------------------------------

    @classmethod
    def exact(cls, task_type: str, handler: Handler, **kw) -> "Route":
        """Match tasks whose ``type`` equals *task_type* exactly."""
        return cls(
            predicate=lambda t, _tt=task_type: t.type == _tt,
            handler=handler,
            name=kw.pop("name", f"exact:{task_type}"),
            **kw,
        )

    @classmethod
    def prefix(cls, prefix: str, handler: Handler, **kw) -> "Route":
        """Match tasks whose ``type`` starts with *prefix*."""
        return cls(
            predicate=lambda t, _p=prefix: t.type.startswith(_p),
            handler=handler,
            name=kw.pop("name", f"prefix:{prefix}"),
            **kw,
        )

    @classmethod
    def regex(cls, pattern: str, handler: Handler, **kw) -> "Route":
        """Match tasks whose ``type`` is matched by the regular expression."""
        compiled = re.compile(pattern)
        return cls(
            predicate=lambda t, _c=compiled: bool(_c.match(t.type)),
            handler=handler,
            name=kw.pop("name", f"regex:{pattern}"),
            **kw,
        )

    @classmethod
    def predicate_match(
        cls,
        predicate: Callable[[Task], bool],
        handler: Handler,
        **kw,
    ) -> "Route":
        """Match tasks that satisfy an arbitrary *predicate* function."""
        return cls(predicate=predicate, handler=handler,
                   name=kw.pop("name", "predicate"), **kw)

    @classmethod
    def wildcard(cls, handler: Handler, **kw) -> "Route":
        """Match every task — typically used as a catch-all fallback."""
        return cls(predicate=lambda _: True, handler=handler,
                   name=kw.pop("name", "wildcard"), **kw)


# ---------------------------------------------------------------------------
# TaskRouter
# ---------------------------------------------------------------------------

class TaskRouter:
    """
    Central dispatcher.  Routes tasks to the first matching handler,
    applies middleware, and manages retries, concurrency, and the DLQ.

    Parameters
    ----------
    workers:
        Size of the thread pool used by :meth:`submit` / :meth:`drain`.
    middleware:
        Stack of middleware applied to every task (outermost → innermost).
        Defaults to :func:`logging_middleware`.
    dead_letter:
        Callback invoked for tasks that exhaust all retry attempts.
        Defaults to accumulating results in :attr:`dead_letter_queue`.
    """

    def __init__(
        self,
        workers: int = 4,
        middleware: Optional[List[Middleware]] = None,
        dead_letter: Optional[Callable[[TaskResult], None]] = None,
        scheduler_tick: float = 0.1,
    ) -> None:
        self._routes: List[Route] = []
        self._middleware: List[Middleware] = (
            middleware if middleware is not None else [logging_middleware]
        )
        self._dead_letter_cb = dead_letter or self._default_dlq
        self._dlq: List[TaskResult] = []
        self._dlq_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="task-router"
        )
        # Dedicated pool for per-attempt timeout enforcement.
        # max_workers=None → auto-sized (always > outer pool) to prevent deadlock.
        self._timeout_executor = ThreadPoolExecutor(
            max_workers=None, thread_name_prefix="task-router-timeout"
        )
        self._heap: List[Task] = []
        self._heap_lock = threading.Lock()

        # Scheduler — (run_at, task) min-heap plus a cancellation set
        self._schedule_heap: List[Tuple[float, Task]] = []
        self._schedule_lock = threading.Lock()
        self._cancelled: set = set()
        self._scheduler_tick = scheduler_tick
        self._scheduler_stop: Optional[threading.Event] = None
        self._scheduler_thread: Optional[threading.Thread] = None

    # ---- Route registration ----------------------------------------------

    def add(self, route: Route) -> "TaskRouter":
        """Append a route.  Routes are evaluated in registration order."""
        self._routes.append(route)
        log.debug("Registered route %r", route.name)
        return self

    def route(
        self,
        task_type: str,
        *,
        retry: Optional[RetryPolicy] = None,
    ):
        """Decorator that registers an exact-match route for *task_type*."""
        def decorator(fn: Handler) -> Handler:
            kw: Dict[str, Any] = {}
            if retry is not None:
                kw["retry_policy"] = retry
            self.add(Route.exact(task_type, fn, **kw))
            return fn
        return decorator

    # ---- Internal helpers ------------------------------------------------

    def _match(self, task: Task) -> Optional[Route]:
        for route in self._routes:
            if route.predicate(task):
                return route
        return None

    def _build_chain(self, handler: Handler) -> Handler:
        """Wrap *handler* with the full middleware stack (outermost executed first)."""
        chain: Handler = handler
        for mw in reversed(self._middleware):
            outer = mw
            inner = chain
            chain = lambda t, _o=outer, _i=inner: _o(t, _i)
        return chain

    def _execute(self, task: Task, route: Route) -> TaskResult:
        """Execute a task through middleware + resilience layers + retry loop."""
        policy = route.retry_policy
        chain  = self._build_chain(route.handler)
        rl     = route.rate_limiter
        cb     = route.circuit_breaker
        t0     = time.perf_counter()

        # Rate limiter: acquired once per dispatch, held across all retry attempts.
        if rl is not None and not rl.acquire():
            log.warning("[%s] rate-limited route=%r", task.id[:8], route.name)
            return TaskResult(
                task=task,
                status=TaskStatus.REJECTED,
                error=RateLimitError(route.name),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            while True:
                # Circuit breaker: checked before every attempt.
                if cb is not None and not cb.allow_request():
                    log.warning("[%s] circuit open route=%r", task.id[:8], route.name)
                    return TaskResult(
                        task=task,
                        status=TaskStatus.REJECTED,
                        error=CircuitOpenError(route.name),
                        duration_ms=(time.perf_counter() - t0) * 1000,
                    )

                try:
                    if route.timeout is not None:
                        fut = self._timeout_executor.submit(chain, task)
                        try:
                            value = fut.result(timeout=route.timeout)
                        except TimeoutError:
                            raise HandlerTimeoutError(route.timeout)
                    else:
                        value = chain(task)

                    if cb is not None:
                        cb.record_success()
                    return TaskResult(
                        task=task,
                        status=TaskStatus.SUCCEEDED,
                        value=value,
                        duration_ms=(time.perf_counter() - t0) * 1000,
                    )

                except Exception as exc:
                    if cb is not None:
                        cb.record_failure()
                    task.attempt += 1
                    if policy.should_retry(task.attempt, exc):
                        delay = policy.delay_for(task.attempt)
                        log.info(
                            "[%s] retry %d/%d in %.2fs — %s",
                            task.id[:8], task.attempt, policy.max_attempts, delay, exc,
                        )
                        time.sleep(delay)
                    else:
                        status = (
                            TaskStatus.DEAD
                            if task.attempt >= policy.max_attempts
                            else TaskStatus.FAILED
                        )
                        result = TaskResult(
                            task=task,
                            status=status,
                            error=exc,
                            duration_ms=(time.perf_counter() - t0) * 1000,
                        )
                        if status == TaskStatus.DEAD:
                            self._dead_letter_cb(result)
                        return result
        finally:
            if rl is not None:
                rl.release()

    # ---- Dispatch --------------------------------------------------------

    def dispatch(self, task: Task) -> TaskResult:
        """Route and execute *task* synchronously.  Returns a :class:`TaskResult`."""
        route = self._match(task)
        if route is None:
            log.warning("No route matched task.type=%r id=%s", task.type, task.id[:8])
            return TaskResult(task=task, status=TaskStatus.REJECTED)
        return self._execute(task, route)

    def submit(self, task: Task) -> Future:
        """Submit *task* for asynchronous execution in the worker pool."""
        return self._executor.submit(self.dispatch, task)

    # ---- Priority queue --------------------------------------------------

    def enqueue(self, task: Task) -> None:
        """Push *task* onto the internal priority queue (thread-safe)."""
        with self._heap_lock:
            heapq.heappush(self._heap, task)

    def drain(self, max_tasks: Optional[int] = None) -> List[Future]:
        """
        Dequeue up to *max_tasks* (all if ``None``) in priority order and
        submit each to the worker pool.  Returns a list of :class:`Future` objects.
        """
        with self._heap_lock:
            tasks: List[Task] = []
            while self._heap and (max_tasks is None or len(tasks) < max_tasks):
                tasks.append(heapq.heappop(self._heap))
        return [self.submit(t) for t in tasks]

    # ---- Scheduled execution ---------------------------------------------

    def schedule(self, task: Task, *, run_at: float) -> str:
        """
        Schedule *task* to run at the given Unix timestamp.  Returns the
        task id (use with :meth:`cancel` to abort before it fires).

        The scheduler thread must be running — call :meth:`start` first
        (the context-manager form does this automatically).
        """
        task.scheduled_at = run_at
        with self._schedule_lock:
            heapq.heappush(self._schedule_heap, (run_at, task))
        return task.id

    def schedule_in(self, task: Task, *, delay: float) -> str:
        """Schedule *task* to run *delay* seconds from now."""
        return self.schedule(task, run_at=time.time() + delay)

    def cancel(self, task_id: str) -> bool:
        """
        Mark a scheduled task for cancellation.  Returns ``True`` if the
        task was still in the schedule heap, ``False`` if it had already
        fired (or was never scheduled).
        """
        with self._schedule_lock:
            present = any(t.id == task_id for _, t in self._schedule_heap)
            if present:
                self._cancelled.add(task_id)
        return present

    def start(self) -> "TaskRouter":
        """Start the background scheduler thread (idempotent)."""
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            return self
        self._scheduler_stop = threading.Event()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="task-router-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()
        return self

    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Stop the background scheduler thread."""
        if self._scheduler_stop is None or self._scheduler_thread is None:
            return
        self._scheduler_stop.set()
        if wait:
            self._scheduler_thread.join(timeout=timeout)
        self._scheduler_thread = None
        self._scheduler_stop = None

    def _scheduler_loop(self) -> None:
        """Background loop that submits scheduled tasks when their time arrives."""
        stop_event = self._scheduler_stop
        assert stop_event is not None
        while not stop_event.is_set():
            now = time.time()
            due: List[Task] = []
            next_at: Optional[float] = None

            with self._schedule_lock:
                while self._schedule_heap and self._schedule_heap[0][0] <= now:
                    _, task = heapq.heappop(self._schedule_heap)
                    if task.id in self._cancelled:
                        self._cancelled.discard(task.id)
                        continue
                    due.append(task)
                if self._schedule_heap:
                    next_at = self._schedule_heap[0][0]

            for task in due:
                try:
                    self.submit(task)
                except RuntimeError:
                    # Executor already shut down — drop remaining tasks.
                    return

            if next_at is None:
                wait_for = self._scheduler_tick
            else:
                wait_for = max(0.001, min(next_at - time.time(), self._scheduler_tick))
            stop_event.wait(wait_for)

    # ---- Dead-letter queue -----------------------------------------------

    def _default_dlq(self, result: TaskResult) -> None:
        with self._dlq_lock:
            self._dlq.append(result)
        log.error(
            "Dead-letter: id=%s type=%s attempts=%d error=%s",
            result.task.id[:8], result.task.type, result.task.attempt, result.error,
        )

    @property
    def dead_letter_queue(self) -> List[TaskResult]:
        """Snapshot of all permanently failed tasks."""
        with self._dlq_lock:
            return list(self._dlq)

    # ---- Lifecycle -------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the scheduler (if running) and worker pools."""
        self.stop(wait=wait)
        self._executor.shutdown(wait=wait)
        self._timeout_executor.shutdown(wait=False)  # abandon any leaked timeout threads

    def __enter__(self) -> "TaskRouter":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"TaskRouter(routes={len(self._routes)}, "
            f"workers={self._executor._max_workers})"
        )
