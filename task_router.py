"""
Task Routing System
===================
Routes tasks to registered handlers by type, prefix, regex, or custom predicate.

Features:
  • Priority queues            – higher-priority tasks run first
  • Retry with backoff/jitter  – configurable per route
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
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    name: str = ""

    # ---- Convenience factories -------------------------------------------

    @classmethod
    def exact(cls, task_type: str, handler: Handler, **kw) -> "Route":
        """Match tasks whose ``type`` equals *task_type* exactly."""
        return cls(
            predicate=lambda t, _tt=task_type: t.type == _tt,
            handler=handler,
            name=f"exact:{task_type}",
            **kw,
        )

    @classmethod
    def prefix(cls, prefix: str, handler: Handler, **kw) -> "Route":
        """Match tasks whose ``type`` starts with *prefix*."""
        return cls(
            predicate=lambda t, _p=prefix: t.type.startswith(_p),
            handler=handler,
            name=f"prefix:{prefix}",
            **kw,
        )

    @classmethod
    def regex(cls, pattern: str, handler: Handler, **kw) -> "Route":
        """Match tasks whose ``type`` is matched by the regular expression."""
        compiled = re.compile(pattern)
        return cls(
            predicate=lambda t, _c=compiled: bool(_c.match(t.type)),
            handler=handler,
            name=f"regex:{pattern}",
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
        return cls(predicate=predicate, handler=handler, name="predicate", **kw)

    @classmethod
    def wildcard(cls, handler: Handler, **kw) -> "Route":
        """Match every task — typically used as a catch-all fallback."""
        return cls(predicate=lambda _: True, handler=handler, name="wildcard", **kw)


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
        self._heap: List[Task] = []
        self._heap_lock = threading.Lock()

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
        """Execute a task through middleware + retry loop."""
        policy = route.retry_policy
        chain  = self._build_chain(route.handler)
        t0     = time.perf_counter()

        while True:
            attempt_start = time.perf_counter()
            try:
                value = chain(task)
                return TaskResult(
                    task=task,
                    status=TaskStatus.SUCCEEDED,
                    value=value,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            except Exception as exc:
                task.attempt += 1
                if policy.should_retry(task.attempt, exc):
                    delay = policy.delay_for(task.attempt)
                    log.info(
                        "[%s] retry %d/%d in %.2fs — %s",
                        task.id[:8], task.attempt, policy.max_attempts, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    # DEAD = exhausted retries; FAILED = non-retryable exception
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
        """Shut down the worker pool."""
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "TaskRouter":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"TaskRouter(routes={len(self._routes)}, "
            f"workers={self._executor._max_workers})"
        )
