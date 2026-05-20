"""
Task Routing System — Live Demo
================================
Run with:  python demo.py

Demonstrates:
  1. Routing by exact type, prefix, regex, and wildcard fallback
  2. Retry with exponential back-off (simulated payment gateway failure)
  3. Priority queue (high-priority items dispatched before low-priority ones)
  4. Dead-letter queue inspection
  5. Metrics middleware
  6. Async submit
  7. Scheduled execution (run_at / delay) and cancellation
"""
import logging
import time

from task_router import (
    Task,
    TaskRouter,
    Route,
    RetryPolicy,
    NO_RETRY,
    FAST_RETRY,
    metrics_middleware,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_email(task: Task):
    to      = task.payload.get("to", "?")
    subject = task.payload.get("subject", "")
    print(f"    [EMAIL]   → {to}  subject={subject!r}")
    return {"sent": True, "to": to}


def handle_sms(task: Task):
    number = task.payload.get("number", "?")
    body   = task.payload.get("body", "")
    print(f"    [SMS]     → {number}  body={body!r}")
    return {"sms_id": "sms-abc-123"}


def handle_push(task: Task):
    platform = task.type.split(".")[1].upper()   # apns / fcm
    token    = task.payload.get("token", "?")
    print(f"    [PUSH/{platform}] → token={token}")
    return {"delivered": True}


def handle_payment(task: Task):
    amount = task.payload.get("amount_cents", 0)
    print(f"    [PAYMENT] → ${amount / 100:.2f}  attempt={task.attempt + 1}")
    if task.payload.get("simulate_failure") and task.attempt < 2:
        raise RuntimeError("Payment gateway timeout — will retry")
    return {"transaction_id": "txn-999", "amount_cents": amount}


def handle_fallback(task: Task):
    print(f"    [MISC]    → unmatched type={task.type!r}")
    return None


order_log: list = []

def track_priority(task: Task):
    order_log.append(task.priority)
    print(f"    [TRACKED] priority={task.priority}")


def scheduled_handler(task: Task):
    elapsed = time.time() - task.payload["queued_at"]
    print(f"    [SCHEDULED] payload={task.payload['label']!r}  fired after {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Build the router
# ---------------------------------------------------------------------------

counters: dict = {}

router = TaskRouter(
    workers=4,
    middleware=[metrics_middleware(counters)],
)

router.add(Route.exact("email.send",      handle_email,   retry_policy=NO_RETRY))
router.add(Route.exact("email.bounce",    handle_email,   retry_policy=NO_RETRY))
router.add(Route.prefix("sms.",           handle_sms,     retry_policy=NO_RETRY))
router.add(Route.regex(r"push\.(apns|fcm)\.send", handle_push, retry_policy=NO_RETRY))
router.add(Route.exact(
    "payment.charge",
    handle_payment,
    retry_policy=RetryPolicy(max_attempts=3, base_delay=0.05, max_delay=1.0, jitter=False),
))
router.add(Route.exact("tracked",   track_priority,    retry_policy=NO_RETRY))
router.add(Route.exact("scheduled", scheduled_handler, retry_policy=NO_RETRY))
router.add(Route.wildcard(handle_fallback, retry_policy=NO_RETRY))


# ---------------------------------------------------------------------------
# 1. Synchronous dispatch
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  1. Synchronous dispatch")
print("=" * 60)

tasks = [
    Task("email.send",     priority=5,  payload={"to": "alice@example.com", "subject": "Welcome!"}),
    Task("sms.otp",        priority=10, payload={"number": "+15550001234",   "body": "Your OTP: 882991"}),
    Task("sms.alert",      priority=8,  payload={"number": "+15559998888",   "body": "Server down!"}),
    Task("push.apns.send", priority=3,  payload={"token": "tok-abc"}),
    Task("push.fcm.send",  priority=3,  payload={"token": "tok-xyz"}),
    Task("payment.charge", priority=7,  payload={"amount_cents": 4999, "simulate_failure": True}),
    Task("webhook.send",   priority=1,  payload={"url": "https://example.com/hook"}),
    Task("unknown.thing",  priority=0,  payload={}),
]

for t in tasks:
    result = router.dispatch(t)
    err = f"  error={result.error}" if result.error else ""
    print(f"    → status={result.status.value:<10} type={t.type}{err}")


# ---------------------------------------------------------------------------
# 2. Priority queue
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  2. Priority queue — items drain in high→low priority order")
print("=" * 60)

for priority in [1, 5, 3, 10, 2]:
    router.enqueue(Task(type="tracked", priority=priority))

futures = router.drain()
[f.result() for f in futures]
print(f"    Dispatch order: {order_log}")
expected = sorted(order_log, reverse=True)
check = "✓ correct" if order_log == expected else "✗ unexpected"
print(f"    Expected:       {expected}  {check}")


# ---------------------------------------------------------------------------
# 3. Dead-letter queue
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  3. Dead-letter queue")
print("=" * 60)

dlq = router.dead_letter_queue
if dlq:
    for r in dlq:
        print(
            f"    id={r.task.id[:8]}  type={r.task.type:<18}"
            f"  attempts={r.task.attempt}  error={r.error}"
        )
else:
    print("    (empty — payment succeeded after retries)")


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  4. Metrics (calls per task type)")
print("=" * 60)

for task_type, count in sorted(counters.items()):
    print(f"    {task_type:<25} {count:>3} call(s)")


# ---------------------------------------------------------------------------
# 5. Async submit
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  5. Async submit")
print("=" * 60)

fut = router.submit(Task("email.send", payload={"to": "bob@example.com", "subject": "Async!"}))
result = fut.result(timeout=5)
print(f"    Future resolved → status={result.status.value}  value={result.value}")


# ---------------------------------------------------------------------------
# 6. Scheduled execution
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  6. Scheduled execution")
print("=" * 60)

router.start()           # start the scheduler thread

now = time.time()
router.schedule_in(Task(type="scheduled", payload={"label": "alpha", "queued_at": now}), delay=0.15)
router.schedule_in(Task(type="scheduled", payload={"label": "beta",  "queued_at": now}), delay=0.30)
router.schedule_in(Task(type="scheduled", payload={"label": "gamma", "queued_at": now}), delay=0.45)

# Cancel the middle one before it fires
cancel_id = router.schedule_in(
    Task(type="scheduled", payload={"label": "CANCELLED", "queued_at": now}),
    delay=0.40,
)
cancelled = router.cancel(cancel_id)
print(f"    Pre-cancellation result for 'CANCELLED' task: {cancelled}")

# Wait for all scheduled tasks to fire
time.sleep(0.7)


router.shutdown()
print("\n✓ Demo complete.\n")
