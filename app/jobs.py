"""One worker thread runs all indexing jobs. Web requests only add to the queue."""
import queue
import threading

from app import answer_cache

_q: "queue.Queue[str]" = queue.Queue()
_pending: dict = {}
_lock = threading.Lock()


def enqueue(key: str, fn, *args):
    """Queue a job. If the same key is already waiting, the newest version replaces it."""
    with _lock:
        already = key in _pending
        _pending[key] = (fn, args)
    if not already:
        _q.put(key)


def _worker():
    while True:
        key = _q.get()
        with _lock:
            fn, args = _pending.pop(key)
        try:
            fn(*args)
        except Exception as e:
            print(f"[jobs] {key} failed: {type(e).__name__}: {e}")
        finally:
            answer_cache.invalidate()  # knowledge changed, so old answers may be wrong


threading.Thread(target=_worker, daemon=True, name="index-worker").start()