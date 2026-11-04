"""Small in-memory cache for repeated, standalone questions."""
import re
import threading
import time
from collections import OrderedDict

TTL = 3600  # one hour
MAX = 200
_lock = threading.Lock()
_data: "OrderedDict[tuple, tuple]" = OrderedDict()
_version = 0


def _key(question: str, lang: str):
    q = re.sub(r"\s+", " ", question.lower()).strip(" ?!.।")
    return (_version, lang, q)


def get(question: str, lang: str):
    k = _key(question, lang)
    with _lock:
        hit = _data.get(k)
        if hit and time.time() - hit[0] < TTL:
            _data.move_to_end(k)
            return hit[1], hit[2]  # answer text, source items
        _data.pop(k, None)
    return None


def put(question: str, lang: str, answer: str, sources: list):
    k = _key(question, lang)
    with _lock:
        _data[k] = (time.time(), answer, sources)
        while len(_data) > MAX:
            _data.popitem(last=False)


def invalidate():
    """Call after any index change; old answers stop matching."""
    global _version
    with _lock:
        _version += 1
        _data.clear()