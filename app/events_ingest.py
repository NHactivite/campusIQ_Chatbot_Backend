"""
Keeps the chatbot index in sync with events created in the RDC & IIC admin panel.
Indexes: event details, PDF documents, and photos (via Gemini OCR).
"""
import os
import tempfile
import threading
from datetime import datetime

import httpx
from dotenv import load_dotenv
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from app.rag import (
    COLLECTION,
    IMAGE_EXTS,
    _store,
    clear_no_text,
    ensure_collection,
    extract_pages,
    get_client,
    mark_no_text,
    ocr_image_bytes,
)

load_dotenv()

PORTAL_URL = os.getenv("PORTAL_URL", "http://localhost:3000").rstrip("/")
INGEST_IMAGES = os.getenv("INGEST_EVENT_IMAGES", "true").lower() == "true"
MAX_BYTES = 15 * 1024 * 1024

# one sync at a time: avoids OCR rate limits and memory spikes
_lock = threading.Lock()


def _fmt_date(d) -> str:
    try:
        return datetime.fromisoformat(str(d).replace("Z", "+00:00")).strftime("%d %B %Y")
    except Exception:
        return str(d or "")


def _event_text(e: dict) -> str:
    lines = [f"Event: {e.get('title', '')}"]
    start, end = _fmt_date(e.get("eventDate")), _fmt_date(e.get("endDate"))
    lines.append(f"Date: {start}" + (f" to {end}" if end and end != start else ""))
    for label, key in [
        ("Category", "category"),
        ("Venue", "venue"),
        ("Organised by", "organiser"),
        ("Participants", "participants"),
        ("Description", "description"),
        ("Outcome", "outcome"),
    ]:
        v = e.get(key)
        if v not in (None, ""):
            lines.append(f"{label}: {str(v).strip()}")
    return "\n".join(lines)


def _download(url: str) -> bytes:
    buf = bytearray()
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            buf += chunk
            if len(buf) > MAX_BYTES:
                raise ValueError("file too large")
    return bytes(buf)


def _ext(m: dict) -> str:
    for src in (m.get("originalName"), m.get("url")):
        if src:
            ext = os.path.splitext(str(src).split("?")[0])[1].lower()
            if ext:
                return ext
    fmt = (m.get("format") or "").lower()
    return f".{fmt}" if fmt else ""


def delete_event(event_id: str):
    ensure_collection()
    get_client().delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="eventId", match=MatchValue(value=event_id))])
        ),
    )
    clear_no_text("eventId", event_id)

def sync_event(e: dict):
    """Re-index one event from scratch. Runs in the background worker."""
    event_id = str(e.get("_id") or e.get("id"))
    title = e.get("title", "Event")
    extra = {"eventId": event_id}

    with _lock:
        try:
            ensure_collection()
            delete_event(event_id)  # also clears the old "no text" markers

            if e.get("isPublished") is False:
                print(f"[events] {title!r} is a draft, removed from the chatbot index")
                return

            page_url = f"{PORTAL_URL}/dashboard/{e.get('slug', '')}"

            # 1) event details
            n = _store([(1, _event_text(e))], title, "events", f"event:{event_id}", page_url, extra)
            print(f"[events] {title!r}: details -> {n} chunks")

            # 2) PDF documents
            for m in e.get("documents") or []:
                name = m.get("originalName") or "document"
                try:
                    if _ext(m) != ".pdf":
                        print(f"[events]   skipped {name} (only PDF is indexed)")
                        continue
                    with tempfile.TemporaryDirectory() as tmp:
                        path = os.path.join(tmp, f"event_{event_id}_{m['_id']}.pdf")
                        with open(path, "wb") as f:
                            f.write(_download(m["url"]))
                        pages = extract_pages(path)
                    n = _store(pages, f"{title} - {name}", "events",
                               f"event:{event_id}:doc:{m['_id']}", m["url"], extra)
                    print(f"[events]   {name} -> {n} chunks")
                except Exception as ex:
                    print(f"[events]   FAILED {name}: {type(ex).__name__}: {ex}")

            # 3) photos (OCR)
            if INGEST_IMAGES:
                for m in e.get("images") or []:
                    name = m.get("originalName") or "photo"
                    try:
                        ext = _ext(m) or ".jpg"
                        mime = IMAGE_EXTS.get(ext)
                        if not mime:
                            print(f"[events]   skipped {name} (unsupported image type)")
                            continue
                        text = ocr_image_bytes(_download(m["url"]), mime, f"event_{event_id}_{m['_id']}")
                        if not text:
                            # OCR ran fine: the photo simply has no text. Remember that.
                            mark_no_text(f"event:{event_id}:img:{m['_id']}", "eventId", event_id)
                            print(f"[events]   {name}: no readable text")
                            continue
                        text = f"Photo from event '{title}' ({_fmt_date(e.get('eventDate'))}):\n{text}"
                        n = _store([(1, text)], f"{title} - photo", "events",
                                   f"event:{event_id}:img:{m['_id']}", m["url"], extra)
                        print(f"[events]   {name} -> {n} chunks")
                    except Exception as ex:
                        print(f"[events]   FAILED {name}: {type(ex).__name__}: {ex}")
        except Exception as ex:
            print(f"[events] sync failed for {title!r}: {type(ex).__name__}: {ex}")