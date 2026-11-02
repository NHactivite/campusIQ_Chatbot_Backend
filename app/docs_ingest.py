"""
Indexes any admin-uploaded university content: notices, circulars, regulations,
features, glance cards, linkages (PDF, photo, or plain text).
"""
import os
import tempfile

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from app.events_ingest import INGEST_IMAGES, _download, _ext, _fmt_date, _lock
from app.rag import (
    COLLECTION,
    IMAGE_EXTS,
    _store,
    ensure_collection,
    extract_pages,
    get_client,
    ocr_image_bytes,
)

VALID_CATEGORIES = {"regulations", "exams", "scholarships", "admission", "facilities", "notices", "other"}


def delete_source(source_id: str):
    ensure_collection()
    get_client().delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="sourceId", match=MatchValue(value=source_id))])
        ),
    )


def sync_document(d: dict):
    """Re-index one notice/record from scratch. Runs in a background thread."""
    sid = str(d.get("_id") or d.get("id"))
    title = d.get("title", "Notice")
    category = d.get("category") if d.get("category") in VALID_CATEGORIES else "other"
    extra = {"sourceId": sid}

    with _lock:
        try:
            ensure_collection()
            delete_source(sid)

            if d.get("isPublished") is False:
                print(f"[docs] {title!r} is a draft, removed from the chatbot index")
                return

            date = _fmt_date(d.get("noticeDate")) if d.get("noticeDate") else ""

            # 1) typed text (and title/date)
            body = (d.get("text") or "").strip()
            head = f"Title: {title}" + (f"\nDate: {date}" if date else "")
            if body or not d.get("files"):
                n = _store([(1, f"{head}\n{body}".strip())], title, category, f"kb:{sid}:text", "", extra)
                print(f"[docs] {title!r}: text -> {n} chunks")

            # 2) attached files
            for m in d.get("files") or []:
                name = m.get("originalName") or "file"
                ext = _ext(m)
                try:
                    if ext == ".pdf":
                        with tempfile.TemporaryDirectory() as tmp:
                            path = os.path.join(tmp, f"kb_{sid}_{m['_id']}.pdf".replace(":", "_"))
                            with open(path, "wb") as f:
                                f.write(_download(m["url"]))
                            pages = extract_pages(path)
                        n = _store(pages, title, category, f"kb:{sid}:{m['_id']}", m["url"], extra)
                    elif ext in IMAGE_EXTS and INGEST_IMAGES:
                        key = f"kb_{sid}_{m['_id']}".replace(":", "_")
                        text = ocr_image_bytes(_download(m["url"]), IMAGE_EXTS[ext], key)
                        if not text:
                            print(f"[docs]   {name}: no readable text")
                            continue
                        n = _store([(1, f"{head}\n{text}")], title, category,
                                   f"kb:{sid}:{m['_id']}", m["url"], extra)
                    else:
                        print(f"[docs]   skipped {name} (only PDF and images are indexed)")
                        continue
                    print(f"[docs]   {name} -> {n} chunks")
                except Exception as ex:
                    print(f"[docs]   FAILED {name}: {type(ex).__name__}: {ex}")
        except Exception as ex:
            print(f"[docs] sync failed for {title!r}: {type(ex).__name__}: {ex}")