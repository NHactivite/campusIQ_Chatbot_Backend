"""
Indexes any admin-uploaded university content: notices, circulars, regulations,
features, glance cards, linkages (PDF, photo, or plain text).

Safe re-sync: nothing is deleted up front. Each file's chunks are replaced by _store only
after the new chunks are ready, and at the end only chunks of files that no longer exist
on the record are removed. A restart or an error in the middle never wipes the old index.
"""
import os
import tempfile

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny, MatchValue

from app.events_ingest import INGEST_IMAGES, _download, _ext, _fmt_date, _lock
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
VALID_CATEGORIES = {"regulations", "exams", "scholarships", "admission", "facilities", "notices", "other"}


def delete_source(source_id: str):
    ensure_collection()
    get_client().delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="sourceId", match=MatchValue(value=source_id))])
        ),
    )
    clear_no_text("sourceId", source_id)


def _delete_stale(source_id: str, keep_ids: set[str]):
    """Remove chunks of this source whose documentId is not in keep_ids (removed files/text)."""
    get_client().delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="sourceId", match=MatchValue(value=source_id))],
                must_not=[FieldCondition(key="documentId", match=MatchAny(any=sorted(keep_ids)))],
            )
        ),
    )


def sync_document(d: dict):
    """Re-index one notice/record. Runs in the background worker."""
    sid = str(d.get("_id") or d.get("id"))
    title = d.get("title", "Notice")
    category = d.get("category") if d.get("category") in VALID_CATEGORIES else "other"
    extra = {"sourceId": sid}

    with _lock:
        try:
            ensure_collection()

            if d.get("isPublished") is False:
                delete_source(sid)
                print(f"[docs] {title!r} is a draft, removed from the chatbot index")
                return

            # reset "no readable text" markers; they are re-set below if a photo still has none
            clear_no_text("sourceId", sid)

            keep: set[str] = set()  # documentIds that belong to the current version of this record
            date = _fmt_date(d.get("noticeDate")) if d.get("noticeDate") else ""

            # 1) typed text (and title/date)
            body = (d.get("text") or "").strip()
            head = f"Title: {title}" + (f"\nDate: {date}" if date else "")
            if body or not d.get("files"):
                doc_id = f"kb:{sid}:text"
                keep.add(doc_id)
                n = _store([(1, f"{head}\n{body}".strip())], title, category, doc_id, "", extra)
                print(f"[docs] {title!r}: text -> {n} chunks")

            # 2) attached files
            for m in d.get("files") or []:
                name = m.get("originalName") or "file"
                ext = _ext(m)
                try:
                    doc_id = f"kb:{sid}:{m['_id']}"
                    if ext == ".pdf":
                        keep.add(doc_id)  # a failed file keeps its old chunks
                        with tempfile.TemporaryDirectory() as tmp:
                            path = os.path.join(tmp, f"kb_{sid}_{m['_id']}.pdf".replace(":", "_"))
                            with open(path, "wb") as f:
                                f.write(_download(m["url"]))
                            pages = extract_pages(path)
                        n = _store(pages, title, category, doc_id, m["url"], extra)
                    elif ext in IMAGE_EXTS and INGEST_IMAGES:
                        keep.add(doc_id)
                        key = f"kb_{sid}_{m['_id']}".replace(":", "_")
                        text = ocr_image_bytes(_download(m["url"]), IMAGE_EXTS[ext], key)
                        if not text:
                            mark_no_text(doc_id, "sourceId", sid)
                            print(f"[docs]   {name}: no readable text")
                            continue
                        n = _store([(1, f"{head}\n{text}")], title, category, doc_id, m["url"], extra)
                    else:
                        print(f"[docs]   skipped {name} (only PDF and images are indexed)")
                        continue
                    print(f"[docs]   {name} -> {n} chunks")
                except Exception as ex:
                    print(f"[docs]   FAILED {name}: {type(ex).__name__}: {ex}")

            # 3) drop chunks of files/text that were removed from the record
            if keep:
                _delete_stale(sid, keep)
        except Exception as ex:
            print(f"[docs] sync failed for {title!r}: {type(ex).__name__}: {ex}")
    """Re-index one notice/record. Runs in the background worker."""
    sid = str(d.get("_id") or d.get("id"))
    title = d.get("title", "Notice")
    category = d.get("category") if d.get("category") in VALID_CATEGORIES else "other"
    extra = {"sourceId": sid}

    with _lock:
        try:
            ensure_collection()

            if d.get("isPublished") is False:
                delete_source(sid)
                print(f"[docs] {title!r} is a draft, removed from the chatbot index")
                return

            keep: set[str] = set()  # documentIds that belong to the current version of this record
            date = _fmt_date(d.get("noticeDate")) if d.get("noticeDate") else ""

            # 1) typed text (and title/date)
            body = (d.get("text") or "").strip()
            head = f"Title: {title}" + (f"\nDate: {date}" if date else "")
            if body or not d.get("files"):
                doc_id = f"kb:{sid}:text"
                keep.add(doc_id)
                n = _store([(1, f"{head}\n{body}".strip())], title, category, doc_id, "", extra)
                print(f"[docs] {title!r}: text -> {n} chunks")

            # 2) attached files
            for m in d.get("files") or []:
                name = m.get("originalName") or "file"
                ext = _ext(m)
                try:
                    doc_id = f"kb:{sid}:{m['_id']}"
                    if ext == ".pdf":
                        keep.add(doc_id)  # a failed file keeps its old chunks
                        with tempfile.TemporaryDirectory() as tmp:
                            path = os.path.join(tmp, f"kb_{sid}_{m['_id']}.pdf".replace(":", "_"))
                            with open(path, "wb") as f:
                                f.write(_download(m["url"]))
                            pages = extract_pages(path)
                        n = _store(pages, title, category, doc_id, m["url"], extra)
                    elif ext in IMAGE_EXTS and INGEST_IMAGES:
                        keep.add(doc_id)
                        key = f"kb_{sid}_{m['_id']}".replace(":", "_")
                        text = ocr_image_bytes(_download(m["url"]), IMAGE_EXTS[ext], key)
                        if not text:
                            print(f"[docs]   {name}: no readable text")
                            continue
                        n = _store([(1, f"{head}\n{text}")], title, category, doc_id, m["url"], extra)
                    else:
                        print(f"[docs]   skipped {name} (only PDF and images are indexed)")
                        continue
                    print(f"[docs]   {name} -> {n} chunks")
                except Exception as ex:
                    print(f"[docs]   FAILED {name}: {type(ex).__name__}: {ex}")

            # 3) drop chunks of files/text that were removed from the record
            if keep:
                _delete_stale(sid, keep)
        except Exception as ex:
            print(f"[docs] sync failed for {title!r}: {type(ex).__name__}: {ex}")