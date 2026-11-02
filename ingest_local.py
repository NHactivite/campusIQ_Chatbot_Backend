"""
Index PDFs and notice photos into Qdrant (Cloud if QDRANT_URL is set, else local folder).

If you use the local folder (no QDRANT_URL), stop uvicorn first: it can be opened
by one process only. Qdrant Cloud has no such limit.

Usage:
    python ingest_local.py docs --category notices                 # whole folder
    python ingest_local.py docs\\exam_notice.pdf --title "Exam Notice 2026" --category exams
    python ingest_local.py docs\\photo1.jpg --title "Hostel Notice" --category notices
"""
import argparse
import os

from app.rag import IMAGE_EXTS, ingest_file

SUPPORTED = {".pdf", *IMAGE_EXTS.keys()}

parser = argparse.ArgumentParser()
parser.add_argument("path", help="a PDF/photo file, or a folder containing them")
parser.add_argument("--title", default=None, help="title shown in citations (single file only)")
parser.add_argument("--category", default="other",
                    help="regulations | exams | scholarships | admission | facilities | notices | other")
parser.add_argument("--id", default=None, help="document id (single file only)")
args = parser.parse_args()

if os.path.isdir(args.path):
    files = sorted(
        os.path.join(args.path, f)
        for f in os.listdir(args.path)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    )
else:
    files = [args.path]

if not files:
    raise SystemExit("No PDF or image files found.")

ok, failed = 0, 0
for f in files:
    title = args.title if (args.title and len(files) == 1) else os.path.splitext(os.path.basename(f))[0]
    print(f"\n== {f}")
    try:
        count = ingest_file(
            f, title=title, category=args.category,
            document_id=args.id if len(files) == 1 else None,
        )
        print(f"  stored {count} chunks for '{title}'")
        ok += 1
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        failed += 1

print(f"\nDone. {ok} file(s) indexed, {failed} failed.")