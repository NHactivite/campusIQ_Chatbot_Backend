"""
rag.py - embeddings + Qdrant + PDF/photo ingestion (Gemini OCR) + search for CampusIQ.

.env settings:
    EMBED_BACKEND=bge            # "bge" (local BAAI/bge-m3) or "gemini"
    GEMINI_API_KEY=...           # needed for OCR and for EMBED_BACKEND=gemini
    GEMINI_OCR_MODEL=gemini-3.1-flash-lite   # tried first; the others in OCR_MODELS are fallbacks
    GEMINI_EMBED_MODEL=models/gemini-embedding-001
    QDRANT_URL=...               # leave empty to use a local folder
    QDRANT_API_KEY=...
    QDRANT_PATH=./qdrant_data
    SCORE_THRESHOLD=0.5          # tune with your own test questions
    OCR_CACHE_DIR=./ocr_cache
"""
import base64
import os
import time
import uuid

import pymupdf as fitz  # PyMuPDF
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

load_dotenv()

# OCR models, tried in this order. If one is overloaded (503), out of quota (429) or does
# not exist (404), the next one is used. Remove names that give 404 for your key.
OCR_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3-flash",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
]

COLLECTION = "campusiq"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "bge").lower()
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
OCR_MODEL = os.getenv("GEMINI_OCR_MODEL", OCR_MODELS[0])
OCR_CACHE_DIR = os.getenv("OCR_CACHE_DIR", "./ocr_cache")

IMAGE_EXTS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL text in this image exactly as written. "
    "The text may be Assamese (অসমীয়া), English, or both. Rules:\n"
    "- Include EVERYTHING: large decorative headings, banners, dates in big or coloured "
    "boxes, text inside shapes, small print, phone numbers and email addresses.\n"
    "- Do NOT translate the text. Keep each language as it appears. "
    "The only addition allowed is the Western-digit copy described below.\n"
    "- Copy every date, year, amount, number and name character by character. "
    "Look at each digit carefully (Assamese digits such as ৩ and ৬, or ১ and ৭, look alike). "
    "Write dates and years exactly as printed.\n"
    "- After every date, year, amount or phone number written in Assamese digits, add the "
    "same value in brackets using Western digits 0-9, for example: ১৫ নবেম্বৰ ২০২৬ (15 November 2026).\n"
    "- Keep the reading order and line breaks; write tables and lists row by row.\n"
    "- Do not add comments, explanations or markdown.\n"
    "- If a part is unreadable write [unclear] instead of guessing.\n"
    "- If the image has no readable text, output exactly: NO_TEXT"
)


# ---------------- Qdrant client ----------------
_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            # Local folder mode: only ONE process can open this folder at a time.
            _client = QdrantClient(path=QDRANT_PATH)
    return _client


# ---------------- Embeddings ----------------
_bge_model = None
_gemini_emb = None


def embed(texts: list[str], is_query: bool = False) -> list[list[float]]:
    global _bge_model, _gemini_emb

    if EMBED_BACKEND == "gemini":
        if _gemini_emb is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            _gemini_emb = GoogleGenerativeAIEmbeddings(
                model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
        if is_query:
            return [_gemini_emb.embed_query(t) for t in texts]
        return _gemini_emb.embed_documents(texts)

    if _bge_model is None:
        from sentence_transformers import SentenceTransformer

        _bge_model = SentenceTransformer("BAAI/bge-m3")
    return _bge_model.encode(texts, normalize_embeddings=True).tolist()


def ensure_collection():
    client = get_client()
    if not client.collection_exists(COLLECTION):
        dim = len(embed(["dimension check"])[0])
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    # Qdrant Cloud needs an index on every field used in a filter.
    for field in ("documentId", "eventId", "sourceId"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:  # already exists, or local mode (indexes have no effect there)
            print(f"[rag] payload index '{field}' skipped: {str(e)[:80]}")


# ---------------- OCR (Gemini, with model fallback) ----------------
_ocr_llms = {}
_ocr_good_model = None  # the model that worked last; it is tried first next time


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _cache_path(key: str) -> str:
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    return os.path.join(OCR_CACHE_DIR, key + ".txt")


def _get_ocr_llm(model_name: str):
    if model_name not in _ocr_llms:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _ocr_llms[model_name] = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    return _ocr_llms[model_name]


def _ocr_model_order() -> list[str]:
    first = _ocr_good_model or OCR_MODEL
    rest = [m for m in [OCR_MODEL, *OCR_MODELS] if m != first]
    return [first] + list(dict.fromkeys(rest))


def ocr_image_bytes(data: bytes, mime: str, cache_key: str) -> str:
    """OCR one image with Gemini, trying several models. Result is cached in
    ocr_cache/<cache_key>.txt. If you edit the .txt file by hand, your text is used."""
    global _ocr_good_model
    path = _cache_path(cache_key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    from langchain_core.messages import HumanMessage

    b64 = base64.b64encode(data).decode("ascii")
    msg = HumanMessage(
        content=[
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
        ]
    )

    text, errors = None, []
    for round_no in range(2):  # two full passes over the model list
        for model_name in _ocr_model_order():
            for attempt in range(2):  # one quick retry for temporary 503s
                try:
                    text = _content_to_text(_get_ocr_llm(model_name).invoke([msg]).content).strip()
                    _ocr_good_model = model_name
                    print(f"  [ocr] done with {model_name}")
                    break
                except Exception as e:
                    err = str(e)
                    errors.append(f"{model_name}: {err[:120]}")
                    print(f"  [ocr] {model_name} failed ({type(e).__name__}: {err[:100]})")
                    if "404" in err or "NOT_FOUND" in err:
                        break  # model name does not exist: next model
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        break  # quota used up: next model
                    if attempt == 0:
                        time.sleep(5)  # 503 / network: retry the same model once
            if text is not None:
                break
        if text is not None:
            break
        if round_no == 0:
            print("  [ocr] all models failed, waiting 30s and trying the list again ...")
            time.sleep(30)

    if text is None:
        raise RuntimeError("OCR failed on every model:\n  " + "\n  ".join(errors[-6:]))

    if text.upper() == "NO_TEXT":
        text = ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------- Extraction + chunking ----------------
_WATERMARKS = ("scanned by camscanner", "camscanner", "scanned with", "scanned by")


def _real_text(raw: str) -> str:
    """Page text with scanner watermark lines removed."""
    lines = [ln for ln in raw.splitlines() if not any(w in ln.lower() for w in _WATERMARKS)]
    return "\n".join(lines).strip()


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Text per page. Pages with almost no real text (scans) are OCR'd."""
    pages = []
    base = os.path.basename(pdf_path)
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf):
            text = _real_text(page.get_text())
            if len(text) < 50:
                print(f"  page {i + 1}: no real text layer, running OCR ...")
                png = page.get_pixmap(dpi=200).tobytes("png")
                text = ocr_image_bytes(png, "image/png", f"{base}.p{i + 1}")
            pages.append((i + 1, text))
    return pages
def extract_image(image_path: str) -> list[tuple[int, str]]:
    ext = os.path.splitext(image_path)[1].lower()
    with open(image_path, "rb") as f:
        data = f.read()
    print("  running OCR ...")
    text = ocr_image_bytes(data, IMAGE_EXTS[ext], os.path.basename(image_path))
    return [(1, text)]


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    chunks, i = [], 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if len(piece) > 20:
            chunks.append(piece)
        i += size - overlap
    return chunks


# ---------------- Ingest / delete ----------------
def delete_document(document_id: str):
    get_client().delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="documentId", match=MatchValue(value=document_id))]
            )
        ),
    )


def _store(pages: list[tuple[int, str]], title, category, document_id, file_url, extra: dict | None = None) -> int:
    texts, metas = [], []
    for page_no, text in pages:
        for c in chunk_text(text):
            texts.append(c)
            metas.append({"page": page_no})

    if not texts:
        print("  no readable text found (blurry photo? check the ocr_cache .txt file)")
        return 0

    client = get_client()
    batch, total = 32, 0
    for start in range(0, len(texts), batch):
        part = texts[start : start + batch]
        # the title is added to what gets embedded, which helps short notices match
        vectors = embed([f"{title}\n{t}" for t in part])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "documentId": document_id,
                    "title": title,
                    "category": category,
                    "page": metas[start + j]["page"],
                    "fileUrl": file_url,
                    "text": part[j],
                    **(extra or {}),
                },
            )
            for j, vec in enumerate(vectors)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
    return total


def ingest_file(
    path: str,
    title: str,
    category: str = "other",
    document_id: str | None = None,
    file_url: str = "",
) -> int:
    """Ingest a PDF or a photo (jpg/jpeg/png/webp). Returns number of chunks stored.
    The default document id comes from the file name, so re-running replaces the old version."""
    ensure_collection()
    document_id = document_id or str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(path)))
    delete_document(document_id)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = extract_pages(path)
    elif ext in IMAGE_EXTS:
        pages = extract_image(path)
    else:
        raise ValueError(f"unsupported file type: {ext}")
    return _store(pages, title, category, document_id, file_url)


def ingest_pdf(pdf_path, title, category="other", document_id=None, file_url="") -> int:
    """Kept for backward compatibility."""
    return ingest_file(pdf_path, title, category, document_id, file_url)


# ---------------- Search ----------------
def search_chunks(query: str, limit: int = 5) -> list[dict]:
    """Return only passages whose similarity is above SCORE_THRESHOLD."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        print(f"[rag] collection '{COLLECTION}' not found, nothing indexed yet")
        return []
    vec = embed([query], is_query=True)[0]
    hits = client.query_points(
        collection_name=COLLECTION, query=vec, limit=limit, with_payload=True
    ).points

    print(f"[rag] query={query!r} scores={[round(h.score, 3) for h in hits]} threshold={SCORE_THRESHOLD}")

    return [
        {
            "text": h.payload["text"],
            "source": h.payload["title"],
            "page": h.payload["page"],
            "url": h.payload.get("fileUrl", ""),
            "score": round(h.score, 3),
        }
        for h in hits
        if h.score >= SCORE_THRESHOLD
    ]