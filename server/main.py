from dotenv import load_dotenv

load_dotenv()  # must run before any os.environ[...] below

import asyncio
import hmac
import os

import jwt
import socketio
import base64
from app.services.tts_service import synthesize_to_mp3
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jwt import PyJWKClient

from app.docs_ingest import delete_source, sync_document
from app.events_ingest import delete_event, sync_event
from app.llm import get_conversation_history, stream_process_msg
from app.history import delete_thread, get_messages, list_threads, save_exchange

# comma-separated, e.g. "https://iic-sibsagar-frontend.vercel.app,http://localhost:3000"
ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
jwks = PyJWKClient(os.environ["CLERK_JWKS_URL"])
INGEST_SECRET = os.environ["INGEST_SECRET"]

sio = socketio.AsyncServer(cors_allowed_origins=ALLOWED, async_mode="asgi")
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/health")
def health():
    return {"ok": True}


# ---------------- sync endpoints (called by the portal backend) ----------------
def _check_secret(secret: str | None):
    if not secret or not hmac.compare_digest(secret, INGEST_SECRET):
        raise HTTPException(status_code=401, detail="unauthorized")


@fastapi_app.post("/internal/events/sync")
def events_sync(event: dict, bg: BackgroundTasks, x_ingest_secret: str | None = Header(None)):
    _check_secret(x_ingest_secret)
    if not (event.get("_id") or event.get("id")):
        raise HTTPException(status_code=400, detail="missing _id")
    bg.add_task(sync_event, event)  # returns at once, indexing runs in the background
    return {"queued": True}


@fastapi_app.delete("/internal/events/{event_id}")
def events_delete(event_id: str, bg: BackgroundTasks, x_ingest_secret: str | None = Header(None)):
    _check_secret(x_ingest_secret)
    bg.add_task(delete_event, event_id)
    return {"queued": True}


@fastapi_app.post("/internal/documents/sync")
def documents_sync(doc: dict, bg: BackgroundTasks, x_ingest_secret: str | None = Header(None)):
    _check_secret(x_ingest_secret)
    if not (doc.get("_id") or doc.get("id")):
        raise HTTPException(status_code=400, detail="missing _id")
    bg.add_task(sync_document, doc)
    return {"queued": True}


@fastapi_app.delete("/internal/documents/{doc_id}")
def documents_delete(doc_id: str, bg: BackgroundTasks, x_ingest_secret: str | None = Header(None)):
    _check_secret(x_ingest_secret)
    bg.add_task(delete_source, doc_id)
    return {"queued": True}


@fastapi_app.get("/internal/index")
def index_status(x_ingest_secret: str | None = Header(None)):
    _check_secret(x_ingest_secret)
    from app.rag import COLLECTION, OCR_CACHE_DIR, get_client

    client = get_client()
    ids: set[str] = set()
    if client.collection_exists(COLLECTION):
        offset = None
        while True:
            pts, offset = client.scroll(
                COLLECTION, limit=256, offset=offset, with_payload=["documentId"], with_vectors=False
            )
            ids.update(p.payload["documentId"] for p in pts)
            if offset is None:
                break

    cache = {}
    if os.path.isdir(OCR_CACHE_DIR):
        cache = {f: os.path.getsize(os.path.join(OCR_CACHE_DIR, f)) for f in os.listdir(OCR_CACHE_DIR)}
    return {"ids": sorted(ids), "cache": cache}

# routes above must be registered on fastapi_app BEFORE it is wrapped here
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


# ---------------- chat over Socket.IO ----------------
def _verify(token: str) -> str:
    key = jwks.get_signing_key_from_jwt(token).key
    claims = jwt.decode(token, key, algorithms=["RS256"], options={"verify_aud": False})
    return claims["sub"]  # Clerk user id


@sio.event
async def connect(sid, environ, auth):
    token = (auth or {}).get("token")
    if not token:
        raise socketio.exceptions.ConnectionRefusedError("unauthorized")
    try:
        user_id = await asyncio.to_thread(_verify, token)
    except Exception as e:
        print("auth failed:", type(e).__name__)
        raise socketio.exceptions.ConnectionRefusedError("unauthorized")
    await sio.save_session(sid, {"user_id": user_id})


def _tid(user_id: str, thread_id: str) -> str:
    return f"{user_id}:{thread_id}"  # a user can only reach their own threads


@sio.on("human_message")
async def human_message(sid, data):
    session = await sio.get_session(sid)
    user_id = session["user_id"]
    text = (data.get("message") or "").strip()[:1000]
    thread = data.get("threadId") or ""
    lang = data.get("language") or "auto"
    if not text or not thread:
        return

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for chunk in stream_process_msg(text, _tid(user_id, thread), lang):
                loop.call_soon_threadsafe(q.put_nowait, chunk)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "msg": str(e)})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    loop.run_in_executor(None, worker)

    answer_parts: list[str] = []
    sources: list = []
    while (chunk := await q.get()) is not None:
        t = chunk["type"]
        if t == "tool_start":
            await sio.emit("tool_start", {"tool": chunk["tool"]}, to=sid)
        elif t == "sources":
            sources = chunk["items"]
            await sio.emit("sources", chunk["items"], to=sid)
        elif t == "ai_message":
            answer_parts.append(chunk["text"])
            await sio.emit("ai_message", {"text": chunk["text"]}, to=sid)
        elif t == "error":
            await sio.emit("chat_error", {"message": "failed"}, to=sid)
    await sio.emit("done", {}, to=sid)

    # saved after "done", so the student never waits for the database
    await asyncio.to_thread(save_exchange, user_id, thread, text, "".join(answer_parts), sources, lang)


@sio.on("list_threads")
async def on_list_threads(sid, data=None):
    session = await sio.get_session(sid)
    await sio.emit("threads", await asyncio.to_thread(list_threads, session["user_id"]), to=sid)


@sio.on("get_history")
async def get_history(sid, data):
    session = await sio.get_session(sid)
    thread = (data or {}).get("threadId") or ""
    history = await asyncio.to_thread(get_messages, session["user_id"], thread)
    await sio.emit("history", history, to=sid)


@sio.on("delete_thread")
async def on_delete_thread(sid, data):
    session = await sio.get_session(sid)
    await asyncio.to_thread(delete_thread, session["user_id"], (data or {}).get("threadId") or "")
    await sio.emit("threads", await asyncio.to_thread(list_threads, session["user_id"]), to=sid)

@sio.on("speak")
async def on_speak(sid, data):
    await sio.get_session(sid)  # only signed-in users get here (token checked on connect)
    data = data or {}
    text = str(data.get("text") or "")[:3000]
    audio = await synthesize_to_mp3(text, data.get("language"))
    payload = {"id": data.get("id")}
    if audio:
        payload["audio"] = base64.b64encode(audio).decode("ascii")
    else:
        payload["error"] = True
    await sio.emit("speech", payload, to=sid)