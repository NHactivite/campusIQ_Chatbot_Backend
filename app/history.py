"""Chat history in MongoDB. Collections: chat_threads (one per conversation), chat_messages."""
import os
import time

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

load_dotenv()

_client = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=5000)
_db = _client[os.getenv("MONGO_DB", "IICSibsagar")]
threads = _db["chat_threads"]
messages = _db["chat_messages"]

try:
    threads.create_index([("userId", ASCENDING), ("threadId", ASCENDING)], unique=True)
    threads.create_index([("userId", ASCENDING), ("updatedAt", DESCENDING)])
    messages.create_index([("userId", ASCENDING), ("threadId", ASCENDING), ("ts", ASCENDING)])
    messages.create_index([("role", ASCENDING), ("answered", ASCENDING), ("ts", DESCENDING)])
except Exception as e:  # never stop the chatbot because of history
    print(f"[history] index setup failed: {type(e).__name__}: {e}")


def save_exchange(user_id: str, thread_id: str, question: str, answer: str, sources: list, lang: str):
    """Saves one question + answer. Failures are logged and never break the chat."""
    try:
        now = time.time() * 1000  # ms, same as JavaScript Date.now()
        answered = bool(sources)
        messages.insert_many(
            [
                {"userId": user_id, "threadId": thread_id, "role": "user", "text": question,
                 "lang": lang, "answered": answered, "ts": now},
                {"userId": user_id, "threadId": thread_id, "role": "ai", "text": answer,
                 "sources": sources, "lang": lang, "ts": now + 1},
            ]
        )
        threads.update_one(
            {"userId": user_id, "threadId": thread_id},
            {
                "$set": {"updatedAt": now, "lang": lang},
                "$setOnInsert": {"title": question[:60], "createdAt": now},
                "$inc": {"count": 1},
            },
            upsert=True,
        )
    except Exception as e:
        print(f"[history] save failed: {type(e).__name__}: {e}")


def list_threads(user_id: str, limit: int = 50) -> list[dict]:
    try:
        rows = threads.find({"userId": user_id}).sort("updatedAt", DESCENDING).limit(limit)
        return [{"id": r["threadId"], "title": r.get("title", "Chat"), "updatedAt": r.get("updatedAt", 0)} for r in rows]
    except Exception as e:
        print(f"[history] list failed: {type(e).__name__}: {e}")
        return []


def get_messages(user_id: str, thread_id: str) -> list[dict]:
    """Only returns messages that belong to this user."""
    try:
        rows = messages.find({"userId": user_id, "threadId": thread_id}).sort("ts", ASCENDING)
        return [{"role": r["role"], "content": r["text"], "sources": r.get("sources", [])} for r in rows]
    except Exception as e:
        print(f"[history] read failed: {type(e).__name__}: {e}")
        return []


def delete_thread(user_id: str, thread_id: str):
    try:
        messages.delete_many({"userId": user_id, "threadId": thread_id})
        threads.delete_one({"userId": user_id, "threadId": thread_id})
    except Exception as e:
        print(f"[history] delete failed: {type(e).__name__}: {e}")