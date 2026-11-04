import os
import re
import sqlite3
from datetime import date
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import answer_cache
from app.rag import search_chunks

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
print(f"[llm] chat model: {CHAT_MODEL}")

HISTORY_MESSAGES = 6  # earlier messages sent to the model


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: NotRequired[str]
    context: NotRequired[str]


llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"))


# ---------------- helpers ----------------
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ASSAMESE = re.compile(r"[\u0980-\u09FF]")
_SMALLTALK = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s+(morning|afternoon|evening)|thanks?|thank\s*you|ok(ay)?|bye|"
    r"নমস্কাৰ|হেল্লো|ধন্যবাদ|শুভ\s*(সকাল|দুপৰীয়া|সন্ধিয়া))\b[\s!.?]*$",
    re.IGNORECASE,
)

NOT_FOUND = {
    "en": "I could not find a verified answer in the university documents currently available. "
          "Please contact the relevant university office for this information.",
    "as": "বর্তমানে উপলব্ধ বিশ্ববিদ্যালয়ৰ নথিপত্ৰত এই প্ৰশ্নৰ কোনো নিৰ্ভৰযোগ্য উত্তৰ পোৱা নগ'ল। "
          "অনুগ্ৰহ কৰি এই বিষয়ে বিশ্ববিদ্যালয়ৰ সংশ্লিষ্ট কাৰ্যালয়ত যোগাযোগ কৰক।",
}
BUSY = {
    "en": "The assistant is busy right now. Please try again in a little while.",
    "as": "সহায়ক এতিয়া ব্যস্ত আছে। অনুগ্ৰহ কৰি অলপ পিছত পুনৰ চেষ্টা কৰক।",
}


def normalize(text: str) -> str:
    return text.translate(_DIGITS).strip()


def pick_lang(language: str | None, question: str) -> str:
    if language in ("as", "en"):
        return language
    return "as" if _ASSAMESE.search(question) else "en"


def is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK.match(text))


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ---------------- prompt ----------------
def format_context(passages: list[dict]) -> str:
    if not passages:
        return ""
    return "\n\n".join(
        f"[{i}] Source: {p['source']}, page {p['page']}\n{p['text']}" for i, p in enumerate(passages, 1)
    )


def build_instructions(language: str | None, context: str) -> str:
    lang_rule = {
        "as": "Always reply in Assamese (অসমীয়া).",
        "en": "Always reply in English.",
    }.get(
        language or "auto",
        "Reply in the language of the user's question. If it is written in Assamese script "
        "(অসমীয়া), your ENTIRE answer must be in Assamese, including the source line, which "
        "must start with 'উৎস:'. If it is written in English, reply in English.",
    )
    today = date.today().strftime("%d %B %Y")
    passages = context if context else "(no passages were found for this question)"
    return (
        "You are CampusIQ, the AI assistant of Sibsagar University, helping students, "
        "faculty and administrative staff.\n"
        f"Today's date is {today}.\n"
        f"{lang_rule}\n\n"
        "Rules:\n"
        "1. For questions about the university, answer ONLY from the PASSAGES below. "
        "Do not add facts from your own knowledge. Never invent dates, fees, names or rules.\n"
        "2. End each grounded answer with the sources you used, in the form: "
        "Source: <document title>, page <number>.\n"
        "3. If there are no passages or they do not contain the answer, say clearly that you "
        "could not find a verified answer in the university documents currently available, "
        "and suggest contacting the relevant university office.\n"
        "4. If a deadline or event date in the passages is earlier than today's date, "
        "say clearly that it has already passed.\n"
        "5. For greetings or small talk, reply briefly and politely and say what you can help with.\n"
        "6. For questions unrelated to the university, politely say you can only help with "
        "university information.\n\n"
        f"PASSAGES:\n{passages}"
    )


# ---------------- graph ----------------
def chat_node(state: ChatState):
    msgs = list(state["messages"])
    last = msgs[-1]
    history = msgs[:-1][-HISTORY_MESSAGES:]
    while history and not isinstance(history[0], HumanMessage):
        history.pop(0)  # the first message sent must be from the user
    prompt = (
        build_instructions(state.get("language"), state.get("context", ""))
        + "\n\nUSER QUESTION:\n"
        + _to_text(last.content)
    )
    response = llm.invoke(history + [HumanMessage(content=prompt)])
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot_db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)


# ---------------- retrieval helpers ----------------
def _sources_from_passages(passages: list[dict]):
    """One chip per document, with all matching pages merged."""
    docs: dict = {}
    for p in passages:
        key = (p.get("source"), p.get("url", ""))
        d = docs.setdefault(key, {"source": p.get("source"), "url": p.get("url", ""), "pages": []})
        if p.get("page") is not None and p["page"] not in d["pages"]:
            d["pages"].append(p["page"])
    return [
        {"source": d["source"], "url": d["url"], "page": ", ".join(str(n) for n in sorted(d["pages"])) or None}
        for d in docs.values()
    ]


def rewrite_question(question: str, past: list) -> str:
    """Turns a follow-up ('what is its last date?') into a standalone question."""
    if not past:
        return question
    convo = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_to_text(m.content)[:300]}"
        for m in past[-4:]
    )
    prompt = (
        "Rewrite the user's LAST question as one standalone question, using the conversation for context. "
        "Keep the same language. If it is already standalone, return it unchanged. "
        "Output only the question, nothing else.\n\n"
        f"Conversation:\n{convo}\n\nLast question: {question}"
    )
    try:
        out = _to_text(llm.invoke([HumanMessage(content=prompt)]).content).strip().strip('"')
        return normalize(out)[:500] if out else question
    except Exception as e:
        print(f"[llm] rewrite failed: {type(e).__name__}")
        return question


# ---------------- stream ----------------
def stream_process_msg(user_input: str, threadId: str, language: str = "auto"):
    """Yields: tool_start, ai_message (many), then sources (only if the answer cited them)."""
    config = {"configurable": {"thread_id": threadId}}
    question = normalize(user_input)
    lang = pick_lang(language, question)
    smalltalk = is_smalltalk(question)

    try:
        past = chatbot.get_state(config).values.get("messages", [])
    except Exception:
        past = []
    has_history = bool(past)

    # repeated standalone question: answer from the cache
    if not smalltalk and not has_history:
        hit = answer_cache.get(question, lang)
        if hit:
            text, items = hit
            yield {"type": "tool_start", "tool": "cache"}
            yield {"type": "ai_message", "text": text}
            yield {"type": "sources", "items": items}
            return

    yield {"type": "tool_start", "tool": "search_university_docs"}

    search_q = question if smalltalk else rewrite_question(question, past)
    if search_q != question:
        print(f"[rag] rewritten: {question!r} -> {search_q!r}")
    try:
        passages = [] if smalltalk else search_chunks(search_q)
    except Exception as e:
        print(f"[rag] search failed: {type(e).__name__}: {e}")
        passages = []

    # nothing relevant: fixed answer, no model call
    if not passages and not smalltalk:
        text = NOT_FOUND[lang]
        try:
            chatbot.update_state(
                config,
                {"messages": [HumanMessage(content=user_input), AIMessage(content=text)]},
                as_node="chat_node",
            )
        except Exception as e:
            print(f"[llm] could not save thread state: {type(e).__name__}")
        yield {"type": "ai_message", "text": text}
        yield {"type": "sources", "items": []}
        return

    answer_parts: list[str] = []
    failed = False
    try:
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)], "language": language,
             "context": format_context(passages)},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):
                text = _to_text(chunk.content)
                if text:
                    answer_parts.append(text)
                    yield {"type": "ai_message", "text": text}
    except Exception as e:
        failed = True
        print(f"[llm] model call failed: {type(e).__name__}: {str(e)[:300]}")
        yield {"type": "ai_message", "text": BUSY[lang]}

    answer = "".join(answer_parts)
    grounded = any(m in answer for m in ("Source:", "উৎস:", "উৎস :"))

    shown = passages
    if passages:
        top = max(p["score"] for p in passages)
        shown = [p for p in passages if p["score"] >= top - 0.12]  # drop weak extras
    items = _sources_from_passages(shown) if grounded else []
    yield {"type": "sources", "items": items}

    if grounded and not failed and not has_history:
        answer_cache.put(question, lang, answer, items)  # only good, standalone answers


# ---------------- old helpers (kept for compatibility) ----------------
def get_conversation_history(threadId: str):
    snapshot = chatbot.get_state({"configurable": {"thread_id": threadId}})
    return [
        {"role": m.type, "content": _to_text(m.content)}
        for m in snapshot.values.get("messages", [])
        if m.type in ("human", "ai") and _to_text(m.content)
    ]