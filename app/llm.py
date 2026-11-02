import os
import sqlite3
from datetime import date
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.rag import search_chunks

load_dotenv()

# Change the model in .env (CHAT_MODEL=...) without touching the code.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
print(f"[llm] chat model: {CHAT_MODEL}")


# ---------------- Chat state -----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: NotRequired[str]  # "en", "as" or "auto"
    context: NotRequired[str]   # retrieved passages for the current question


# ---------------- LLM -----------------
llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)


# ---------------- Prompt -----------------
def format_context(passages: list[dict]) -> str:
    if not passages:
        return ""
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"[{i}] Source: {p['source']}, page {p['page']}\n{p['text']}")
    return "\n\n".join(parts)


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


# ---------------- Graph node -----------------
def chat_node(state: ChatState):
    msgs = list(state["messages"])
    last = msgs[-1]  # the new user question
    prompt = (
        build_instructions(state.get("language"), state.get("context", ""))
        + "\n\nUSER QUESTION:\n"
        + _to_text(last.content)
    )
    # Instructions go inside the user turn, so this works with every model
    # (some models, e.g. Gemma, do not accept a separate system message).
    response = llm.invoke(msgs[:-1] + [HumanMessage(content=prompt)])
    return {"messages": [response]}


# ---------------- Checkpointer -----------------
conn = sqlite3.connect(database="chatbot_db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# ---------------- Graph -----------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------- Helpers -----------------
def _to_text(content) -> str:
    """Gemini may return content as a list of parts; always return a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _sources_from_passages(passages: list[dict]):
    """One chip per document, with all matching pages merged."""
    docs: dict = {}
    for p in passages:
        key = (p.get("source"), p.get("url", ""))
        d = docs.setdefault(key, {"source": p.get("source"), "url": p.get("url", ""), "pages": []})
        if p.get("page") is not None and p["page"] not in d["pages"]:
            d["pages"].append(p["page"])
    return [
        {
            "source": d["source"],
            "url": d["url"],
            "page": ", ".join(str(n) for n in sorted(d["pages"])) or None,
        }
        for d in docs.values()
    ]


def stream_process_msg(user_input: str, threadId: str, language: str = "auto"):
    """Yields dicts: tool_start, ai_message, then sources (only if the answer used them)."""
    config = {"configurable": {"thread_id": threadId}}

    try:
        passages = search_chunks(user_input)
    except Exception as e:
        print(f"[rag] search failed: {type(e).__name__}: {e}")
        passages = []

    yield {"type": "tool_start", "tool": "search_university_docs"}

    answer_parts: list[str] = []
    try:
        for chunk, _ in chatbot.stream(
            {
                "messages": [HumanMessage(content=user_input)],
                "language": language,
                "context": format_context(passages),
            },
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):
                text = _to_text(chunk.content)
                if text:
                    answer_parts.append(text)
                    yield {"type": "ai_message", "text": text}
    except Exception as e:
        print(f"[llm] model call failed: {type(e).__name__}: {str(e)[:300]}")
        yield {
            "type": "ai_message",
            "text": "The assistant is busy right now. Please try again in a little while.",
        }

    # Show sources only when the model actually cited them (rule 2 in the prompt adds a
    # "Source:" / "উৎস:" line to grounded answers). "Not found" answers get no chips.
    answer = "".join(answer_parts)
    grounded = any(m in answer for m in ("Source:", "উৎস:", "উৎস :"))

    shown = passages
    if passages:
        top = max(p["score"] for p in passages)
        shown = [p for p in passages if p["score"] >= top - 0.12]  # drop weak extras

    yield {"type": "sources", "items": _sources_from_passages(shown) if grounded else []}
# ---------------- Conversation history -----------------
def get_conversation_history(threadId: str):
    config = {"configurable": {"thread_id": threadId}}
    snapshot = chatbot.get_state(config)
    state = snapshot.values
    return [
        {"role": msg.type, "content": _to_text(msg.content)}
        for msg in state.get("messages", [])
        if msg.type in ("human", "ai") and _to_text(msg.content)
    ]


# ---------------- Retrieve old threads -----------------
def retriveAllThreads():
    allThreads = set()
    for checkpoint in checkpointer.list(None):
        allThreads.add(checkpoint.config["configurable"]["thread_id"])
    return list(allThreads)