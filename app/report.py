"""Builds a detailed PDF report for one event (admin use)."""
import io
import json
import re
from datetime import datetime
from xml.sax.saxutils import escape

from PIL import Image
from qdrant_client.models import FieldCondition, Filter, MatchValue
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from app.events_ingest import _download, _fmt_date
from app.llm import _to_text, llm
from app.rag import COLLECTION, get_client

NAVY, GOLD = colors.HexColor("#0b1e36"), colors.HexColor("#c69a36")
MAX_PHOTOS = 6
MAX_SOURCE_CHARS = 12000


def _indexed_text(event_id: str) -> list[tuple[str, str]]:
    """(label, text) for the event's PDFs and photos, taken from what is already in Qdrant."""
    c = get_client()
    if not c.collection_exists(COLLECTION):
        return []
    out, offset = [], None
    flt = Filter(must=[FieldCondition(key="eventId", match=MatchValue(value=event_id))])
    while True:
        pts, offset = c.scroll(COLLECTION, scroll_filter=flt, limit=128, offset=offset,
                               with_payload=["documentId", "text", "title"], with_vectors=False)
        for p in pts:
            doc_id = p.payload.get("documentId", "")
            if ":doc:" in doc_id:
                out.append((f"Document: {p.payload.get('title', '')}", p.payload["text"]))
            elif ":img:" in doc_id:
                out.append(("Photo text", p.payload["text"]))
        if offset is None:
            break
    return out


def _facts(e: dict) -> str:
    start, end = _fmt_date(e.get("eventDate")), _fmt_date(e.get("endDate"))
    lines = [
        f"Title: {e.get('title', '')}",
        f"Category: {e.get('category', '')}",
        f"Date: {start}" + (f" to {end}" if end and end != start else ""),
        f"Venue: {e.get('venue') or 'Not provided'}",
        f"Organiser: {e.get('organiser') or 'Not provided'}",
        f"Participants: {e.get('participants') if e.get('participants') is not None else 'Not provided'}",
        f"Description: {e.get('description') or 'Not provided'}",
        f"Outcome: {e.get('outcome') or 'Not provided'}",
        f"Photos attached: {len(e.get('images') or [])}",
        f"Documents attached: {len(e.get('documents') or [])}",
        f"Videos attached: {len(e.get('videos') or [])}",
    ]
    return "\n".join(lines)


def _write_sections(e: dict, sources: list[tuple[str, str]]) -> dict:
    src = "\n\n".join(f"[{label}]\n{text}" for label, text in sources)[:MAX_SOURCE_CHARS] or "(none)"
    prompt = (
        "You write official event reports for Sibsagar University's RDC and IIC.\n"
        "Use ONLY the facts below. Never invent names, numbers, speakers, dates or results. "
        "If something is not in the facts, write \"Not provided\" or leave the list empty.\n"
        "Reply with JSON only (no markdown fences) with these keys:\n"
        '"overview": string (3-5 sentences),\n'
        '"objectives": list of short strings (only if stated or clearly implied),\n'
        '"activities": list of short strings (what took place),\n'
        '"outcomes": list of short strings,\n'
        '"participation": string,\n'
        '"document_findings": list of short strings summarising the attached documents and photo text,\n'
        '"suggestions": list of 2-4 practical follow-up suggestions, each starting with "Suggestion:",\n'
        '"conclusion": string (2-3 sentences).\n\n'
        f"EVENT FACTS:\n{_facts(e)}\n\nTEXT READ FROM ATTACHED FILES:\n{src}"
    )
    raw = _to_text(llm.invoke(prompt).content).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    return data if isinstance(data, dict) else {}


def _photo(url: str):
    try:
        im = Image.open(io.BytesIO(_download(url))).convert("RGB")
        im.thumbnail((900, 900))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=75)
        buf.seek(0)
        w, h = im.size
        box_w, box_h = 82 * mm, 62 * mm
        scale = min(box_w / w, box_h / h)
        return RLImage(buf, width=w * scale, height=h * scale)
    except Exception as ex:
        print(f"[report] photo skipped: {type(ex).__name__}: {ex}")
        return None


def build_event_report(e: dict) -> bytes:
    event_id = str(e.get("_id") or e.get("id"))
    sources = _indexed_text(event_id)
    sec = _write_sections(e, sources)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10.5, leading=15)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=12, spaceAfter=4)
    title = ParagraphStyle("title", parent=ss["Title"], textColor=NAVY, fontSize=20, leading=24, alignment=0)
    small = ParagraphStyle("small", parent=body, fontSize=9, textColor=colors.grey)

    def P(text, style=body):
        return Paragraph(escape(str(text)), style)

    def bullets(items):
        items = [i for i in (items or []) if str(i).strip()]
        return [Paragraph("• " + escape(str(i)), body) for i in items] or [P("Not provided")]

    story = [
        P("SIBSAGAR UNIVERSITY · RDC & IIC", small),
        Spacer(1, 3),
        P(e.get("title", "Event report"), title),
        P("Event report", small),
        Spacer(1, 8),
    ]

    start, end = _fmt_date(e.get("eventDate")), _fmt_date(e.get("endDate"))
    meta = [
        ["Category", e.get("category") or "-"],
        ["Date", start + (f" to {end}" if end and end != start else "")],
        ["Venue", e.get("venue") or "Not provided"],
        ["Organiser", e.get("organiser") or "Not provided"],
        ["Participants", str(e.get("participants")) if e.get("participants") is not None else "Not provided"],
        ["Status", "Published" if e.get("isPublished") is not False else "Draft"],
        ["Attachments", f"{len(e.get('images') or [])} photos, {len(e.get('documents') or [])} documents, "
                        f"{len(e.get('videos') or [])} videos"],
    ]
    t = Table([[P(k, small), P(v)] for k, v in meta], colWidths=[34 * mm, 136 * mm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f7f9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t]

    def section(name, content):
        story.append(KeepTogether([Paragraph(name, h2)] + content[:1]))
        story.extend(content[1:])

    section("1. Overview", [P(sec.get("overview") or e.get("description") or "Not provided")])
    section("2. Objectives", bullets(sec.get("objectives")))
    section("3. Activities", bullets(sec.get("activities")))
    section("4. Participation", [P(sec.get("participation") or "Not provided")])
    section("5. Outcomes", bullets(sec.get("outcomes") or ([e["outcome"]] if e.get("outcome") else [])))
    section("6. Attached documents and photo text", bullets(sec.get("document_findings")))

    # photos, two per row
    photos = [p for p in (_photo(i["url"]) for i in (e.get("images") or [])[:MAX_PHOTOS] if i.get("url")) if p]
    if photos:
        story.append(Paragraph("7. Photographs", h2))
        rows = [photos[i:i + 2] + [""] * (2 - len(photos[i:i + 2])) for i in range(0, len(photos), 2)]
        pt = Table(rows, colWidths=[85 * mm, 85 * mm])
        pt.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.append(pt)

    section("8. Suggestions", bullets(sec.get("suggestions")))
    section("9. Conclusion", [P(sec.get("conclusion") or "Not provided")])

    story += [Spacer(1, 10), P(
        "Prepared from the event record and the documents attached to it. "
        "Suggestions are recommendations and not part of the official record.", small)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(1.2)
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(18 * mm, 9 * mm, f"Generated by CampusIQ on {datetime.now():%d %B %Y, %H:%M}")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=20 * mm, title=e.get("title", "Event report")).build(
        story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()