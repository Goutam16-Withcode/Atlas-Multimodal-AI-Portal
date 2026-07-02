"""
Tools the agent can call. In a real deployment these would hit actual
systems (a CMMS/SCADA API, a ticketing system like Jira/ServiceNow, a
vector DB). Here they're realistic simulations with the same interface,
so swapping in real backends later is a one-line change per tool.
"""
import uuid
import time
import math
import datetime
import requests
from langchain_core.tools import tool

from config import settings
from logger import get_logger
from asset_export import save_remote_asset

logger = get_logger("tools")

# ---------------------------------------------------------------------
# Simulated equipment/knowledge base (stand-in for a vector DB / CMMS)
# ---------------------------------------------------------------------
_EQUIPMENT_DB = {
    "conveyor-belt-3": {
        "status": "operational",
        "last_maintenance": "2026-05-14",
        "manual_ref": "Section 4.2: Belt tension should be 80-90 PSI",
    },
    "hydraulic-press-1": {
        "status": "operational",
        "last_maintenance": "2026-06-01",
        "manual_ref": "Section 7.1: Max pressure 3000 PSI, check seals monthly",
    },
    "cooling-pump-2": {
        "status": "under maintenance",
        "last_maintenance": "2026-06-28",
        "manual_ref": "Section 2.5: Coolant flow rate should be 40-60 L/min",
    },
}

_TICKETS = {}


@tool
def calculator(expression: str) -> str:
    """Evaluate basic or advanced mathematical and engineering expressions.
    Supports numbers, +, -, *, /, (, ), and functions/constants like:
    sqrt(), pow(), pi, sin(), cos(), tan(), log(), exp(), abs(), round(), ceil(), floor()."""
    cleaned_expression = expression.lower()
    valid_words = ["sqrt", "pow", "pi", "sin", "cos", "tan", "log", "exp", "abs", "round", "ceil", "floor", "math."]
    for word in valid_words:
        cleaned_expression = cleaned_expression.replace(word, "")

    allowed_chars = set("0123456789+-*/(). ,e")
    if not set(cleaned_expression) <= allowed_chars:
        return "Error: expression contains disallowed characters or unauthorized functions."

    try:
        safe_namespace = {
            "math": math, "sqrt": math.sqrt, "pow": math.pow, "pi": math.pi,
            "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
            "exp": math.exp, "abs": abs, "round": round, "ceil": math.ceil, "floor": math.floor
        }
        eval_expr = expression.lower()
        for word in ["sqrt", "pow", "sin", "cos", "tan", "log", "exp", "ceil", "floor"]:
            if word in eval_expr and f"math.{word}" not in eval_expr:
                eval_expr = eval_expr.replace(word, f"math.{word}")
        if "pi" in eval_expr and "math.pi" not in eval_expr:
            eval_expr = eval_expr.replace("pi", "math.pi")

        return str(eval(eval_expr, {"__builtins__": {}}, safe_namespace))
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def check_equipment_status(equipment_id: str) -> str:
    """Look up the live status, last maintenance date, and relevant manual
    reference for a piece of industrial equipment by its ID
    (e.g. 'conveyor-belt-3', 'hydraulic-press-1', 'cooling-pump-2')."""
    key = equipment_id.strip().lower()
    if key not in _EQUIPMENT_DB:
        known = ", ".join(_EQUIPMENT_DB.keys())
        return f"No record found for '{equipment_id}'. Known equipment: {known}"
    info = _EQUIPMENT_DB[key]
    return (
        f"Equipment: {equipment_id}\n"
        f"Status: {info['status']}\n"
        f"Last maintenance: {info['last_maintenance']}\n"
        f"Manual reference: {info['manual_ref']}"
    )


# ---------------------------------------------------------------------
# Hybrid RAG knowledge base (unchanged from prior version)
# ---------------------------------------------------------------------
_KNOWLEDGE_BASE_DOCS = [
    {"id": "SOP-LOTO-001", "title": "Lockout/Tagout Safety Protocol (LOTO)",
     "content": "Lockout/Tagout (LOTO) must be applied before servicing any rotating or pressurized equipment. All personnel must verify a zero energy state before beginning work. Secure all padlocks, tag out keys, and bleed residual pressure from hydraulic lines.",
     "category": "safety"},
    {"id": "SOP-HAND-002", "title": "Shift Handover SOP",
     "content": "Shift handover requires: equipment status log, open tickets, any anomalies observed, and signed handover checklist. The incoming team must visually inspect the primary lines.",
     "category": "operations"},
    {"id": "SOP-QUAL-003", "title": "Quality Control Inspection Guidelines",
     "content": "Quality inspection occurs every 500 units on the primary line, and every 200 units on the secondary line. Check caliper alignments and reject any units with deviation above 0.05 mm.",
     "category": "quality"},
    {"id": "SOP-MTR-004", "title": "Conveyor Motor Troubleshooting",
     "content": "If a conveyor motor trips, immediately verify temperature logs. For temperatures above 90°C, shut down line, apply LOTO on conveyor-belt-3, check belt tension (80-90 PSI), and inspect motor ventilation cover.",
     "category": "maintenance"},
    {"id": "SOP-HYD-005", "title": "Hydraulic Press Seal Maintenance",
     "content": "Hydraulic Press 1 operates at a maximum pressure of 3000 PSI. Monthly checks are required for cylinder seals. In case of fluid leak warnings, depressurize system, clean catch basin, check seals, and raise critical support ticket.",
     "category": "maintenance"},
    {"id": "SOP-PMP-006", "title": "Cooling Pump Flow Rates",
     "content": "Cooling pump 2 coolant flow rate must be maintained between 40 to 60 L/min. If flow rate drops below 30 L/min, high temperature alarms will trigger. Check inlet valve seal and clear pipe clogging.",
     "category": "operations"}
]

SYNONYMS = {
    "leak": ["leak", "fluid", "seal", "press", "hydraulic", "catch basin"],
    "overheat": ["temperature", "hot", "overheating", "motor", "pump", "coolant"],
    "trip": ["stopped", "motor", "tension", "conveyor", "electrical"],
    "safety": ["loto", "lockout", "tagout", "energy", "padlock", "rules"],
    "maintenance": ["check", "inspect", "service", "ticket", "technician"]
}


class BM25Retriever:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(self._tokenize(d["content"])) for d in docs]
        self.avg_doc_len = sum(self.doc_len) / len(docs) if docs else 1
        self.doc_freqs = []
        self.idf = {}
        self._initialize()

    def _tokenize(self, text):
        return [w.lower().strip(",.!?()\"';:") for w in text.split() if len(w) > 2]

    def _initialize(self):
        num_docs = len(self.docs)
        word_df = {}
        for d in self.docs:
            tokens = set(self._tokenize(d["content"]) + self._tokenize(d["title"]))
            for t in tokens:
                word_df[t] = word_df.get(t, 0) + 1
        for word, df in word_df.items():
            self.idf[word] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        for d in self.docs:
            tokens = self._tokenize(d["content"]) + self._tokenize(d["title"])
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

    def get_scores(self, query_tokens):
        scores = [0.0] * len(self.docs)
        for i, freqs in enumerate(self.doc_freqs):
            d_len = self.doc_len[i]
            score = 0.0
            for token in query_tokens:
                if token in freqs:
                    tf = freqs[token]
                    idf = self.idf.get(token, 0.0)
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avg_doc_len))
                    score += idf * (num / den)
            scores[i] = score
        return scores


class DenseTfidfRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vocab = {}
        self.doc_vectors = []
        self._initialize()

    def _tokenize(self, text):
        return [w.lower().strip(",.!?()\"';:") for w in text.split() if len(w) > 2]

    def _initialize(self):
        all_words = set()
        for d in self.docs:
            all_words.update(self._tokenize(d["content"]) + self._tokenize(d["title"]))
        self.vocab = {word: i for i, word in enumerate(sorted(all_words))}

        num_docs = len(self.docs)
        df = [0] * len(self.vocab)
        for d in self.docs:
            tokens = set(self._tokenize(d["content"]) + self._tokenize(d["title"]))
            for t in tokens:
                if t in self.vocab:
                    df[self.vocab[t]] += 1

        self.idf = []
        for count in df:
            self.idf.append(math.log(1 + (num_docs / (count + 1))))

        for d in self.docs:
            tokens = self._tokenize(d["content"]) + self._tokenize(d["title"])
            vec = [0.0] * len(self.vocab)
            for t in tokens:
                if t in self.vocab:
                    vec[self.vocab[t]] += 1
            for idx in range(len(vec)):
                vec[idx] *= self.idf[idx]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            self.doc_vectors.append(vec)

    def get_scores(self, query_tokens):
        q_vec = [0.0] * len(self.vocab)
        for t in query_tokens:
            if t in self.vocab:
                q_vec[self.vocab[t]] += 1
        for idx in range(len(q_vec)):
            q_vec[idx] *= self.idf[idx]
        q_norm = math.sqrt(sum(v * v for v in q_vec))
        if q_norm > 0:
            q_vec = [v / q_norm for v in q_vec]

        scores = []
        for doc_vec in self.doc_vectors:
            cos_sim = sum(qv * dv for qv, dv in zip(q_vec, doc_vec))
            scores.append(cos_sim)
        return scores


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal operations knowledge base for procedures, safety
    protocols, and standard operating procedures (SOPs). Uses a hybrid
    BM25 and vector-style TF-IDF semantic matcher with automatic query translation."""
    query_tokens = [w.lower().strip(",.!?()\"';:") for w in query.split() if len(w) > 2]
    expanded_tokens = list(query_tokens)
    for token in query_tokens:
        for key, syns in SYNONYMS.items():
            if token == key or key in token:
                expanded_tokens.extend(syns)
    expanded_tokens = list(set(expanded_tokens))

    bm25 = BM25Retriever(_KNOWLEDGE_BASE_DOCS)
    bm25_scores = bm25.get_scores(expanded_tokens)

    dense = DenseTfidfRetriever(_KNOWLEDGE_BASE_DOCS)
    dense_scores = dense.get_scores(expanded_tokens)

    def normalize(scores):
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s == 0:
            return [1.0 / len(scores)] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    norm_bm25 = normalize(bm25_scores)
    norm_dense = normalize(dense_scores)

    hybrid_scores = [0.5 * b_s + 0.5 * d_s for b_s, d_s in zip(norm_bm25, norm_dense)]
    ranked_indices = sorted(range(len(hybrid_scores)), key=lambda idx: hybrid_scores[idx], reverse=True)

    results = []
    for i in range(min(3, len(ranked_indices))):
        idx = ranked_indices[i]
        if hybrid_scores[idx] > 0.05:
            doc = _KNOWLEDGE_BASE_DOCS[idx]
            results.append(
                f"[{doc['id']}] Title: {doc['title']}\n"
                f"Category: {doc['category']}\n"
                f"Content: {doc['content']}\n"
                f"Match Score: {hybrid_scores[idx]:.3f}"
            )

    if not results:
        return "No matching entry found in the operations knowledge base. Please check keywords or escalate."
    return "\n---\n".join(results)


# ---------------------------------------------------------------------
# Image generation — now persists the asset locally so it's downloadable
# ---------------------------------------------------------------------
@tool
def generate_image_asset(prompt: str) -> str:
    """Generate a high-quality creative image, diagram, or poster asset using AI.
    Provide a detailed prompt of what the image should contain (e.g. 'schematic diagram of a hydraulic pump').
    Returns a stable, locally-hosted, downloadable URL."""
    import urllib.parse
    encoded = urllib.parse.quote(prompt.strip())
    remote_url = f"{settings.IMAGE_GEN_BASE_URL}/{encoded}?width=800&height=600&nologo=true&private=true"

    headers = {}
    if settings.POLLINATIONS_API_TOKEN:
        headers["Authorization"] = f"Bearer {settings.POLLINATIONS_API_TOKEN}"

    try:
        saved = save_remote_asset(remote_url, kind="image", headers=headers)
        local_url = saved["url"]
        return (
            f"Image generated and saved successfully.\n"
            f"Download URL: {local_url}\n"
            f"HTML Tag: <img src=\"{local_url}\" alt=\"{prompt}\" class=\"rounded-lg shadow-lg my-4 max-w-full\" data-downloadable=\"true\" data-asset-type=\"image\" />"
        )
    except Exception as e:
        logger.warning(f"Failed to persist generated image locally, falling back to remote URL: {e}")
        return (
            f"Image generated successfully (remote-hosted; may expire).\n"
            f"URL: {remote_url}\n"
            f"HTML Tag: <img src=\"{remote_url}\" alt=\"{prompt}\" class=\"rounded-lg shadow-lg my-4 max-w-full\" />"
        )


# ---------------------------------------------------------------------
# Video generation — Replicate API (async predict + poll)
# ---------------------------------------------------------------------
@tool
def generate_video_asset(prompt: str, duration_hint: str = "short") -> str:
    """Generate a short AI video clip from a text prompt (e.g. 'slow pan across
    a factory floor with robotic arms welding'). Use for presentations or
    explainers that specifically call for motion/video rather than a static image.
    This can take up to a few minutes. Returns a stable, locally-hosted,
    downloadable URL once ready."""
    if not settings.REPLICATE_API_TOKEN:
        return ("Error: video generation is not configured. Set REPLICATE_API_TOKEN "
                "in the server environment to enable this tool. Use generate_image_asset "
                "instead for now.")

    headers = {
        "Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "version": settings.REPLICATE_VIDEO_MODEL_VERSION.split(":")[-1],
        "input": {"prompt": prompt.strip()},
    }

    try:
        create_res = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers, json=payload, timeout=30,
        )
        if create_res.status_code not in (200, 201):
            logger.error(f"Replicate prediction creation failed: {create_res.text}")
            return f"Error: video generation request failed ({create_res.status_code}): {create_res.text[:200]}"

        prediction = create_res.json()
        get_url = prediction.get("urls", {}).get("get")
        if not get_url:
            return "Error: video generation service did not return a valid job URL."

        elapsed = 0.0
        while elapsed < settings.VIDEO_GEN_MAX_POLL_SECONDS:
            time.sleep(settings.VIDEO_GEN_POLL_INTERVAL_SECONDS)
            elapsed += settings.VIDEO_GEN_POLL_INTERVAL_SECONDS
            poll_res = requests.get(get_url, headers=headers, timeout=30)
            poll_data = poll_res.json()
            status = poll_data.get("status")

            if status == "succeeded":
                output = poll_data.get("output")
                video_url = output[0] if isinstance(output, list) and output else output
                if not video_url:
                    return "Error: video generation succeeded but no output URL was returned."
                try:
                    saved = save_remote_asset(video_url, kind="video")
                    local_url = saved["url"]
                    return (
                        f"Video generated and saved successfully.\n"
                        f"Download URL: {local_url}\n"
                        f"HTML Tag: <video src=\"{local_url}\" controls class=\"rounded-lg shadow-lg my-4 max-w-full\" data-downloadable=\"true\" data-asset-type=\"video\"></video>"
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist generated video locally: {e}")
                    return f"Video generated successfully (remote-hosted).\nURL: {video_url}"

            if status == "failed" or status == "canceled":
                err = poll_data.get("error", "unknown error")
                return f"Error: video generation {status}: {err}"

        return "Error: video generation timed out. The job may still complete server-side; try again shortly."

    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return f"Error generating video: {str(e)}"


@tool
def create_support_ticket(issue_summary: str, priority: str = "medium") -> str:
    """Create a support/maintenance ticket in the industrial ticketing system.
    Use this when an issue cannot be resolved through information alone and
    requires human/technician follow-up. priority should be 'low', 'medium',
    'high', or 'critical'."""
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    _TICKETS[ticket_id] = {
        "summary": issue_summary,
        "priority": priority,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "status": "open",
    }
    return f"Ticket created: {ticket_id} (priority: {priority}). A technician will be notified."


ALL_TOOLS = [
    calculator,
    check_equipment_status,
    search_knowledge_base,
    generate_image_asset,
    generate_video_asset,
    create_support_ticket,
]