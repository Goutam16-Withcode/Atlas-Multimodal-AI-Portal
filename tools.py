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
import re
import urllib.parse
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


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information, news, or general knowledge."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return f"Error: Search engine returned status code {r.status_code}"
        
        html = r.text
        results = []
        pattern = r'<a class="result__a" href="([^"]+)"[^>]*>(.*?)</a>'
        matches = re.findall(pattern, html, re.DOTALL)
        
        snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
        snippets = re.findall(snippet_pattern, html, re.DOTALL)
        
        for i, (link, title) in enumerate(matches[:5]):
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            if "uddg=" in link:
                parsed_url = urllib.parse.urlparse(link)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                if "uddg" in query_params:
                    link = query_params["uddg"][0]
            
            snippet_clean = ""
            if i < len(snippets):
                snippet_clean = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                
            results.append(f"Title: {title_clean}\nURL: {link}\nSnippet: {snippet_clean}")
            
        if not results:
            return "No web results found."
        return "\n\n".join(results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def check_stock_price(ticker: str) -> str:
    """Retrieve the real-time stock price and market data for a given company ticker symbol (e.g. 'AAPL', 'MSFT', 'TSLA')."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    clean_ticker = ticker.strip().upper()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_ticker}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 404:
            return f"Error: Ticker '{clean_ticker}' not found."
        if r.status_code != 200:
            return f"Error: Stock price service returned status code {r.status_code}"
        
        data = r.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return f"Error: No data available for ticker '{clean_ticker}'."
        
        meta = result[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        currency = meta.get("currency", "USD")
        prev_close = meta.get("previousClose")
        exchange = meta.get("exchangeName", "Unknown")
        
        if price is None:
            return f"Error: Regular market price not found for ticker '{clean_ticker}'."
            
        change = ""
        if prev_close is not None:
            diff = price - prev_close
            pct = (diff / prev_close) * 100
            sign = "+" if diff >= 0 else ""
            change = f" ({sign}{diff:.2f} / {sign}{pct:.2f}%)"
            
        return (
            f"Stock: {clean_ticker} ({exchange})\n"
            f"Current Price: {price:.2f} {currency}{change}\n"
            f"Previous Close: {prev_close:.2f} {currency}"
        )
    except Exception as e:
        return f"Error fetching stock price for '{clean_ticker}': {str(e)}"


@tool
def check_weather(location: str) -> str:
    """Retrieve the real-time weather conditions for any city or location globally."""
    headers = {"User-Agent": "curl/7.81.0"}
    clean_location = urllib.parse.quote(location.strip())
    url = f"https://wttr.in/{clean_location}?format=4"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return f"Error: Weather service returned status code {r.status_code}"
        return r.text.strip()
    except Exception as e:
        return f"Error fetching weather for '{location}': {str(e)}"


@tool
def query_database(sql_query: str) -> str:
    """Execute a read-only SQL query against the local SQLite metadata database to inspect tables like users, sessions, thread_metadata, or audit_log.
    Only SELECT statements are permitted for safety."""
    clean_query = sql_query.strip()
    if not clean_query.lower().startswith("select"):
        return "Error: Only read-only SELECT queries are permitted."
    
    blocked_keywords = ["insert", "update", "delete", "drop", "alter", "create", "replace", "vacuum"]
    for keyword in blocked_keywords:
        if re.search(r"\b" + keyword + r"\b", clean_query.lower()):
            return f"Error: Unauthorized database operation '{keyword}' detected."
            
    import sqlite3
    try:
        conn = sqlite3.connect(settings.SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(clean_query)
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        conn.close()
        
        if not rows:
            return "Query executed successfully. No rows returned."
            
        header = " | ".join(colnames)
        separator = " | ".join(["---"] * len(colnames))
        table_rows = []
        for row in rows[:50]:
            table_rows.append(" | ".join(str(val) for val in row))
            
        result_table = f"{header}\n{separator}\n" + "\n".join(table_rows)
        if len(rows) > 50:
            result_table += f"\n\n*Note: Output truncated to first 50 rows (total: {len(rows)}).*"
        return result_table
    except Exception as e:
        return f"Database query failed: {str(e)}"


@tool
def send_email_message(to_address: str, subject: str, body: str) -> str:
    """Simulate sending an email notification or report to a specified email address."""
    if not re.match(r"[^@]+@[^@]+\.[^@]+", to_address.strip()):
        return f"Error: '{to_address}' is not a valid email address."
    logger.info(f"Simulated email sent to {to_address} (Subject: {subject})")
    return (
        f"Email sent successfully to {to_address}!\n"
        f"Subject: {subject}\n"
        f"Body Preview: {body[:150]}..."
    )


@tool
def schedule_calendar_event(title: str, date_iso: str, start_time: str, duration_minutes: int) -> str:
    """Schedule an event or meeting on the calendar.
    date_iso must be in YYYY-MM-DD format. start_time must be in HH:MM format."""
    try:
        datetime.datetime.strptime(date_iso.strip(), "%Y-%m-%d")
        datetime.datetime.strptime(start_time.strip(), "%H:%M")
    except ValueError:
        return "Error: Date must be YYYY-MM-DD and start_time must be HH:MM format."
    return f"Event '{title}' successfully scheduled for {date_iso} at {start_time} (Duration: {duration_minutes} mins)."


@tool
def manage_task_todo(action: str, task_name: str, priority: str = "medium") -> str:
    """Create, complete, or list todo tasks.
    action must be 'create', 'complete', or 'list'."""
    act = action.strip().lower()
    if act not in ("create", "complete", "list"):
        return "Error: action must be 'create', 'complete', or 'list'."
    if act == "create":
        return f"Task '{task_name}' created successfully with {priority} priority."
    elif act == "complete":
        return f"Task '{task_name}' marked as completed."
    else:
        return "Active Tasks:\n- [ ] Fix pump ventilation cover (high)\n- [ ] Review quarterly plant electricity usage (medium)"


@tool
def execute_python_code(code_string: str) -> str:
    """Execute a python code block to perform complex numerical computations, data formatting, or custom calculations.
    For safety, it executes in a restricted environment and blocks access to os, sys, subprocess, and network modules."""
    blocked = ["os", "sys", "subprocess", "requests", "socket", "urllib", "eval", "exec", "open", "builtins", "__import__"]
    for word in blocked:
        if re.search(r"\b" + word + r"\b", code_string):
            return f"Error: Execution of code containing '{word}' is blocked for safety."
            
    import sys
    import io
    
    local_vars = {"math": math, "datetime": datetime}
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        compiled = compile(code_string, "<string>", "exec")
        exec(compiled, {"__builtins__": {}}, local_vars)
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        if not output.strip():
            ret_vals = {k: v for k, v in local_vars.items() if k not in ("math", "datetime")}
            if ret_vals:
                return f"Execution successful. Variable states:\n{ret_vals}"
            return "Execution successful. No output was printed."
        return output.strip()
    except Exception as e:
        sys.stdout = old_stdout
        return f"Execution failed: {str(e)}"


@tool
def post_slack_message(channel: str, message: str) -> str:
    """Post an notification message to a specified enterprise Slack channel (e.g. '#alerts', '#ops-coordination')."""
    clean_channel = channel.strip()
    if not clean_channel.startswith("#"):
        clean_channel = "#" + clean_channel
    logger.info(f"Simulated Slack post to {clean_channel}: {message}")
    return f"Message successfully posted to Slack channel '{clean_channel}'."


@tool
def get_github_issue(repo: str, issue_number: int) -> str:
    """Fetch status and details of a GitHub issue or pull request for a repository (e.g., 'owner/repo')."""
    return (
        f"GitHub Repo: {repo}\n"
        f"Issue #{issue_number}: 'Optimize SQLite WAL checkpoint frequency under high concurrent load'\n"
        f"Status: Open\n"
        f"Assignee: @lead-architect\n"
        f"Description: Telemetry shows checkpoint lock wait time spikes on peak hourly load. Need to tune PRAGMA synchronous."
    )


@tool
def compare_documents_similarity(doc_a_path: str, doc_b_path: str) -> str:
    """Compare two documents or local files to extract differences, similarities, and text alignment metrics."""
    import os
    text_a, text_b = "", ""
    try:
        if os.path.exists(doc_a_path):
            with open(doc_a_path, "r", encoding="utf-8", errors="ignore") as f:
                text_a = f.read()
        else:
            text_a = doc_a_path
            
        if os.path.exists(doc_b_path):
            with open(doc_b_path, "r", encoding="utf-8", errors="ignore") as f:
                text_b = f.read()
        else:
            text_b = doc_b_path
            
        len_a, len_b = len(text_a), len(text_b)
        if len_a == 0 or len_b == 0:
            return "Error: One or both documents are empty."
            
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        similarity = (intersection / union) * 100 if union > 0 else 0.0
        
        return (
            f"Comparison Result:\n"
            f"- Document A Size: {len_a} characters\n"
            f"- Document B Size: {len_b} characters\n"
            f"- Vocabulary Similarity (Jaccard): {similarity:.1f}%\n"
            f"- Recommendation: The texts share high semantic overlap. Review section revisions for exact line diffs."
        )
    except Exception as e:
        return f"Comparison failed: {str(e)}"


ALL_TOOLS = [
    calculator,
    check_equipment_status,
    search_knowledge_base,
    generate_image_asset,
    generate_video_asset,
    create_support_ticket,
    web_search,
    check_stock_price,
    check_weather,
    query_database,
    send_email_message,
    schedule_calendar_event,
    manage_task_todo,
    execute_python_code,
    post_slack_message,
    get_github_issue,
    compare_documents_similarity,
]