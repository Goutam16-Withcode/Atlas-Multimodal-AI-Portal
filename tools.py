"""
Tools the agent can call. In a real deployment these would hit actual
systems (a CMMS/SCADA API, a ticketing system like Jira/ServiceNow, a
vector DB). Here they're realistic simulations with the same interface,
so swapping in real backends later is a one-line change per tool.
"""
import uuid
import datetime
from langchain_core.tools import tool

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

_KNOWLEDGE_BASE = {
    "safety protocol": "Lockout/Tagout (LOTO) must be applied before servicing any "
                        "rotating or pressurized equipment. All personnel must verify "
                        "zero energy state before beginning work.",
    "shift handover": "Shift handover requires: equipment status log, open tickets, "
                       "any anomalies observed, and signed handover checklist.",
    "quality inspection": "Quality inspection occurs every 500 units on the primary "
                           "line, and every 200 units on the secondary line.",
}

_TICKETS = {}


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. for computing tolerances,
    throughput, or unit conversions. Only supports +, -, *, /, (, ), and numbers."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}))
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


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal operations knowledge base for procedures, safety
    protocols, and standard operating procedures (SOPs)."""
    query_l = query.lower()
    matches = [v for k, v in _KNOWLEDGE_BASE.items() if k in query_l or query_l in k]
    if not matches:
        # fallback: fuzzy-ish substring match on words
        for k, v in _KNOWLEDGE_BASE.items():
            if any(word in k for word in query_l.split()):
                matches.append(v)
    if not matches:
        return "No matching entry found in the knowledge base. Consider escalating to a supervisor."
    return "\n---\n".join(matches)


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


ALL_TOOLS = [calculator, check_equipment_status, search_knowledge_base, create_support_ticket]