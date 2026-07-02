"""
All graph nodes for the industrial support chatbot.

Design mirrors a real production support workflow:
  1. classify_intent   -> route the query
  2. compress_history   -> summarize old turns so context never blows up
  3. agent_node          -> ReAct-style LLM call, can invoke tools
  4. tool_node           -> executes any tool calls the agent requested
  5. escalation_check    -> flags safety/urgent keywords for human handoff
  6. handle_error        -> catches LLM/tool failures, retries with backoff
"""
import time
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from state import ChatState
from tools import ALL_TOOLS
from config import settings
from logger import get_logger

logger = get_logger(__name__)

llm = ChatGroq(
    model=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    api_key=settings.GROQ_API_KEY or None,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

tool_node = ToolNode(ALL_TOOLS)

SYSTEM_PROMPT = """You are Atlas, a real-world operations copilot for general support, troubleshooting, planning, calculations, and escalation.
You help plant staff with equipment status checks, safety/SOP lookups,
calculations, ticket creation, summaries, and multi-part requests.

Always answer in this exact structure:
## Summary
- one short sentence that answers the request
## Checks
- list the checks performed, including tool calls or verification steps
## Findings
- list the verified facts or results
## Recommendation
- list the next action the user should take
## Escalation
- Yes or No, plus a short reason

If the user asks for multiple things at once, solve all of them in order and keep each part separate.
If a request is general, answer it directly instead of forcing a narrow support template.

Rules:
- Use tools whenever a question needs live data (equipment status), a
  calculation, or a knowledge-base lookup. Do not guess numbers or statuses.
- If an issue sounds urgent, unsafe, or unresolved after checking available
  info, create a support ticket via create_support_ticket.
- If you cannot verify something, write "Unable to verify" rather than
  inventing an answer.
- Be concise, precise, and use plant-floor language. No fluff.
- If you don't know something and no tool can answer it, say so plainly and
  suggest escalation instead of fabricating an answer.
"""


# ---------------------------------------------------------------------
# 1. Intent classification (cheap, deterministic-ish routing)
# ---------------------------------------------------------------------
def classify_intent(state: ChatState) -> dict:
    last_msg = state["messages"][-1]
    text = last_msg.content.lower() if isinstance(last_msg.content, str) else ""

    if any(kw in text for kw in settings.ESCALATE_ON_KEYWORDS):
        intent = "escalation"
    elif any(kw in text for kw in ["status", "equipment", "pump", "press", "belt", "sensor"]):
        intent = "technical_support"
    elif any(kw in text for kw in ["protocol", "sop", "procedure", "policy", "handover"]):
        intent = "knowledge_query"
    else:
        intent = "general"

    logger.info(f"Intent classified as: {intent}")
    return {"intent": intent, "needs_escalation": intent == "escalation"}


# ---------------------------------------------------------------------
# 2. Context compression: summarize once history grows too long
# ---------------------------------------------------------------------
def compress_history(state: ChatState) -> dict:
    messages = state["messages"]
    if len(messages) <= settings.MAX_MESSAGES_BEFORE_SUMMARY:
        return {}

    keep_n = settings.KEEP_LAST_N_AFTER_SUMMARY
    to_summarize = messages[:-keep_n]
    existing_summary = state.get("summary") or ""

    summary_prompt = [
        SystemMessage(content="Summarize the following support conversation concisely, "
                               "preserving equipment names, ticket IDs, and unresolved issues."),
        HumanMessage(content=f"Existing summary: {existing_summary}\n\n"
                              f"New messages to fold in:\n" +
                              "\n".join(f"{m.type}: {m.content}" for m in to_summarize)),
    ]
    result = llm.invoke(summary_prompt)
    logger.info("History compressed into rolling summary.")

    # LangGraph's add_messages reducer needs explicit removal semantics;
    # simplest safe approach: keep only the last N messages going forward.
    return {"summary": result.content, "messages": messages[-keep_n:]}


# ---------------------------------------------------------------------
# 3. Main ReAct agent node (with retry/backoff for resilience)
# ---------------------------------------------------------------------
def agent_node(state: ChatState) -> dict:
    messages = state["messages"]
    summary = state.get("summary")

    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\nConversation summary so far:\n{summary}"
    if state.get("intent"):
        system_content += f"\n\nDetected intent: {state['intent']}"

    full_messages = [SystemMessage(content=system_content)] + messages

    retries = 0
    last_error = None
    while retries < settings.MAX_LLM_RETRIES:
        try:
            response = llm_with_tools.invoke(full_messages)
            return {"messages": [response], "retry_count": 0, "error": None}
        except Exception as e:
            last_error = str(e)
            retries += 1
            logger.warning(f"LLM call failed (attempt {retries}/{settings.MAX_LLM_RETRIES}): {e}")
            time.sleep(settings.RETRY_BACKOFF_SECONDS * retries)

    logger.error(f"LLM call failed after {settings.MAX_LLM_RETRIES} retries: {last_error}")
    fallback = AIMessage(
        content="I'm having trouble reaching the language model right now. "
                "I've logged this issue — please retry shortly, or contact "
                "support directly if this is urgent."
    )
    return {"messages": [fallback], "retry_count": retries, "error": last_error}


# ---------------------------------------------------------------------
# 4. Routing helpers (conditional edges)
# ---------------------------------------------------------------------
def route_after_agent(state: ChatState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def route_after_intent(state: ChatState) -> str:
    # Everything currently funnels through the same ReAct agent, which has
    # all tools available. Intent is used for prompting/logging/escalation,
    # but kept as a distinct branch point so new specialized nodes (e.g. a
    # dedicated compliance-review node) can be slotted in per intent later.
    if state.get("needs_escalation"):
        return "escalation_check"
    return "compress_history"


# ---------------------------------------------------------------------
# 5. Escalation check — ensures urgent/safety issues always get a ticket
# ---------------------------------------------------------------------
def escalation_check(state: ChatState) -> dict:
    logger.info("Escalation path triggered — ensuring ticket creation is prioritized.")
    notice = SystemMessage(
        content="URGENT/SAFETY-RELATED MESSAGE DETECTED. You must create a "
                "support ticket with priority='critical' or 'high' using the "
                "create_support_ticket tool, in addition to any other help you give."
    )
    return {"messages": [notice]}