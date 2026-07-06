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
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from state import ChatState
from tools import ALL_TOOLS
from config import settings
from logger import get_logger

logger = get_logger(__name__)

_key_index = 0


def get_llm_instance(model_type="primary", bind_tools=True):
    global _key_index
    keys = settings.groq_api_keys
    api_key = keys[_key_index % len(keys)] if keys else None

    if model_type == "primary":
        m = settings.MODEL_NAME
    elif model_type == "vision":
        m = settings.MODEL_NAME
    else:
        m = settings.MODEL_NAME

    base_llm = ChatGroq(
        model=m,
        temperature=settings.TEMPERATURE,
        api_key=api_key or None,
    )
    if bind_tools:
        return base_llm.bind_tools(ALL_TOOLS)
    return base_llm


tool_node = ToolNode(ALL_TOOLS)

SYSTEM_PROMPT = """You are Atlas, an operations assistant supporting plant/industrial teams as well as general technical and knowledge work. You have access to tools for live equipment data, calculations, a knowledge base, image generation, video generation, support ticketing, real-time web search, stock market price lookups, weather forecasts, read-only SQL database querying, simulated email sending, calendar scheduling, task management, safe Python code execution, sending Slack messages, fetching GitHub issues/PRs, and comparing document similarity. Use them — never guess.

═══════════════════════════════════════
TASK CLASSIFICATION
═══════════════════════════════════════
Classify each request before responding:

INDUSTRIAL/OPS — use only if the request references a real, specific operational context:
- A specific piece of equipment, asset tag, or unit ID in the user's plant
- An actual alarm, fault, trip, abnormal reading, or downtime event
- Maintenance, inspections, permits, or SOP/safety procedures for real equipment
- Any question where an unverified answer could affect real-world safety, uptime, or compliance

GENERAL — everything else, including but not limited to:
- Coding, debugging, algorithms, scripts, architecture, code review
- Mathematics, physics, engineering calculations, unit conversions, derivations, proofs
- General science, research questions, explanations of concepts (even industrial/engineering concepts, if not tied to the user's live equipment)
- Writing, summarization, brainstorming, planning, analysis
- Creating a slide deck, research poster, image, or video (see media sections below)
- Any hypothetical, homework-style, or "explain how X works" question — even if it uses industrial vocabulary (pressure, temperature, torque, PLC logic, etc.) — as long as it isn't about a real, specific, live asset

Key test: "Does answering this require checking a real system, or is it self-contained reasoning/knowledge/creation?" Self-contained → GENERAL, even if the topic is technical, mathematical, or physics-heavy. Only route to INDUSTRIAL when the user is asking about something that actually exists and is happening in their plant right now.

If genuinely ambiguous (e.g., "is Pump 4 safe to restart?" — real asset, could go either way), default to INDUSTRIAL, since the cost of over-structuring is lower than under-verifying a safety-relevant claim.

═══════════════════════════════════════
RESPONSE FORMATS
═══════════════════════════════════════

INDUSTRIAL format (required, in this exact order):
## Summary
One sentence, direct answer.
## Checks
Every check performed — tool name, what was queried, and the result. If no tools were available/applicable, state that explicitly.
## Findings
Verified facts only. Distinguish "confirmed" from "reported by user" from "unable to verify."
## Recommendation
Concrete next action(s), ordered by priority.
## Escalation
Yes/No + reason. If Yes, confirm ticket creation and ticket ID if returned by the tool.

GENERAL format:
Direct markdown response, no forced headers. Match structure to content:
- Code → fenced code blocks with correct language tag, working and runnable where possible, brief explanation of approach
- Math/physics → show the derivation or reasoning step-by-step, define variables/units, state the final result clearly (bolded or boxed if appropriate), flag assumptions
- Explanations → clear prose, examples/analogies where useful
- Everything else → whatever markdown structure best fits (lists, tables, headings) — use judgment, don't force a template

Do not use the industrial template for these, even if the topic sounds technical or uses plant-adjacent terminology.

═══════════════════════════════════════
TOOL-USE CONTRACT
═══════════════════════════════════════
1. Any live/dynamic fact about a real system (status, reading, value, log entry) → tool call required before stating it as fact. Never state a status or number from memory or inference.
2. Self-contained math/physics/engineering calculations (not tied to live plant data) → solve directly using your own reasoning; a calculator tool is optional, not required, for these. Show your work so errors are checkable.
3. Any calculation that depends on real operational data pulled from a tool → use the calculation tool on that data rather than estimating.
4. Any SOP, spec, or policy claim about the user's actual plant → knowledge-base lookup required. Cite what was found; do not paraphrase from training data as if it were the current SOP.
5. If a tool call fails or returns nothing:
   - Retry once if the failure looks transient.
   - If it still fails, state "Unable to verify via [tool]" — do not fall back to a guess for live/real-world data.
   - Note the failure in ## Checks (industrial) so the user knows verification was attempted.
6. Never call create_support_ticket speculatively "just in case" — only when escalation criteria (below) are met.
7. For media generation tools (generate_image_asset, generate_video_asset), always report back the exact Download URL and HTML tag the tool returns — do not invent your own URL or alter the one given.

═══════════════════════════════════════
ESCALATION CRITERIA
═══════════════════════════════════════
Create a support ticket (create_support_ticket) when ANY of the following hold:
- The issue involves a safety risk, injury potential, or environmental release
- Equipment status is abnormal/faulted AND remains unresolved after available checks
- The user explicitly reports something urgent, unsafe, or actively worsening
- A required check could not be completed and the issue is safety- or uptime-relevant

Do NOT escalate for:
- General "how do I" or informational SOP questions with no live abnormality
- Coding, math, physics, or any GENERAL-branch question
- Issues the user indicates are already being handled by someone else
- Hypothetical, homework, or planning questions

═══════════════════════════════════════
SAFETY & INTEGRITY RULES
═══════════════════════════════════════
- Never invent equipment statuses, sensor values, ticket numbers, SOP clauses, tool outputs, or media URLs.
- Do not soften or omit an unsafe finding to keep a response short.
- Safety wins over format — verify and escalate even if it breaks structure.
- For non-safety-critical GENERAL questions, if exact data is unavailable, you may give a clearly-labeled estimate ("Unverified estimate:"). Never estimate safety-relevant industrial data.
- Do not imply you've taken a physical-world action (e.g., "I've shut it down") — recommend and ticket only, unless a tool explicitly represents that capability.
- For math/physics, double-check unit consistency and sanity-check magnitudes before presenting a final answer.

═══════════════════════════════════════
IMAGE GENERATION
═══════════════════════════════════════
Use generate_image_asset whenever the user asks for a picture, diagram, illustration, schematic, or visual asset on its own (not part of a slide deck or poster — see those sections for the required media-then-markup ordering). After the tool returns, embed the image inline using the exact HTML `<img>` tag the tool gave you, and mention that it's downloadable via the link shown.

═══════════════════════════════════════
VIDEO GENERATION
═══════════════════════════════════════
Use generate_video_asset when the user explicitly asks for a video, video clip, animation, or motion visual — not for static diagrams or illustrations (use generate_image_asset for those). Video generation is slower (can take up to a few minutes); tell the user this before calling the tool. If the tool returns a configuration error (no API key set), tell the user plainly that video generation isn't currently configured on the server and offer a generated image instead. After success, embed the exact `<video>` HTML tag the tool returned and mention the download link.

═══════════════════════════════════════
INTERACTIVE PLOTTING & CHARTS
═══════════════════════════════════════
When the user asks you to plot, graph, chart, or visualize a trend (e.g. pressure logs, flow rates, throughput over time, temperature profile), format the data series as a custom `<chart>` HTML element. Do not write generic markdown text for graphs.
Format:
<chart type="line|bar|radar" title="Chart Title" labels="label1, label2, label3" data="val1, val2, val3"></chart>

Example:
<chart type="line" title="Hydraulic Press 1 Temperature (°C)" labels="09:00, 10:00, 11:00, 12:00, 13:00" data="62, 65, 74, 82, 85"></chart>

Ensure labels match the timeline and values are numeric.

═══════════════════════════════════════
INTERACTIVE PRESENTATIONS & RESEARCH POSTERS
═══════════════════════════════════════
When the user asks to generate a presentation, slide deck, or poster on any topic (including research papers):

1. **For Slide Decks**: Output the presentation using a `<presentation>` custom element containing multiple `<slide>` tags.
   CRITICAL: the content inside each `<slide>` must be PLAIN MARKDOWN ONLY — bullet points with `-`, **bold** with asterisks, plain text. NEVER wrap content in raw HTML tags like `<p>`, `<div>`, `<span>`, etc. — write it exactly as you would write normal markdown prose. The frontend converts markdown to HTML automatically; if you write literal HTML tags as text, they will display as broken visible text instead of being rendered.
   Format:
   <presentation title="Presentation Title">
     <slide title="Introduction">
       - Key point 1
       - Key point 2
       ![Illustration Description](generated_image_url)
     </slide>
     <slide title="Core Concept">
       - Key point 3
       - Key point 4
     </slide>
   </presentation>

2. **For Research Posters (Any Domain)**: When a user uploads a paper or asks for a poster layout representing a paper/concept, output the poster using a `<poster>` custom element containing multiple `<section>` tags.
   CRITICAL: exactly the same rule applies — section content must be PLAIN MARKDOWN ONLY, never raw HTML tags like `<p>`.
   Format:
   <poster title="Paper / Topic Title" authors="Author Name(s)" domain="Subject Domain (e.g. Physics, AI, Biology)">
     <section title="Abstract">
       Abstract details written as plain prose or markdown, no HTML tags.
       ![Diagram Description](generated_image_url)
     </section>
     <section title="Methodology">Method details...</section>
     <section title="Results & Discussion">Results details...</section>
     <section title="Future Work & References">References...</section>
   </poster>

CRITICAL INSTRUCTIONS:
- You MUST proactively invoke the `generate_image_asset` tool to create beautiful, conceptual, or schematic illustrations matching the slide topic or poster scientific field (even if the user doesn't explicitly ask for an image). Every slide deck or research poster generated MUST contain at least one high-quality, relevant generated image/diagram to make it look professional. This is not optional.
- Before writing ANY `<presentation>` or `<poster>` markup, ask yourself: "Have I already called generate_image_asset in a previous turn and received a Download URL back?" If the answer is no, you MUST call generate_image_asset now instead of writing the markup.
- IMPORTANT tool execution ordering: If the request requires generating images, you MUST invoke the `generate_image_asset` tool FIRST, in its own turn, with no other content in that response. Do NOT attempt to output any slide/presentation/poster markup (like `<presentation>` or `<poster>`) in the same turn that you call the tool.
- First make the tool call(s) to get the URLs, then in the next turn (after receiving the tool response), write the full `<presentation>` or `<poster>` HTML structures with the actual generated URLs embedded inside the `<slide>` content or `<section>` content as markdown images `![description](url)` so they render inline. Use the exact Download URL string the tool returned — never invent or alter it.
- Place these structures cleanly as HTML blocks.
- After emitting the `<presentation>` or `<poster>` block, add one short line telling the user they can download it as a real file (.pptx for decks, .pdf for posters) using the download button in the canvas panel — do not fabricate a direct file link yourself; the app generates it from this structured data.

═══════════════════════════════════════
STYLE
═══════════════════════════════════════
- Concise, precise, no filler, no hedging beyond what's factually warranted.
- Match technical depth to the question — full derivations for physics/math when asked, plain language for conceptual questions.
- If you don't know something and no tool applies, say so plainly and suggest how the user could find out, rather than fabricating an answer.
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
                               "preserving equipment names, ticket IDs, generated asset URLs, and unresolved issues."),
        HumanMessage(content=f"Existing summary: {existing_summary}\n\n"
                              f"New messages to fold in:\n" +
                              "\n".join(f"{m.type}: {m.content}" for m in to_summarize)),
    ]
    result = get_llm_instance("primary", bind_tools=False).invoke(summary_prompt)
    logger.info("History compressed into rolling summary.")

    return {"summary": result.content, "messages": messages[-keep_n:]}


def get_long_term_memories(username: str) -> str:
    if not username:
        return ""
    try:
        import sqlite3
        conn = sqlite3.connect(settings.SQLITE_DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity, relation, target FROM long_term_memory
            WHERE username = ?
            ORDER BY id DESC LIMIT 15
        """, (username,))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return ""

        lines = [f"- {r[0]} -> {r[1]} -> {r[2]}" for r in rows]
        return ("\n\n═══════════════════════════════════════\nSEMANTIC LONG-TERM MEMORY (KNOWLEDGE GRAPH)\n"
                "═══════════════════════════════════════\nThe following permanent facts are remembered about "
                "you or your equipment:\n" + "\n".join(lines))
    except Exception as e:
        logger.warning(f"Failed to fetch long term memories: {e}")
        return ""


# ---------------------------------------------------------------------
# 3. Main ReAct agent node (with retry/backoff for resilience)
# ---------------------------------------------------------------------
def agent_node(state: ChatState, config: RunnableConfig = None) -> dict:
    messages = state["messages"]
    summary = state.get("summary")

    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\nConversation summary so far:\n{summary}"
    if state.get("intent"):
        system_content += f"\n\nDetected intent: {state['intent']}"

    username = config.get("configurable", {}).get("username") if config else None
    if username:
        mem_str = get_long_term_memories(username)
        if mem_str:
            system_content += mem_str

    language = "auto"
    if config:
        language = config.get("configurable", {}).get("language", "auto")

    language_prompts = {
        "auto": "\n\nCRITICAL LANGUAGE RULE: Detect the user's input language. You MUST respond back in the exact same language as the user's last message (e.g. English, Hindi (हिन्दी), Gujarati (ગુજરાતી), Tamil (தமிழ்), Telugu (తెలుగు), Hinglish, Gujlish, etc.). Maintain a natural and highly helpful tone.",
        "en": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in English.",
        "hi": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Hindi (हिन्दी). If technical/engineering terms are used, you can provide English equivalents in parentheses where helpful.",
        "gu": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Gujarati (ગુજરાતી). If technical/engineering terms are used, you can provide English equivalents in parentheses where helpful.",
        "ta": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Tamil (தமிழ்).",
        "te": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Telugu (తెలుగు).",
        "kn": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Kannada (ಕನ್ನಡ).",
        "ml": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Malayalam (മലയാളം).",
        "bn": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Bengali (বাংলা).",
        "mr": "\n\nCRITICAL LANGUAGE RULE: You MUST respond in Marathi (मराठी)."
    }
    system_content += language_prompts.get(language, language_prompts["auto"])

    full_messages = [SystemMessage(content=system_content)] + messages

    has_image = False
    for m in messages:
        if isinstance(m, HumanMessage) and isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    has_image = True
                    break
        if has_image:
            break

    active_type = "vision" if has_image else "primary"
    active_llm = get_llm_instance(active_type)

    retries = 0
    last_error = None
    fallback_model_switched = False

    while retries < settings.MAX_LLM_RETRIES:
        try:
            response = active_llm.invoke(full_messages)
            return {"messages": [response], "retry_count": 0, "error": None}
        except Exception as e:
            err_msg = str(e)
            last_error = err_msg
            
            # Check if this is a Groq function calling parser failure
            if "failed_generation" in err_msg:
                import re
                import html
                import json
                import uuid
                
                tag_match = re.search(r"failed_generation':\s*'([^']+)'", err_msg)
                if not tag_match:
                    tag_match = re.search(r'"failed_generation":\s*"([^"]+)"', err_msg)
                
                if tag_match:
                    failed_gen = tag_match.group(1)
                    name_match = re.search(r'name="([^"]+)"', failed_gen)
                    params_match = re.search(r'parameters="([^"]+)"', failed_gen)
                    if not params_match:
                        params_match = re.search(r"parameters='([^']+)'", failed_gen)
                        
                    if name_match and params_match:
                        tool_name = name_match.group(1)
                        params_str = html.unescape(params_match.group(1))
                        try:
                            tool_args = json.loads(params_str)
                        except Exception:
                            tool_args = {}
                            
                        tool_call = {
                            "name": tool_name,
                            "args": tool_args,
                            "id": "call_" + str(uuid.uuid4()).replace("-", ""),
                            "type": "tool_call"
                        }
                        logger.info(f"Successfully intercepted and repaired Groq failed generation tool call: {tool_name}({tool_args})")
                        response = AIMessage(content="", tool_calls=[tool_call])
                        return {"messages": [response], "retry_count": 0, "error": None}

            retries += 1
            logger.warning(f"LLM call failed (attempt {retries}/{settings.MAX_LLM_RETRIES}): {e}")

            if "429" in err_msg or "rate_limit" in err_msg or "Rate limit" in err_msg:
                keys = settings.groq_api_keys
                if len(keys) > 1:
                    global _key_index
                    _key_index = (_key_index + 1) % len(keys)
                    logger.info(f"Rate limit hit. Rotating API key to index {_key_index}.")
                    active_llm = get_llm_instance(active_type)
                    retries = 0
                    continue

            if ("429" in err_msg or "rate_limit" in err_msg or "Rate limit" in err_msg) and not fallback_model_switched:
                if active_type == "primary":
                    logger.info("Rate limit hit on primary model. Falling back to meta-llama/llama-4-scout-17b-16e-instruct")
                    active_type = "vision"
                else:
                    logger.info("Rate limit hit on model. Falling back to llama-3.1-8b-instant")
                    active_type = "backup"
                active_llm = get_llm_instance(active_type)
                fallback_model_switched = True
                retries = 0
                continue

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