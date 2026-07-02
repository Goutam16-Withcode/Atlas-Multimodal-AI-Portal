"""
FastAPI service exposing the chatbot as a real HTTP API and a polished
browser UI for local use.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /               -> web UI
    POST /chat           -> send a message, get a reply
    GET  /history/{thread} -> fetch full message history for a thread
    POST /reset/{thread} -> clear a thread's state (new checkpoint)
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph import chatbot
from logger import get_logger

logger = get_logger("api")
app = FastAPI(title="Industrial Support Chatbot API", version="1.0.0")


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    intent: str | None = None
    needs_escalation: bool = False




def build_ui_html() -> str:
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Atlas Chat</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --panel: rgba(13, 18, 32, 0.86);
      --panel-strong: #11182a;
      --line: rgba(148, 163, 184, 0.16);
      --text: #e6edf8;
      --muted: #96a3bb;
      --accent: #43d3ff;
      --accent-2: #7c6cff;
      --success: #3ddc97;
      --danger: #ff6b81;
      --shadow: 0 22px 70px rgba(0, 0, 0, 0.34);
      --radius: 22px;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background:
        radial-gradient(circle at top left, rgba(67, 211, 255, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(124, 108, 255, 0.18), transparent 32%),
        linear-gradient(180deg, #070b16 0%, #0b1020 100%);
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .sidebar {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow: hidden;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      padding-bottom: 6px;
    }

    .brand-badge {
      width: 46px;
      height: 46px;
      border-radius: 16px;
      display: grid;
      place-items: center;
      font-weight: 900;
      color: #07111f;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: 0 10px 28px rgba(67, 211, 255, 0.22);
    }

    h1, h2, h3, h4, p { margin: 0; }

    .eyebrow {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.75rem;
    }

    .brand h1 {
      font-size: 1.1rem;
      line-height: 1.1;
    }

    .card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(148, 163, 184, 0.12);
      border-radius: 18px;
      padding: 14px;
    }

    .card-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 10px;
    }

    .card h3 {
      font-size: 0.94rem;
      margin-bottom: 4px;
    }

    .helper {
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }

    .field {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }

    .field label {
      color: var(--muted);
      font-size: 0.8rem;
    }

    input, textarea, button {
      font: inherit;
    }

    input, textarea {
      width: 100%;
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 14px;
      padding: 12px 14px;
      color: var(--text);
      background: #0a1020;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
    }

    input:focus, textarea:focus {
      border-color: rgba(67, 211, 255, 0.6);
      box-shadow: 0 0 0 4px rgba(67, 211, 255, 0.12);
    }

    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }

    .chip {
      border: 1px solid rgba(148, 163, 184, 0.15);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      border-radius: 999px;
      padding: 9px 11px;
      cursor: pointer;
      transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease;
    }

    .chip:hover { transform: translateY(-1px); }
    .chip.active {
      background: linear-gradient(135deg, rgba(67, 211, 255, 0.15), rgba(124, 108, 255, 0.15));
      border-color: rgba(67, 211, 255, 0.35);
    }

    .preset-list {
      display: grid;
      gap: 8px;
    }

    .preset {
      text-align: left;
      border: 1px solid rgba(148, 163, 184, 0.12);
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      border-radius: 14px;
      padding: 11px 12px;
      cursor: pointer;
    }

    .preset strong { display: block; margin-bottom: 2px; }
    .preset span { color: var(--muted); font-size: 0.83rem; }

    .btn-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
    }

    .button {
      border: 0;
      border-radius: 14px;
      padding: 11px 14px;
      cursor: pointer;
      transition: transform 0.12s ease, opacity 0.12s ease, filter 0.12s ease;
    }

    .button:hover { transform: translateY(-1px); }
    .button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

    .button.primary {
      color: #06101d;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent), #9fe8ff);
    }

    .button.secondary {
      color: var(--text);
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }

    .button.ghost {
      color: var(--text);
      background: transparent;
      border: 1px solid rgba(148, 163, 184, 0.14);
    }

    .main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
      min-height: calc(100vh - 32px);
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent);
    }

    .title-block {
      display: grid;
      gap: 6px;
    }

    .title-block h2 {
      font-size: 1.2rem;
    }

    .top-status {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 9px 12px;
      background: rgba(148, 163, 184, 0.1);
      border: 1px solid rgba(148, 163, 184, 0.15);
      color: var(--text);
      font-size: 0.86rem;
    }

    .pill.interactive {
      cursor: pointer;
    }

    .pill.interactive:hover {
      border-color: rgba(67, 211, 255, 0.4);
      box-shadow: 0 0 0 3px rgba(67, 211, 255, 0.08);
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--success);
      box-shadow: 0 0 0 4px rgba(61, 220, 151, 0.12);
    }

    .chat-shell {
      display: grid;
      grid-template-rows: 1fr auto;
      min-height: 0;
      background:
        radial-gradient(circle at top center, rgba(67, 211, 255, 0.05), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.01), transparent 120px);
    }

    .messages {
      overflow: auto;
      padding: 24px 18px 12px;
      display: grid;
      gap: 16px;
      align-content: start;
    }

    .welcome {
      max-width: 820px;
      margin: 22px auto 8px;
      text-align: center;
      color: var(--muted);
      padding: 10px 14px;
    }

    .welcome h3 {
      color: var(--text);
      font-size: 1.5rem;
      margin-bottom: 8px;
      letter-spacing: -0.02em;
    }

    .welcome p {
      max-width: 640px;
      margin: 0 auto;
      line-height: 1.6;
    }

    .message {
      width: min(860px, 100%);
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      line-height: 1.55;
      animation: rise 0.16s ease-out;
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message.user {
      margin-left: auto;
      background: linear-gradient(135deg, rgba(67, 211, 255, 0.12), rgba(124, 108, 255, 0.1));
      border-color: rgba(67, 211, 255, 0.16);
    }

    .message.assistant {
      background: rgba(10, 15, 28, 0.92);
    }

    .message.tool {
      background: rgba(255, 107, 129, 0.08);
      border-color: rgba(255, 107, 129, 0.18);
      color: #ffd0d8;
    }

    .message-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 0.82rem;
    }

    .message-body {
      white-space: pre-wrap;
      color: var(--text);
    }

    .section-grid {
      display: grid;
      gap: 10px;
    }

    .section-card {
      padding: 12px 13px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(148, 163, 184, 0.1);
    }

    .section-card h4 {
      color: #8be9ff;
      font-size: 0.92rem;
      margin-bottom: 8px;
    }

    .section-card ul {
      margin: 0;
      padding-left: 18px;
      color: var(--text);
    }

    .section-card p {
      margin: 0;
      white-space: pre-wrap;
      color: var(--text);
    }

    .meta-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.82rem;
    }

    .composer {
      border-top: 1px solid var(--line);
      padding: 14px 18px 18px;
      background: linear-gradient(180deg, transparent, rgba(6, 10, 18, 0.78));
    }

    .composer-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
    }

    textarea {
      min-height: 112px;
      resize: vertical;
      line-height: 1.5;
    }

    .composer-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      align-items: center;
      justify-content: space-between;
    }

    .hint {
      color: var(--muted);
      font-size: 0.82rem;
    }

    .empty {
      margin: 48px auto;
      text-align: center;
      color: var(--muted);
      max-width: 520px;
      padding: 24px 18px;
    }

    .empty h3 {
      color: var(--text);
      font-size: 1.1rem;
      margin-bottom: 8px;
    }

    @media (max-width: 1040px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { order: 2; }
      .main { min-height: 72vh; }
    }

    @media (max-width: 720px) {
      .app { padding: 10px; gap: 10px; }
      .topbar, .messages, .composer { padding-left: 12px; padding-right: 12px; }
      .composer-grid { grid-template-columns: 1fr; }
      .top-status { justify-content: flex-start; }
      .message { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel sidebar">
      <div class="brand">
        <div class="brand-badge">A</div>
        <div>
          <div class="eyebrow">Atlas Chat</div>
          <h1>Real-world operations assistant</h1>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h3>Thread</h3>
            <div class="helper">Keep the same thread ID to preserve history and context.</div>
          </div>
        </div>
        <div class="field">
          <label for="threadId">Thread ID</label>
          <input id="threadId" value="plant-floor-001" autocomplete="off" />
        </div>
        <div class="btn-row">
          <button class="button primary" id="saveThread" type="button">Save</button>
          <button class="button secondary" id="loadHistory" type="button">Load</button>
          <button class="button ghost" id="resetThread" type="button">Reset</button>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h3>Multi-task builder</h3>
            <div class="helper">Select one or more tasks and generate a combined request.</div>
          </div>
        </div>
        <div class="chip-row" id="taskChips">
          <button class="chip" type="button" data-task="status">Status</button>
          <button class="chip" type="button" data-task="sop">SOP</button>
          <button class="chip" type="button" data-task="calculate">Calculate</button>
          <button class="chip" type="button" data-task="ticket">Ticket</button>
          <button class="chip" type="button" data-task="summarize">Summarize</button>
          <button class="chip" type="button" data-task="general">General</button>
        </div>
        <div class="btn-row">
          <button class="button primary" id="buildPrompt" type="button">Build prompt</button>
          <button class="button secondary" id="clearTasks" type="button">Clear</button>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h3>Quick starts</h3>
            <div class="helper">Use a preset, then send or edit the message.</div>
          </div>
        </div>
        <div class="preset-list">
          <button class="preset" data-prompt="Check the status of cooling-pump-2 and tell me the next step."><strong>Status check</strong><span>Equipment health and next action</span></button>
          <button class="preset" data-prompt="Explain the safety protocol for servicing a press and summarize the key steps."><strong>SOP summary</strong><span>Safety procedure and checklist</span></button>
          <button class="preset" data-prompt="Create a combined answer: check the equipment, explain the SOP, and recommend the next step."><strong>Combined task</strong><span>Multiple parts in one response</span></button>
        </div>
      </div>

      <div class="card">
        <h3>System status</h3>
        <div class="helper" id="systemStatus">Ready</div>
        <div class="helper" style="margin-top:8px;">Endpoints: /chat, /history, /reset</div>
      </div>
    </aside>

    <main class="panel main">
      <header class="topbar">
        <div class="title-block">
          <div class="eyebrow">ChatGPT-style workspace</div>
          <h2>Ask multiple things at once and get a structured response</h2>
          <p class="helper">Type one request, select a few task chips, or use a quick start to build a multi-part prompt.</p>
        </div>
        <div class="top-status">
          <div class="pill"><span class="dot" id="healthDot"></span><span id="healthLabel">Online</span></div>
          <div class="pill interactive" id="intentChip" role="button" tabindex="0" aria-label="Current intent. Click for a follow-up prompt.">Intent: <span id="intentLabel">n/a</span></div>
          <div class="pill interactive" id="escalationChip" role="button" tabindex="0" aria-label="Escalation status. Click for a follow-up prompt.">Escalation: <span id="escalationLabel">false</span></div>
        </div>
      </header>

      <section class="chat-shell">
        <div class="messages" id="messages">
          <div class="welcome" id="welcomeCard">
            <h3>Start a support thread</h3>
            <p>Use the prompt builder, pick one or more tasks, and send a request. Responses are rendered as sections so the important parts are easy to scan.</p>
          </div>
        </div>

        <div class="composer">
          <form id="chatForm">
            <div class="composer-grid">
              <div class="field" style="margin-top:0;">
                <label for="message">Message</label>
                <textarea id="message" placeholder="Describe the issue, ask for help, or combine multiple tasks in one message."></textarea>
              </div>
              <button class="button primary" id="sendButton" type="submit">Send</button>
            </div>
            <div class="composer-tools">
              <div class="btn-row" style="margin-top:0;">
                <button class="button secondary" type="button" id="clearMessage">Clear message</button>
              </div>
              <div class="hint" id="chatHint">History is saved per thread and can be reloaded anytime.</div>
            </div>
          </form>
        </div>
      </section>
    </main>
  </div>

  <script>
    const threadInput = document.getElementById('threadId');
    const messageInput = document.getElementById('message');
    const chatForm = document.getElementById('chatForm');
    const messages = document.getElementById('messages');
    const welcomeCard = document.getElementById('welcomeCard');
    const sendButton = document.getElementById('sendButton');
    const saveThreadButton = document.getElementById('saveThread');
    const loadHistoryButton = document.getElementById('loadHistory');
    const resetThreadButton = document.getElementById('resetThread');
    const clearMessageButton = document.getElementById('clearMessage');
    const buildPromptButton = document.getElementById('buildPrompt');
    const clearTasksButton = document.getElementById('clearTasks');
    const intentChip = document.getElementById('intentChip');
    const escalationChip = document.getElementById('escalationChip');
    const intentLabel = document.getElementById('intentLabel');
    const escalationLabel = document.getElementById('escalationLabel');
    const systemStatus = document.getElementById('systemStatus');
    const chatHint = document.getElementById('chatHint');
    const healthLabel = document.getElementById('healthLabel');
    const healthDot = document.getElementById('healthDot');

    const storageKey = 'atlas-thread-id';

    function currentThreadId() {
      return threadInput.value.trim() || 'default-session';
    }

    function setBusy(isBusy) {
      sendButton.disabled = isBusy;
      sendButton.textContent = isBusy ? 'Sending...' : 'Send';
      systemStatus.textContent = isBusy ? 'Working on reply' : 'Ready';
    }

    function scrollToBottom() {
      messages.scrollTop = messages.scrollHeight;
    }

    function removeWelcome() {
      const node = document.getElementById('welcomeCard');
      if (node) {
        node.remove();
      }
    }

    function ensureWelcome() {
      if (document.querySelector('.message') || document.getElementById('welcomeCard')) {
        return;
      }
      const node = document.createElement('div');
      node.className = 'welcome';
      node.id = 'welcomeCard';
      node.innerHTML = '<h3>Start a support thread</h3><p>Use the prompt builder, pick one or more tasks, and send a request. Responses are rendered as sections so the important parts are easy to scan.</p>';
      messages.prepend(node);
    }

    function parseSections(text) {
      const lines = String(text || '').split('\r\n').join('\n').split('\n');
      const sections = [];
      let current = null;

      for (const line of lines) {
        if (line.startsWith('## ')) {
          if (current) sections.push(current);
          current = { title: line.slice(3).trim(), lines: [] };
          continue;
        }
        if (!current) {
          current = { title: 'Reply', lines: [] };
        }
        current.lines.push(line);
      }

      if (current) sections.push(current);
      return sections.filter((section) => section.lines.length || section.title !== 'Reply');
    }

    function renderTextBlock(container, text) {
      const lines = String(text || '').split('\n').filter((line) => line.trim().length);
      const bulletOnly = lines.length > 0 && lines.every((line) => {
        const trimmed = line.trim();
        return trimmed.startsWith('- ') || trimmed.startsWith('* ');
      });

      if (bulletOnly) {
        const list = document.createElement('ul');
        lines.forEach((line) => {
          const item = document.createElement('li');
          item.textContent = line.trim().slice(2);
          list.appendChild(item);
        });
        container.appendChild(list);
        return;
      }

      const paragraph = document.createElement('p');
      paragraph.textContent = text;
      container.appendChild(paragraph);
    }

    function appendMessage(role, content, meta) {
      removeWelcome();

      const message = document.createElement('article');
      message.className = 'message ' + role;

      const header = document.createElement('div');
      header.className = 'message-header';
      const left = document.createElement('span');
      left.textContent = role === 'user' ? 'You' : role === 'assistant' ? 'Atlas' : 'System';
      const right = document.createElement('span');
      right.textContent = currentThreadId();
      header.appendChild(left);
      header.appendChild(right);
      message.appendChild(header);

      const body = document.createElement('div');
      body.className = 'message-body';
      if (role === 'assistant') {
        const sections = parseSections(content);
        if (sections.length > 1) {
          const grid = document.createElement('div');
          grid.className = 'section-grid';
          sections.forEach((section) => {
            const card = document.createElement('section');
            card.className = 'section-card';
            const heading = document.createElement('h4');
            heading.textContent = section.title;
            card.appendChild(heading);
            renderTextBlock(card, section.lines.join('\n').trim());
            grid.appendChild(card);
          });
          body.appendChild(grid);
        } else {
          renderTextBlock(body, content);
        }
      } else {
        renderTextBlock(body, content);
      }
      message.appendChild(body);

      if (meta && meta.length) {
        const metaRow = document.createElement('div');
        metaRow.className = 'meta-row';
        meta.forEach((item) => {
          const pill = document.createElement('span');
          pill.className = 'pill';
          pill.textContent = item;
          metaRow.appendChild(pill);
        });
        message.appendChild(metaRow);
      }

      messages.appendChild(message);
      scrollToBottom();
    }

    function clearMessages() {
      messages.innerHTML = '';
      ensureWelcome();
    }

    function selectedTasks() {
      return Array.from(document.querySelectorAll('.chip.active')).map((button) => button.dataset.task);
    }

    function taskPrompt(task) {
      const prompts = {
        status: 'Check the equipment status for the item I mention and tell me the next step.',
        sop: 'Explain the relevant SOP or safety procedure and summarize the key steps.',
        calculate: 'Do the calculation I need and show the result clearly.',
        ticket: 'If the issue cannot be resolved, create a support ticket and report the ticket details.',
        summarize: 'Summarize the issue, findings, and what should happen next.',
        general: 'Help me with this request in a clear, practical way.'
      };
      return prompts[task] || prompts.general;
    }

    function buildCombinedPrompt() {
      const tasks = selectedTasks();
      if (!tasks.length) {
        messageInput.value = 'Help me solve this as a general operations assistant. If there are multiple parts, handle all of them in order.';
      } else if (tasks.length === 1) {
        messageInput.value = taskPrompt(tasks[0]);
      } else {
        const lines = tasks.map((task, index) => String(index + 1) + '. ' + taskPrompt(task));
        messageInput.value = ['Handle these together in one reply:', ...lines, '', 'Use separate sections for each part and keep it practical.'].join('\n');
      }
      messageInput.focus();
      messageInput.setSelectionRange(messageInput.value.length, messageInput.value.length);
      chatHint.textContent = 'Combined request inserted. Edit it if needed, then send.';
    }

    function suggestFollowUp(kind) {
      const intent = intentLabel.textContent.trim();
      const escalated = escalationLabel.textContent.trim() === 'true';
      if (kind === 'intent') {
        messageInput.value = intent === 'n/a'
          ? 'Please summarize the current issue in summary, checks, findings, recommendation, and escalation format.'
          : 'Explain the current ' + intent + ' issue in summary, checks, findings, recommendation, and escalation format.';
      } else {
        messageInput.value = escalated
          ? 'Continue with escalation steps and provide a structured incident summary.'
          : 'No escalation is needed right now. Provide the next operational step and any verification checks.';
      }
      messageInput.focus();
      messageInput.setSelectionRange(messageInput.value.length, messageInput.value.length);
      chatHint.textContent = 'Follow-up prompt inserted. Edit it and send when ready.';
    }

    async function loadHistory() {
      const threadId = currentThreadId();
      localStorage.setItem(storageKey, threadId);
      systemStatus.textContent = 'Loading history';
      try {
        const response = await fetch('/history/' + encodeURIComponent(threadId));
        if (!response.ok) throw new Error('Failed to load history');
        const data = await response.json();
        clearMessages();
        (data.messages || []).forEach((entry) => appendMessage(entry.role, entry.content));
        chatHint.textContent = data.messages && data.messages.length ? 'Loaded ' + data.messages.length + ' message' + (data.messages.length === 1 ? '' : 's') + ' for ' + threadId + '.' : 'No stored messages for ' + threadId + '.';
        systemStatus.textContent = 'History loaded';
      } catch (error) {
        systemStatus.textContent = 'History unavailable';
        appendMessage('tool', 'Could not load history for ' + threadId + ': ' + error.message);
      }
    }

    async function sendMessage(messageText) {
      const threadId = currentThreadId();
      localStorage.setItem(storageKey, threadId);
      appendMessage('user', messageText);
      setBusy(true);
      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ thread_id: threadId, message: messageText }),
        });
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || 'Request failed');
        }
        const data = await response.json();
        appendMessage('assistant', data.reply, [data.intent ? 'Intent: ' + data.intent : 'Intent: n/a', 'Escalation: ' + Boolean(data.needs_escalation)]);
        intentLabel.textContent = data.intent || 'n/a';
        escalationLabel.textContent = String(Boolean(data.needs_escalation));
        chatHint.textContent = 'Reply stored in thread ' + threadId + '.';
      } catch (error) {
        appendMessage('tool', 'Chat request failed: ' + error.message);
        healthLabel.textContent = 'Offline';
        healthDot.style.background = 'var(--danger)';
        systemStatus.textContent = 'Send failed';
      } finally {
        setBusy(false);
      }
    }

    async function resetThread() {
      const threadId = currentThreadId();
      localStorage.setItem(storageKey, threadId);
      systemStatus.textContent = 'Resetting thread';
      try {
        const response = await fetch('/reset/' + encodeURIComponent(threadId), { method: 'POST' });
        if (!response.ok) throw new Error('Reset failed');
        clearMessages();
        intentLabel.textContent = 'n/a';
        escalationLabel.textContent = 'false';
        chatHint.textContent = 'Thread ' + threadId + ' was reset.';
        systemStatus.textContent = 'Thread reset';
      } catch (error) {
        appendMessage('tool', 'Could not reset ' + threadId + ': ' + error.message);
      }
    }

    async function checkHealth() {
      try {
        const response = await fetch('/health');
        if (!response.ok) throw new Error('Health check failed');
        healthLabel.textContent = 'Online';
        healthDot.style.background = 'var(--success)';
      } catch (error) {
        healthLabel.textContent = 'Offline';
        healthDot.style.background = 'var(--danger)';
      }
    }

    chatForm.addEventListener('submit', async function (event) {
      event.preventDefault();
      const messageText = messageInput.value.trim();
      if (!messageText) return;
      messageInput.value = '';
      await sendMessage(messageText);
    });

    saveThreadButton.addEventListener('click', async function () {
      localStorage.setItem(storageKey, currentThreadId());
      await loadHistory();
    });

    loadHistoryButton.addEventListener('click', loadHistory);
    resetThreadButton.addEventListener('click', resetThread);
    clearMessageButton.addEventListener('click', function () { messageInput.value = ''; });
    buildPromptButton.addEventListener('click', buildCombinedPrompt);
    clearTasksButton.addEventListener('click', function () {
      document.querySelectorAll('.chip.active').forEach((button) => button.classList.remove('active'));
      chatHint.textContent = 'Task choices cleared.';
    });
    intentChip.addEventListener('click', function () { suggestFollowUp('intent'); });
    escalationChip.addEventListener('click', function () { suggestFollowUp('escalation'); });

    document.querySelectorAll('.chip').forEach(function (button) {
      button.addEventListener('click', function () { button.classList.toggle('active'); });
    });

    document.querySelectorAll('.preset').forEach(function (button) {
      button.addEventListener('click', function () {
        messageInput.value = button.dataset.prompt || '';
        messageInput.focus();
      });
    });

    window.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !sendButton.disabled) {
        chatForm.requestSubmit();
      }
    });

    const storedThread = localStorage.getItem(storageKey);
    if (storedThread) {
      threadInput.value = storedThread;
    }

    ensureWelcome();
    checkHealth();
    loadHistory();
  </script>
</body>
</html>
    '''



@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=build_ui_html())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    config = {"configurable": {"thread_id": req.thread_id}}
    try:
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail="internal chatbot error") from e

    reply = result["messages"][-1]
    return ChatResponse(
        thread_id=req.thread_id,
        reply=reply.content,
        intent=result.get("intent"),
        needs_escalation=result.get("needs_escalation", False),
    )


@app.get("/history/{thread_id}")
def history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    if not state or not state.values.get("messages"):
        return {"thread_id": thread_id, "messages": []}

    out = []
    for m in state.values["messages"]:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        else:
            role = "system"
        out.append({"role": role, "content": m.content})
    return {"thread_id": thread_id, "messages": out}


@app.post("/reset/{thread_id}")
def reset(thread_id: str):
    # Overwrite state with an empty message list for a fresh checkpoint.
    config = {"configurable": {"thread_id": thread_id}}
    chatbot.update_state(config, {"messages": [], "summary": None, "needs_escalation": False})
    return {"thread_id": thread_id, "status": "reset"}


@app.get("/health")
def health():
    return {"status": "ok"}
