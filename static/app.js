/* SGLang Workbench v2 front-end — vanilla JS, no build step */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  settings: {},
  chats: [],
  chatId: null,
  messages: [],
  streaming: false,
  abort: null,
  tools: [],
  mcp: [],
  server: {},
  job: {},
  engine: {},
  attachments: [],
  optimistic: null,
  personas: [],
  stats: null,
  harnesses: [],
  harness: null,
  h2RunId: null,
  h2Timer: null,
  lessons: [],
  paletteIndex: 0,
};

const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/* Background news (job state, artifacts, server events) goes through notify(): the same
   message is never shown twice within its cooldown. User-initiated actions use toast()
   directly — those are a response to a click and should always appear. */
const NOTIFY_SEEN = new Map();
function notify(msg, kind = '', cooldown = 8000) {
  const now = Date.now();
  if (now - (NOTIFY_SEEN.get(msg) || 0) < cooldown) return false;
  NOTIFY_SEEN.set(msg, now);
  if (NOTIFY_SEEN.size > 60) NOTIFY_SEEN.clear();
  toast(msg, kind);
  return true;
}

function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), 3800);
}

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) throw new Error((await r.text()).slice(0, 300) || r.statusText);
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r.text();
}

/* --------------------------- remembered state --------------------------- */
/* Everything the user changed about the workspace is remembered twice:
   localStorage for an instant restore, and /api/ui-state so a new browser or
   another machine on the LAN opens the same layout. */

const UI_KEY = 'wb.ui.v1';
const UI_DEFAULTS = {
  chatId: null, tab: 'model', panel: false, sidebar: true, theme: 'dark',
  density: 'comfortable', sideW: 280, panelW: 400, measure: 800,
  modelSource: 'cache', modelTask: '', modelQuery: '',
  harness: null, persona: null, drafts: {}, scroll: {}, filesTab: 'chats',
  jobAck: null,
};

const UI = {
  state: { ...UI_DEFAULTS },
  _saveTimer: null,
  _remote: null,

  load() {
    try {
      const raw = localStorage.getItem(UI_KEY);
      if (raw) Object.assign(this.state, JSON.parse(raw));
    } catch (e) { /* private mode */ }
    return this.state;
  },

  async hydrate() {
    // ask the server only when this browser has never seen the workspace
    if (localStorage.getItem(UI_KEY)) return this.state;
    try {
      const r = await api('/api/ui-state');
      if (r && r.state) Object.assign(this.state, r.state);
    } catch (e) { /* offline */ }
    return this.state;
  },

  set(patch, { save = true } = {}) {
    Object.assign(this.state, patch);
    if (save) this.save();
  },

  save() {
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      try { localStorage.setItem(UI_KEY, JSON.stringify(this.state)); } catch (e) { /* quota */ }
      this.pushRemote();
    }, 350);
  },

  pushRemote() {
    clearTimeout(this._remote);
    this._remote = setTimeout(() => {
      const { drafts, scroll, ...durable } = this.state;   // keep the payload small
      api('/api/ui-state', { method: 'POST', body: durable }).catch(() => {});
    }, 2500);
  },

  draft(chatId, text) {
    if (!chatId) return;
    if (text) this.state.drafts[chatId] = text;
    else delete this.state.drafts[chatId];
    this.save();
  },

  rememberScroll(chatId, top) {
    if (!chatId) return;
    this.state.scroll[chatId] = Math.round(top);
    this.save();
  },
};

function applyLayout() {
  const app = $('#app');
  if (!app) return;
  const s = UI.state;
  document.documentElement.dataset.theme = s.theme || 'dark';
  app.dataset.density = s.density || 'comfortable';
  app.style.setProperty('--side-w', `${Math.max(220, Math.min(460, s.sideW || 288))}px`);
  app.style.setProperty('--panel-w', `${Math.max(320, Math.min(760, s.panelW || 410))}px`);
  app.style.setProperty('--measure', `${Math.max(620, Math.min(1200, s.measure || 830))}px`);
  app.classList.toggle('no-side', !s.sidebar);
  app.classList.toggle('with-panel', !!s.panel);
  $('#panel').classList.toggle('open', !!s.panel);
  $('#panelResize').hidden = !s.panel;
  const openBtn = $('#openPanel');
  if (openBtn) openBtn.classList.toggle('on', !!s.panel);
  const scrimEl = $('#scrim');
  if (scrimEl) scrimEl.hidden = !s.panel;
  $('#sideResize').hidden = !s.sidebar;
  const btn = $('#densityBtn');
  if (btn) btn.classList.toggle('on', s.density === 'compact');
}

function bindLayout() {
  // drag a handle, persist the width
  const drag = (handle, { onMove, onEnd }) => {
    if (!handle) return;
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      handle.classList.add('dragging');
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => onMove(ev);
      const up = () => {
        handle.classList.remove('dragging');
        handle.removeEventListener('pointermove', move);
        handle.removeEventListener('pointerup', up);
        onEnd && onEnd();
      };
      handle.addEventListener('pointermove', move);
      handle.addEventListener('pointerup', up);
    });
  };

  drag($('#panelResize'), {
    onMove: (ev) => {
      const w = Math.round(window.innerWidth - ev.clientX);
      UI.state.panelW = Math.max(320, Math.min(760, w));
      $('#app').style.setProperty('--panel-w', `${UI.state.panelW}px`);
    },
    onEnd: () => UI.save(),
  });

  drag($('#sideResize'), {
    onMove: (ev) => {
      UI.state.sideW = Math.max(220, Math.min(460, Math.round(ev.clientX)));
      $('#app').style.setProperty('--side-w', `${UI.state.sideW}px`);
    },
    onEnd: () => UI.save(),
  });

  const scrim = $('#scrim');
  const syncScrim = () => {
    if (!scrim) return;
    const open = $('#panel').classList.contains('open')
      || $('#sidebar').classList.contains('open');
    scrim.hidden = !open;
  };
  if (scrim) scrim.onclick = () => {
    $('#panel').classList.remove('open');
    $('#app').classList.remove('with-panel');
    $('#sidebar').classList.remove('open');
    $('#panelResize').hidden = true;
    syncScrim();
    UI.set({ panel: false });
  };

  $('#panelResize').addEventListener('dblclick', () => { UI.set({ panelW: 410 }); applyLayout(); });
  $('#sideResize').addEventListener('dblclick', () => { UI.set({ sideW: 288 }); applyLayout(); });

  $('#densityBtn').onclick = () => {
    UI.set({ density: UI.state.density === 'compact' ? 'comfortable' : 'compact' });
    applyLayout();
  };

  // remember the composer draft of the open chat
  const input = $('#input');
  input.addEventListener('input', () => UI.draft(state.chatId, input.value));

  // remember scroll position per chat
  let scrollTimer = null;
  $('#messages').addEventListener('scroll', () => {
    clearTimeout(scrollTimer);
    scrollTimer = setTimeout(() => UI.rememberScroll(state.chatId, $('#messages').scrollTop), 300);
  });
}

/* ----------------------------- markdown ----------------------------- */
/* Rendered with marked (CommonMark + GFM), sanitised with DOMPurify, code
   highlighted with highlight.js. Everything is vendored: no network. */

const LANG_ALIASES = { js: 'javascript', ts: 'typescript', py: 'python', sh: 'bash', yml: 'yaml', md: 'markdown' };

/* Models that inline their chain of thought: move it into the reasoning block. */
function splitThink(msg) {
  let text = msg.content || '';
  const m = text.match(/^\s*<think(?:ing)?>([\s\S]*?)<\/(?:think|thinking)>\s*/i);
  if (m) {
    if (!msg.reasoning) msg.reasoning = m[1].trim();
    text = text.slice(m[0].length);
  }
  text = text.replace(/<\/?(?:think|thinking)>/gi, '');
  return text;
}

function codeBlock(code, infostring = '') {
  if (!code || !code.trim()) return '';   // a model that emits empty fences should not draw empty boxes
  const rawLang = (infostring || '').trim().split(/\s+/)[0].toLowerCase();
  const lang = LANG_ALIASES[rawLang] || rawLang;
  if (rawLang === 'mermaid') {
    return `<div class="mermaid-wrap" data-mermaid="${esc(code)}"><pre class="mermaid-src" hidden>${esc(code)}</pre></div>`;
  }
  let body;
  if (window.hljs && lang && window.hljs.getLanguage(lang)) {
    try {
      body = window.hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
    } catch (e) {
      body = esc(code);
    }
  } else if (window.hljs) {
    try { body = window.hljs.highlightAuto(code).value; } catch (e) { body = esc(code); }
  } else {
    body = esc(code);
  }
  return `<pre data-code="${esc(code)}"><code class="hljs" data-lang="${esc(rawLang)}">${body}</code></pre>`;
}

function configureMarked() {
  if (!window.marked || window.marked.__configured) return;
  const renderer = new window.marked.Renderer();
  renderer.code = (code, infostring) => codeBlock(code, infostring);
  renderer.codespan = (code) => `<code class="inline">${esc(code)}</code>`;
  renderer.link = (href, title, text) => {
    const safe = /^(https?:|mailto:|#)/i.test(href || '') ? href : '#';
    return `<a href="${esc(safe)}" target="_blank" rel="noopener noreferrer">${text}</a>`;
  };
  renderer.image = (href, title, text) =>
    /^(https?:|data:image\/)/i.test(href || '')
      ? `<img src="${esc(href)}" alt="${esc(text || '')}" title="${esc(title || '')}" loading="lazy">`
      : `<span class="muted">[image blocked: ${esc(href || '')}]</span>`;
  window.marked.use({ renderer, gfm: true, breaks: true, headerIds: false, mangle: false });
  window.marked.__configured = true;
}

function md(src) {
  if (!src) return '';
  if (!window.marked || !window.DOMPurify) return `<pre data-code="${esc(src)}"><code>${esc(src)}</code></pre>`;
  configureMarked();
  let html;
  try {
    html = window.marked.parse(String(src));
  } catch (e) {
    return `<pre>${esc(String(src))}</pre>`;
  }
  return window.DOMPurify.sanitize(html, {
    ADD_ATTR: ['target', 'rel', 'data-lang', 'data-code', 'data-mermaid', 'class', 'loading', 'hidden'],
    FORBID_TAGS: ['script', 'style', 'object', 'embed', 'form'],
    FORBID_ATTR: ['onerror', 'onload', 'onclick', 'srcdoc'],
    ALLOW_DATA_ATTR: true,
  });
}


/* --------------------------- clear history --------------------------- */
async function clearHistory() {
  const n = state.chats.length;
  if (!n) return toast('No chats to clear');
  const ok = confirm(
    `Delete all ${n} conversation${n === 1 ? '' : 's'}?\n\n` +
    'Messages and generated files go with them. Settings, memory, prompts and personas are kept.'
  );
  if (!ok) return;
  try {
    const r = await api('/api/chats?confirm=all', { method: 'DELETE' });
    UI.set({ chatId: null, drafts: {}, scroll: {} });
    state.chats = [];
    state.messages = [];
    state.chatId = null;
    renderChats();
    await newChat();
    notify(`cleared ${r.deleted_chats} chats · ${r.deleted_messages} messages`);
  } catch (e) {
    toast(e.message, 'err');
  }
}

/* ------------------------------ chats ------------------------------ */
function renderChats() {
  const q = $('#chatSearch').value.trim().toLowerCase();
  const list = $('#chatList');
  const items = state.chats.filter((c) => !q || (c.title || '').toLowerCase().includes(q));
  list.innerHTML = '';
  if (!items.length) {
    list.innerHTML = `<div class="muted" style="padding:10px">${q ? 'No match' : 'No chats yet'}</div>`;
    return;
  }
  for (const c of items) {
    const el = document.createElement('div');
    el.className = 'chat-item' + (c.id === state.chatId ? ' active' : '');
    el.innerHTML = `${c.pinned ? '<span class="pin" title="Pinned">●</span>' : ''}
      <span class="t"></span><span class="x" title="Pin / unpin">◎</span><span class="x" title="Delete">×</span>`;
    el.querySelector('.t').textContent = c.title || 'Untitled';
    el.onclick = (e) => {
      if (e.target.classList.contains('x')) return;
      openChat(c.id);
    };
    const [pin, del] = el.querySelectorAll('.x');
    pin.onclick = async (e) => {
      e.stopPropagation();
      await api(`/api/chats/${c.id}`, { method: 'PATCH', body: { pinned: !c.pinned } });
      await loadChats();
    };
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${c.title}"?`)) return;
      await api(`/api/chats/${c.id}`, { method: 'DELETE' });
      if (state.chatId === c.id) { state.chatId = null; state.messages = []; renderMessages(); }
      await loadChats();
      toast('Chat deleted');
    };
    list.appendChild(el);
  }
}

async function loadChats() {
  state.chats = await api('/api/chats');
  renderChats();
}

/* ----------------------------- messages ----------------------------- */
function msgEl(m, idx) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = m.role === 'user' ? 'You' : 'AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (m.reasoning) {
    const th = document.createElement('div');
    th.className = 'think';
    th.textContent = m.reasoning;
    bubble.appendChild(th);
  }
  if (m.tools?.rounds) {
    const tl = document.createElement('div');
    tl.className = 'tool-log';
    tl.innerHTML = `<div class="th"><span class="tico"></span><b>${m.tools.rounds} tool round(s)</b></div>`;
    const events = m.tools.events || [];
    if (events.length) {
      const pre = document.createElement('pre');
      pre.textContent = events.map((e) =>
        `${e.tool} (${e.server})\n  args: ${JSON.stringify(e.arguments)}\n  → ${(e.result || '').slice(0, 600)}`
      ).join('\n\n');
      tl.appendChild(pre);
    }
    bubble.appendChild(tl);
  }
  const body = document.createElement('div');
  body.className = 'body';
  if (m.role === 'assistant') body.innerHTML = md(m.content);
  else body.textContent = m.content;
  bubble.appendChild(body);

  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  const mk = (label, fn, title) => {
    const b = document.createElement('button');
    b.className = 'mini';
    b.textContent = label;
    if (title) b.title = title;
    b.onclick = fn;
    actions.appendChild(b);
    return b;
  };

  mk('Copy', async () => { await navigator.clipboard.writeText(m.content); toast('Copied'); });
  if (m.role === 'user' && m.id && !m.id.startsWith('tmp-')) {
    mk('Edit', () => startEdit(m, body, actions));
  }
  if (!m.id.startsWith('tmp-')) {
    mk('Fork here', async () => {
      const r = await api(`/api/chats/${state.chatId}/fork`, { method: 'POST', body: { message_id: m.id } });
      await loadChats();
      await openChat(r.id);
      toast(`Forked (${r.messages} messages)`);
    });
    mk('Delete', async () => {
      if (!confirm('Delete this message?')) return;
      await api(`/api/messages/${m.id}`, { method: 'DELETE' });
      await openChat(state.chatId);
    });
  }
  if (m.role === 'assistant' && idx === state.messages.length - 1) {
    mk('Regenerate', regenerate);
  }
  if (m.meta?.usage?.completion_tokens) {
    const s = document.createElement('span');
    s.className = 'mini';
    s.style.cursor = 'default';
    s.textContent = `${m.meta.usage.prompt_tokens}→${m.meta.usage.completion_tokens} tok · ${m.meta.elapsed_s}s`;
    actions.appendChild(s);
  }
  if (m.artifacts && m.artifacts.length) {
    for (const a of m.artifacts) bubble.appendChild(artCard(a, { inline: true }));
  }
  if (m.edited) {
    const e = document.createElement('span');
    e.className = 'edited';
    e.textContent = 'edited';
    actions.appendChild(e);
  }
  bubble.appendChild(actions);
  wrap.append(avatar, bubble);
  if (m.role === 'assistant') { attachCodeSave(bubble); enrich(bubble); }
  return wrap;
}

function startEdit(m, body, actions) {
  const box = document.createElement('textarea');
  box.className = 'edit-box';
  box.value = m.content;
  body.replaceWith(box);
  box.focus();
  const row = document.createElement('div');
  row.className = 'msg-actions';
  const save = document.createElement('button');
  save.className = 'mini on';
  save.textContent = 'Save';
  save.onclick = async () => {
    await api(`/api/messages/${m.id}`, { method: 'PATCH', body: { content: box.value } });
    toast('Message updated — regenerate to apply');
    await openChat(state.chatId);
  };
  const cancel = document.createElement('button');
  cancel.className = 'mini';
  cancel.textContent = 'Cancel';
  cancel.onclick = () => openChat(state.chatId);
  row.append(save, cancel);
  actions.replaceWith(row);
}

function renderMessages() {
  const box = $('#messages');
  box.innerHTML = '';
  if (!state.messages.length) {
    box.appendChild(emptyState());
    updateCtxMeter();
    return;
  }
  const thread = document.createElement('div');
  thread.className = 'thread';
  state.messages.forEach((m, i) => thread.appendChild(msgEl(m, i)));
  box.appendChild(thread);
  box.scrollTop = box.scrollHeight;
  updateCtxMeter();
}

function emptyState() {
  const tpl = document.createElement('div');
  tpl.innerHTML = `
    <div class="empty" id="emptyState">
      <div class="empty-logo">S</div>
      <h2>What can I help with?</h2>
      <p>Ask anything — streaming answers, memory, MCP tools, model switching and attachments, all served locally by SGLang.</p>
      <div class="suggestions">
        <button class="sug">Explain what the memory store injects into each prompt</button>
        <button class="sug">Which context window can this machine actually handle, and why?</button>
        <button class="sug">Write a Python function that tails a log and flags OOM crashes</button>
        <button class="sug">Summarise what MCP is in five lines</button>
      </div>
    </div>`;
  tpl.querySelectorAll('.sug').forEach((b) => {
    b.onclick = () => { $('#input').value = b.textContent; autosize(); $('#input').focus(); };
  });
  return tpl.firstElementChild;
}

/* --------------------------- chat flow --------------------------- */
async function openChat(id) {
  state.chatId = id;
  UI.set({ chatId: id });
  state.messages = await api(`/api/chats/${id}/messages`);
  const chat = state.chats.find((c) => c.id === id);
  $('#chatTitle').textContent = chat?.title || 'Chat';
  renderMessages();
  renderChats();
  $('#sidebar').classList.remove('open');
  $('#scrim').hidden = !$('#panel').classList.contains('open');
  // restore what this chat looked like the last time it was open
  const input = $('#input');
  input.value = UI.state.drafts[id] || '';
  input.style.height = 'auto';
  const top = UI.state.scroll[id];
  const scroller = $('#messages');
  scroller.scrollTop = typeof top === 'number' ? top : scroller.scrollHeight;
}

async function newChat() {
  const c = await api('/api/chats', { method: 'POST', body: {} });
  state.chatId = c.id;
  state.messages = [];
  $('#chatTitle').textContent = 'New chat';
  renderMessages();
  await loadChats();
  $('#sidebar').classList.remove('open');
  $('#scrim').hidden = !$('#panel').classList.contains('open');
  $('#input').focus();
}

function addLocal(role, content) {
  const m = { id: 'tmp-' + Math.random(), role, content };
  state.messages.push(m);
  const box = $('#messages');
  if (!box.querySelector('.thread')) {
    box.innerHTML = '';
    const t = document.createElement('div');
    t.className = 'thread';
    box.appendChild(t);
  }
  const el = msgEl(m, state.messages.length - 1);
  box.querySelector('.thread').appendChild(el);
  box.scrollTop = box.scrollHeight;
  return { m, el };
}

function autosize() {
  const t = $('#input');
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 220) + 'px';
  updateCtxMeter();
}

function updateCtxMeter() {
  const ctx = Number(state.settings.context_length) || 12288;
  const chars = state.messages.reduce((a, m) => a + (m.content || '').length, 0) +
                (state.settings.system_prompt || '').length + $('#input').value.length +
                state.attachments.reduce((a, f) => a + f.text.length, 0);
  const tokens = Math.ceil(chars / 3);
  const pct = Math.min(999, Math.round((tokens / ctx) * 100));
  const el = $('#ctxMeter');
  el.textContent = `ctx ${pct}%`;
  el.style.color = pct > 95 ? 'var(--danger)' : pct > 75 ? 'var(--warn)' : 'var(--fg-mute)';
  el.title = `~${tokens} prompt tokens of ${ctx}`;
}

async function sendMessage(textOverride) {
  if (state.streaming) return;
  if (JOB_ACTIVE.includes(state.job.state) && !(state.server && state.server.up)) {
    return toast(`model is ${state.job.state} — wait for ready`, 'err');
  }
  const text = (textOverride ?? $('#input').value).trim();
  if (!text) return;
  if (!state.chatId) await newChat();

  $('#input').value = '';
  autosize();
  UI.draft(state.chatId, '');   // the prompt left the composer; do not restore it as a draft
  addLocal('user', text + (state.attachments.length ? `\n\n[${state.attachments.length} attachment(s)]` : ''));
  const attachments = [...state.attachments];
  state.attachments = [];
  renderAttachments();

  const assistant = { id: 'tmp-' + Math.random(), role: 'assistant', content: '', reasoning: '' };
  state.messages.push(assistant);
  const aEl = msgEl(assistant, state.messages.length - 1);
  aEl.classList.add('streaming');
  $('#messages .thread').appendChild(aEl);
  const bodyEl = aEl.querySelector('.body');
  bodyEl.hidden = true;
  bodyEl.innerHTML = '';
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.innerHTML = '<span class="td"></span><span class="td"></span><span class="td"></span><em>generating</em>';
  aEl.querySelector('.bubble').insertBefore(typing, bodyEl);
  let revealed = false;
  const reveal = () => {
    if (revealed) return;
    revealed = true;
    typing.classList.add('out');
    bodyEl.hidden = false;
    setTimeout(() => typing.remove(), 320);
  };
  const thinkEl = document.createElement('details');
  thinkEl.className = 'think';
  thinkEl.open = true;
  thinkEl.innerHTML = '<summary><span class="th-label">thinking</span><span class="th-dots">…</span></summary><div class="think-body"></div>';
  thinkEl.style.display = 'none';
  const thinkBody = thinkEl.querySelector('.think-body');
  aEl.querySelector('.bubble').prepend(thinkEl);

  state.streaming = true;
  state.abort = new AbortController();
  setSendState();
  const t0 = performance.now();
  let chars = 0;

  try {
    const res = await openChatStream({ chat_id: state.chatId, message: text, attachments });

    const listEl = $('#messages');
    const hintEl = $('#streamHint');
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        const ev = /^event: (.*)$/m.exec(part)?.[1];
        const dataRaw = /^data: (.*)$/m.exec(part)?.[1];
        if (!ev || !dataRaw) continue;
        const data = JSON.parse(dataRaw);
        if (ev === 'delta') {
          reveal();
          assistant.content += data.text;
          chars += data.text.length;
          const now = performance.now();
          // re-render and re-layout at most every 120 ms: marked + syntax
          // highlighting over the whole answer on every token is quadratic
          if (now - (assistant._lastRender || 0) > 120) {
            assistant._lastRender = now;
            bodyEl.innerHTML = md(assistant.content) + '<span class="cursor"></span>';
            const secs = Math.max(0.001, (now - t0) / 1000);
            hintEl.textContent = `~${Math.round(chars / 4 / secs)} tok/s`;
            listEl.scrollTop = listEl.scrollHeight;
          }
        } else if (ev === 'reasoning') {
          assistant.reasoning += data.text;
          thinkEl.style.display = 'block';
          thinkBody.innerHTML = md(assistant.reasoning);
        } else if (ev === 'tool') {
          if (data.state === 'running') {
            const tl = document.createElement('div');
            tl.className = 'tool-log';
            tl.innerHTML = `<div class="th"><span class="tico"></span><b>${esc(data.tool)}</b> · ${esc(data.server)} · running…</div><pre></pre>`;
            tl.querySelector('pre').textContent = JSON.stringify(data.arguments, null, 2);
            aEl.querySelector('.bubble').appendChild(tl);
          } else {
            const last = [...aEl.querySelectorAll('.tool-log')].pop();
            if (last) {
              last.querySelector('.th').innerHTML = `<span class="tico"></span><b>${esc(data.tool)}</b> · ${esc(data.server)} · done`;
              last.querySelector('pre').textContent = data.result || '(no output)';
            }
          }
          listEl.scrollTop = listEl.scrollHeight;
        } else if (ev === 'artifact') {
          assistant.artifacts = assistant.artifacts || [];
          assistant.artifacts.push(data);
          aEl.querySelector('.bubble').appendChild(artCard(data, { inline: true }));
          notify(`File written: ${data.name}`, '', 3000);
        } else if (ev === 'error') {
          toast(data.message, 'err');
          reveal();
          assistant.content += `\n\n**Error:** ${data.message}`;
          bodyEl.innerHTML = md(assistant.content);
        } else if (ev === 'done') {
          reveal();
          assistant.content = data.text || assistant.content;
          assistant.meta = { usage: data.usage, elapsed_s: data.elapsed_s };
          bodyEl.innerHTML = md(assistant.content);
          $('#statsBar').textContent =
            `${data.usage?.prompt_tokens ?? '?'} prompt · ${data.usage?.completion_tokens ?? '?'} completion · ${data.elapsed_s}s` +
            (data.rounds ? ` · ${data.rounds} tool round(s)` : '');
        }
      }
    }
  } catch (e) {
    reveal();
    if (e.name === 'AbortError') {
      assistant.content += '\n\n_[stopped]_';
      bodyEl.innerHTML = md(assistant.content);
    } else {
      toast(e.message, 'err');
      assistant.content += `\n\n**Error:** ${e.message}`;
      bodyEl.innerHTML = md(assistant.content);
    }
  } finally {
    state.streaming = false;
    state.abort = null;
    setSendState();
    $('#streamHint').textContent = '';
    await openChat(state.chatId);
    await loadChats();
  }
}

async function openChatStream(body) {
  /** POST /api/chat and hand back the raw SSE response (caller reads the body). */
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: state.abort.signal,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.text()).slice(0, 300));
  return res;
}

async function regenerate() {
  if (state.streaming || !state.chatId) return;
  state.streaming = true;
  state.abort = new AbortController();
  setSendState();
  try {
    // the answer is rendered from the server afterwards, so the stream only
    // has to be consumed — but it must not be re-implemented here
    const res = await openChatStream({ chat_id: state.chatId, regenerate: true });
    await res.text();
  } catch (e) {
    if (e.name !== 'AbortError') toast(e.message, 'err');
  } finally {
    state.streaming = false;
    state.abort = null;
    setSendState();
    await openChat(state.chatId);
  }
}

function setSendState() {
  const b = $('#sendBtn');
  b.classList.toggle('stop', state.streaming);
  b.innerHTML = state.streaming
    ? '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="M4 12l16-8-6 16-3-6-7-2z"/></svg>';
}

/* --------------------------- attachments --------------------------- */
function renderAttachments() {
  const row = $('#attachRow');
  row.innerHTML = '';
  if (!state.attachments.length) { row.hidden = true; updateCtxMeter(); return; }
  row.hidden = false;
  state.attachments.forEach((f, i) => {
    const chip = document.createElement('span');
    chip.className = 'file-chip';
    chip.innerHTML = `<span></span><button title="Remove">×</button>`;
    const kind = f.kind && f.kind !== 'text'
      ? `${f.kind.toUpperCase()}${f.pages ? ` · ${f.pages}p` : f.slides ? ` · ${f.slides} slides` : ''}`
      : `${Math.round(f.text.length / 1024)} KB`;
    chip.querySelector('span').textContent = `${f.name} — ${kind}${f.truncated ? ' · trimmed' : ''}`;
    chip.title = `${f.chars || f.text.length} chars extracted`;
    chip.querySelector('button').onclick = () => {
      state.attachments.splice(i, 1);
      renderAttachments();
    };
    row.appendChild(chip);
  });
  updateCtxMeter();
}

const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|ya?ml|toml|py|js|ts|tsx|jsx|html|css|sh|sql|rs|go|java|c|cpp|h|log|xml|ini|conf)$/i;
const MAX_UPLOAD = 25 * 1024 * 1024;

function contextCharBudget() {
  const ctx = Number(state.settings.context_length) || 12288;
  // ~3 chars per token, leaving room for the answer -- the server trims to
  // the same budget, so this is only about not shipping text we will drop
  return Math.max(4000, Math.floor(ctx * 3 - 3000));
}

async function uploadDocument(file) {
  if (!state.chatId) await newChat();
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`/api/chats/${state.chatId}/uploads`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error((await res.text()).slice(0, 300));
  return res.json();
}

async function addFiles(files) {
  const budget = contextCharBudget();
  for (const file of files) {
    if (file.size > MAX_UPLOAD) { toast(`${file.name} is too large (25 MB max)`, 'err'); continue; }
    try {
      if (TEXT_EXT.test(file.name) || !file.name.includes('.')) {
        const text = await file.text();
        state.attachments.push({ name: file.name, text, kind: 'text', chars: text.length });
      } else {
        toast(`Extracting ${file.name}…`);
        const r = await uploadDocument(file);
        let text = r.text;
        let truncated = false;
        if (text.length > budget) { text = text.slice(0, budget); truncated = true; }
        state.attachments.push({
          name: r.name, text, kind: r.kind, pages: r.pages, slides: r.slides,
          chars: r.chars, truncated,
          // keep the path: the model needs it to run a tool on the file
          path: (r.artifact && r.artifact.path) || null,
          manifest: r.manifest || null,
        });
        const warn = r.warning ? ` — ${r.warning}` : '';
        toast(`${r.name}: ${r.chars.toLocaleString()} chars${r.pages ? ` from ${r.pages} pages` : ''}${truncated ? ' (trimmed to context)' : ''}${warn}`);
        loadFiles();
      }
    } catch (e) {
      toast(e.message, 'err');
    }
  }
  renderAttachments();
}


/* ---------- rich post-processing: mermaid, html preview, code bar ---------- */
let mermaidSeq = 0;

function initMermaid() {
  if (!window.mermaid || window.mermaid.__ready) return;
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: document.documentElement.dataset.theme === 'light' ? 'default' : 'dark',
    themeVariables: { fontFamily: 'ui-sans-serif, system-ui' },
  });
  window.mermaid.__ready = true;
}

async function renderMermaid(scope) {
  initMermaid();
  if (!window.mermaid) return;
  const nodes = [...scope.querySelectorAll('.mermaid-wrap')].filter((n) => !n.dataset.done);
  if (!nodes.length) return;
  for (const node of nodes) {
    const srcEl = node.querySelector('.mermaid-src');
    const src = srcEl ? srcEl.textContent : node.dataset.mermaid;
    if (!src || !src.trim()) continue;
    const id = `mmd-${++mermaidSeq}`;
    try {
      const { svg } = await window.mermaid.render(id, src.trim());
      node.innerHTML = svg;
      const svgEl = node.querySelector('svg');
      if (svgEl) {
        const vb = (svgEl.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number);
        if (vb.length === 4 && vb[2] > 0 && vb[3] > 0) {
          svgEl.setAttribute('width', Math.round(vb[2]));
          svgEl.setAttribute('height', Math.round(vb[3]));
          svgEl.style.maxWidth = '100%';
          svgEl.style.height = 'auto';
        }
      }
      node.dataset.done = '1';
    } catch (e) {
      node.dataset.done = '1';
      node.innerHTML = `<div class="mermaid-err">mermaid error: ${esc(String(e.message || e)).slice(0, 300)}</div>`;
    }
  }
}

function attachCodeBars(scope) {
  scope.querySelectorAll('pre[data-code]').forEach((pre) => {
    if (pre.querySelector('.code-bar')) return;
    const code = pre.dataset.code;
    const bar = document.createElement('div');
    bar.className = 'code-bar';
    const copy = document.createElement('button');
    copy.textContent = 'copy';
    copy.onclick = async (e) => { e.stopPropagation(); await navigator.clipboard.writeText(code); toast('Copied'); };
    const save = document.createElement('button');
    save.textContent = 'save file';
    save.onclick = (e) => {
      e.stopPropagation();
      const lang = (pre.querySelector('code')?.dataset.lang || '').toLowerCase();
      const ext = { python: 'py', py: 'py', js: 'js', javascript: 'js', ts: 'ts', json: 'json',
                    bash: 'sh', sh: 'sh', html: 'html', css: 'css', md: 'md', sql: 'sql',
                    yaml: 'yml', yml: 'yml', csv: 'csv', mermaid: 'mmd' }[lang] || 'txt';
      const name = prompt('File name', `snippet.${ext}`);
      if (name) saveTextAsFile(name, code);
    };
    bar.append(copy, save);
    pre.appendChild(bar);
  });
}

function attachHtmlPreviews(scope) {
  scope.querySelectorAll('pre[data-code]').forEach((pre) => {
    const lang = (pre.querySelector('code')?.dataset.lang || '').toLowerCase();
    if (!['html', 'svg'].includes(lang)) return;
    if (pre.nextElementSibling?.classList.contains('html-preview')) return;
    const wrap = document.createElement('div');
    wrap.className = 'html-preview';
    wrap.innerHTML = `<div class="hp-head"><span>sandboxed preview</span><span class="grow"></span>
      <button class="mini">hide</button></div><iframe sandbox="allow-scripts" loading="lazy"></iframe>`;
    const frame = wrap.querySelector('iframe');
    const probe = `<script>
      const send=()=>{const b=document.body,c=document.documentElement;
        const h=Math.min(1400,Math.max(48,Math.max(b.scrollHeight,b.offsetHeight,c.scrollHeight)));
        parent.postMessage({__htmlPreview:1,h},'*')};
      addEventListener('load',()=>{send();setTimeout(send,150);setTimeout(send,600)});
    <\/script>`;
    frame.srcdoc = pre.dataset.code + probe;
    frame.style.height = '90px';
    wrap.querySelector('button').onclick = () => {
      frame.hidden = !frame.hidden;
      wrap.querySelector('button').textContent = frame.hidden ? 'show' : 'hide';
    };
    pre.after(wrap);
  });
}

function enrich(scope) {
  attachCodeBars(scope);
  attachHtmlPreviews(scope);
  renderMermaid(scope);
}

/* --------------------------- model switcher --------------------------- */
async function openModelMenu() {
  const menu = $('#modelMenu');
  if (!menu.hidden) { menu.hidden = true; return; }
  menu.innerHTML = '<div class="m-hint">loading models…</div>';
  menu.hidden = false;
  const serving = (state.server && state.server.model) || null;
  let models = [];
  try {
    models = await api('/api/models?source=cache');
  } catch (e) {
    menu.innerHTML = `<div class="m-hint">${esc(e.message)}</div>`;
    return;
  }
  const loadable = state.canLoadModels !== false;
  const busy = JOB_ACTIVE.includes((state.job || {}).state);
  const rows = models
    .filter((m) => !/gguf|onnx|embed|rerank|whisper|clip|docling/i.test(m.id))
    .slice(0, 12)
    .map((m) => {
      const cur = m.id === serving;
      return `<button class="m-item" data-model="${esc(m.id)}" ${cur || !loadable || busy ? 'disabled' : ''}>
        ${cur ? '<span class="cur"></span>' : '<span style="width:7px"></span>'}
        <span class="nm">${esc(m.id)}</span>
        <span class="sz">${m.size_gb ? m.size_gb + ' GB' : ''}${cur ? ' · serving' : ''}</span>
      </button>`;
    }).join('');

  menu.innerHTML = `
    <div class="m-head">${busy ? `busy: ${esc(state.job.state)} ${esc((state.job.model || '').split('/').pop())}` : 'Switch model — one click restarts the server'}</div>
    ${rows || '<div class="m-hint">No downloaded models found</div>'}
    ${loadable ? '' : `<div class="m-hint">⚠ model loading is disabled in this deployment (run the host loader)</div>`}
    <div class="m-sep"></div>
    <button class="m-item" data-act="search"><span style="width:7px"></span><span class="nm">Search &amp; download models…</span></button>
    <button class="m-item" data-act="stop" ${state.server && state.server.up ? '' : 'disabled'}>
      <span style="width:7px"></span><span class="nm">Stop server</span></button>`;

  menu.querySelectorAll('[data-model]').forEach((b) => {
    b.onclick = () => switchModel(b.dataset.model);
  });
  menu.querySelector('[data-act="search"]').onclick = () => { menu.hidden = true; openPanel('model'); };
  menu.querySelector('[data-act="stop"]').onclick = async () => {
    menu.hidden = true;
    await api('/api/server/stop', { method: 'POST', body: {} });
    toast('Server stopped');
    refreshState();
  };
}

// One place decides what the header/composer look like while a model changes.
// Sources: the click that started the switch (optimistic) or an in-flight job
// observed after a reload — both must show the same thing.
function applySwitchingUI() {
  const chip = $('#modelChip');
  const target = state.optimistic
    || (JOB_ACTIVE.includes((state.job || {}).state) ? state.job.model : null);
  const name = (target || '').split('/').pop();
  if (target) {
    chip.classList.add('switching');
    chip.firstChild.textContent = `→ ${name}`;
    $('#input').disabled = true;
    $('#input').placeholder = `switching to ${name}…`;
    $('#sendBtn').disabled = true;
  } else {
    chip.classList.remove('switching');
    $('#input').disabled = false;
    $('#input').placeholder = 'Send a message…  (⏎ send · ⇧⏎ newline)';
    $('#sendBtn').disabled = false;
  }
}

function setSwitching(on, model) {
  state.optimistic = on ? model : null;
  applySwitchingUI();
}

async function switchModel(model) {
  $('#modelMenu').hidden = true;
  if (JOB_ACTIVE.includes(state.job.state)) {
    return toast(`already ${state.job.state} ${state.job.model || ''}`.trim(), 'err');
  }
  const payload = {
    model,
    context_length: +($('#ctxInput')?.value || state.settings.context_length),
    mem_fraction_static: +($('#memInput')?.value || state.settings.mem_fraction_static),
    max_running_requests: +($('#reqInput')?.value || state.settings.max_running_requests),
    mlx_cache_gb: +($('#cacheInput')?.value || state.settings.mlx_cache_gb),
  };
  try {
    setSwitching(true, model);
    await api('/api/server/start', { method: 'POST', body: payload });
    $('#jobBanner').hidden = false;
    await pollJob();
    setSwitching(false);
    refreshState();
  } catch (e) {
    setSwitching(false);
    toast(e.message, 'err');
  }
}


/* --------------------------- model load banner --------------------------- */
// An engine this container does not supervise (compose `gpu` profile, host
// loader, remote box) still has a lifecycle worth showing: image pull, model
// download, warm-up. /api/state carries it as `engine`.
const ENGINE_ACTIVE = ['pulling', 'downloading', 'starting', 'waiting'];
const ENGINE_LABELS = {
  pulling: 'pulling image',
  downloading: 'downloading model',
  starting: 'starting engine',
  waiting: 'waiting for engine',
  failed: 'engine failed',
  ready: 'engine ready',
};
let engineWatched = false;

const engineLabel = (e) => ENGINE_LABELS[(e || {}).phase] || (e || {}).phase || 'engine';
const engineMessage = (e) => {
  const msg = String((e || {}).message || '');
  const norm = (s) => String(s || '').toLowerCase()
    .replace(/\b(the|a|an)\b/g, '').replace(/[^a-z]+/g, ' ').trim();
  return norm(msg) === norm(engineLabel(e)) ? '' : msg;
};
const fmtGB = (n) => n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${Math.max(1, Math.round(n / 1e6))} MB`;
const fmtRate = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB/s` : `${Math.max(1, Math.round(n / 1e3))} kB/s`;

function renderJobBanner() {
  const bar = $('#jobBanner');
  if (!bar) return;
  const j = state.job || {};
  const e = state.engine || {};
  const active = JOB_ACTIVE.includes(j.state);
  const failed = j.state === 'failed' && !jobErrDismissed;
  const engPhase = e.phase || '';
  const engActive = ENGINE_ACTIVE.includes(engPhase);
  const engFailed = engPhase === 'failed';

  if (engActive) engineWatched = true;
  if (!active && !failed && !engActive && !engFailed && engPhase === 'ready' && engineWatched) {
    engineWatched = false;
    notify(`engine ready${e.model ? ` · ${String(e.model).split('/').pop()}` : ''}`);
  }

  if (!active && !failed && !engActive && !engFailed) {
    bar.hidden = true;
    bar.classList.remove('err', 'done');
    bar.title = '';
    const key = `${j.state}|${j.model || ''}|${j.finished || 0}`;
    if (j.state === 'ready' && j.finished && key !== UI.state.jobAck) {
      UI.set({ jobAck: key });
      // only announce a load this tab actually watched; a page load is not news
      if (jobWatched) {
        const name = String(j.model || '').split('/').pop();
        const pool = Number(j.pool || j.max_total_num_tokens || 0);
        notify(`ready · ${name}${pool ? ` · pool ${pool.toLocaleString()} tok` : ''}`);
      }
    }
    jobWatched = false;
    return;
  }

  // Engine provisioned elsewhere: no cancel, no local log — just progress.
  if (!active && !failed && (engActive || engFailed)) {
    const pct = Number.isFinite(e.progress) ? Math.max(0, Math.min(100, e.progress)) : null;
    bar.hidden = false;
    bar.classList.toggle('err', engFailed);
    bar.classList.remove('done');
    bar.title = e.hint || '';
    $('#jbState').textContent = engineLabel(e);
    $('#jbModel').textContent = e.model ? String(e.model).split('/').pop() : '';
    $('#jbCancel').hidden = true;
    $('#jbLog').hidden = true;

    const fill = $('#jbFill');
    fill.classList.toggle('indet', pct === null);
    fill.style.width = pct === null ? '' : `${pct}%`;

    const parts = [];
    if (e.bytes_done) {
      parts.push(`${fmtGB(e.bytes_done)}${e.bytes_total ? ` / ${fmtGB(e.bytes_total)}` : ''}`);
    }
    if (e.rate_bps) parts.push(fmtRate(e.rate_bps));
    if (e.elapsed_s) parts.push(`${Math.round(e.elapsed_s)}s`);
    const extra = engPhase === 'pulling' ? engineMessage(e) : '';
    if (!e.bytes_done && extra) parts.push(extra);
    if (engPhase === 'waiting' && e.hint) parts.push(e.hint);
    if (engFailed && e.hint) parts.push(e.hint);
    $('#jbNum').textContent = parts.join(' · ');
    return;
  }

  $('#jbLog').hidden = false;
  if (active) jobWatched = true;

  bar.hidden = false;
  bar.classList.toggle('err', !!failed);
  bar.classList.remove('done');
  bar.title = '';
  $('#jbState').textContent = failed ? 'failed' : j.state;
  $('#jbModel').textContent = j.model ? j.model.split('/').pop() : '';
  $('#jbCancel').hidden = !active;

  const pct = Number(j.progress || 0);
  const fill = $('#jbFill');
  // download progress is real; starting/stopping has none — show an indeterminate sweep
  fill.classList.toggle('indet', active && j.state !== 'downloading');
  fill.style.width = `${active && j.state === 'downloading' ? pct : Math.max(6, pct)}%`;

  const parts = [];
  if (j.bytes_total && j.bytes_done && j.state === 'downloading') {
    parts.push(`${(j.bytes_done / 1e9).toFixed(2)} / ${(j.bytes_total / 1e9).toFixed(1)} GB`);
  } else if (j.message) {
    parts.push(j.message);
  }
  if (failed && !j.message) parts.push('load failed');
  $('#jbNum').textContent = parts.join(' · ');
}

/* ----------------------------- state ----------------------------- */
const JOB_ACTIVE = ['preparing', 'downloading', 'starting', 'stopping'];
let jobErrDismissed = false;
let jobWatched = false;   // did this tab see the load in progress?

async function refreshState() {
  try {
    const s = await api('/api/state');
    state.settings = s.settings;
    state.tools = s.tools || [];
    state.mcp = s.mcp || [];
    state.server = s.server || {};
    state.job = s.job || {};
    state.engine = s.engine || {};
    state.canLoadModels = s.can_load_models !== false;
    const up = !!s.server?.up;
    const jobActive = JOB_ACTIVE.includes(state.job.state);
    const eng = state.engine;
    const engActive = ENGINE_ACTIVE.includes(eng.phase);
    const engFailed = eng.phase === 'failed';
    const engPct = Number.isFinite(eng.progress) ? ` ${Math.round(eng.progress)}%` : '';

    $('#statusDot').className = 'dot ' + (jobActive || engActive ? 'busy' : up ? 'on' : 'off');
    $('#statusText').textContent = jobActive
      ? `${state.job.state} ${state.job.progress ? Math.round(state.job.progress) + '%' : ''}`.trim()
      : engActive ? `${engineLabel(eng)}${engPct}`.trim()
      : engFailed ? 'engine failed'
      : up ? `${((s.server && s.server.model) || state.settings.model || '').split('/').pop()} · up` : 'server down';
    $('#statusText').title = engActive || engFailed ? (eng.hint || eng.message || '') : '';
    $('#brandSub').textContent = up
      ? `local · Metal · ${(s.server?.context_length || state.settings.context_length).toLocaleString()} ctx`
      : engActive ? `local · ${engineLabel(eng)}${engPct}`
      : engFailed ? 'local · engine failed'
      : 'local · Metal · offline';

    $('#modelChip').textContent = s.server?.model || (engActive && eng.model) || state.settings.model;
    $('#ctxChip').textContent = `ctx ${Number(s.server?.context_length || s.settings.context_length).toLocaleString()}`;
    $('#memChip').textContent = s.settings.memory_enabled ? 'memory on' : 'memory off';
    if (s.harness && $('#harnessChip')) $('#harnessChip').textContent = `harness ${s.harness.name}`;
    $('#memChip').className = 'chip' + (s.settings.memory_enabled ? ' on' : '');
    $('#toolCount').textContent = mcpAvailableToolCount();
    $('#toolsToggle').classList.toggle('on', !!s.settings.tools_enabled);
    $('#toolsToggle').title = mcpToolsTooltip();
    renderMcpRow();
    $('#thinkToggle').classList.toggle('on', !!s.settings.enable_thinking);
    const chipBusy = jobActive || engActive;
    $('#jobChip').hidden = !(chipBusy || engFailed);
    if (chipBusy || engFailed) {
      const pct = state.job.state === 'downloading' ? ` ${Math.round(state.job.progress || 0)}%` : '';
      $('#jobChip').className = 'chip ' + (engFailed ? 'err' : 'busy');
      $('#jobChip').textContent = jobActive
        ? `${state.job.state}${pct}`
        : `${engFailed ? 'engine failed' : engineLabel(eng) + engPct}`.trim();
    }
    if (JOB_ACTIVE.includes(state.job.state)) jobErrDismissed = false;
    // rendering problems are UI problems: they must not flip the connection pill
    for (const name of ['renderServerCard', 'renderJobBanner', 'applySwitchingUI', 'applySettingsToForm']) {
      const fn = globalThis[name];
      if (typeof fn !== 'function') continue;
      try { fn(); } catch (e) { console.warn('[ui]', name, e && e.message); }
    }
  } catch (e) {
    const why = (e && e.message) || String(e);
    $('#statusDot').className = 'dot off';
    $('#statusText').textContent = 'workbench offline';
    $('#statusText').title = why;
    $('#serverPill').dataset.err = why;
  }
}

function renderServerCard() {
  const j = state.job || {};
  const e = state.engine || {};
  const s = state.server || {};
  const up = !!s.up;
  const engActive = ENGINE_ACTIVE.includes(e.phase);
  const engPct = Number.isFinite(e.progress) ? Math.max(0, Math.min(100, e.progress)) : null;
  $('#scDot').className = 'dot ' + (JOB_ACTIVE.includes(j.state) || engActive ? 'busy' : up ? 'on' : 'off');
  $('#scModel').textContent = s.model || e.model || state.settings.model || '—';
  const pool = Number(s.max_total_num_tokens || 0);
  const poolTxt = pool ? ` · pool ${pool.toLocaleString()} tok` : '';
  $('#scMeta').textContent = up
    ? `ctx ${Number(s.context_length || 0).toLocaleString()}${poolTxt} · RSS ${s.rss_gb ?? '?'} GB`
    : engActive
      ? `${engineLabel(e)}${engPct === null ? '' : ` ${Math.round(engPct)}%`}${engPhase === 'pulling' && engineMessage(e) ? ` · ${engineMessage(e)}` : ''}`
      : e.phase === 'failed'
        ? `engine failed${e.hint ? `: ${e.hint}` : ''}`
        : (j.state !== 'idle' ? `${j.state}: ${j.message || ''}` : 'server not running');
  $('#scMeta').classList.toggle('warn', up && pool > 0 && pool < 4096);
  $('#scMeta').title = up && pool > 0 && pool < 4096
    ? 'KV pool is too small for real conversations — lower the context window or pick a smaller model'
    : (e.hint || '');

  const prog = $('#loadProgress');
  const bar = $('#loadBar');
  const active = JOB_ACTIVE.includes(j.state);
  prog.hidden = !(active || engActive);
  bar.classList.toggle('indet', engActive && engPct === null);
  bar.style.width = `${Math.round(engActive ? (engPct ?? 0) : (j.progress || 0))}%`;

  const err = $('#loadError');
  if (j.state === 'failed' && j.error) { err.hidden = false; err.textContent = j.error; }
  else if (e.phase === 'failed' && e.hint) { err.hidden = false; err.textContent = e.hint; }
  else err.hidden = true;

  $('#cancelLoad').hidden = !active;
  $('#stopServer').disabled = !up || state.canLoadModels === false;
  $('#stopServer').title = state.canLoadModels === false
    ? 'the engine is managed outside the app' : '';
}

function setVal(sel, v) { const el = $(sel); if (el) el.value = v; }
function setCheck(sel, v) { const el = $(sel); if (el) el.checked = !!v; }
function setTxt(sel, v) { const el = $(sel); if (el) el.textContent = v; }

function applySettingsToForm() {
  const s = state.settings || {};
  setVal('#ctxInput', s.context_length);
  setVal('#memInput', s.mem_fraction_static);
  setVal('#reqInput', s.max_running_requests);
  setVal('#cacheInput', s.mlx_cache_gb);
  setVal('#sysPrompt', s.system_prompt || '');
  setVal('#tempRange', s.temperature); setTxt('#tempVal', s.temperature);
  setVal('#toppRange', s.top_p); setTxt('#toppVal', s.top_p);
  setVal('#topkRange', s.top_k); setTxt('#topkVal', s.top_k);
  setVal('#maxtokRange', s.max_tokens); setTxt('#maxtokVal', s.max_tokens);
  setVal('#repRange', s.repeat_penalty); setTxt('#repVal', s.repeat_penalty);
  setVal('#roundsRange', s.max_tool_rounds); setTxt('#roundsVal', s.max_tool_rounds);
  setVal('#seedInput', s.seed ?? '');
  setVal('#stopInput', Array.isArray(s.stop) ? s.stop.join(',') : (s.stop || ''));
  setCheck('#memEnabled', s.memory_enabled);
  setCheck('#toolsEnabled', s.tools_enabled);
  setCheck('#thinkEnabled', s.enable_thinking);
  setCheck('#jsonEnabled', s.json_mode);
  setCheck('#streamEnabled', s.stream);
  setCheck('#autoTitleEnabled', s.auto_title);
  try { updateCtxWarn(); } catch (e) { /* panel not rendered yet */ }
}

function updateCtxWarn() {
  const v = +$('#ctxInput').value;
  const w = $('#ctxWarn');
  if (v > 14336) w.textContent = '⚠ The API refuses anything above 14336: 12288 is tested, 16384 dies with a Metal OOM.';
  else if (v > 12288) w.textContent = '⚠ 12288 is the tested-safe maximum on this Mac.';
  else w.textContent = '';
}

/* ---------------------------- model tab ---------------------------- */
async function loadModels() {
  const src = $('#modelSource').value;
  const q = $('#modelQuery').value.trim();
  const task = $('#modelTask').value;
  const list = $('#modelList');
  list.innerHTML = '<div class="muted" style="padding:8px">loading…</div>';
  try {
    const models = await api(`/api/models?source=${src}&q=${encodeURIComponent(q)}&task=${encodeURIComponent(task)}`);
    list.innerHTML = '';
    if (!models.length) { list.innerHTML = '<div class="muted" style="padding:8px">nothing found</div>'; return; }
    for (const m of models) {
      const el = document.createElement('div');
      el.className = 'item' + (m.id === state.settings.model ? ' active' : '');
      const left = document.createElement('div');
      left.style.minWidth = '0';
      left.innerHTML = `<div class="mid"></div><div class="meta"></div>`;
      left.querySelector('.mid').textContent = m.id;
      const bits = [];
      if (m.size_gb) bits.push(`${m.size_gb} GB`);
      if (m.downloads) bits.push(`${m.downloads.toLocaleString()} dl`);
      if (m.gated) bits.push('gated');
      left.querySelector('.meta').textContent = bits.join(' · ') || m.source;
      const right = document.createElement('div');
      right.className = 'row';
      right.style.flex = '0 0 auto';
      right.innerHTML = `<span class="badge ${m.downloaded ? 'dl' : ''}">${m.downloaded ? 'local' : 'remote'}</span>
        <button class="mini">Load</button>`;
      right.querySelector('button').onclick = (e) => {
        e.stopPropagation();
        startServer(m.id);
      };
      el.append(left, right);
      el.onclick = () => {
        state.settings.model = m.id;
        $('#modelChip').textContent = m.id;
        toast(`Selected ${m.id}`);
        loadModels();
      };
      list.appendChild(el);
    }
  } catch (e) {
    list.innerHTML = `<div class="muted" style="padding:8px">${esc(e.message)}</div>`;
  }
}

async function startServer(model) {
  if (state.canLoadModels === false) {
    const e = state.engine || {};
    const busy = ENGINE_ACTIVE.includes(e.phase);
    const pct = Number.isFinite(e.progress) ? ` ${Math.round(e.progress)}%` : '';
    toast(busy
      ? `${engineLabel(e)}${pct} — wait for the engine before loading another model`
      : 'model loading is disabled here: the engine is managed outside the app (compose MODEL in .env)', 'err');
    openPanel('model');
    return;
  }
  openPanel('model');
  $('#jobBanner').hidden = false;
  const payload = {
    model,
    context_length: +$('#ctxInput').value,
    mem_fraction_static: +$('#memInput').value,
    max_running_requests: +$('#reqInput').value,
    mlx_cache_gb: +$('#cacheInput').value,
  };
  try {
    await api('/api/server/start', { method: 'POST', body: payload });
    pollJob();
  } catch (e) {
    toast(e.message, 'err');
  }
}

async function pollJob() {
  let guard = 0;
  while (guard++ < 400) {
    await refreshState();
    if (!JOB_ACTIVE.includes(state.job.state)) {
      if (state.job.state === 'ready') loadModels();
      return;
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

/* ---------------------------- memory ---------------------------- */
async function loadMemory() {
  const rows = await api('/api/memory');
  const list = $('#memList');
  list.innerHTML = '';
  if (!rows.length) { list.innerHTML = '<div class="muted" style="padding:8px">No memories yet</div>'; return; }
  for (const r of rows) {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `<span></span><button class="mini">delete</button>`;
    el.firstElementChild.textContent = r.text;
    el.querySelector('button').onclick = async () => {
      await api(`/api/memory/${r.id}`, { method: 'DELETE' });
      loadMemory(); toast('Memory removed');
    };
    list.appendChild(el);
  }
}

/* ---------------------------- prompts ---------------------------- */
async function loadPrompts() {
  const rows = await api('/api/prompts');
  const list = $('#promptList');
  list.innerHTML = '';
  if (!rows.length) { list.innerHTML = '<div class="muted" style="padding:8px">No prompts saved</div>'; return; }
  for (const r of rows) {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `<div><div class="mid"></div><div class="meta"></div></div>
      <div class="row" style="flex:0 0 auto"><button class="mini u">use</button><button class="mini d">×</button></div>`;
    el.querySelector('.mid').textContent = r.title;
    el.querySelector('.meta').textContent = r.body.slice(0, 60);
    el.querySelector('.u').onclick = () => {
      $('#input').value = r.body;
      autosize();
      $('#input').focus();
      toast('Prompt inserted');
    };
    el.querySelector('.d').onclick = async () => {
      await api(`/api/prompts/${r.id}`, { method: 'DELETE' });
      loadPrompts();
    };
    list.appendChild(el);
  }
}

/* ------------------------------ MCP ------------------------------ */
let mcpOpenServer = null;   // which server's tools are expanded in the composer

function mcpServerLabel(s) {
  const n = (s.tools || []).length;
  if (s.enabled === false) return 'off';
  if (!s.connected) return 'not connected';
  return `${n} tool${n === 1 ? '' : 's'}`;
}

function mcpAvailableToolCount() {
  return (state.mcp || []).reduce(
    (n, s) => n + (s.connected && s.enabled !== false ? (s.tools || []).length : 0), 0);
}

function mcpToolsTooltip() {
  const servers = state.mcp || [];
  const on = !!(state.settings || {}).tools_enabled;
  if (!servers.length) return 'No MCP servers registered';
  const lines = servers.map((s) => {
    const names = (s.tools || []).map((t) => t.name).join(', ');
    const head = `${s.name} (${mcpServerLabel(s)})`;
    return names ? `${head}: ${names}` : head;
  });
  const total = mcpAvailableToolCount();
  return `${on ? `Tools are sent to the model (${total}).` : 'Tools are NOT sent to the model.'}\n\n${lines.join('\n')}`;
}

function renderMcpDetail() {
  const box = $('#mcpDetail');
  if (!box) return;
  const s = (state.mcp || []).find((x) => x.name === mcpOpenServer);
  if (!s) { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = '';

  const hd = document.createElement('div');
  hd.className = 'hd';
  const nm = document.createElement('span');
  nm.textContent = s.name;
  const sub = document.createElement('span');
  sub.className = 'sub';
  const where = s.bundled ? 'built in' : 'added';
  const how = s.enabled === false ? 'disabled' : (s.connected ? 'connected' : 'not connected');
  sub.textContent = `${where} · ${how}`;
  hd.append(nm, sub);
  box.appendChild(hd);

  const tools = s.tools || [];
  if (!tools.length) {
    const none = document.createElement('div');
    none.className = 'none';
    none.textContent = s.enabled === false
      ? 'This server is disabled — turn it on in the MCP panel to use its tools.'
      : 'No tools discovered. Connect the server from the MCP panel.';
    box.appendChild(none);
    return;
  }
  for (const t of tools) {
    const row = document.createElement('div');
    row.className = 'mcp-tool';
    const code = document.createElement('code');
    code.textContent = t.name;
    const desc = document.createElement('div');
    desc.className = 'desc';
    // the first line of the description is the summary; the rest is detail
    const lines = (t.description || '').split('\n').filter((l) => l.trim());
    const summary = lines[0] || '';
    desc.appendChild(document.createTextNode(summary));
    if (t.params && t.params.length) {
      const p = document.createElement('div');
      p.className = 'params';
      const req = new Set(t.required || []);
      p.textContent = '(' + t.params.map((x) => (req.has(x) ? x : `[${x}]`)).join(', ') + ')';
      desc.appendChild(p);
    }
    row.append(code, desc);
    box.appendChild(row);
  }
}

function renderMcpRow() {
  const row = $('#mcpRow');
  if (!row) return;
  const servers = state.mcp || [];
  row.innerHTML = '';
  if (!servers.length) {
    row.hidden = true;
    mcpOpenServer = null;
    renderMcpDetail();
    return;
  }
  row.hidden = false;
  for (const s of servers) {
    const chip = document.createElement('button');
    chip.type = 'button';
    const cls = s.enabled === false ? 'off' : (s.connected ? 'on' : 'idle');
    chip.className = 'mcp-chip ' + cls + (s.name === mcpOpenServer ? ' active' : '');
    chip.title = s.connected
      ? `${s.name}: ${mcpServerLabel(s)} — click to list the tools`
      : `${s.name}: ${mcpServerLabel(s)}`;
    const dot = document.createElement('span');
    dot.className = 'dot';
    const nm = document.createElement('span');
    nm.className = 'nm';
    nm.textContent = s.name;
    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = mcpServerLabel(s);
    chip.append(dot, nm, n);
    chip.onclick = () => {
      mcpOpenServer = mcpOpenServer === s.name ? null : s.name;
      renderMcpRow();
      renderMcpDetail();
    };
    row.appendChild(chip);
  }
  const manage = document.createElement('button');
  manage.type = 'button';
  manage.className = 'mcp-manage';
  manage.textContent = 'manage';
  manage.title = 'Open the MCP panel';
  manage.onclick = () => openPanel('mcp');
  row.appendChild(manage);
  renderMcpDetail();
}

async function loadMcp() {
  const rows = await api('/api/mcp/servers');
  const list = $('#mcpList');
  list.innerHTML = '';
  if (!rows.length) { list.innerHTML = '<div class="muted" style="padding:8px">No MCP servers configured</div>'; return; }
  for (const r of rows) {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `<div><div class="mid"></div><div class="meta"></div></div>
      <div class="row" style="flex:0 0 auto">
        <button class="mini t">${r.enabled === false ? 'off' : 'on'}</button>
        <button class="mini c">connect</button><button class="mini d">×</button>
      </div>`;
    el.querySelector('.mid').textContent = r.bundled ? `${r.name} (built in)` : r.name;
    el.querySelector('.meta').textContent = r.connected
      ? `${r.tools.length} tools: ${r.tools.join(', ')}` : r.command;
    el.querySelector('.t').onclick = async () => {
      // a missing "enabled" means on; only an explicit false is off
      const nowOn = r.enabled !== false;
      await api(`/api/mcp/servers/${encodeURIComponent(r.name)}`, {
        method: 'PATCH', body: { enabled: !nowOn },
      });
      loadMcp(); refreshState();
    };
    el.querySelector('.c').onclick = async () => {
      try {
        const res = await api(`/api/mcp/servers/${encodeURIComponent(r.name)}/connect`, { method: 'POST', body: {} });
        toast(`${r.name}: ${res.tools.length} tools`);
        loadMcp(); refreshState();
      } catch (e) { toast(e.message, 'err'); }
    };
    el.querySelector('.d').onclick = async () => {
      if (r.bundled) {
        toast('Built-in servers cannot be deleted — disable it instead', 'err');
        return;
      }
      try {
        await api(`/api/mcp/servers/${encodeURIComponent(r.name)}`, { method: 'DELETE' });
        loadMcp(); refreshState();
      } catch (e) { toast(e.message, 'err'); }
    };
    list.appendChild(el);
  }
}

/* --------------------------- personas --------------------------- */
async function loadPersonas() {
  state.personas = await api('/api/personas');
  const sel = $('#personaSelect');
  sel.innerHTML = '<option value="">— none —</option>';
  for (const p of state.personas) {
    const o = document.createElement('option');
    o.value = p.id;
    o.textContent = p.name;
    sel.appendChild(o);
  }
  if (state.settings.persona) sel.value = state.settings.persona;
}


/* --------------------------- harness (c1..c8) --------------------------- */
const JSON_FIELDS = {
  hToolGuide: 'tool_guidance', hSkills: 'skills', hSubagents: 'subagents',
  hScripts: 'scripts', hPlugins: 'plugins',
};

async function loadHarnesses() {
  const data = await api('/api/harnesses');
  state.harnesses = data.harnesses || [];
  const sel = $('#harnessSelect');
  const current = sel.value || (state.harness && state.harness.id);
  sel.innerHTML = '';
  for (const h of state.harnesses) {
    const o = document.createElement('option');
    o.value = h.id;
    o.textContent = h.name;
    sel.appendChild(o);
  }
  if (current && state.harnesses.some((h) => h.id === current)) sel.value = current;
  fillHarnessForm();
}

function currentHarness() {
  const id = $('#harnessSelect').value;
  return state.harnesses.find((h) => h.id === id) || state.harnesses[0] || {};
}

function fillHarnessForm() {
  const h = currentHarness();
  state.harness = h;
  const j = (v) => JSON.stringify(v ?? (Array.isArray(v) ? [] : {}), null, 1);
  $('#hSystem').value = h.system_prompt || '';
  $('#hGuard').value = h.guardrails || '';
  $('#hMemory').value = h.memory || '';
  $('#hToolGuide').value = j(h.tool_guidance || {});
  $('#hSkills').value = j(h.skills || []);
  $('#hSubagents').value = j(h.subagents || []);
  $('#hScripts').value = j(h.scripts || []);
  $('#hPlugins').value = j(h.plugins || []);
  const rules = h.rules || {};
  $('#hBlocked').value = (rules.blocked_input || []).join('\n');
  $('#hForbid').value = (rules.forbid_output || []).join('\n');
  $('#hMaxTok').value = rules.max_prompt_tokens || 0;
  $('#hAllow').value = (rules.tool_allowlist || []).join(',');
  $('#hRequireJson').checked = !!rules.require_json;
  $('#hLessons').checked = rules.include_lessons !== false;
  $('#hReport').textContent = `${h.name} · ${JSON.stringify(h).length} chars of definition`;
  loadComposition();
}

function parseField(id) {
  const raw = $('#' + id).value.trim();
  if (!raw) return JSON_FIELDS[id] === 'tool_guidance' ? {} : [];
  try {
    return JSON.parse(raw);
  } catch (e) {
    throw new Error(`${id}: invalid JSON — ${e.message}`);
  }
}

function harnessPayload() {
  const payload = {
    name: currentHarness().name,
    system_prompt: $('#hSystem').value,
    guardrails: $('#hGuard').value,
    memory: $('#hMemory').value,
    tool_guidance: parseField('hToolGuide'),
    skills: parseField('hSkills'),
    subagents: parseField('hSubagents'),
    scripts: parseField('hScripts'),
    plugins: parseField('hPlugins'),
    rules: {
      blocked_input: $('#hBlocked').value.split('\n').map((x) => x.trim()).filter(Boolean),
      forbid_output: $('#hForbid').value.split('\n').map((x) => x.trim()).filter(Boolean),
      require_json: $('#hRequireJson').checked,
      max_prompt_tokens: +$('#hMaxTok').value || 0,
      tool_allowlist: $('#hAllow').value.split(',').map((x) => x.trim()).filter(Boolean),
      include_lessons: $('#hLessons').checked,
    },
  };
  return payload;
}

async function saveHarness() {
  const h = currentHarness();
  if (!h.id) return toast('No harness selected', 'err');
  try {
    await api(`/api/harnesses/${h.id}`, { method: 'PATCH', body: harnessPayload() });
    toast('Harness saved');
    await loadHarnesses();
    refreshState();
  } catch (e) { toast(e.message, 'err'); }
}

async function previewHarness() {
  const h = currentHarness();
  const q = $('#input').value.trim();
  const r = await api(`/api/harnesses/${h.id}/preview?q=${encodeURIComponent(q)}`);
  const rows = Object.entries(r.report)
    .filter(([k, v]) => k !== 'total' && v && typeof v === 'object' && 'tokens' in v)
    .map(([k, v]) => `${k}: ${v.tokens} tok`).join(' · ');
  const plugins = r.plugin_count ? `\n${r.profile} · ${r.plugin_count} plugins · ${r.services.length} services` : '';
  $('#hReport').textContent = `${rows}${plugins}\n\n${r.system}`;
}

/* ----------------------- composition (DSH plugin tree) ----------------------- */
let COMPOSITION = null;

async function loadComposition(force = false) {
  const h = currentHarness();
  if (!h.id) return;
  if (COMPOSITION && COMPOSITION.harness_id === h.id && !force) { renderComposition(COMPOSITION); return; }
  try {
    const r = await api(`/api/harnesses/${h.id}/composition`);
    COMPOSITION = { harness_id: h.id, ...r };
    renderComposition(COMPOSITION);
  } catch (e) {
    $('#composeSub').textContent = e.message;
  }
}

function renderComposition(r) {
  $('#composeSub').textContent = `${r.profile} · ${(r.bundles || []).join(' + ')} · ${r.rows.length} plugins`;
  const state = $('#composeState');
  state.textContent = r.failed.length ? `${r.failed.length} failed` : `${r.services.length} services`;
  state.className = 'badge ' + (r.failed.length ? 'err' : 'dl');
  const box = $('#composeRows');
  box.innerHTML = '';
  for (const row of r.rows) {
    const chip = document.createElement('button');
    chip.className = 'rc-chip';
    chip.innerHTML = `<b>${esc(row.name)}</b> <span class="cstate ${esc(row.state)}">${esc(row.state)}</span>`;
    chip.title = (row.provides && row.provides.length) ? `provides: ${row.provides.join(', ')}` : row.id;
    chip.onclick = () => {
      const log = $('#composeLog');
      log.hidden = !log.hidden;
      log.textContent = JSON.stringify({
        id: row.id, name: row.name, state: row.state,
        provides: row.provides, config: row.config,
      }, null, 2);
    };
    box.appendChild(chip);
  }
  if (r.failed.length) {
    const log = $('#composeLog');
    log.hidden = false;
    log.textContent = r.failed.map((f) => `${f.id}: ${f.error}`).join('\n');
  }
}

async function compactChat() {
  if (!state.chatId) return toast('Open a chat first', 'err');
  toast('condensing…');
  try {
    const r = await api(`/api/chats/${state.chatId}/compact`, { method: 'POST', body: {} });
    if (!r.condensed) return toast(`nothing to condense (${r.before_tokens} tok)`);
    toast(`condensed ${r.items || 0} items · ${r.before_tokens}→${r.after_tokens} tok`);
  } catch (e) { toast(e.message, 'err'); }
}

async function probeHarness() {
  if (!state.chatId) return toast('Open a chat first', 'err');
  const h = currentHarness();
  $('#hProbeOut').textContent = 'probing…';
  try {
    const r = await api('/api/harness/probe', { method: 'POST', body: { chat_id: state.chatId, harness_id: h.id } });
    const g = r.guardrails && r.guardrails.length ? `\n\nguardrail flags: ${r.guardrails.join('; ')}` : '';
    $('#hProbeOut').textContent = `[${r.harness}] ${r.elapsed_s}s · ${r.prompt_tokens ?? '?'} prompt tok${g}\n\n${r.text}`;
  } catch (e) { $('#hProbeOut').textContent = e.message; }
}

async function bindHarness() {
  if (!state.chatId) return toast('Open a chat first', 'err');
  const h = currentHarness();
  await api(`/api/chats/${state.chatId}/harness`, { method: 'POST', body: { harness_id: h.id } });
  toast(`Chat bound to ${h.name}`);
  refreshState();
}

/* ------------------- recursive refinement (Harness² PPC) ------------------- */
const H2_ACTIVE = ['queued', 'probing', 'proposing', 'judging', 'composing', 'reflecting'];

function lastUserTask() {
  const m = [...(state.messages || [])].reverse().find((x) => x.role === 'user');
  return (m && m.content) || '';
}

function renderH2Result(result, job) {
  $('#rcResult').hidden = false;
  const v = result.verdict || {};
  const lesson = (result.lesson && result.lesson.lesson) || '';
  const winner = v.winner || '—';
  const hasEdits = (result.steps || []).some((s) => (s.edits || []).length);
  const adoptable = result.changed === true || (result.changed === undefined && hasEdits);
  const held = adoptable ? '' : '\nbaseline held · no harness change to save';
  $('#rcVerdict').textContent =
    `winner: ${winner} · ${v.rationale || 'no judge note'}` + (lesson ? `\nlesson: ${lesson}` : '') + held;
  $('#rcAdopt').disabled = !adoptable;
  $('#rcAdoptBind').disabled = !adoptable;
  $('#rcChips').innerHTML = (result.candidates || []).map((c) => {
    const guard = c.guardrails && c.guardrails.length ? ' · ⚠ guardrail' : '';
    return `<span class="rc-chip" title="${esc(c.note || '')}"><b>${esc(c.id)}</b>` +
      ` ${c.score ?? '—'}${guard}</span>`;
  }).join('');
  $('#rcPreview').textContent = result.system_preview || '';
  state.h2RunId = (job && job.run_id) || state.h2RunId;
}

function renderH2Job(job) {
  const stateName = job.state || 'idle';
  const active = H2_ACTIVE.includes(stateName);
  const badge = $('#rcState');
  badge.textContent = stateName;
  badge.className = 'badge ' + (active ? 'busy' : stateName === 'ready' ? 'dl' : stateName === 'failed' ? 'err' : '');
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  $('#rcProgress').hidden = !(active || stateName === 'ready');
  $('#rcBar').style.width = `${progress}%`;
  $('#rcLog').textContent = job.error || job.message
    || (stateName === 'idle' ? 'Run one Propose–Probe–Compose step to adapt this harness to the current task.' : '…');
  $('#rcRun').disabled = active;
  $('#rcCancel').hidden = !active;
  if (stateName === 'ready' && job.result) {
    state.h2RunId = job.run_id;
    renderH2Result(job.result, job);
  } else if (stateName === 'failed' || stateName === 'cancelled') {
    $('#rcResult').hidden = true;
  }
}

async function loadH2Job() {
  const task = lastUserTask();
  const input = $('#rcTask');
  if (input) {
    input.placeholder = task
      ? `default: ${task.replace(/\s+/g, ' ').slice(0, 90)}${task.length > 90 ? '…' : ''}`
      : 'Task — defaults to the last user turn';
  }
  try {
    let job = await api('/api/harness/recursion/job');
    const current = currentHarness();
    if (job.state === 'ready' && job.harness_id && job.harness_id !== current.id) {
      job = { ...job, state: 'idle', result: null, progress: 0, message: '' };
      $('#rcResult').hidden = true;
      state.h2RunId = null;
    }
    renderH2Job(job);
    if (H2_ACTIVE.includes(job.state)) { pollH2(); return; }
    // After a reload the in-memory job is idle but the run is still in SQLite.
    if (job.state === 'idle') {
      $('#rcResult').hidden = true;
      state.h2RunId = null;
      const runs = await api('/api/harness/runs?limit=8');
      const last = runs.find((r) => r.state === 'ready' && r.harness_id === current.id);
      if (last) {
        const run = await api(`/api/harness/runs/${last.id}`);
        if (run.result) {
          state.h2RunId = run.id;
          renderH2Result(run.result, { run_id: run.id });
          $('#rcLog').textContent =
            `last run · ${run.mode} k=${run.k} · ${new Date(run.created * 1000).toLocaleString()}`;
          const badge = $('#rcState');
          badge.textContent = 'ready';
          badge.className = 'badge dl';
          $('#rcProgress').hidden = false;
          $('#rcBar').style.width = '100%';
        }
      }
    }
  } catch (e) { /* harness panel stays usable */ }
}

async function pollH2() {
  clearTimeout(state.h2Timer);
  try {
    const job = await api('/api/harness/recursion/job');
    renderH2Job(job);
    if (H2_ACTIVE.includes(job.state)) state.h2Timer = setTimeout(pollH2, 1200);
  } catch (e) {
    state.h2Timer = setTimeout(pollH2, 2500);
  }
}

async function startH2() {
  const h = currentHarness();
  if (!h.id) return toast('Select a harness first', 'err');
  const task = $('#rcTask').value.trim() || lastUserTask();
  if (!task) return toast('Open a chat with a user turn, or type a task', 'err');
  $('#rcRun').disabled = true;
  $('#rcResult').hidden = true;
  $('#rcLog').textContent = 'starting…';
  try {
    const r = await api('/api/harness/recursion', {
      method: 'POST',
      body: {
        chat_id: state.chatId,
        harness_id: h.id,
        task,
        mode: $('#rcMode').value,
        k: +$('#rcK').value || 1,
      },
    });
    state.h2RunId = r.run_id;
    toast(`refining ${h.name} (${r.mode}, k=${r.k})`);
    pollH2();
  } catch (e) {
    $('#rcRun').disabled = false;
    toast(e.message, 'err');
  }
}

async function adoptH2(bind) {
  if (!state.h2RunId) return toast('Run a refinement first', 'err');
  try {
    const r = await api(`/api/harness/runs/${state.h2RunId}/adopt`, {
      method: 'POST', body: { bind: !!bind },
    });
    await loadHarnesses();
    $('#harnessSelect').value = r.id;
    fillHarnessForm();
    if (bind) refreshState();
    toast(`saved ${r.name}${bind ? ' · bound to this chat' : ''}`);
  } catch (e) { toast(e.message, 'err'); }
}

async function loadLessons() {
  state.lessons = await api('/api/lessons');
  const list = $('#lessonList');
  list.innerHTML = '';
  if (!state.lessons.length) { list.innerHTML = '<div class="muted" style="padding:8px">No lessons yet</div>'; return; }
  for (const l of state.lessons) {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `<div><div class="mid"></div><div class="meta"></div></div><button class="mini">×</button>`;
    el.querySelector('.mid').textContent = l.lesson;
    el.querySelector('.meta').textContent = `${l.domain} · ${l.confidence || '?'}${l.cue ? ' · ' + l.cue : ''}`;
    el.querySelector('button').onclick = async () => {
      await api(`/api/lessons/${l.id}`, { method: 'DELETE' });
      loadLessons();
    };
    list.appendChild(el);
  }
}


/* --------------------------- artifacts / files --------------------------- */
function artCard(a, { inline = false } = {}) {
  const el = document.createElement('div');
  el.className = 'art-card';
  const size = a.size > 1024 ? (a.size / 1024).toFixed(1) + ' KB' : (a.size || 0) + ' B';
  el.innerHTML = `<div class="art-head">
      <b></b><span class="sz"></span><span class="grow"></span>
      <button class="mini p">preview</button>
      <a class="mini d" href="/api/artifacts/${encodeURIComponent(a.id)}" download>download</a>
      <button class="mini x" title="delete">×</button>
    </div><pre hidden></pre>`;
  el.querySelector('b').textContent = a.name;
  el.querySelector('.sz').textContent = size;
  el.querySelector('.p').onclick = async () => {
    const pre = el.querySelector('pre');
    if (!pre.hidden) { pre.hidden = true; return; }
    const r = await api(`/api/artifacts/${a.id}/preview`);
    pre.textContent = r.text + (r.truncated ? '\n… (truncated)' : '');
    pre.hidden = false;
  };
  el.querySelector('.x').onclick = async () => {
    await api(`/api/artifacts/${a.id}`, { method: 'DELETE' });
    el.remove();
    if (!inline) loadFiles();
    toast('File deleted');
  };
  return el;
}

async function loadFiles() {
  const list = $('#fileList');
  if (!list) return;
  if (!state.chatId) { list.innerHTML = '<div class="muted" style="padding:8px">Open a chat first</div>'; return; }
  const rows = await api(`/api/chats/${state.chatId}/artifacts`);
  state.artifacts = rows;
  list.innerHTML = '';
  if (!rows.length) { list.innerHTML = '<div class="muted" style="padding:8px">No files in this chat yet</div>'; return; }
  rows.forEach((a) => list.appendChild(artCard(a)));
}

async function saveTextAsFile(name, text) {
  if (!state.chatId) await newChat();
  const r = await api(`/api/chats/${state.chatId}/artifacts`, { method: 'POST', body: { name, content: text } });
  toast(`Saved ${r.name}`);
  loadFiles();
  return r;
}

// every fenced code block in a reply gets a save button
function attachCodeSave(scope) {
  scope.querySelectorAll('pre > code').forEach((code) => {
    const pre = code.parentElement;
    if (pre.querySelector('.code-save')) return;
    const btn = document.createElement('button');
    btn.className = 'code-save';
    btn.textContent = 'save file';
    btn.onclick = (e) => {
      e.stopPropagation();
      const lang = (code.dataset.lang || '').toLowerCase();
      const ext = { python: 'py', py: 'py', js: 'js', javascript: 'js', ts: 'ts', json: 'json',
                    bash: 'sh', sh: 'sh', html: 'html', css: 'css', md: 'md', sql: 'sql',
                    yaml: 'yml', yml: 'yml', csv: 'csv' }[lang] || 'txt';
      const name = prompt('File name', `snippet.${ext}`);
      if (name) saveTextAsFile(name, code.textContent);
    };
    pre.appendChild(btn);
  });
}

/* ----------------------------- modals ----------------------------- */
async function showLogs() {
  const lines = await api('/api/server/logs?tail=400');
  $('#bigLog').textContent = lines.join('\n') || '(empty)';
  $('#logsModal').classList.add('open');
  $('#bigLog').scrollTop = $('#bigLog').scrollHeight;
}

async function showStats() {
  const s = await api('/api/stats');
  state.stats = s;
  $('#statGrid').innerHTML = `
    <div class="stat"><div class="v">${s.chats}</div><div class="l">chats</div></div>
    <div class="stat"><div class="v">${s.messages}</div><div class="l">messages</div></div>
    <div class="stat"><div class="v">${s.prompt_tokens.toLocaleString()}</div><div class="l">prompt tok</div></div>
    <div class="stat"><div class="v">${s.completion_tokens.toLocaleString()}</div><div class="l">completion tok</div></div>`;
  const max = Math.max(1, ...s.days.map((d) => d.prompt + d.completion));
  $('#statBars').innerHTML = s.days.slice(-21).map((d) => {
    const total = d.prompt + d.completion;
    return `<div class="bar-row"><span>${d.date}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(total / max) * 100}%"></span></span>
      <span>${total.toLocaleString()} tok · ${d.messages} msg</span></div>`;
  }).join('') || '<div class="muted">No usage yet</div>';
  $('#statsModal').classList.add('open');
}

window.addEventListener('message', (e) => {
  const d = e.data;
  if (d && d.__htmlPreview && typeof d.h === 'number') {
    document.querySelectorAll('.html-preview iframe').forEach((f) => {
      if (f.contentWindow === e.source) f.style.height = Math.min(1200, Math.max(60, d.h + 8)) + 'px';
    });
  }
});

/* --------------------------- command palette --------------------------- */
function paletteItems() {
  const cmds = [
    { label: 'New chat', kind: 'action', run: newChat },
    { label: 'Clear all chat history', kind: 'action', run: clearHistory },
    { label: 'Open model manager', kind: 'action', run: () => openPanel('model') },
    { label: 'Open settings', kind: 'action', run: () => openPanel('params') },
    { label: 'Open memory', kind: 'action', run: () => openPanel('memory') },
    { label: 'Open prompts', kind: 'action', run: () => openPanel('prompts') },
    { label: 'Open MCP servers', kind: 'action', run: () => openPanel('mcp') },
    { label: 'Open harness editor', kind: 'action', run: () => openPanel('harness') },
    { label: 'Condense this chat (/compact)', kind: 'action', run: compactChat },
    { label: 'Open files', kind: 'action', run: () => openPanel('files') },
    { label: 'Usage stats', kind: 'action', run: showStats },
    { label: 'Server logs', kind: 'action', run: showLogs },
    { label: 'Export chat', kind: 'action', run: exportChat },
    { label: 'Toggle theme', kind: 'action', run: toggleTheme },
    { label: 'API docs (Swagger)', kind: 'link', run: () => window.open('/docs', '_blank') },
    ...state.chats.slice(0, 30).map((c) => ({
      label: c.title || 'Untitled', kind: 'chat', run: () => openChat(c.id),
    })),
  ];
  const q = $('#paletteInput').value.trim().toLowerCase();
  return cmds.filter((c) => !q || c.label.toLowerCase().includes(q));
}

function renderPalette() {
  const items = paletteItems();
  state.paletteIndex = Math.min(state.paletteIndex, Math.max(0, items.length - 1));
  const list = $('#paletteList');
  list.innerHTML = '';
  items.forEach((it, i) => {
    const el = document.createElement('div');
    el.className = 'pal-item' + (i === state.paletteIndex ? ' sel' : '');
    el.innerHTML = `<span></span><span class="k">${it.kind}</span>`;
    el.firstElementChild.textContent = it.label;
    el.onclick = () => { closePalette(); it.run(); };
    list.appendChild(el);
  });
}

function openPalette() {
  $('#palette').classList.add('open');
  $('#paletteInput').value = '';
  state.paletteIndex = 0;
  renderPalette();
  $('#paletteInput').focus();
}

function closePalette() { $('#palette').classList.remove('open'); }

/* ------------------------------ wiring ------------------------------ */
function openPanel(tab = 'model') {
  UI.set({ panel: true, tab });
  $('#app').classList.add('with-panel');
  $('#panelResize').hidden = false;
  if ($('#scrim')) $('#scrim').hidden = false;
  $('#panel').classList.add('open');
  $$('#panelTabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  $$('.tab-pane').forEach((p) => p.classList.toggle('active', p.dataset.pane === tab));
  if (tab === 'model') loadModels();
  if (tab === 'memory') loadMemory();
  if (tab === 'prompts') loadPrompts();
  if (tab === 'mcp') loadMcp();
  if (tab === 'params') loadPersonas();
  if (tab === 'harness') { loadHarnesses(); loadLessons(); loadH2Job(); }
  if (tab === 'files') loadFiles();
}

function exportChat() {
  if (!state.chatId) return toast('No chat to export', 'err');
  window.location = `/api/chats/${state.chatId}/export?fmt=md`;
}

async function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  UI.set({ theme: next });
  if (window.mermaid) window.mermaid.__ready = false;   // re-init for the new palette
  document.querySelectorAll('.mermaid-wrap[data-done]').forEach((n) => {
    const src = n.dataset.mermaid;
    if (src) { n.dataset.done = ''; n.innerHTML = `<pre class="mermaid-src" hidden>${esc(src)}</pre>`; }
  });
  renderMermaid(document.body);
  await api('/api/settings', { method: 'POST', body: { theme: next } });
}

function bind() {
  $('#newChat').onclick = newChat;
  $('#clearHistory').onclick = clearHistory;
  $('#chatSearch').oninput = renderChats;
  $('#sendBtn').onclick = () => (state.streaming ? state.abort?.abort() : sendMessage());
  $('#input').addEventListener('input', autosize);
  $('#input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  $('#attachBtn').onclick = () => $('#fileInput').click();
  $('#fileInput').onchange = async (e) => { await addFiles([...e.target.files]); e.target.value = ''; };
  $('#input').addEventListener('dragover', (e) => e.preventDefault());
  $('#input').addEventListener('drop', async (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) await addFiles([...e.dataTransfer.files]);
  });

  $('#thinkToggle').onclick = async () => {
    state.settings.enable_thinking = !state.settings.enable_thinking;
    await api('/api/settings', { method: 'POST', body: { enable_thinking: state.settings.enable_thinking } });
    refreshState();
  };
  $('#toolsToggle').onclick = async () => {
    state.settings.tools_enabled = !state.settings.tools_enabled;
    await api('/api/settings', { method: 'POST', body: { tools_enabled: state.settings.tools_enabled } });
    refreshState();
  };
  $('#themeBtn').onclick = toggleTheme;
  $('#exportBtn').onclick = exportChat;
  $('#statsBtn').onclick = showStats;
  $('#logsBtn').onclick = showLogs;
  $('#forkBtn').onclick = async () => {
    if (!state.chatId) return;
    const last = [...state.messages].reverse().find((m) => !String(m.id).startsWith('tmp-'));
    const r = await api(`/api/chats/${state.chatId}/fork`, { method: 'POST', body: { message_id: last?.id } });
    await loadChats();
    await openChat(r.id);
    toast('Chat forked');
  };
  $('#closeLogs').onclick = () => $('#logsModal').classList.remove('open');
  $('#refreshLogs').onclick = showLogs;
  $('#closeStats').onclick = () => $('#statsModal').classList.remove('open');
  $('#openPanel').onclick = () => openPanel(UI.state.tab || 'model');
  $('#paletteBtn').onclick = openPalette;
  $('#closePanel').onclick = () => {
    $('#panel').classList.remove('open');
    $('#app').classList.remove('with-panel');
    $('#panelResize').hidden = true;
    $('#scrim').hidden = !$('#sidebar').classList.contains('open');
    UI.set({ panel: false });
  };
  $('#toggleSidebar').onclick = () => {
    const open = $('#sidebar').classList.toggle('open');
    $('#scrim').hidden = !(open || $('#panel').classList.contains('open'));
  };
  $$('#panelTabs button').forEach((b) => (b.onclick = () => openPanel(b.dataset.tab)));

  $('#modelChip').onclick = (e) => { e.stopPropagation(); openModelMenu(); };
  document.addEventListener('click', (e) => {
    const menu = $('#modelMenu');
    if (!menu || menu.hidden) return;
    if (!e.target.closest('#modelMenu') && !e.target.closest('#modelChip')) menu.hidden = true;
  });
  $('#jbCancel').onclick = async () => {
    await api('/api/server/cancel', { method: 'POST', body: {} });
    toast('Cancelling…');
  };
  $('#jbLog').onclick = showLogs;
  $('#jobBanner').addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    if (state.job.state === 'failed') { jobErrDismissed = true; renderJobBanner(); }
  });
  $('#jobBanner').onclick = (e) => { if (e.target.closest('button')) return; openPanel('model'); };

  // model tab
  $('#modelSearch').onclick = loadModels;
  ['#modelSource', '#modelTask'].forEach((sel) => {
    const el = $(sel);
    if (!el) return;
    el.onchange = () => {
      UI.set(sel === '#modelSource' ? { modelSource: el.value } : { modelTask: el.value });
      loadModels();
    };
  });
  const mq = $('#modelQuery');
  if (mq) mq.oninput = () => UI.set({ modelQuery: mq.value });
  $('#modelSource').onchange = loadModels;
  $('#modelTask').onchange = loadModels;
  $('#modelQuery').onkeydown = (e) => { if (e.key === 'Enter') loadModels(); };
  $('#ctxInput').oninput = updateCtxWarn;
  $('#stopServer').onclick = async () => {
    await api('/api/server/stop', { method: 'POST', body: {} });
    toast('Server stopped');
    refreshState();
  };
  $('#cancelLoad').onclick = async () => {
    await api('/api/server/cancel', { method: 'POST', body: {} });
    toast('Cancelling…');
  };

  // settings tab
  const bindRange = (id, valId, key) => {
    $('#' + id).oninput = (e) => {
      $('#' + valId).textContent = e.target.value;
      state.settings[key] = +e.target.value;
    };
  };
  bindRange('tempRange', 'tempVal', 'temperature');
  bindRange('toppRange', 'toppVal', 'top_p');
  bindRange('topkRange', 'topkVal', 'top_k');
  bindRange('maxtokRange', 'maxtokVal', 'max_tokens');
  bindRange('repRange', 'repVal', 'repeat_penalty');
  bindRange('roundsRange', 'roundsVal', 'max_tool_rounds');

  $('#saveParams').onclick = async () => {
    const body = {
      system_prompt: $('#sysPrompt').value,
      memory_enabled: $('#memEnabled').checked,
      tools_enabled: $('#toolsEnabled').checked,
      enable_thinking: $('#thinkEnabled').checked,
      json_mode: $('#jsonEnabled').checked,
      stream: $('#streamEnabled').checked,
      auto_title: $('#autoTitleEnabled').checked,
      temperature: +$('#tempRange').value,
      top_p: +$('#toppRange').value,
      top_k: +$('#topkRange').value,
      max_tokens: +$('#maxtokRange').value,
      repeat_penalty: +$('#repRange').value,
      max_tool_rounds: +$('#roundsRange').value,
      seed: $('#seedInput').value === '' ? null : +$('#seedInput').value,
      stop: $('#stopInput').value.split(',').map((s) => s.trim()).filter(Boolean),
      persona: $('#personaSelect').value || null,
    };
    await api('/api/settings', { method: 'POST', body });
    toast('Settings saved');
    refreshState();
  };

  $('#personaSelect').onchange = async (e) => {
    const p = state.personas.find((x) => x.id === e.target.value);
    if (!p) return;
    $('#sysPrompt').value = p.system_prompt;
    let params = {};
    try { params = JSON.parse(p.params || '{}'); } catch (err) { params = {}; }
    for (const k of ['temperature', 'top_p', 'top_k', 'max_tokens', 'repeat_penalty']) {
      if (params[k] != null) {
        const map = { temperature: ['tempRange', 'tempVal'], top_p: ['toppRange', 'toppVal'],
                      top_k: ['topkRange', 'topkVal'], max_tokens: ['maxtokRange', 'maxtokVal'],
                      repeat_penalty: ['repRange', 'repVal'] };
        const [rid, vid] = map[k];
        $('#' + rid).value = params[k];
        $('#' + vid).textContent = params[k];
      }
    }
    $('#saveParams').click();
  };

  $('#personaSave').onclick = async () => {
    const name = prompt('Persona name');
    if (!name) return;
    await api('/api/personas', {
      method: 'POST',
      body: {
        name,
        system_prompt: $('#sysPrompt').value,
        params: {
          temperature: +$('#tempRange').value, top_p: +$('#toppRange').value,
          top_k: +$('#topkRange').value, max_tokens: +$('#maxtokRange').value,
          repeat_penalty: +$('#repRange').value,
        },
      },
    });
    await loadPersonas();
    toast(`Persona "${name}" saved`);
  };

  // memory tab
  $('#memAdd').onclick = async () => {
    const text = $('#memInputText').value.trim();
    if (!text) return;
    await api('/api/memory', { method: 'POST', body: { text } });
    $('#memInputText').value = '';
    loadMemory(); toast('Memory saved');
  };
  $('#memInputText').onkeydown = (e) => { if (e.key === 'Enter') $('#memAdd').click(); };

  // prompts tab
  $('#promptAdd').onclick = async () => {
    const title = $('#promptTitle').value.trim();
    const body = $('#promptBody').value.trim();
    if (!title || !body) return toast('title and body required', 'err');
    await api('/api/prompts', { method: 'POST', body: { title, body } });
    $('#promptTitle').value = $('#promptBody').value = '';
    loadPrompts(); toast('Prompt saved');
  };

  // harness tab
  $('#harnessSelect').onchange = async () => { fillHarnessForm(); await loadH2Job(); };
  $('#hSave').onclick = saveHarness;
  $('#hPreview').onclick = () => previewHarness().catch((e) => toast(e.message, 'err'));
  $('#composeRefresh').onclick = () => loadComposition(true).catch((e) => toast(e.message, 'err'));
  $('#compactNow').onclick = compactChat;
  $('#hProbe').onclick = probeHarness;
  $('#rcRun').onclick = startH2;
  $('#rcCancel').onclick = async () => {
    try {
      await api('/api/harness/recursion/cancel', { method: 'POST', body: {} });
      toast('cancelling…');
    } catch (e) { toast(e.message, 'err'); }
  };
  $('#rcAdopt').onclick = () => adoptH2(false);
  $('#rcAdoptBind').onclick = () => adoptH2(true);
  $('#hBind').onclick = bindHarness;
  $('#harnessNew').onclick = async () => {
    const name = prompt('Harness name');
    if (!name) return;
    const r = await api('/api/harnesses', { method: 'POST', body: { name } });
    await loadHarnesses();
    $('#harnessSelect').value = r.id;
    fillHarnessForm();
  };
  $('#harnessDup').onclick = async () => {
    const h = currentHarness();
    if (!h.id) return;
    const r = await api(`/api/harnesses/${h.id}/duplicate`, { method: 'POST', body: {} });
    await loadHarnesses();
    $('#harnessSelect').value = r.id;
    fillHarnessForm();
  };
  $('#harnessDel').onclick = async () => {
    const h = currentHarness();
    if (!h.id || !confirm(`Delete harness "${h.name}"?`)) return;
    try {
      await api(`/api/harnesses/${h.id}`, { method: 'DELETE' });
      await loadHarnesses();
      toast('Harness deleted');
    } catch (e) { toast(e.message, 'err'); }
  };
  $('#lessonAdd').onclick = async () => {
    const lesson = $('#lessonText').value.trim();
    if (!lesson) return;
    await api('/api/lessons', { method: 'POST', body: { lesson, domain: 'general' } });
    $('#lessonText').value = '';
    loadLessons();
  };
  $('#lessonReflect').onclick = async () => {
    if (!state.chatId) return toast('Open a chat first', 'err');
    $('#lessonReflect').disabled = true;
    try {
      const r = await api('/api/lessons/reflect', { method: 'POST', body: { chat_id: state.chatId } });
      toast(`Lesson: ${(r.lesson || '').slice(0, 60)}…`);
      loadLessons();
    } catch (e) { toast(e.message, 'err'); }
    $('#lessonReflect').disabled = false;
  };

  // files tab
  $('#fileCreate').onclick = async () => {
    const name = $('#fileName').value.trim() || 'note.txt';
    const body = $('#fileBody').value;
    if (!body) return toast('nothing to save', 'err');
    await saveTextAsFile(name, body);
    $('#fileBody').value = '';
  };

  // mcp tab
  $('#mcpAdd').onclick = async () => {
    const name = $('#mcpName').value.trim();
    const command = $('#mcpCmd').value.trim();
    const args = $('#mcpArgs').value.trim().split(/\s+/).filter(Boolean);
    if (!name || !command) return toast('name and command required', 'err');
    await api('/api/mcp/servers', { method: 'POST', body: { name, command, args } });
    $('#mcpName').value = $('#mcpCmd').value = $('#mcpArgs').value = '';
    loadMcp(); toast('Server added — press connect');
  };

  // palette
  $('#paletteInput').oninput = () => { state.paletteIndex = 0; renderPalette(); };
  $('#paletteInput').onkeydown = (e) => {
    if (e.key === 'Escape') return closePalette();
    const items = paletteItems();
    if (e.key === 'ArrowDown') { e.preventDefault(); state.paletteIndex = Math.min(items.length - 1, state.paletteIndex + 1); renderPalette(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); state.paletteIndex = Math.max(0, state.paletteIndex - 1); renderPalette(); }
    if (e.key === 'Enter') {
      const it = items[state.paletteIndex];
      if (it) { closePalette(); it.run(); }
    }
  };
  $('#palette').onclick = (e) => { if (e.target.id === 'palette') closePalette(); };

  document.addEventListener('keydown', (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); }
    if (mod && e.key.toLowerCase() === 'j') { e.preventDefault(); openPanel('params'); }
    if (e.key === 'Escape') {
      $('#modelMenu').hidden = true;
      closePalette();
      $('#panel').classList.remove('open');
      $('#logsModal').classList.remove('open');
      $('#statsModal').classList.remove('open');
    }
  });
}

async function boot() {
  // 1. restore the workspace shape before anything paints
  UI.load();
  applyLayout();
  bind();
  bindLayout();

  const st = await api('/api/settings').catch(() => null);
  if (!UI.state.theme) UI.state.theme = st?.theme || 'dark';
  if (!localStorage.getItem(UI_KEY)) await UI.hydrate();
  applyLayout();
  document.documentElement.dataset.theme = UI.state.theme || 'dark';

  await refreshState();
  await loadChats();

  // 2. reopen the chat that was open, else the most recent one
  const remembered = UI.state.chatId && state.chats.some((c) => c.id === UI.state.chatId);
  const recent = state.chats[0];
  if (remembered) await openChat(UI.state.chatId);
  else if (recent) await openChat(recent.id);
  else await newChat();

  // filters
  if ($('#modelSource')) $('#modelSource').value = UI.state.modelSource || 'cache';
  if ($('#modelTask')) $('#modelTask').value = UI.state.modelTask || '';
  if ($('#modelQuery')) $('#modelQuery').value = UI.state.modelQuery || '';

  // 3. restore the panel/tab last used
  if (UI.state.panel) openPanel(UI.state.tab || 'model');
  // model-load progress is polled on its own cheap endpoint, every second
  setInterval(async () => {
    try {
      const j = await api('/api/server/job');
      const changed = j.state !== state.job.state || j.progress !== state.job.progress;
      state.job = j;
      if (JOB_ACTIVE.includes(j.state)) jobErrDismissed = false;
      renderJobBanner();
      if (changed) {
        renderJobBanner();
        renderServerCard();
        applySwitchingUI();
        const activeJob = JOB_ACTIVE.includes(j.state);
        const eng = state.engine || {};
        const engActive = ENGINE_ACTIVE.includes(eng.phase);
        const engPct = Number.isFinite(eng.progress) ? ` ${Math.round(eng.progress)}%` : '';
        $('#statusDot').className = 'dot ' + (activeJob || engActive ? 'busy' : (state.server && state.server.up ? 'on' : 'off'));
        const pctTxt = j.state === 'downloading' && j.progress ? ` ${Math.round(j.progress)}%` : '';
        $('#statusText').textContent = activeJob
          ? `${j.state}${pctTxt}`
          : engActive ? `${engineLabel(eng)}${engPct}`.trim()
          : (state.server && state.server.up ? `${(state.server.model || state.settings.model || '').split('/').pop()} · up` : 'server down');
        $('#jobChip').hidden = !(activeJob || engActive);
        if (activeJob || engActive) {
          const pct = j.state === 'downloading' ? ` ${Math.round(j.progress || 0)}%` : '';
          $('#jobChip').className = 'chip busy';
          $('#jobChip').textContent = activeJob ? `${j.state}${pct}` : `${engineLabel(eng)}${engPct}`.trim();
        }
      }
    } catch (e) { /* workbench offline */ }
  }, 1000);

  (async function tick() {
    try { await refreshState(); } catch (e) { /* offline */ }
    const fast = JOB_ACTIVE.includes(state.job.state)
      || ENGINE_ACTIVE.includes((state.engine || {}).phase);
    setTimeout(tick, fast ? 1500 : 8000);
  })();

  const chatParam = new URLSearchParams(location.search).get('chat');
  if (chatParam && state.chats.some((c) => c.id === chatParam)) await openChat(chatParam);

  const ask = new URLSearchParams(location.search).get('ask');
  if (ask) { await newChat(); sendMessage(ask); }
  const panel = new URLSearchParams(location.search).get('panel');
  if (panel) openPanel(panel);
  if (new URLSearchParams(location.search).get('menu') === 'models') openModelMenu();
  const sw = new URLSearchParams(location.search).get('switch');
  if (sw) switchModel(sw);   // deep link: /?switch=org/model
}

boot();
