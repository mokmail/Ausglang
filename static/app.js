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
  server: {},
  job: {},
  attachments: [],
  optimistic: null,
  personas: [],
  stats: null,
  harnesses: [],
  harness: null,
  lessons: [],
  paletteIndex: 0,
};

const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

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
  density: 'comfortable', sideW: 288, panelW: 410, measure: 830,
  modelSource: 'cache', modelTask: '', modelQuery: '',
  harness: null, persona: null, drafts: {}, scroll: {}, filesTab: 'chats',
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
    <div class="empty">
      <div class="empty-logo">S</div>
      <h2>Local inference, properly harnessed</h2>
      <p>Streaming, memory, MCP tools, model switching, attachments, personas and a context window you control — all served by SGLang on this Mac.</p>
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
  const tokens = Math.ceil(chars / 4);
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
  addLocal('user', text + (state.attachments.length ? `\n\n[${state.attachments.length} attachment(s)]` : ''));
  const attachments = [...state.attachments];
  state.attachments = [];
  renderAttachments();

  const assistant = { id: 'tmp-' + Math.random(), role: 'assistant', content: '', reasoning: '' };
  state.messages.push(assistant);
  const aEl = msgEl(assistant, state.messages.length - 1);
  $('#messages .thread').appendChild(aEl);
  const bodyEl = aEl.querySelector('.body');
  bodyEl.innerHTML = '<span class="cursor"></span>';
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
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: state.abort.signal,
      body: JSON.stringify({ chat_id: state.chatId, message: text, attachments }),
    });
    if (!res.ok) throw new Error((await res.text()).slice(0, 300));

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
          assistant.content += data.text;
          chars += data.text.length;
          const now = performance.now();
          if (now - (assistant._lastRender || 0) > 120) {
            assistant._lastRender = now;
            bodyEl.innerHTML = md(assistant.content) + '<span class="cursor"></span>';
          }
          const secs = Math.max(0.001, (performance.now() - t0) / 1000);
          $('#streamHint').textContent = `~${Math.round(chars / 4 / secs)} tok/s`;
          $('#messages').scrollTop = $('#messages').scrollHeight;
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
          $('#messages').scrollTop = $('#messages').scrollHeight;
        } else if (ev === 'artifact') {
          assistant.artifacts = assistant.artifacts || [];
          assistant.artifacts.push(data);
          aEl.querySelector('.bubble').appendChild(artCard(data, { inline: true }));
          toast(`File written: ${data.name}`);
        } else if (ev === 'error') {
          toast(data.message, 'err');
          assistant.content += `\n\n**Error:** ${data.message}`;
          bodyEl.innerHTML = md(assistant.content);
        } else if (ev === 'done') {
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

async function regenerate() {
  if (state.streaming || !state.chatId) return;
  state.streaming = true;
  state.abort = new AbortController();
  setSendState();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: state.abort.signal,
      body: JSON.stringify({ chat_id: state.chatId, regenerate: true }),
    });
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
  return Math.max(4000, Math.floor((ctx - 2500) * 3.4));
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
        });
        toast(`${r.name}: ${r.chars.toLocaleString()} chars${r.pages ? ` from ${r.pages} pages` : ''}${truncated ? ' (trimmed to context)' : ''}`);
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
    $('#input').placeholder = 'Message the model…  (⏎ send · ⇧⏎ newline · ⌘K palette)';
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
    toast(`Switching to ${model.split('/').pop()}…`);
    $('#jobBanner').hidden = false;
    await pollJob();
    setSwitching(false);
    refreshState();
  } catch (e) {
    setSwitching(false);
    toast(e.message, 'err');
  }
}

/* ----------------------------- state ----------------------------- */
const JOB_ACTIVE = ['preparing', 'downloading', 'starting', 'stopping'];

async function refreshState() {
  try {
    const s = await api('/api/state');
    state.settings = s.settings;
    state.tools = s.tools || [];
    state.server = s.server || {};
    state.job = s.job || {};
    state.canLoadModels = s.can_load_models !== false;
    const up = !!s.server?.up;
    const jobActive = JOB_ACTIVE.includes(state.job.state);

    $('#statusDot').className = 'dot ' + (jobActive ? 'busy' : up ? 'on' : 'off');
    $('#statusText').textContent = jobActive
      ? `${state.job.state} ${state.job.progress ? Math.round(state.job.progress) + '%' : ''}`.trim()
      : up ? `${((s.server && s.server.model) || state.settings.model || '').split('/').pop()} · up` : 'server down';
    $('#brandSub').textContent = up
      ? `local · Metal · ${(s.server?.context_length || state.settings.context_length).toLocaleString()} ctx`
      : 'local · Metal · offline';

    $('#modelChip').textContent = s.server?.model || s.settings.model;
    $('#ctxChip').textContent = `ctx ${Number(s.server?.context_length || s.settings.context_length).toLocaleString()}`;
    $('#memChip').textContent = s.settings.memory_enabled ? 'memory on' : 'memory off';
    if (s.harness && $('#harnessChip')) $('#harnessChip').textContent = `harness ${s.harness.name}`;
    $('#memChip').className = 'chip' + (s.settings.memory_enabled ? ' on' : '');
    $('#toolCount').textContent = state.tools.length;
    $('#toolsToggle').classList.toggle('on', !!s.settings.tools_enabled);
    $('#thinkToggle').classList.toggle('on', !!s.settings.enable_thinking);
    $('#jobChip').hidden = !jobActive;
    if (jobActive) {
      const pct = state.job.state === 'downloading' ? ` ${Math.round(state.job.progress || 0)}%` : '';
      $('#jobChip').className = 'chip busy';
      $('#jobChip').textContent = `${state.job.state}${pct}`;
    }
    if (state.job.state === 'ready' && !jobDoneUntil) jobDoneUntil = Date.now() + 4000;
    if (JOB_ACTIVE.includes(state.job.state)) jobErrDismissed = false;
    // rendering problems are UI problems: they must not flip the connection pill
    for (const fn of [renderServerCard, renderJobBanner, applySwitchingUI, applySettingsToForm]) {
      try { fn(); } catch (e) { console.warn('[ui]', fn.name, e && e.message); }
    }
  } catch (e) {
    $('#statusDot').className = 'dot off';
    $('#statusText').textContent = 'workbench offline';
  }
}

function renderServerCard() {
  const j = state.job || {};
  const s = state.server || {};
  const up = !!s.up;
  $('#scDot').className = 'dot ' + (JOB_ACTIVE.includes(j.state) ? 'busy' : up ? 'on' : 'off');
  $('#scModel').textContent = s.model || state.settings.model || '—';
  const pool = Number(s.max_total_num_tokens || 0);
  const poolTxt = pool ? ` · pool ${pool.toLocaleString()} tok` : '';
  $('#scMeta').textContent = up
    ? `ctx ${Number(s.context_length || 0).toLocaleString()}${poolTxt} · RSS ${s.rss_gb ?? '?'} GB`
    : (j.state !== 'idle' ? `${j.state}: ${j.message || ''}` : 'server not running');
  $('#scMeta').classList.toggle('warn', up && pool > 0 && pool < 4096);
  $('#scMeta').title = up && pool > 0 && pool < 4096
    ? 'KV pool is too small for real conversations — lower the context window or pick a smaller model'
    : '';

  const prog = $('#loadProgress');
  const bar = $('#loadBar');
  const active = JOB_ACTIVE.includes(j.state);
  prog.hidden = !active;
  bar.style.width = `${Math.round(j.progress || 0)}%`;

  const err = $('#loadError');
  if (j.state === 'failed' && j.error) { err.hidden = false; err.textContent = j.error; }
  else err.hidden = true;

  $('#cancelLoad').hidden = !active;
  $('#stopServer').disabled = !up && !active;
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
    toast(`Loading ${model}…`);
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
      if (state.job.state === 'ready') { toast(`${state.job.model} ready`); loadModels(); }
      if (state.job.state === 'failed') toast('Load failed — see the error in the model tab', 'err');
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
    el.querySelector('.mid').textContent = r.name;
    el.querySelector('.meta').textContent = r.connected
      ? `${r.tools.length} tools: ${r.tools.join(', ')}` : r.command;
    el.querySelector('.t').onclick = async () => {
      await api(`/api/mcp/servers/${encodeURIComponent(r.name)}`, {
        method: 'PATCH', body: { enabled: r.enabled === false },
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
      await api(`/api/mcp/servers/${encodeURIComponent(r.name)}`, { method: 'DELETE' });
      loadMcp(); refreshState();
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
  const rows = Object.entries(r.report).filter(([k]) => k !== 'total')
    .map(([k, v]) => `${k}: ${v.tokens} tok`).join(' · ');
  $('#hReport').textContent = `${rows}\n\n${r.system}`;
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
    { label: 'Open model manager', kind: 'action', run: () => openPanel('model') },
    { label: 'Open settings', kind: 'action', run: () => openPanel('params') },
    { label: 'Open memory', kind: 'action', run: () => openPanel('memory') },
    { label: 'Open prompts', kind: 'action', run: () => openPanel('prompts') },
    { label: 'Open MCP servers', kind: 'action', run: () => openPanel('mcp') },
    { label: 'Open harness editor', kind: 'action', run: () => openPanel('harness') },
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
  $('#panel').classList.add('open');
  $$('#panelTabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  $$('.tab-pane').forEach((p) => p.classList.toggle('active', p.dataset.pane === tab));
  if (tab === 'model') loadModels();
  if (tab === 'memory') loadMemory();
  if (tab === 'prompts') loadPrompts();
  if (tab === 'mcp') loadMcp();
  if (tab === 'params') loadPersonas();
  if (tab === 'harness') { loadHarnesses(); loadLessons(); }
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
  $('#openPanel').onclick = () => openPanel('model');
  $('#paletteBtn').onclick = openPalette;
  $('#closePanel').onclick = () => {
    $('#panel').classList.remove('open');
    $('#app').classList.remove('with-panel');
    $('#panelResize').hidden = true;
    UI.set({ panel: false });
  };
  $('#toggleSidebar').onclick = () => $('#sidebar').classList.toggle('open');
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
  $('#harnessSelect').onchange = fillHarnessForm;
  $('#hSave').onclick = saveHarness;
  $('#hPreview').onclick = () => previewHarness().catch((e) => toast(e.message, 'err'));
  $('#hProbe').onclick = probeHarness;
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
      if (changed) {
        renderJobBanner();
        renderServerCard();
        applySwitchingUI();
        const activeJob = JOB_ACTIVE.includes(j.state);
        $('#statusDot').className = 'dot ' + (activeJob ? 'busy' : (state.server && state.server.up ? 'on' : 'off'));
        const pctTxt = j.state === 'downloading' && j.progress ? ` ${Math.round(j.progress)}%` : '';
        $('#statusText').textContent = activeJob
          ? `${j.state}${pctTxt}`
          : (state.server && state.server.up ? `${(state.server.model || state.settings.model || '').split('/').pop()} · up` : 'server down');
        $('#jobChip').hidden = !activeJob;
        if (activeJob) {
          const pct = j.state === 'downloading' ? ` ${Math.round(j.progress || 0)}%` : '';
          $('#jobChip').className = 'chip busy';
          $('#jobChip').textContent = `${j.state}${pct}`;
        }
      }
    } catch (e) { /* workbench offline */ }
  }, 1000);

  (async function tick() {
    try { await refreshState(); } catch (e) { /* offline */ }
    setTimeout(tick, JOB_ACTIVE.includes(state.job.state) ? 1000 : 8000);
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
