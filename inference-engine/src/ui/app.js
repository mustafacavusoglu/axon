'use strict';

// Axon Console: model list + inference playground.
// All dynamic content is inserted via textContent (never innerHTML).

const KEY_STORAGE = 'axon.apiKey';
const DATATYPES = ['FP32', 'INT32', 'INT64', 'BYTES'];
const MAX_PREFILL = 256;
const OUTPUT_PREVIEW_CHARS = 4000;

const state = {
  apiKey: '',
  models: [],          // [{name, version, state, platform, device}]
  selected: null,      // model name
  meta: null,          // metadata of the selected model/version
  version: null,       // selected version string
  inputs: [],          // editable input specs
  filter: '',
};

try { state.apiKey = sessionStorage.getItem(KEY_STORAGE) || ''; } catch (_) { /* storage blocked */ }

// ---------------------------------------------------------------- helpers

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

const $ = (id) => document.getElementById(id);

class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function api(path, options = {}) {
  const headers = { Accept: 'application/json', ...(options.headers || {}) };
  if (state.apiKey) headers.Authorization = 'Bearer ' + state.apiKey;
  let res;
  try {
    res = await fetch(path, { ...options, headers });
  } catch (e) {
    throw new ApiError(0, 'Cannot reach the server: ' + e.message);
  }
  let body = null;
  const text = await res.text();
  if (text) { try { body = JSON.parse(text); } catch (_) { body = text; } }
  if (res.status === 401) {
    setStatus('err', 'API key required');
    $('api-key').focus();
  }
  if (!res.ok) {
    const msg = body && body.error ? body.error : (typeof body === 'string' && body) || res.statusText;
    throw new ApiError(res.status, msg);
  }
  return body;
}

function setStatus(kind, text) {
  $('status-dot').className = 'dot ' + kind;
  $('status-text').textContent = text;
}

function shapeProduct(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function parseShape(text) {
  const parts = text.split(/[\s,\[\]x×]+/).filter(Boolean);
  const shape = parts.map(Number);
  if (!shape.length && text.trim() === '' ) return [];
  if (shape.some((n) => !Number.isInteger(n) || n < 0)) return null;
  return shape;
}

// Accepts a JSON value, or bare comma/space separated values ("1, 2, 3").
function parseData(text) {
  const t = text.trim();
  if (!t) return { error: 'enter some data' };
  try { return { value: JSON.parse(t) }; } catch (_) { /* fall through */ }
  try { return { value: JSON.parse('[' + t + ']') }; } catch (e) {
    return { error: 'not valid JSON: ' + e.message };
  }
}

function flatten(value, out = []) {
  if (Array.isArray(value)) value.forEach((v) => flatten(v, out));
  else out.push(value);
  return out;
}

function defaultInput(t) {
  const shape = (t.shape || []).map((d) => (d > 0 ? d : 1));
  const n = shapeProduct(shape);
  const dtype = DATATYPES.includes(t.datatype) ? t.datatype : 'FP32';
  let data = '[]';
  if (n > 0 && n <= MAX_PREFILL) {
    const fill = dtype === 'BYTES' ? 'text' : 0;
    data = JSON.stringify(Array(n).fill(fill));
  }
  return { name: t.name || '', datatype: dtype, shape: shape.join(', '), data };
}

function validateInput(inp) {
  const shape = parseShape(inp.shape);
  if (!inp.name.trim()) return { bad: true, msg: 'name is required' };
  if (!shape) return { bad: true, msg: 'shape must be non-negative integers, e.g. 1, 30' };
  const parsed = parseData(inp.data);
  if (parsed.error) return { bad: true, msg: parsed.error };
  const flat = flatten(parsed.value);
  const want = shapeProduct(shape);
  const isText = inp.datatype === 'BYTES';
  const wrongType = flat.some((v) => (isText ? typeof v !== 'string' : typeof v !== 'number'));
  if (wrongType) return { bad: true, msg: isText ? 'BYTES values must be strings' : 'values must be numbers' };
  if (flat.length !== want) {
    return { bad: true, msg: flat.length + ' values, but shape needs ' + want };
  }
  return { bad: false, shape, data: parsed.value, msg: flat.length + ' values · matches shape' };
}

function buildRequest() {
  const inputs = [];
  for (const inp of state.inputs) {
    const v = validateInput(inp);
    if (v.bad) return { error: (inp.name || 'input') + ': ' + v.msg };
    inputs.push({ name: inp.name.trim(), shape: v.shape, datatype: inp.datatype, data: v.data });
  }
  if (!inputs.length) return { error: 'add at least one input' };
  return { body: { inputs } };
}

function inferPath() {
  return '/v2/models/' + encodeURIComponent(state.selected) +
    '/versions/' + encodeURIComponent(state.version) + '/infer';
}

function curlFor(body) {
  const esc = (s) => s.replace(/'/g, "'\\''");
  const lines = ["curl -s -X POST '" + esc(location.origin + inferPath()) + "'",
    "  -H 'Content-Type: application/json'"];
  if (state.apiKey) lines.push('  -H "Authorization: Bearer $AXON_API_KEY"');
  lines.push("  -d '" + esc(JSON.stringify(body)) + "'");
  return lines.join(' \\\n');
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const old = button.textContent;
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = old; }, 1200);
  } catch (_) {
    button.textContent = 'Copy failed';
  }
}

function deviceBadge(device) {
  if (!device) return null;
  const gpu = device.includes('cuda') || device.includes('trt');
  return h('span', { class: 'badge ' + (gpu ? 'gpu' : 'cpu') }, device);
}

// ---------------------------------------------------------------- model list

function groupedModels() {
  const map = new Map();
  for (const m of state.models) {
    if (!map.has(m.name)) map.set(m.name, { name: m.name, versions: [], platform: m.platform, devices: new Set() });
    const g = map.get(m.name);
    g.versions.push(m.version);
    if (m.device) g.devices.add(m.device);
  }
  return [...map.values()].sort((a, b) => a.name.localeCompare(b.name));
}

function renderList() {
  const list = $('model-list');
  list.replaceChildren();
  const q = state.filter.trim().toLowerCase();
  const groups = groupedModels().filter((g) => !q || g.name.toLowerCase().includes(q));
  $('model-count').textContent = state.models.length ? groups.length + ' / ' + groupedModels().length : '';
  for (const g of groups) {
    const badges = h('div', { class: 'badges' },
      h('span', { class: 'badge' }, g.platform),
      [...g.devices].map(deviceBadge),
      h('span', { class: 'badge' }, 'v' + g.versions.join(', v')));
    list.append(h('li', { class: 'model-item' + (g.name === state.selected ? ' active' : '') },
      h('button', { type: 'button', onclick: () => selectModel(g.name) },
        h('span', { class: 'model-name' }, g.name), badges)));
  }
  $('list-hint').textContent = state.models.length ? '' : 'No models loaded yet.';
}

async function loadModels() {
  try {
    state.models = await api('/v2/models');
    renderList();
    if (state.selected && !state.models.some((m) => m.name === state.selected)) {
      state.selected = null;
      renderEmpty('That model is no longer loaded.');
    }
  } catch (e) {
    if (e.status !== 401) setStatus('err', e.message);
    state.models = [];
    renderList();
    $('list-hint').textContent = e.status === 401 ? 'Enter the API key to list models.' : e.message;
  }
}

async function loadStatus() {
  try {
    const res = await fetch('/v2/health/ready');
    if (res.ok) {
      let version = '';
      try { version = (await api('/v2')).version; } catch (_) { /* needs key */ }
      setStatus('ok', 'ready' + (version ? ' · v' + version : ''));
    } else {
      setStatus('warn', 'no models loaded');
    }
  } catch (_) {
    setStatus('err', 'server unreachable');
  }
}

// ---------------------------------------------------------------- model view

function renderEmpty(msg) {
  $('content').replaceChildren(h('div', { class: 'empty' },
    h('h1', null, 'Select a model'),
    h('p', null, msg || 'Pick a model on the left to see its schema and try an inference.')));
}

async function selectModel(name) {
  state.selected = name;
  renderList();
  try {
    const latest = await api('/v2/models/' + encodeURIComponent(name));
    const versions = latest.versions.map(Number).sort((a, b) => a - b);
    state.version = String(versions[versions.length - 1]);
    await loadVersion(latest.versions);
  } catch (e) {
    $('content').replaceChildren(h('div', { class: 'card' }, h('pre', { class: 'data err' }, e.message)));
  }
}

async function loadVersion(allVersions) {
  const meta = await api('/v2/models/' + encodeURIComponent(state.selected) +
    '/versions/' + encodeURIComponent(state.version));
  state.meta = meta;
  state.versions = allVersions || state.versions || [state.version];
  state.inputs = (meta.inputs || []).map(defaultInput);
  renderModel();
}

function schemaTable(title, tensors) {
  return h('div', null,
    h('h3', null, title),
    tensors.length
      ? h('table', { class: 'schema' },
          h('thead', null, h('tr', null, ['Name', 'Type', 'Shape'].map((c) => h('th', null, c)))),
          h('tbody', null, tensors.map((t) => h('tr', null,
            h('td', { class: 'mono' }, t.name),
            h('td', { class: 'mono' }, t.datatype),
            h('td', { class: 'mono' }, '[' + t.shape.join(', ') + ']')))))
      : h('p', { class: 'muted' }, 'No schema in the model config.'));
}

function renderModel() {
  const meta = state.meta;
  const versionSelect = h('select', {
    'aria-label': 'Version',
    onchange: (e) => { state.version = e.target.value; loadVersion().catch(showError); },
  }, [...state.versions].sort((a, b) => a - b).map((v) =>
    h('option', { value: v, selected: v === state.version }, 'v' + v)));

  const header = h('div', { class: 'model-title' },
    h('h1', null, meta.name),
    versionSelect,
    h('span', { class: 'badge' }, meta.platform),
    deviceBadge(meta.device));

  const schema = h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', null, 'Schema')),
    h('div', { class: 'schema-grid' },
      schemaTable('Inputs', meta.inputs || []),
      h('div', { class: 'gap' }),
      schemaTable('Outputs', meta.outputs || [])));

  const inputsHost = h('div', { id: 'inputs-host' });
  const resultHost = h('div', { id: 'result-host' });
  const runBtn = h('button', { class: 'btn', id: 'run', type: 'button', onclick: () => runInference(runBtn) }, 'Run inference');

  const playground = h('div', { class: 'card' },
    h('div', { class: 'card-head' },
      h('h2', null, 'Try it'),
      h('div', { class: 'spacer' }),
      h('button', { class: 'btn btn-ghost btn-small', type: 'button', onclick: addInput }, '+ Add input'),
      h('button', { class: 'btn btn-ghost btn-small', type: 'button', onclick: resetInputs }, 'Reset')),
    inputsHost,
    h('div', { class: 'actions' }, runBtn,
      h('button', { class: 'btn btn-ghost', type: 'button', onclick: (e) => copyCurl(e.target) }, 'Copy as curl')),
  );

  $('content').replaceChildren(header, schema, playground, resultHost);
  renderInputs();
}

function showError(e) {
  $('result-host')?.replaceChildren(h('div', { class: 'card' }, h('pre', { class: 'data err' }, e.message)));
}

// ---------------------------------------------------------------- inputs editor

function addInput() {
  state.inputs.push({ name: '', datatype: 'FP32', shape: '1', data: '[0]' });
  renderInputs();
}

function resetInputs() {
  state.inputs = (state.meta.inputs || []).map(defaultInput);
  renderInputs();
}

function renderInputs() {
  const host = $('inputs-host');
  host.replaceChildren();
  if (!state.inputs.length) {
    host.append(h('p', { class: 'muted' }, 'This model has no declared inputs. Add one to send a request.'));
    return;
  }
  state.inputs.forEach((inp, i) => {
    const check = h('div', { class: 'check' });
    const update = () => {
      const v = validateInput(inp);
      check.textContent = v.msg;
      check.className = 'check ' + (v.bad ? 'bad' : 'good');
    };
    const on = (key) => (e) => { inp[key] = e.target.value; update(); };
    host.append(h('div', { class: 'input-card' },
      h('div', { class: 'input-row' },
        h('div', { class: 'field' }, h('label', null, 'Name'),
          h('input', { value: inp.name, class: 'mono', spellcheck: 'false', oninput: on('name') })),
        h('div', { class: 'field' }, h('label', null, 'Type'),
          h('select', { onchange: on('datatype') },
            DATATYPES.map((d) => h('option', { value: d, selected: d === inp.datatype }, d)))),
        h('div', { class: 'field' }, h('label', null, 'Shape'),
          h('input', { value: inp.shape, class: 'mono', spellcheck: 'false', oninput: on('shape') })),
        h('button', {
          class: 'btn btn-ghost btn-small', type: 'button', 'aria-label': 'Remove input',
          onclick: () => { state.inputs.splice(i, 1); renderInputs(); },
        }, 'Remove')),
      h('div', { class: 'field' }, h('label', null, 'Data (JSON, flat or nested)'),
        h('textarea', { spellcheck: 'false', oninput: on('data') }, inp.data)),
      check));
    update();
  });
}

function copyCurl(button) {
  const req = buildRequest();
  if (req.error) { button.textContent = 'Fix inputs first'; setTimeout(() => { button.textContent = 'Copy as curl'; }, 1500); return; }
  copyText(curlFor(req.body), button);
}

// ---------------------------------------------------------------- run + result

function outputCard(o) {
  const full = JSON.stringify(o.data);
  const truncated = full.length > OUTPUT_PREVIEW_CHARS;
  const pre = h('pre', { class: 'data' }, truncated ? full.slice(0, OUTPUT_PREVIEW_CHARS) + ' …' : full);
  const head = h('div', { class: 'out-head' },
    h('strong', { class: 'mono' }, o.name),
    h('span', { class: 'badge' }, o.datatype),
    h('span', { class: 'badge mono' }, '[' + o.shape.join(', ') + ']'),
    h('span', { class: 'muted' }, flatten(o.data).length + ' values'),
    h('div', { class: 'spacer' }));
  if (truncated) {
    const more = h('button', { class: 'btn btn-ghost btn-small', type: 'button' }, 'Show all');
    more.addEventListener('click', () => { pre.textContent = full; more.remove(); });
    head.append(more);
  }
  const copy = h('button', { class: 'btn btn-ghost btn-small', type: 'button' }, 'Copy');
  copy.addEventListener('click', () => copyText(full, copy));
  head.append(copy);
  return h('div', { class: 'out' }, head, pre);
}

async function runInference(button) {
  const host = $('result-host');
  const req = buildRequest();
  if (req.error) {
    host.replaceChildren(h('div', { class: 'card' }, h('pre', { class: 'data err' }, req.error)));
    return;
  }
  button.disabled = true;
  button.textContent = 'Running…';
  const started = performance.now();
  try {
    const res = await api(inferPath(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const ms = performance.now() - started;
    host.replaceChildren(h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Result')),
      h('div', { class: 'result-meta' },
        h('span', { class: 'pill ok' }, '200 OK'),
        h('span', null, ms.toFixed(1) + ' ms round trip'),
        h('span', { class: 'muted' }, res.model_name + ' v' + res.model_version)),
      res.outputs.map(outputCard)));
  } catch (e) {
    host.replaceChildren(h('div', { class: 'card' },
      h('div', { class: 'result-meta' },
        h('span', { class: 'pill err' }, e.status ? 'HTTP ' + e.status : 'Error'),
        h('span', { class: 'muted' }, (performance.now() - started).toFixed(1) + ' ms')),
      h('pre', { class: 'data err' }, e.message)));
  } finally {
    button.disabled = false;
    button.textContent = 'Run inference';
  }
}

// ---------------------------------------------------------------- boot

$('filter').addEventListener('input', (e) => { state.filter = e.target.value; renderList(); });

$('key-form').addEventListener('submit', (e) => {
  e.preventDefault();
  state.apiKey = $('api-key').value.trim();
  try {
    if (state.apiKey) sessionStorage.setItem(KEY_STORAGE, state.apiKey);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch (_) { /* storage blocked */ }
  $('api-key').value = '';
  $('api-key').placeholder = state.apiKey ? 'API key set (session only)' : 'API key (if enabled)';
  loadStatus();
  loadModels();
});

if (state.apiKey) $('api-key').placeholder = 'API key set (session only)';

loadStatus();
loadModels();
setInterval(() => { loadStatus(); loadModels(); }, 10000);
