// 2D2VR player: Gaussian splats (.splat) and splat videos (.splv) on the desktop and in VR (WebXR).
//
// The splat rasteriser (vertex/fragment shaders, instanced quads, worker depth sort) is adapted from
// splaTV / splat by Kevin Kwok (MIT, see LICENSE-splatv.txt). 2D2VR adds the .splv splat video format
// (streamed with HTTP range requests, decoded in a pool of workers one keyframe group each, played at the
// video's speed with its sound as the clock), OpenGL/WebXR camera conventions, the levelling transform
// of the app, VR locomotion and a size control (controller A/B or X/Y) for when the real-world scale
// of a scene is off.

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const sceneUrl = new URL(q.get('scene') || 'scene.splat', location.href).href;
const sceneName = q.get('name') || sceneUrl.split('/').pop();
$('title').textContent = '2D2VR — ' + sceneName;
const nums = (k, n) => { const v = (q.get(k) || '').split(',').map(Number); return v.length === n && v.every(isFinite) ? v : null; };
const fail = (msg) => { $('err').textContent = msg; console.error(msg); };
const status = (msg) => { $('status').textContent = msg || ''; };

// ---------------------------------------------------------------- math (column-major 4x4)
function mul4(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
  return o;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
function lookAt(eye, target, up) {
  const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]), -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1]);
}
function fromQuatPos(qx, qy, qz, qw, p) {        // three.js / GL convention, quaternion x,y,z,w
  return new Float32Array([
    1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw), 0,
    2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qx * qw), 0,
    2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy), 0, p[0], p[1], p[2], 1]);
}

// Scenes are stored in OpenCV coordinates (x right, y down, z forward) with the photo's / video's camera
// at the origin. The app sends the levelling transform (q = rotation x,y,z,w; p = translation) into GL
// coordinates (y up, looking down -z). Without it, a 180° turn about X does the same, unlevelled.
const sq = nums('q', 4) || [1, 0, 0, 0];
const model0 = fromQuatPos(sq[0], sq[1], sq[2], sq[3], nums('p', 3) || [0, 0, 0]);
const depth0 = Math.max(parseFloat(q.get('depth') || '2'), 0.05);
const target0 = nums('t', 3) || [0, 0, -depth0];

// ---------------------------------------------------------------- size (scale about the camera's place)
const SIZE_KEY = 'size:' + sceneName;
const size = { value: 1, initial: 1 };
{
  let v = parseFloat(q.get('s') || '');
  if (!(v > 0)) { try { v = parseFloat(localStorage.getItem(SIZE_KEY) || ''); } catch (e) { v = NaN; } }
  if (v > 0) size.value = size.initial = Math.min(Math.max(v, 0.2), 5);
}
function setSize(v) {
  size.value = Math.min(Math.max(v, 0.2), 5);
  $('size').textContent = Math.round(size.value * 100) + ' %';
  try { localStorage.setItem(SIZE_KEY, String(size.value)); } catch (e) { /* private mode */ }
}
setSize(size.value);
function model() {
  const s = size.value;
  return mul4(new Float32Array([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]), model0);
}
$('smaller').onclick = () => setSize(size.value / 1.05);
$('bigger').onclick = () => setSize(size.value * 1.05);
$('sizereset').onclick = () => setSize(1);

// ---------------------------------------------------------------- sort worker
// Input: the splat texture data (8 words per splat, words 0-2 = position) and a view row; output: the
// splat indices nearest first (16-bit counting sort). Every frame of a video has the same splats in the
// same order, so an order sorted for the previous frame is still a valid (almost right) order.
function sortWorker(self) {
  let f32 = null, n = 0, zrow = null, lastZ = null, dirty = false, busy = false;
  function sort() {
    const keys = new Float32Array(n);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const o = 8 * i;
      const d = -(zrow[0] * f32[o] + zrow[1] * f32[o + 1] + zrow[2] * f32[o + 2] + zrow[3]);
      keys[i] = d;
      if (d < mn) mn = d;
      if (d > mx) mx = d;
    }
    const B = 65536, scale = (B - 1) / Math.max(mx - mn, 1e-9);
    const counts = new Uint32Array(B), q16 = new Uint16Array(n);
    for (let i = 0; i < n; i++) { const k = ((keys[i] - mn) * scale) | 0; q16[i] = k; counts[k]++; }
    const starts = new Uint32Array(B);
    for (let k = 1; k < B; k++) starts[k] = starts[k - 1] + counts[k - 1];
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[starts[q16[i]]++] = i;
    lastZ = zrow; dirty = false;
    self.postMessage({ order, count: n }, [order.buffer]);
  }
  function maybeSort() {
    if (!f32 || !zrow) return;
    if (!dirty && lastZ) {
      const d = Math.hypot(zrow[0] - lastZ[0], zrow[1] - lastZ[1], zrow[2] - lastZ[2]);
      if (d < 0.002 && Math.abs(zrow[3] - lastZ[3]) < 0.002) return;
    }
    if (busy) return;
    busy = true;
    sort();
    setTimeout(() => { busy = false; maybeSort(); }, 0);
  }
  self.onmessage = (e) => {
    const d = e.data;
    if (d.tex) { f32 = new Float32Array(d.tex.buffer || d.tex); n = d.count; dirty = true; }
    if (d.zrow) zrow = d.zrow;
    maybeSort();
  };
}

// ---------------------------------------------------------------- decode worker (.splv)
// One keyframe group at a time: fetch its bytes (HTTP range), inflate each frame, undo the filters
// (temporal: + previous frame; spatial: running sum), dequantize with lookup tables and pack the splat
// texture (words: x, y, z, quaternion bytes, scale x, y, z, rgba bytes).
function decodeWorker(self) {
  const FIELDS = [['u', 2], ['v', 2], ['z', 2], ['s0', 1], ['s1', 1], ['s2', 1], ['q0', 1], ['q1', 1], ['q2', 1],
    ['q3', 1], ['r', 1], ['g', 1], ['b', 1], ['a', 1]];
  let url = '', dataStart = 0, count = 0, index = null, LU, LV, LZ, LS, gen = 0;
  function luts(quant) {
    const lin = (lo, hi, levels, f) => { const t = new Float32Array(levels + 1); for (let i = 0; i <= levels; i++) t[i] = f(lo + i / levels * (hi - lo)); return t; };
    LU = lin(quant.u[0], quant.u[1], 65535, (x) => x);
    LV = lin(quant.v[0], quant.v[1], 65535, (x) => x);
    LZ = lin(quant.logz[0], quant.logz[1], 65535, Math.exp);
    LS = lin(quant.logs[0], quant.logs[1], 255, Math.exp);
  }
  async function inflate(bytes) {
    const ds = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(ds).arrayBuffer());
  }
  async function decode(chunk, prev) {
    const modes = chunk.subarray(0, 14);
    const raw = await inflate(chunk.subarray(14));
    const out = {};
    let o = 0;
    for (let f = 0; f < 14; f++) {
      const [name, size] = FIELDS[f];
      const mode = modes[f];
      const p = prev ? prev[name] : null;
      if ((mode & 1) && !p) throw new Error('a frame needs the previous one');
      if (size === 2) {
        const a = new Uint16Array(count);
        let acc = 0;
        for (let i = 0; i < count; i++) {
          let v = raw[o + i] | (raw[o + count + i] << 8);
          if (mode & 2) { acc = (acc + v) & 0xffff; v = acc; }
          if (mode & 1) v = (v + p[i]) & 0xffff;
          a[i] = v;
        }
        o += 2 * count;
        out[name] = a;
      } else {
        const a = new Uint8Array(count);
        let acc = 0;
        for (let i = 0; i < count; i++) {
          let v = raw[o + i];
          if (mode & 2) { acc = (acc + v) & 0xff; v = acc; }
          if (mode & 1) v = (v + p[i]) & 0xff;
          a[i] = v;
        }
        o += count;
        out[name] = a;
      }
    }
    return out;
  }
  function pack(qa, rows) {
    const words = new Uint32Array(rows * 1024 * 8);
    const f = new Float32Array(words.buffer);
    const { u, v, z, s0, s1, s2, q0, q1, q2, q3, r, g, b, a } = qa;
    for (let i = 0; i < count; i++) {
      const o = 8 * i, d = LZ[z[i]];
      f[o] = LU[u[i]] * d; f[o + 1] = LV[v[i]] * d; f[o + 2] = d;
      words[o + 3] = q0[i] | (q1[i] << 8) | (q2[i] << 16) | (q3[i] << 24);
      f[o + 4] = LS[s0[i]] * d; f[o + 5] = LS[s1[i]] * d; f[o + 6] = LS[s2[i]] * d;
      words[o + 7] = r[i] | (g[i] << 8) | (b[i] << 16) | (a[i] << 24);
    }
    return words;
  }
  self.onmessage = async (e) => {
    const d = e.data;
    if (d.init) { ({ url, dataStart, count, index } = d.init); luts(d.init.quant); return; }
    if (d.abort !== undefined) { gen = d.abort; return; }
    if (d.gop) {
      const my = gen = d.gen;
      const { from, to, rows } = d.gop;
      try {
        const a = dataStart + index[from][0], b = dataStart + index[to][0] + index[to][1] - 1;
        const res = await fetch(url, { headers: { Range: `bytes=${a}-${b}` } });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        let buf = new Uint8Array(await res.arrayBuffer());
        if (res.status === 200) buf = buf.subarray(a, b + 1);      // server ignored the range
        let prev = null;
        for (let k = from; k <= to; k++) {
          if (gen !== my) return;
          const off = dataStart + index[k][0] - a;
          const qa = await decode(buf.subarray(off, off + index[k][1]), index[k][2] ? null : prev);
          prev = qa;
          if (gen !== my) return;
          const tex = pack(qa, rows);
          self.postMessage({ gen: my, k, tex }, [tex.buffer]);
        }
        self.postMessage({ gen: my, done: true });
      } catch (err) {
        self.postMessage({ gen: my, error: String(err), done: true });
      }
    }
  };
}
const workerOf = (fn) => new Worker(URL.createObjectURL(new Blob(['(', fn.toString(), ')(self)'], { type: 'application/javascript' })));

// ---------------------------------------------------------------- shaders
const VS = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D u_texture;
uniform mat4 projection, modelView;
uniform vec2 focal;
uniform vec2 viewport;
in vec2 position;
in int index;
out vec4 vColor;
out vec2 vPosition;
void main () {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  ivec2 base = ivec2(int((uint(index) & 0x3ffu) << 1), int(uint(index) >> 10));
  uvec4 A = texelFetch(u_texture, base, 0);
  vec4 cam = modelView * vec4(uintBitsToFloat(A.xyz), 1.0);
  if (cam.z > -0.01) return;
  vec4 pos = projection * cam;
  float clip = 1.2 * pos.w;
  if (pos.x < -clip || pos.x > clip || pos.y < -clip || pos.y > clip) return;
  uvec4 B = texelFetch(u_texture, base + ivec2(1, 0), 0);
  vec4 rot = vec4(float(A.w & 0xffu), float((A.w >> 8) & 0xffu), float((A.w >> 16) & 0xffu), float(A.w >> 24)) / 127.5 - 1.0;
  rot /= max(length(rot), 1e-12);
  float w = rot.x, x = rot.y, y = rot.z, z = rot.w;
  mat3 R = mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
                2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
                2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
  vec3 s = uintBitsToFloat(B.xyz);
  mat3 M = R * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 J = mat3(focal.x / -cam.z, 0.0, 0.0,
                0.0, focal.y / -cam.z, 0.0,
                focal.x * cam.x / (cam.z * cam.z), focal.y * cam.y / (cam.z * cam.z), 0.0);
  mat3 T = J * mat3(modelView) * M;
  mat3 cov = T * transpose(T);
  float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = mid - rad;
  if (l2 < 0.0) return;
  vec2 dir = abs(b) < 1e-9 ? (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(b, l1 - a));
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dir;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x);
  uint rgba = B.w;
  vColor = vec4(float(rgba & 0xffu), float((rgba >> 8) & 0xffu), float((rgba >> 16) & 0xffu),
                float((rgba >> 24) & 0xffu)) / 255.0;
  gl_Position = vec4(pos.xy / pos.w + (position.x * major + position.y * minor) * 2.0 / viewport, 0.0, 1.0);
  vPosition = position;
}`;
const FS = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vPosition;
out vec4 fragColor;
void main () {
  float A = -dot(vPosition, vPosition);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  fragColor = vec4(B * vColor.rgb, B);
}`;

// ---------------------------------------------------------------- GL setup
const canvas = $('canvas');
const gl = canvas.getContext('webgl2', { antialias: false, xrCompatible: true });
if (!gl) fail('This browser has no WebGL2.');
function shader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) fail(gl.getProgramInfoLog(prog));
gl.useProgram(prog);
const U = {};
for (const k of ['projection', 'modelView', 'focal', 'viewport', 'u_texture']) U[k] = gl.getUniformLocation(prog, k);
gl.uniform1i(U.u_texture, 0);
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, 'position');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
const indexBuf = gl.createBuffer();
const aIdx = gl.getAttribLocation(prog, 'index');
gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
gl.enableVertexAttribArray(aIdx);
gl.vertexAttribIPointer(aIdx, 1, gl.INT, 0, 0);
gl.vertexAttribDivisor(aIdx, 1);
const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, texture);
for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

let drawCount = 0, texRows = 0;
const sorter = workerOf(sortWorker);
sorter.onmessage = (e) => {
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
  gl.bufferData(gl.ARRAY_BUFFER, e.data.order, gl.DYNAMIC_DRAW);
  drawCount = e.data.count;
};
// Upload splat texture data (8 words per splat, 1024 splats per row) and hand it to the sorter.
function upload(words, count) {
  const rows = words.length / 8192;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  if (rows !== texRows) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, 2048, rows, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, words);
    texRows = rows;
  } else {
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 2048, rows, gl.RGBA_INTEGER, gl.UNSIGNED_INT, words);
  }
  sorter.postMessage({ tex: words, count }, [words.buffer]);
}

function draw(projection, view, width, height) {
  if (!drawCount) return;
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniformMatrix4fv(U.projection, false, projection);
  gl.uniformMatrix4fv(U.modelView, false, mul4(view, model()));
  gl.uniform2f(U.focal, projection[0] * width / 2, projection[5] * height / 2);
  gl.uniform2f(U.viewport, width, height);
  gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, drawCount);
}
const rowZ = (m) => [m[2], m[6], m[10], m[14]];

// ---------------------------------------------------------------- splat video playback
// The clock is the sound when there is sound (it never drifts), else the frame timer. When the next
// frame is not decoded yet, playback waits (sound paused) instead of skipping: stable speed, no jumps.
const audio = $('audio');
const vid = {
  on: false, frames: 1, fps: 24, count: 0, rows: 0, index: null, keys: [], clock: 0, playing: false,
  loop: true, shown: -1, ready: new Map(), gen: 0, workers: [], nextGop: 0, inflight: 0, maxAhead: 24,
  waiting: false, hasAudio: false, ended: false,
};
function frameAt(t) { return Math.min(Math.max(Math.floor(t * vid.fps + 1e-6), 0), vid.frames - 1); }
function gopOf(k) { let g = 0; for (let i = 0; i < vid.keys.length; i++) if (vid.keys[i] <= k) g = i; return g; }
function gopRange(g) { return [vid.keys[g], g + 1 < vid.keys.length ? vid.keys[g + 1] - 1 : vid.frames - 1]; }
function pump() {
  for (const w of vid.workers) {
    if (w.busy || vid.nextGop >= vid.keys.length) continue;
    const [from, to] = gopRange(vid.nextGop);
    if (vid.ready.size + vid.inflight >= vid.maxAhead) return;     // enough decoded or on the way
    w.busy = true;
    vid.inflight += to - from + 1;
    w.pending = to - from + 1;
    w.postMessage({ gen: vid.gen, gop: { from, to, rows: vid.rows } });
    vid.nextGop++;
  }
}
function restartAt(k) {
  vid.gen++;
  vid.ready.clear();
  vid.inflight = 0;
  for (const w of vid.workers) { w.postMessage({ abort: vid.gen }); w.busy = false; w.pending = 0; }
  vid.nextGop = gopOf(k);
  pump();
}
function onDecoded(w, e) {
  const d = e.data;
  if (d.gen !== vid.gen) return;
  if (d.tex) {
    w.pending--; vid.inflight--;
    if (d.k >= frameAt(vid.clock) || vid.shown < 0) vid.ready.set(d.k, d.tex);
  }
  if (d.error) fail('Could not decode the splat video: ' + d.error);
  if (d.done) { vid.inflight -= w.pending; w.pending = 0; w.busy = false; }
  pump();
}
function setPlaying(on) {
  if (!vid.on) return;
  if (on && vid.ended) { seek(0); vid.ended = false; }
  vid.playing = on;
  $('play').textContent = on ? 'Pause' : 'Play';
  if (vid.hasAudio) {
    if (on && !vid.waiting) audio.play().catch(() => {});
    else audio.pause();
  }
}
function seek(t) {
  vid.clock = Math.min(Math.max(t, 0), (vid.frames - 0.5) / vid.fps);
  if (vid.hasAudio) audio.currentTime = vid.clock;
  const k = frameAt(vid.clock);
  if (!vid.ready.has(k)) restartAt(k);
}
// Advance the clock and show the right frame (called once per displayed frame, desktop or VR).
function tick(dt) {
  if (!vid.on) return;
  if (vid.playing && !vid.waiting) {
    const next = vid.hasAudio && !audio.paused ? audio.currentTime : vid.clock + dt;
    if (next >= vid.frames / vid.fps) {
      if (vid.loop) { vid.clock = 0; if (vid.hasAudio) audio.currentTime = 0; restartAt(0); }
      else { vid.clock = (vid.frames - 0.5) / vid.fps; vid.ended = true; setPlaying(false); }
    } else vid.clock = Math.max(next, vid.clock);
  }
  const want = frameAt(vid.clock);
  for (const key of vid.ready.keys()) if (key < want) vid.ready.delete(key);
  if (want !== vid.shown) {
    const tex = vid.ready.get(want);
    if (tex) {
      vid.ready.delete(want);
      upload(tex, vid.count);
      vid.shown = want;
      if (vid.waiting) { vid.waiting = false; status(''); if (vid.playing && vid.hasAudio) audio.play().catch(() => {}); }
    } else if (vid.playing && !vid.waiting && vid.shown >= 0) {
      vid.waiting = true;                // not decoded yet: wait for it (sound too)
      status('buffering…');
      if (vid.hasAudio) audio.pause();
      if (!vid.inflight && vid.ready.size === 0) restartAt(want);
    }
  }
  if (vid.waiting && vid.hasAudio) vid.clock = Math.min(vid.clock, (want + 0.999) / vid.fps);
  pump();
  updateBar();
}
function updateBar() {
  const k = frameAt(vid.clock);
  $('seek').value = k;
  $('time').textContent = `${vid.clock.toFixed(1)} / ${(vid.frames / vid.fps).toFixed(1)} s`;
}
$('play').onclick = () => setPlaying(!vid.playing);
$('seek').oninput = (e) => { seek((+e.target.value + 0.5) / vid.fps); updateBar(); };
$('loop').onchange = (e) => { vid.loop = e.target.checked; };

// ---------------------------------------------------------------- desktop camera
const orbit = {};
function resetView() {
  const off = sub([0, 0, 0], target0);
  orbit.target = target0.slice();
  orbit.r = Math.hypot(...off) || depth0;
  orbit.base = orbit.r;
  orbit.yaw = Math.atan2(off[0], off[2]);
  orbit.pitch = Math.asin(Math.max(-1, Math.min(1, off[1] / orbit.r)));
}
resetView();
function desktopView() {
  const cp = Math.cos(orbit.pitch), t = orbit.target;
  const eye = [t[0] + orbit.r * cp * Math.sin(orbit.yaw), t[1] + orbit.r * Math.sin(orbit.pitch),
    t[2] + orbit.r * cp * Math.cos(orbit.yaw)];
  return lookAt(eye, t, [0, 1, 0]);
}
let drag = null;
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.button !== 0 || e.shiftKey }; });
window.addEventListener('pointerup', () => { drag = null; });
window.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const v = desktopView(), s = orbit.r * 0.0015;
    for (let i = 0; i < 3; i++) orbit.target[i] += (-dx * v[i * 4] + dy * v[i * 4 + 1]) * s;
  } else {
    orbit.yaw -= dx * 0.005;
    orbit.pitch = Math.max(-1.5, Math.min(1.5, orbit.pitch + dy * 0.005));
  }
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  orbit.r = Math.max(orbit.base * 0.02, Math.min(orbit.base * 20, orbit.r * Math.pow(1.1, e.deltaY / 100)));
}, { passive: false });
window.addEventListener('keydown', (e) => {
  if (e.key === ' ' && vid.on) { setPlaying(!vid.playing); e.preventDefault(); }
  else if (e.key === ',' && vid.on) { setPlaying(false); seek((frameAt(vid.clock) - 0.5) / vid.fps); updateBar(); }
  else if (e.key === '.' && vid.on) { setPlaying(false); seek((frameAt(vid.clock) + 1.5) / vid.fps); updateBar(); }
  else if (e.key === 'r' || e.key === 'R') resetView();
  else if (e.key === '+' || e.key === '=') setSize(size.value * 1.05);
  else if (e.key === '-' || e.key === '_') setSize(size.value / 1.05);
  else if (e.key === '0') setSize(1);
});

// ---------------------------------------------------------------- desktop loop
let lastT = 0;
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
}
window.addEventListener('resize', resize);
resize();
function desktopFrame(now) {
  if (xr.session) return;
  requestAnimationFrame(desktopFrame);
  const dt = lastT ? Math.min((now - lastT) / 1000, 0.25) : 0;
  lastT = now;
  tick(dt);
  const proj = perspective(60 * Math.PI / 180, canvas.width / canvas.height, 0.02, 1000);
  const view = desktopView();
  sorter.postMessage({ zrow: rowZ(mul4(view, model())) });
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  draw(proj, view, canvas.width, canvas.height);
}

// ---------------------------------------------------------------- WebXR
// Moves the user by offsetting the XR reference space ('local': the origin is where the headset was
// when ENTER VR was pressed, i.e. you start where the camera stood, at eye height).
const xr = { session: null, base: null, space: null, pos: [0, 0, 0], yaw: 0, lastT: 0, snap: true, reset: false };
const SPEED = Math.min(Math.max(depth0 * 0.6, 0.5), 5);
function applyRig() {
  const h = -xr.yaw / 2, qi = { x: 0, y: Math.sin(h), z: 0, w: Math.cos(h) };
  const c = Math.cos(-xr.yaw), s = Math.sin(-xr.yaw);
  const p = xr.pos.map((v) => -v);
  const pr = [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]];
  xr.space = xr.base.getOffsetReferenceSpace(new XRRigidTransform({ x: pr[0], y: pr[1], z: pr[2], w: 1 }, qi));
}
const dz = (v) => (Math.abs(v) < 0.15 ? 0 : v);
const pressed = (gp, i) => !!(gp.buttons[i] && gp.buttons[i].pressed);
function xrFrame(t, frame) {
  const session = frame.session;
  session.requestAnimationFrame(xrFrame);
  const dt = xr.lastT ? Math.min((t - xr.lastT) / 1000, 0.25) : 0;
  xr.lastT = t;
  let mx = 0, mz = 0, my = 0, turn = 0, reset = false, grow = 0;
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp) continue;
    const ax = dz(gp.axes.length >= 4 ? gp.axes[2] : (gp.axes[0] || 0));
    const ay = dz(gp.axes.length >= 4 ? gp.axes[3] : (gp.axes[1] || 0));
    if (src.handedness === 'left') { mx += ax; mz += ay; } else { turn = ax; my -= ay; }
    if (pressed(gp, 4)) grow -= 1;      // A / X: smaller
    if (pressed(gp, 5)) grow += 1;      // B / Y: bigger
    if (pressed(gp, 3)) reset = true;   // stick click: back to the start, original size
  }
  let changed = false;
  if (grow) setSize(size.value * Math.exp(0.5 * grow * dt));
  if (reset && !xr.reset) { xr.pos = [0, 0, 0]; xr.yaw = 0; setSize(1); changed = true; }
  xr.reset = reset;
  const pose0 = frame.getViewerPose(xr.space);
  if ((mx || mz || my) && pose0) {
    const m = pose0.transform.matrix;                   // head forward (-z) in the rig frame
    let f = [-m[8], 0, -m[10]];
    const l = Math.hypot(f[0], f[2]) || 1;
    f = [f[0] / l, 0, f[2] / l];
    const c = Math.cos(xr.yaw), s = Math.sin(xr.yaw);
    const fw = [c * f[0] + s * f[2], 0, -s * f[0] + c * f[2]];     // into the base frame
    const right = [-fw[2], 0, fw[0]];
    for (let i = 0; i < 3; i += 2) xr.pos[i] += (-mz * fw[i] + mx * right[i]) * SPEED * dt;
    xr.pos[1] += my * SPEED * 0.5 * dt;
    changed = true;
  }
  if (Math.abs(turn) > 0.7 && xr.snap) {
    const pb = frame.getViewerPose(xr.base);
    const hp = pb ? pb.transform.position : { x: 0, y: 0, z: 0 };
    const rot = (v, a) => [Math.cos(a) * v[0] + Math.sin(a) * v[2], v[1], -Math.sin(a) * v[0] + Math.cos(a) * v[2]];
    const head = [hp.x, hp.y, hp.z];
    const before = rot(head, xr.yaw).map((v, i) => v + xr.pos[i]);
    xr.yaw += (turn > 0 ? -1 : 1) * Math.PI / 6;
    const after = rot(head, xr.yaw);
    xr.pos = before.map((v, i) => v - after[i]);
    xr.snap = false;
    changed = true;
  } else if (Math.abs(turn) < 0.3) {
    xr.snap = true;
  }
  if (changed) applyRig();
  tick(dt);
  const pose = frame.getViewerPose(xr.space);
  if (!pose) return;
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  sorter.postMessage({ zrow: rowZ(mul4(pose.transform.inverse.matrix, model())) });
  for (const view of pose.views) {
    const vp = layer.getViewport(view);
    gl.viewport(vp.x, vp.y, vp.width, vp.height);
    draw(view.projectionMatrix, view.transform.inverse.matrix, vp.width, vp.height);
  }
}
async function enterVR() {
  if (vid.on && !vid.playing) setPlaying(true);      // a click: the sound may start now
  try {
    const session = await navigator.xr.requestSession('immersive-vr');
    await gl.makeXRCompatible();
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl, { antialias: false }), depthNear: 0.02, depthFar: 1000 });
    xr.base = await session.requestReferenceSpace('local');
    xr.session = session;
    xr.pos = [0, 0, 0]; xr.yaw = 0; xr.lastT = 0;
    applyRig();
    $('vr').textContent = 'EXIT VR';
    session.addEventListener('select', () => setPlaying(!vid.playing));     // trigger (a user gesture: sound allowed)
    session.addEventListener('end', () => {
      xr.session = null;
      $('vr').textContent = 'ENTER VR';
      lastT = 0;
      requestAnimationFrame(desktopFrame);
    });
    session.requestAnimationFrame(xrFrame);
  } catch (e) {
    fail('Could not start VR: ' + e);
  }
}
if (navigator.xr) {
  navigator.xr.isSessionSupported('immersive-vr').then((ok) => {
    if (!ok) return;
    $('vr').style.display = 'block';
    $('vr').onclick = () => (xr.session ? xr.session.end() : enterVR());
  });
}

// ---------------------------------------------------------------- loading
async function fetchRange(a, b) {
  const res = await fetch(sceneUrl, { headers: { Range: `bytes=${a}-${b}` } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  return res.status === 200 ? buf.subarray(a, b + 1) : buf;
}
async function loadVideo() {
  let head = await fetchRange(0, 65535);
  const dv = new DataView(head.buffer, head.byteOffset, head.byteLength);
  const hl = dv.getUint32(8, true);
  if (12 + hl > head.length) head = await fetchRange(0, 12 + hl - 1);
  const header = JSON.parse(new TextDecoder().decode(head.subarray(12, 12 + hl)));
  const dataStart = 12 + hl + ((8 - ((12 + hl) % 8)) % 8);
  vid.on = true;
  vid.frames = header.frames;
  vid.fps = header.fps;
  vid.count = header.count;
  vid.rows = Math.ceil(header.count / 1024);
  vid.index = header.index;
  vid.keys = header.index.map((e, k) => (e[2] ? k : -1)).filter((k) => k >= 0);
  if (!vid.keys.length || vid.keys[0] !== 0) throw new Error('the splat video does not start with a keyframe');
  const frameBytes = vid.rows * 1024 * 32;
  vid.maxAhead = Math.max(Math.min(Math.floor(1.2e9 / frameBytes), Math.ceil(4 * vid.fps)), Math.ceil(2 * vid.fps) + 2, 4);
  const nWorkers = Math.min(Math.max((navigator.hardwareConcurrency || 4) >> 2, 2), 4);
  for (let i = 0; i < nWorkers; i++) {
    const w = workerOf(decodeWorker);
    w.busy = false; w.pending = 0;
    w.onmessage = (e) => onDecoded(w, e);
    w.postMessage({ init: { url: sceneUrl, dataStart, count: vid.count, index: vid.index, quant: header.quant } });
    vid.workers.push(w);
  }
  document.querySelectorAll('.vid').forEach((el) => { el.style.display = 'inline'; });
  $('video').style.display = 'flex';
  $('seek').max = vid.frames - 1;
  if (header.audio) {
    const bytes = await fetchRange(dataStart + header.audio.offset, dataStart + header.audio.offset + header.audio.size - 1);
    audio.src = URL.createObjectURL(new Blob([bytes], { type: header.audio.mime || 'audio/mp4' }));
    vid.hasAudio = true;
    audio.addEventListener('ended', () => { if (!vid.loop) { vid.ended = true; setPlaying(false); } });
  }
  status(header.frames > 1 ? `${header.frames} frames · ${header.fps} fps · ${(header.count / 1000).toFixed(0)}k splats${vid.hasAudio ? ' · sound' : ''} — press Play` : '');
  restartAt(0);
  updateBar();
  $('progress').style.display = 'none';
}
async function loadSplat() {
  const res = await fetch(sceneUrl);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const total = +res.headers.get('Content-Length') || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    if (total) $('progress').style.width = (100 * got / total).toFixed(1) + '%';
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  // .splat: 32 bytes per splat = position (3 float), scale (3 float), rgba (4 byte), rotation (4 byte, w x y z
  // as q * 128 + 128) → the player's layout (quaternion bytes as (q + 1) * 127.5)
  const n = Math.floor(got / 32);
  const f = new Float32Array(buf.buffer, 0, n * 8);
  const u8 = buf;
  const rows = Math.ceil(n / 1024);
  const words = new Uint32Array(rows * 1024 * 8);
  const wf = new Float32Array(words.buffer);
  for (let i = 0; i < n; i++) {
    const o8 = 8 * i, b = 32 * i;
    wf[o8] = f[o8]; wf[o8 + 1] = f[o8 + 1]; wf[o8 + 2] = f[o8 + 2];
    const qb = (k) => Math.min(Math.max(Math.round(((u8[b + 28 + k] - 128) / 128 + 1) * 127.5), 0), 255);
    words[o8 + 3] = qb(0) | (qb(1) << 8) | (qb(2) << 16) | (qb(3) << 24);
    wf[o8 + 4] = f[o8 + 3]; wf[o8 + 5] = f[o8 + 4]; wf[o8 + 6] = f[o8 + 5];
    words[o8 + 7] = u8[b + 24] | (u8[b + 25] << 8) | (u8[b + 26] << 16) | (u8[b + 27] << 24);
  }
  upload(words, n);
  $('progress').style.display = 'none';
}
async function load() {
  const head = await fetchRange(0, 3).catch(() => null);
  const isVideo = head && head.length >= 4 && String.fromCharCode(...head.subarray(0, 4)) === 'SPLV';
  if (isVideo) await loadVideo(); else await loadSplat();
}
document.querySelectorAll('.vid').forEach((el) => { el.style.display = 'none'; });
load().then(() => requestAnimationFrame(desktopFrame)).catch((e) => fail('Could not load the scene: ' + e));
window.player = { vid, orbit, xr, size, setSize, setPlaying, seek, audio, stats: () => ({ drawCount, texRows }) };
