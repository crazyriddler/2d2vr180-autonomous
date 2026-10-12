// 2D2VR180 4D player: plays a .splatv 4D Gaussian splat scene on the desktop and in VR (WebXR).
//
// The splat rasteriser (vertex/fragment shaders, worker depth sort, the .splatv layout of 16 uint32
// per splat) is adapted from splaTV by Kevin Kwok (MIT, see LICENSE-splatv.txt). 2D2VR180 adds
// OpenGL/WebXR camera conventions (stereo views with off-axis projections), frame-accurate
// playback with a timeline, time-bucketed sorting (only the splats visible at the current time are
// sorted and drawn), the levelling transform of the app and VR locomotion.

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const sceneUrl = q.get('scene') || 'scene.splatv';
$('title').textContent = '2D2VR180 — ' + (q.get('name') || sceneUrl);
const nums = (k, n) => { const v = (q.get(k) || '').split(',').map(Number); return v.length === n && v.every(isFinite) ? v : null; };
const fail = (msg) => { $('err').textContent = msg; console.error(msg); };

// ---------------------------------------------------------------- math (column-major 4x4)
function mul4(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
  return o;
}
function inv4(a) {
  const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4];
  const b03 = a[1] * a[6] - a[2] * a[5], b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6];
  const b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12], b08 = a[8] * a[15] - a[11] * a[12];
  const b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  const d = 1 / det;
  return new Float32Array([
    (a[5] * b11 - a[6] * b10 + a[7] * b09) * d, (a[2] * b10 - a[1] * b11 - a[3] * b09) * d,
    (a[13] * b05 - a[14] * b04 + a[15] * b03) * d, (a[10] * b04 - a[9] * b05 - a[11] * b03) * d,
    (a[6] * b08 - a[4] * b11 - a[7] * b07) * d, (a[0] * b11 - a[2] * b08 + a[3] * b07) * d,
    (a[14] * b02 - a[12] * b05 - a[15] * b01) * d, (a[8] * b05 - a[10] * b02 + a[11] * b01) * d,
    (a[4] * b10 - a[5] * b08 + a[7] * b06) * d, (a[1] * b08 - a[0] * b10 - a[3] * b06) * d,
    (a[12] * b04 - a[13] * b02 + a[15] * b00) * d, (a[9] * b02 - a[8] * b04 - a[11] * b00) * d,
    (a[5] * b07 - a[4] * b09 - a[6] * b06) * d, (a[0] * b09 - a[1] * b07 + a[2] * b06) * d,
    (a[13] * b01 - a[12] * b03 - a[14] * b00) * d, (a[8] * b03 - a[9] * b01 + a[10] * b00) * d]);
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
const FLIP_YZ = new Float32Array([1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]);

// Scenes are stored in OpenCV coordinates (x right, y down, z forward). The app sends the levelling
// transform (q = rotation x,y,z,w; p = translation) into GL coordinates (y up, camera looking down -z)
// with the filming camera at the origin. Without it, a 180° turn about X does the same unlevelled.
const sq = nums('q', 4) || [1, 0, 0, 0];
const model = fromQuatPos(sq[0], sq[1], sq[2], sq[3], nums('p', 3) || [0, 0, 0]);
const depth0 = Math.max(parseFloat(q.get('depth') || '2'), 0.05);
const target0 = nums('t', 3) || [0, 0, -depth0];

// ---------------------------------------------------------------- sort worker
function sortWorker(self) {
  let pos = null, n = 0, bins = 1, staticIdx = null, bucketStart = null, bucketIdx = null;
  let zrow = null, bin = 0, lastZ = null, lastBin = -1, busy = false, again = false;
  const LN = Math.sqrt(-Math.log(0.02));   // temporal opacity below 0.02 is not drawn
  function build(tc, ts, framesMode, F) {
    bins = framesMode ? F : 128;
    const lo = new Int32Array(n), hi = new Int32Array(n);
    let nStatic = 0;
    const counts = new Uint32Array(bins + 1);
    for (let i = 0; i < n; i++) {
      const w = ts[i] * LN;
      let a, b;
      if (framesMode) { a = Math.ceil((tc[i] - w) * F - 0.5); b = Math.floor((tc[i] + w) * F - 0.5); }
      else { a = Math.floor((tc[i] - w) * bins); b = Math.floor((tc[i] + w) * bins); }
      a = Math.max(a, 0); b = Math.min(b, bins - 1);
      if (a <= 0 && b >= bins - 1) { lo[i] = -1; nStatic++; continue; }
      lo[i] = a; hi[i] = b;
      for (let k = a; k <= b; k++) counts[k]++;
    }
    staticIdx = new Uint32Array(nStatic);
    bucketStart = new Uint32Array(bins + 1);
    for (let k = 0; k < bins; k++) bucketStart[k + 1] = bucketStart[k] + counts[k];
    bucketIdx = new Uint32Array(bucketStart[bins]);
    const fill = bucketStart.slice(0, bins);
    let s = 0;
    for (let i = 0; i < n; i++) {
      if (lo[i] < 0) { staticIdx[s++] = i; continue; }
      if (lo[i] > hi[i]) continue;
      for (let k = lo[i]; k <= hi[i]; k++) bucketIdx[fill[k]++] = i;
    }
  }
  function sort() {
    if (!pos || !zrow) return;
    const dyn = bucketIdx.subarray(bucketStart[bin], bucketStart[bin + 1]);
    const m = staticIdx.length + dyn.length;
    const keys = new Float32Array(m);
    let mn = Infinity, mx = -Infinity;
    const at = (j) => (j < staticIdx.length ? staticIdx[j] : dyn[j - staticIdx.length]);
    for (let j = 0; j < m; j++) {
      const i = at(j);
      const d = -(zrow[0] * pos[3 * i] + zrow[1] * pos[3 * i + 1] + zrow[2] * pos[3 * i + 2] + zrow[3]);
      keys[j] = d;
      if (d < mn) mn = d;
      if (d > mx) mx = d;
    }
    // 16-bit counting sort, nearest first (front-to-back blending)
    const B = 65536, scale = (B - 1) / Math.max(mx - mn, 1e-9);
    const counts = new Uint32Array(B), q16 = new Uint16Array(m);
    for (let j = 0; j < m; j++) { const k = ((keys[j] - mn) * scale) | 0; q16[j] = k; counts[k]++; }
    const starts = new Uint32Array(B);
    for (let k = 1; k < B; k++) starts[k] = starts[k - 1] + counts[k - 1];
    const order = new Uint32Array(m);
    for (let j = 0; j < m; j++) order[starts[q16[j]]++] = at(j);
    lastZ = zrow; lastBin = bin;
    self.postMessage({ order, count: m }, [order.buffer]);
  }
  function maybeSort() {
    if (busy) { again = true; return; }
    if (lastZ && bin === lastBin) {
      const d = Math.hypot(zrow[0] - lastZ[0], zrow[1] - lastZ[1], zrow[2] - lastZ[2]);
      if (d < 0.002 && Math.abs(zrow[3] - lastZ[3]) < 0.002) return;
    }
    busy = true;
    sort();
    setTimeout(() => { busy = false; if (again) { again = false; maybeSort(); } }, 0);
  }
  self.onmessage = (e) => {
    const d = e.data;
    if (d.positions) { pos = d.positions; n = pos.length / 3; build(d.tc, d.ts, d.framesMode, d.frames); maybeSort(); }
    if (d.zrow) { zrow = d.zrow; bin = Math.min(Math.max(d.bin | 0, 0), bins - 1); maybeSort(); }
  };
}

// ---------------------------------------------------------------- shaders
const VS = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D u_texture;
uniform mat4 projection, modelView;
uniform vec2 focal;
uniform vec2 viewport;
uniform float time;
in vec2 position;
in int index;
out vec4 vColor;
out vec2 vPosition;
void main () {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  ivec2 base = ivec2(int((uint(index) & 0x3ffu) << 2), int(uint(index) >> 10));
  uvec4 motion1 = texelFetch(u_texture, base + ivec2(3, 0), 0);
  vec2 trbf = unpackHalf2x16(motion1.w);
  float dt = time - trbf.x;
  float u = dt / trbf.y;
  float topacity = exp(-u * u);
  if (topacity < 0.02) return;
  uvec4 motion0 = texelFetch(u_texture, base + ivec2(2, 0), 0);
  uvec4 static0 = texelFetch(u_texture, base, 0);
  vec2 m0 = unpackHalf2x16(motion0.x), m1 = unpackHalf2x16(motion0.y), m2 = unpackHalf2x16(motion0.z),
       m3 = unpackHalf2x16(motion0.w), m4 = unpackHalf2x16(motion1.x);
  vec3 tpos = vec3(m0.xy, m1.x) * dt + vec3(m1.y, m2.xy) * dt * dt + vec3(m3.xy, m4.x) * dt * dt * dt;
  vec4 cam = modelView * vec4(uintBitsToFloat(static0.xyz) + tpos, 1.0);
  if (cam.z > -0.01) return;
  vec4 pos = projection * cam;
  float clip = 1.2 * pos.w;
  if (pos.x < -clip || pos.x > clip || pos.y < -clip || pos.y > clip) return;
  uvec4 static1 = texelFetch(u_texture, base + ivec2(1, 0), 0);
  vec4 rot = vec4(unpackHalf2x16(static0.w), unpackHalf2x16(static1.x))
           + vec4(unpackHalf2x16(motion1.y), unpackHalf2x16(motion1.z)) * dt;
  rot /= max(length(rot), 1e-12);
  float w = rot.x, x = rot.y, y = rot.z, z = rot.w;
  mat3 R = mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
                2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
                2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
  vec3 s = vec3(unpackHalf2x16(static1.y), unpackHalf2x16(static1.z).x);
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
  uint rgba = static1.w;
  vColor = vec4(float(rgba & 0xffu), float((rgba >> 8) & 0xffu), float((rgba >> 16) & 0xffu),
                float((rgba >> 24) & 0xffu)) / 255.0;
  vColor.a *= topacity;
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
for (const k of ['projection', 'modelView', 'focal', 'viewport', 'time', 'u_texture']) U[k] = gl.getUniformLocation(prog, k);
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
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

let drawCount = 0;
const worker = new Worker(URL.createObjectURL(new Blob(['(', sortWorker.toString(), ')(self)'], { type: 'application/javascript' })));
worker.onmessage = (e) => {
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
  gl.bufferData(gl.ARRAY_BUFFER, e.data.order, gl.DYNAMIC_DRAW);
  drawCount = e.data.count;
};

function draw(projection, view, width, height, t) {
  if (!drawCount) return;
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniformMatrix4fv(U.projection, false, projection);
  gl.uniformMatrix4fv(U.modelView, false, mul4(view, model));
  gl.uniform2f(U.focal, projection[0] * width / 2, projection[5] * height / 2);
  gl.uniform2f(U.viewport, width, height);
  gl.uniform1f(U.time, t);
  gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, drawCount);
}

// ---------------------------------------------------------------- playback
const play = { frames: 1, fps: 12, framesMode: true, clock: 0, playing: true, loop: true, speed: 1, cams: [] };
function frameIndex() { return Math.min(Math.floor(play.clock * play.fps), play.frames - 1); }
function sceneTime() {
  if (play.framesMode) return (frameIndex() + 0.5) / play.frames;
  return Math.min(play.clock * play.fps / play.frames, 1);
}
function sortBin() { return play.framesMode ? frameIndex() : Math.floor(sceneTime() * 128); }
function advance(dt) {
  if (!play.playing) return;
  const dur = play.frames / play.fps;
  play.clock += dt * play.speed;
  if (play.clock >= dur) {
    if (play.loop) play.clock %= dur;
    else { play.clock = dur - 1e-6; setPlaying(false); }
  }
}
function setPlaying(on) { play.playing = on; $('play').textContent = on ? 'Pause' : 'Play'; }
function seekFrame(k) { play.clock = (Math.max(0, Math.min(k, play.frames - 1)) + 0.5) / play.fps; }
$('play').onclick = () => setPlaying(!play.playing);
$('seek').oninput = (e) => { setPlaying(false); seekFrame(+e.target.value); };
$('loop').onchange = (e) => { play.loop = e.target.checked; };
$('speed').onchange = (e) => { play.speed = +e.target.value; };
function updateBar() {
  const k = frameIndex();
  $('seek').value = k;
  $('time').textContent = `${(play.clock).toFixed(2)} s · frame ${k + 1}/${play.frames}`;
}

// ---------------------------------------------------------------- desktop camera
const orbit = {};
function resetView() {
  const off = sub([0, 0, 0], target0);
  orbit.target = target0.slice();
  orbit.r = Math.hypot(...off) || depth0;
  orbit.base = orbit.r;
  orbit.yaw = Math.atan2(off[0], off[2]);
  orbit.pitch = Math.asin(Math.max(-1, Math.min(1, off[1] / orbit.r)));
  orbit.follow = false;
}
resetView();
function desktopView() {
  if (orbit.follow && play.cams.length) {   // the filming camera at this frame (OpenCV c2w → GL view)
    const c = play.cams[Math.min(frameIndex(), play.cams.length - 1)];
    const c2w = new Float32Array(16);
    for (let r = 0; r < 4; r++) for (let col = 0; col < 4; col++) c2w[col * 4 + r] = c[r][col];
    return inv4(mul4(mul4(model, c2w), FLIP_YZ));
  }
  const cp = Math.cos(orbit.pitch);
  const eye = [orbit.target[0] + orbit.r * cp * Math.sin(orbit.yaw), orbit.target[1] + orbit.r * Math.sin(orbit.pitch),
    orbit.target[2] + orbit.r * cp * Math.cos(orbit.yaw)];
  return lookAt(eye, orbit.target, [0, 1, 0]);
}
let drag = null;
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.button !== 0 || e.shiftKey }; orbit.follow = false; });
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
  if (e.key === ' ') { setPlaying(!play.playing); e.preventDefault(); }
  else if (e.key === ',') { setPlaying(false); seekFrame(frameIndex() - 1); }
  else if (e.key === '.') { setPlaying(false); seekFrame(frameIndex() + 1); }
  else if (e.key === 'r' || e.key === 'R') resetView();
  else if (e.key === 'f' || e.key === 'F') orbit.follow = !orbit.follow;
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
  const dt = lastT ? Math.min((now - lastT) / 1000, 0.1) : 0;
  lastT = now;
  advance(dt);
  updateBar();
  const proj = perspective(60 * Math.PI / 180, canvas.width / canvas.height, 0.02, 1000);
  const view = desktopView();
  worker.postMessage({ zrow: rowZ(mul4(view, model)), bin: sortBin() });
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  draw(proj, view, canvas.width, canvas.height, sceneTime());
}
function rowZ(m) { return [m[2], m[6], m[10], m[14]]; }

// ---------------------------------------------------------------- WebXR
// Moves the user by offsetting the XR reference space ('local': the origin is where the headset was
// when ENTER VR was pressed, i.e. you start where the filming camera stood, at eye height).
const xr = { session: null, base: null, space: null, pos: [0, 0, 0], yaw: 0, lastT: 0, snap: true, reset: false, trig: false };
const SPEED = Math.min(Math.max(depth0 * 0.6, 0.5), 5);
function applyRig() {
  const h = -xr.yaw / 2, qi = { x: 0, y: Math.sin(h), z: 0, w: Math.cos(h) };
  const c = Math.cos(-xr.yaw), s = Math.sin(-xr.yaw);
  const p = xr.pos.map((v) => -v);
  const pr = [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]];
  xr.space = xr.base.getOffsetReferenceSpace(new XRRigidTransform({ x: pr[0], y: pr[1], z: pr[2], w: 1 }, qi));
}
const dz = (v) => (Math.abs(v) < 0.15 ? 0 : v);
function xrFrame(t, frame) {
  const session = frame.session;
  session.requestAnimationFrame(xrFrame);
  const dt = xr.lastT ? Math.min((t - xr.lastT) / 1000, 0.1) : 0;
  xr.lastT = t;
  let mx = 0, mz = 0, my = 0, turn = 0, reset = false, trig = false;
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp) continue;
    const ax = dz(gp.axes.length >= 4 ? gp.axes[2] : (gp.axes[0] || 0));
    const ay = dz(gp.axes.length >= 4 ? gp.axes[3] : (gp.axes[1] || 0));
    if (src.handedness === 'left') { mx += ax; mz += ay; } else { turn = ax; my -= ay; }
    if (gp.buttons[4] && gp.buttons[4].pressed) reset = true;     // A / X
    if (gp.buttons[0] && gp.buttons[0].pressed) trig = true;      // trigger
  }
  let changed = false;
  if (reset && !xr.reset) { xr.pos = [0, 0, 0]; xr.yaw = 0; changed = true; }
  xr.reset = reset;
  if (trig && !xr.trig) setPlaying(!play.playing);
  xr.trig = trig;
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
  advance(dt);
  const pose = frame.getViewerPose(xr.space);
  if (!pose) return;
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  worker.postMessage({ zrow: rowZ(mul4(pose.transform.inverse.matrix, model)), bin: sortBin() });
  const tScene = sceneTime();
  for (const view of pose.views) {
    const vp = layer.getViewport(view);
    gl.viewport(vp.x, vp.y, vp.width, vp.height);
    draw(view.projectionMatrix, view.transform.inverse.matrix, vp.width, vp.height, tScene);
  }
}
async function enterVR() {
  try {
    const session = await navigator.xr.requestSession('immersive-vr');
    await gl.makeXRCompatible();
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl, { antialias: false }), depthNear: 0.02, depthFar: 1000 });
    xr.base = await session.requestReferenceSpace('local');
    xr.session = session;
    xr.pos = [0, 0, 0]; xr.yaw = 0; xr.lastT = 0;
    applyRig();
    $('vr').textContent = 'EXIT VR';
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
async function load() {
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
  const head = new Uint32Array(buf.buffer, 0, 2);
  if (head[0] !== 0x674b) throw new Error('not a .splatv file');
  const chunks = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + head[1])));
  const ch = chunks.find((c) => c.type === 'splat');
  const off = 8 + head[1];
  const tex = new Uint32Array(buf.buffer.slice(off, off + ch.size));
  const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (ch.texheight > maxTex) throw new Error(`scene too large for this GPU (${ch.texheight} > ${maxTex} texture rows)`);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, ch.texwidth, ch.texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, tex);
  const n = ch.count || Math.floor(tex.length / 16);
  const f32 = new Float32Array(tex.buffer);
  const positions = new Float32Array(3 * n), tc = new Float32Array(n), ts = new Float32Array(n);
  const half = (h) => {                       // IEEE half → float
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * 5.960464477539063e-8;
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  };
  for (let i = 0; i < n; i++) {
    positions[3 * i] = f32[16 * i]; positions[3 * i + 1] = f32[16 * i + 1]; positions[3 * i + 2] = f32[16 * i + 2];
    const w = tex[16 * i + 15];
    tc[i] = half(w & 0xffff);
    ts[i] = half(w >>> 16);
  }
  play.frames = Math.max(1, ch.frames || 1);
  play.fps = ch.fps || 12;
  play.framesMode = (ch.playback || 'continuous') === 'frames';
  play.cams = (ch.frame_cameras || []).map((c) => c.c2w);
  play.clock = 0.5 / play.fps;
  $('seek').max = play.frames - 1;
  if (play.frames <= 1) { $('bar').style.display = 'none'; setPlaying(false); }
  worker.postMessage({ positions, tc, ts, framesMode: play.framesMode, frames: play.frames },
    [positions.buffer, tc.buffer, ts.buffer]);
  $('progress').style.display = 'none';
}
load().then(() => requestAnimationFrame(desktopFrame)).catch((e) => fail('Could not load the scene: ' + e));
window.player = { play, orbit, xr };
