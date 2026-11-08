/* Minimal 3D Gaussian splat renderer on three.js (WebGL2). Written for this supplementary; no third-party code.
 * EWA projection of each Gaussian to a screen-space ellipse (3-sigma quad), 0.3 px^2 low-pass dilation (opacity
 * compensation as in gsplat's antialiased mode is available through uComp), back-to-front premultiplied alpha blending after a 16-bit counting sort of view depths.
 * Only the per-instance index buffer is re-uploaded after a sort; Gaussian data live in float textures. */
(function () {
  'use strict';
  const TEX_W = 2048;

  const VERT = `
precision highp float;
precision highp int;
uniform highp sampler2D uCenter;
uniform highp sampler2D uQuat;
uniform highp sampler2D uScale;
uniform float uMinRatio;
uniform float uSize;
uniform highp sampler2D uColor;
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform float uZMax;
uniform float uComp;
uniform float uRMax;
attribute float splatIndex;
varying vec4 vColor;
varying vec2 vPos;
void main() {
  int idx = int(splatIndex + 0.5);
  ivec2 tc = ivec2(idx % ${TEX_W}, idx / ${TEX_W});
  vec3 c = texelFetch(uCenter, tc, 0).xyz;
  vec4 cam = modelViewMatrix * vec4(c, 1.0);
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  if (c.z > uZMax || length(c - vec3(0.0, 0.0, 1.6)) > uRMax) return;   // optional height / range clip
  if (cam.z > -0.05) return;
  vec4 clip = projectionMatrix * cam;
  float bound = 1.3 * clip.w;
  if (abs(clip.x) > bound || abs(clip.y) > bound) return;
  // display regulariser: smallest scale >= uMinRatio x middle scale, all scales x uSize (0 / 1 = raw export)
  vec3 sc = texelFetch(uScale, tc, 0).xyz;
  float smid = max(min(sc.x, sc.y), min(max(sc.x, sc.y), sc.z));
  sc = max(sc, vec3(uMinRatio * smid)) * uSize;
  vec4 q = texelFetch(uQuat, tc, 0);                    // (w, x, y, z), unit
  float qw = q.x, qx = q.y, qy = q.z, qz = q.w;
  mat3 R = mat3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy),
                2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx),
                2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
  mat3 M = R * mat3(sc.x, 0.0, 0.0, 0.0, sc.y, 0.0, 0.0, 0.0, sc.z);
  mat3 S = M * transpose(M);
  float z = cam.z;
  mat3 J = mat3(uFocal.x / -z, 0.0, 0.0,
                0.0, uFocal.y / -z, 0.0,
                uFocal.x * cam.x / (z * z), uFocal.y * cam.y / (z * z), 0.0);
  mat3 T = J * mat3(modelViewMatrix);
  mat3 C = T * S * transpose(T);
  float c00 = C[0][0], c01 = C[0][1], c11 = C[1][1];
  float det0 = c00 * c11 - c01 * c01;
  c00 += 0.3; c11 += 0.3;
  float det1 = c00 * c11 - c01 * c01;
  float comp = sqrt(max(det0 / det1, 0.0));
  float mid = 0.5 * (c00 + c11);
  float rad = length(vec2(0.5 * (c00 - c11), c01));
  float l1 = mid + rad, l2 = mid - rad;
  if (l2 <= 0.0) return;
  vec2 v1 = abs(c01) > 1e-7 ? normalize(vec2(c01, l1 - c00)) : (c00 >= c11 ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 v2 = vec2(-v1.y, v1.x);
  vec2 major = min(3.0 * sqrt(l1), 2048.0) * v1;
  vec2 minor = min(3.0 * sqrt(l2), 2048.0) * v2;
  vec4 col = texelFetch(uColor, tc, 0);
  col.a *= uComp > 0.5 ? comp : 1.0;     // gsplat 'antialiased' compensation, off by default (classic 3DGS dilation)
  if (col.a < 1.0 / 255.0) return;
  vColor = col;
  vPos = position.xy;
  vec2 off = position.x * major + position.y * minor;
  gl_Position = vec4(clip.xy / clip.w + 2.0 * off / uViewport, 0.0, 1.0);
}`;

  const FRAG = `
precision highp float;
varying vec4 vColor;
varying vec2 vPos;
void main() {
  float r2 = dot(vPos, vPos);
  if (r2 > 1.0) discard;
  float alpha = vColor.a * exp(-4.5 * r2);
  if (alpha < 1.0 / 255.0) discard;
  gl_FragColor = vec4(vColor.rgb * alpha, alpha);
}`;

  function b64ToBytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  /* Packed format written by gs_viewer_export.py: 'PDWS' | u32 header length | JSON header |
     pos u16x3 | log-scale u8x3 | rgba u8x4 | quaternion (w,x,y,z) u8x4. */
  function decodePacked(b64) {
    const bytes = b64ToBytes(b64);
    const dv = new DataView(bytes.buffer);
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== 'PDWS') throw new Error('bad scene file');
    const hlen = dv.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + hlen)));
    const n = header.n;
    let off = 8 + hlen;
    const pos = new Uint16Array(bytes.buffer.slice(off, off + 6 * n)); off += 6 * n;
    const sc = bytes.subarray(off, off + 3 * n); off += 3 * n;
    const rgba = bytes.subarray(off, off + 4 * n); off += 4 * n;
    const qu = bytes.subarray(off, off + 4 * n);
    const H = Math.ceil(n / TEX_W);
    const center = new Float32Array(TEX_W * H * 4);
    const quat = new Float32Array(TEX_W * H * 4);
    const scale = new Float32Array(TEX_W * H * 4);
    const color = new Uint8Array(TEX_W * H * 4);
    color.set(rgba);
    const lo = header.box_min, hi = header.box_max;
    const ls0 = header.log_scale_min, lsr = (header.log_scale_max - header.log_scale_min) / 255;
    const xyz = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) {
        const v = lo[k] + (pos[3 * i + k] / 65535) * (hi[k] - lo[k]);
        center[4 * i + k] = v; xyz[3 * i + k] = v;
        scale[4 * i + k] = Math.exp(ls0 + sc[3 * i + k] * lsr);
      }
      let w = qu[4 * i] / 127.5 - 1, x = qu[4 * i + 1] / 127.5 - 1, y = qu[4 * i + 2] / 127.5 - 1,
        z = qu[4 * i + 3] / 127.5 - 1;
      const qn = Math.hypot(w, x, y, z) || 1;
      quat[4 * i] = w / qn; quat[4 * i + 1] = x / qn; quat[4 * i + 2] = y / qn; quat[4 * i + 3] = z / qn;
    }
    return { header, n, H, center, quat, scale, color, xyz };
  }

  function floatTex(data, H) {
    const t = new THREE.DataTexture(data, TEX_W, H, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.needsUpdate = true;
    return t;
  }

  class SplatMesh extends THREE.Mesh {
    constructor(data) {
      const geo = new THREE.InstancedBufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(
        new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
      geo.setIndex([0, 1, 2, 0, 2, 3]);
      const order = new Float32Array(data.n);
      for (let i = 0; i < data.n; i++) order[i] = i;
      const attr = new THREE.InstancedBufferAttribute(order, 1);
      attr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('splatIndex', attr);
      geo.instanceCount = data.n;
      const ct = new THREE.DataTexture(data.color, TEX_W, data.H, THREE.RGBAFormat, THREE.UnsignedByteType);
      ct.minFilter = ct.magFilter = THREE.NearestFilter; ct.generateMipmaps = false; ct.needsUpdate = true;
      const mat = new THREE.ShaderMaterial({
        vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthTest: false, depthWrite: false,
        blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
        blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
        blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
        uniforms: {
          uCenter: { value: floatTex(data.center, data.H) }, uQuat: { value: floatTex(data.quat, data.H) },
          uScale: { value: floatTex(data.scale, data.H) }, uColor: { value: ct },
          uMinRatio: { value: 0 }, uSize: { value: 1 },
          uFocal: { value: new THREE.Vector2(1, 1) }, uViewport: { value: new THREE.Vector2(1, 1) },
          uZMax: { value: 1e4 }, uRMax: { value: 1e4 }, uComp: { value: 0 }
        }
      });
      super(geo, mat);
      this.frustumCulled = false;
      this.n = data.n; this.xyz = data.xyz; this.header = data.header;
      this.depth = new Float32Array(data.n); this.keys = new Uint32Array(data.n);
      this.counts = new Uint32Array(65536); this.lastView = null;
    }

    /* Call before each render: updates focal/viewport and re-sorts when the view moved. */
    update(camera, renderer) {
      const size = renderer.getDrawingBufferSize(new THREE.Vector2());
      const fy = size.y / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
      this.material.uniforms.uFocal.value.set(fy, fy);
      this.material.uniforms.uViewport.value.copy(size);
      camera.updateMatrixWorld();
      const m = camera.matrixWorldInverse.elements;
      const v = [m[2], m[6], m[10], m[14]];
      if (this.lastView && Math.abs(v[0] - this.lastView[0]) + Math.abs(v[1] - this.lastView[1]) +
          Math.abs(v[2] - this.lastView[2]) + 0.02 * Math.abs(v[3] - this.lastView[3]) < 1e-3) return;
      this.lastView = v;
      const n = this.n, xyz = this.xyz, depth = this.depth;
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const d = v[0] * xyz[3 * i] + v[1] * xyz[3 * i + 1] + v[2] * xyz[3 * i + 2] + v[3];
        depth[i] = d; if (d < lo) lo = d; if (d > hi) hi = d;
      }
      const scale = 65535 / Math.max(hi - lo, 1e-6);
      const counts = this.counts; counts.fill(0);
      const keys = this.keys;
      for (let i = 0; i < n; i++) { const k = ((depth[i] - lo) * scale) | 0; keys[i] = k; counts[k]++; }
      for (let k = 1; k < 65536; k++) counts[k] += counts[k - 1];
      const order = this.geometry.attributes.splatIndex.array;
      for (let i = n - 1; i >= 0; i--) order[--counts[keys[i]]] = i;   // ascending view z = far first
      this.geometry.attributes.splatIndex.needsUpdate = true;
    }

    dispose() {
      this.geometry.dispose();
      for (const u of ['uCenter', 'uQuat', 'uScale', 'uColor']) this.material.uniforms[u].value.dispose();
      this.material.dispose();
    }
  }

  THREE.SplatMesh = SplatMesh;
  THREE.decodePackedSplats = decodePacked;
})();
