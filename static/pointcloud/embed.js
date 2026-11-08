/* Embedded single-scene viewer (one per card of index.html). Scene data are classic scripts (scenes/*.js), so it
   also works from file://. Settings (view, splat size, raw, background) arrive from the parent page by postMessage.
   Renders only when the view or a setting changes. */
(function () {
  'use strict';
  const IDX = window.PANDWM_INDEX;
  const q = new URLSearchParams(location.search);
  const SCENE = +(q.get('scene') || IDX.scenes[0]), FRAME = +(q.get('frame') || IDX.frames[0]);
  const stage = document.getElementById('stage');
  if (q.get('shot') === '1') document.body.classList.add('is-shot');   // preview capture: no overlays
  const statusEl = document.getElementById('status');
  const setStatus = (s) => { statusEl.textContent = s; };
  const post = (msg) => { try { window.parent.postMessage(Object.assign({ scene: SCENE }, msg), '*'); } catch (e) { /* standalone */ } };
  if (!document.createElement('canvas').getContext('webgl2')) {
    setStatus('This viewer needs WebGL 2 (any recent Chrome, Edge, Firefox or Safari).');
    post({ type: 'loaded', ok: false });
    return;
  }
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, preserveDrawingBuffer: true });
  renderer.setClearColor(0x000000, 0);                      // transparent: the card gradient shows through
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  stage.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const HFOV = 75;
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 600);
  camera.up.set(0, 0, 1);                                   // ego frame: x forward, y left, z up
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.12; controls.screenSpacePanning = true;
  controls.maxDistance = 200; controls.minDistance = 0.3;
  // small cards: gentler orbit, and the camera never goes below the road (polar angle from +z <= 89 deg)
  controls.rotateSpeed = 0.3; controls.maxPolarAngle = 0.495 * Math.PI;
  // web page: keep the camera near street level (at most ~36 deg above the orbit centre) and within 40 m; from high
  // above, the ray-thin Gaussians of building facades read as streaks. ?free=1 lifts both limits (top-down preset).
  if (q.get('free') !== '1' && (q.get('view') || 'driver') !== 'top') { controls.minPolarAngle = 0.3 * Math.PI; controls.maxDistance = 40; }
  const egoBox = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(4.6, 1.9, 1.6)),
    new THREE.LineBasicMaterial({ color: 0x3273dc }));
  scene.add(egoBox);

  // display regulariser (shader only, the data are unchanged): smallest scale >= ratio x middle scale, size gain
  const REG = { on: true, ratio: 1.5, size: 1.6 };
  const state = { view: q.get('view') || 'driver', mesh: null };
  function presets(h) {
    const c = h.rig_centre, z0 = h.road_z;
    const back = (hgt, pitch, behind) => {
      const p = THREE.MathUtils.degToRad(pitch), eye = [c[0] - behind, c[1], z0 + hgt], d = hgt / Math.sin(p);
      return { pos: eye, target: [eye[0] + d * Math.cos(p), eye[1], eye[2] - d * Math.sin(p)] };
    };
    return {
      driver: { pos: [c[0], c[1], c[2]], target: [c[0] + 6, c[1], c[2] - 6 * Math.tan(THREE.MathUtils.degToRad(3))] },   // orbit centre 6 m ahead
      chase: back(5.0, 13.0, -4.0),             // 5 m up, 13 deg down, 4 m ahead of the rig
      behind: back(4.0, 13.5, 6.0),             // 6 m behind the car
      high: back(7.0, 20.0, 9.0),
      overview: back(+(q.get('oh') || 15.0), +(q.get('op') || 30.0), +(q.get('ob') || 20.0)),   // raised view down the road (page default)
      top: { pos: [c[0] + 7.6, c[1], z0 + 55], target: [c[0] + 8, c[1], z0] }
    };
  }
  let dirty = true;
  controls.addEventListener('change', () => { dirty = true; });
  function setView(name) {
    state.view = name;
    if (!state.mesh) return;
    const v = presets(state.mesh.header)[name];
    if (!v) return;
    // per-scene zoom-out for the page's overview (same as scrolling back): eye moves away from the orbit centre
    const k = name === 'overview' ? (+(q.get('ok')) || ({ 75: 1.3, 101: 1.3 }[SCENE] || 1.0)) : 1.0;
    if (k !== 1.0) v.pos = v.pos.map((p, i) => v.target[i] + (p - v.target[i]) * k);
    camera.position.set(...v.pos); controls.target.set(...v.target); controls.update(); dirty = true;
  }
  function setBackground(dark) { stage.classList.toggle('is-dark', !!dark); }

  // hint hand: hidden after the first interaction
  const hand = document.getElementById('hand');
  const hideHand = () => { if (hand) hand.classList.add('is-hidden'); };
  renderer.domElement.addEventListener('pointerdown', hideHand, { once: true });
  renderer.domElement.addEventListener('wheel', hideHand, { once: true, passive: true });
  const fsBtn = document.getElementById('fs');
  if (fsBtn) {
    if (!document.fullscreenEnabled) fsBtn.style.display = 'none';
    fsBtn.addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen(); else stage.requestFullscreen().catch(() => {});
    });
  }

  window.addEventListener('message', (ev) => {
    const m = ev.data || {};
    if (m.type === 'view') setView(m.view);
    else if (m.type === 'reg') { REG.on = !!m.on; if (m.size !== undefined) REG.size = +m.size; dirty = true; }
    else if (m.type === 'bg') setBackground(m.dark);
  });

  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.fov = 2 * THREE.MathUtils.radToDeg(Math.atan(Math.tan(THREE.MathUtils.degToRad(HFOV / 2)) / camera.aspect));
    camera.updateProjectionMatrix();
    if (state.mesh) state.mesh.lastView = null;
    dirty = true;
  }
  new ResizeObserver(resize).observe(stage);
  resize();

  function loop() {
    requestAnimationFrame(loop);
    const moved = controls.update();
    if (!(dirty || moved)) return;
    dirty = false;
    const c = camera.position;
    egoBox.visible = !(c.x > -1.2 && c.x < 3.8 && Math.abs(c.y) < 1.15 && c.z < egoBox.position.z + 1.0);
    if (state.mesh) {
      const u = state.mesh.material.uniforms;
      u.uMinRatio.value = REG.on ? REG.ratio : 0.0;
      u.uSize.value = REG.on ? REG.size : 1.0;
      state.mesh.update(camera, renderer);
    }
    renderer.render(scene, camera);
  }

  const name = `s${String(SCENE).padStart(3, '0')}_f${String(FRAME).padStart(3, '0')}`;
  function loadScript(n) {
    return new Promise((resolve, reject) => {
      const el = document.createElement('script');
      el.src = `scenes/${n}.js`;
      el.onload = () => { el.remove(); resolve(); };
      el.onerror = () => { el.remove(); reject(new Error(`missing scenes/${n}.js`)); };
      document.head.appendChild(el);
    });
  }
  (async function () {
    try {
      setStatus('Loading ...');
      await loadScript(name);
      await new Promise((r) => setTimeout(r, 0));
      const data = THREE.decodePackedSplats(window.PANDWM_SPLATS[name]);
      delete window.PANDWM_SPLATS[name];
      const mesh = new THREE.SplatMesh(data);
      state.mesh = mesh; scene.add(mesh);
      egoBox.position.set(1.3, 0, mesh.header.road_z + 0.8);
      setView(state.view);
      setStatus(`${mesh.n.toLocaleString('en-US')} Gaussians`);
      statusEl.classList.add('is-quiet');
      loop();
      post({ type: 'loaded', ok: true, n: mesh.n });
    } catch (e) {
      setStatus(String(e.message || e));
      post({ type: 'loaded', ok: false });
    }
  })();
  window.addEventListener('keydown', (ev) => {
    if (ev.key !== 'c' || !state.mesh) return;
    const r = (v) => v.toArray().map((n) => +n.toFixed(2));
    const s = JSON.stringify({ scene: SCENE, pos: r(camera.position), target: r(controls.target) });
    console.log(s); try { navigator.clipboard.writeText(s); } catch (e) { /* ignore */ }
    setStatus('Camera copied: ' + s);
  });
  window.PANDWM_VIEWER = { setView, REG, state, camera, controls, renderer };
})();
