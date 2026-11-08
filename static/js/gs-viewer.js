/* Single interactive 3D Gaussian viewer under the comparison videos. It follows the scene picker: whenever the
   selected scene changes (project.js dispatches 'pandwm:scene'), the viewer for that scene is loaded in place
   (static/pointcloud/embed.html, 60-80 MB per scene). Only one scene is live at a time. Loading starts once the
   viewer is near the viewport. */
'use strict';
(() => {
  const frame = document.getElementById('gs-frame');
  if (!frame) return;
  const preview = document.getElementById('gs-preview');
  const loading = document.getElementById('gs-loading');
  const pad = n => String(n).padStart(3, '0');
  let current = null;      // scene number wanted
  let loaded = null;       // scene number in the iframe
  let iframe = null;
  let visible = !('IntersectionObserver' in window);

  function unload() {
    if (!iframe) return;
    iframe.src = 'about:blank';
    iframe.remove();
    iframe = null;
    loaded = null;
  }

  function show(scene) {
    current = scene;
    frame.classList.remove('is-live');
    preview.src = `static/pointcloud/previews/s${pad(scene)}_overview.jpg`;
    preview.alt = `Scene ${scene}: driver view of the Gaussian scene`;
    loading.textContent = 'Loading 3D scene ...';
    if (visible) load();
  }

  function load() {
    if (current === null || loaded === current) return;
    unload();
    iframe = document.createElement('iframe');
    iframe.title = `Interactive Gaussian scene ${current}`;
    iframe.setAttribute('allowfullscreen', '');
    iframe.src = `static/pointcloud/embed.html?scene=${current}&view=overview`;
    frame.appendChild(iframe);
    frame.classList.add('is-loading');
    loaded = current;
  }

  window.addEventListener('message', ev => {
    const m = ev.data || {};
    if (m.type !== 'loaded' || !iframe || ev.source !== iframe.contentWindow) return;
    frame.classList.remove('is-loading');
    if (m.ok) frame.classList.add('is-live');
    else loading.textContent = 'The 3D scene could not be loaded (WebGL 2 required).';
  });

  document.addEventListener('pandwm:scene', ev => show(ev.detail.scene));
  // project.js may already have selected the first scene before this script ran.
  const first = (window.PANDWM_SCENES || [])[0];
  if (first && current === null) show(first.scene);

  if (!visible) {
    new IntersectionObserver(entries => {
      if (!entries.some(e => e.isIntersecting)) return;
      visible = true;
      load();
    }, { rootMargin: '200px 0px' }).observe(frame);
  }
})();
