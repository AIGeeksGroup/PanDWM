'use strict';
(() => {
  const videos = [...document.querySelectorAll('.comparison-view video')];
  if (!videos.length) return;

  const zoom = 4;
  const lens = document.createElement('div');
  lens.className = 'video-magnifier';
  lens.hidden = true;
  lens.setAttribute('aria-hidden', 'true');
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { alpha: false });
  if (!context) return;
  const badge = document.createElement('span');
  badge.className = 'video-magnifier-scale';
  badge.textContent = `${zoom}×`;
  lens.append(canvas, badge);
  document.body.append(lens);

  let video = null;
  let pointerX = 0;
  let pointerY = 0;
  let animationFrame = 0;

  function hide() {
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    video = null;
    lens.hidden = true;
  }

  function render() {
    animationFrame = 0;
    if (!video || document.hidden) { hide(); return; }
    const rect = video.getBoundingClientRect();
    if (!rect.width || !rect.height || pointerX < rect.left || pointerX > rect.right ||
        pointerY < rect.top || pointerY > rect.bottom) {
      hide();
      return;
    }
    if (video.readyState < 2 || !video.videoWidth) {
      lens.hidden = true;
      animationFrame = requestAnimationFrame(render);
      return;
    }

    const margin = 12;
    const width = Math.min(320, innerWidth - margin * 2, rect.width * zoom);
    const height = Math.min(180, innerHeight - margin * 2, rect.height * zoom);
    const pixelRatio = Math.min(devicePixelRatio || 1, 2);
    const bufferWidth = Math.round(width * pixelRatio);
    const bufferHeight = Math.round(height * pixelRatio);
    if (canvas.width !== bufferWidth || canvas.height !== bufferHeight) {
      canvas.width = bufferWidth;
      canvas.height = bufferHeight;
    }
    lens.style.width = `${width}px`;
    lens.style.height = `${height}px`;
    let left = pointerX + 20;
    if (left + width > innerWidth - margin) left = pointerX - width - 20;
    let top = pointerY - height - 20;
    if (top < margin) top = pointerY + 20;
    lens.style.left = `${Math.max(margin, Math.min(innerWidth - width - margin, left))}px`;
    lens.style.top = `${Math.max(margin, Math.min(innerHeight - height - margin, top))}px`;

    const sourceWidth = width * video.videoWidth / (rect.width * zoom);
    const sourceHeight = height * video.videoHeight / (rect.height * zoom);
    const centerX = (pointerX - rect.left) / rect.width * video.videoWidth;
    const centerY = (pointerY - rect.top) / rect.height * video.videoHeight;
    const sourceX = Math.max(0, Math.min(video.videoWidth - sourceWidth, centerX - sourceWidth / 2));
    const sourceY = Math.max(0, Math.min(video.videoHeight - sourceHeight, centerY - sourceHeight / 2));
    // Sample the displayed video's current frame; no second player or download.
    context.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, bufferWidth, bufferHeight);
    lens.hidden = false;
    animationFrame = requestAnimationFrame(render);
  }

  function follow(event) {
    if (event.pointerType === 'touch') return;
    video = event.currentTarget;
    pointerX = event.clientX;
    pointerY = event.clientY;
    if (!animationFrame) render();
  }

  videos.forEach(item => {
    item.addEventListener('pointerenter', follow);
    item.addEventListener('pointermove', follow);
    item.addEventListener('pointerleave', hide);
    item.addEventListener('pointercancel', hide);
    item.addEventListener('emptied', () => { if (video === item) hide(); });
    item.addEventListener('error', () => { if (video === item) hide(); });
  });
  window.addEventListener('scroll', hide, { passive: true, capture: true });
  window.addEventListener('resize', hide);
  window.addEventListener('blur', hide);
  document.addEventListener('visibilitychange', () => { if (document.hidden) hide(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') hide(); });
})();
