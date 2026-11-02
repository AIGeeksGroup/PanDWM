'use strict';
(() => {
  const scenes = window.PANDWM_SCENES || [];
  if (!scenes.length) return;
  const $ = id => document.getElementById(id);
  const videos = [...document.querySelectorAll('#comparison-shell video')];
  const master = videos[0];
  const picker = $('scene-picker');
  const status = $('comparison-status');
  const playButton = $('play-comparison');
  const restartButton = $('restart-comparison');
  const progress = $('comparison-progress');
  const clock = $('comparison-time');
  const controls = [playButton, restartButton, progress];
  let index = 0;
  let duration = scenes[0].duration;
  let loadToken = 0;
  let playbackToken = 0;
  let controller = null;
  let objectUrls = new Set();
  let ready = false;
  let playing = false;
  let starting = false;
  let scrubbing = false;
  let buffering = false;
  let visible = !('IntersectionObserver' in window);
  let wantsPlayback = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let animationFrame = 0;

  function showStatus(message = '') {
    status.hidden = !message;
    status.textContent = message;
  }

  function updateTime(time = master.currentTime || 0) {
    const bounded = Math.max(0, Math.min(duration, time));
    if (!scrubbing) progress.value = bounded;
    clock.textContent = `${bounded.toFixed(1)} / ${duration.toFixed(1)} s`;
    progress.setAttribute('aria-valuetext', `${bounded.toFixed(1)} of ${duration.toFixed(1)} seconds`);
  }

  function pauseAll(changeIntent = true) {
    playbackToken++;
    if (changeIntent) wantsPlayback = false;
    playing = false;
    starting = false;
    cancelAnimationFrame(animationFrame);
    videos.forEach(video => video.pause());
    playButton.textContent = 'Play all';
  }

  function alignTo(time, force = false) {
    videos.forEach(video => {
      if (force || (!video.seeking && Math.abs(video.currentTime - time) > .08)) {
        video.currentTime = time;
      }
    });
    updateTime(time);
  }

  async function playAll() {
    wantsPlayback = true;
    if (!ready || starting || playing || scrubbing || document.hidden) return;
    starting = true;
    const token = ++playbackToken;
    const time = master.currentTime >= duration - .02 ? 0 : master.currentTime;
    alignTo(time);
    try {
      await Promise.all(videos.map(video => video.play()));
      if (token !== playbackToken) return;
      starting = false;
      playing = true;
      buffering = false;
      playButton.textContent = 'Pause all';
      showStatus();
      animationFrame = requestAnimationFrame(tick);
    } catch (error) {
      if (token !== playbackToken) return;
      pauseAll();
      showStatus('Playback is paused. Press Play all to start the five videos together.');
    }
  }

  function tick() {
    if (!playing) return;
    if (master.ended || master.currentTime >= duration - .02) {
      advanceScene();
      return;
    }
    // GT is the common clock; keep drift below one 10 fps frame.
    if (!videos.some(video => video.seeking)) alignTo(master.currentTime);
    else updateTime();
    animationFrame = requestAnimationFrame(tick);
  }

  function advanceScene() {
    if (!ready || !playing || !wantsPlayback) return;
    selectScene(index + 1);
  }

  function restart() {
    if (!ready) return;
    pauseAll(false);
    alignTo(0, true);
    playAll();
  }

  function releaseMedia() {
    pauseAll(false);
    ready = false;
    buffering = false;
    scrubbing = false;
    controls.forEach(control => { control.disabled = true; });
    videos.forEach(video => {
      video.removeAttribute('src');
      video.load();
    });
    objectUrls.forEach(url => URL.revokeObjectURL(url));
    objectUrls = new Set();
  }

  function prepareVideo(video, source, signal) {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        video.removeEventListener('loadeddata', loaded);
        video.removeEventListener('error', failed);
        signal.removeEventListener('abort', aborted);
      };
      const loaded = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error(`Could not load ${video.dataset.kind}`)); };
      const aborted = () => { cleanup(); reject(new DOMException('Cancelled', 'AbortError')); };
      if (signal.aborted) { aborted(); return; }
      video.addEventListener('loadeddata', loaded, { once: true });
      video.addEventListener('error', failed, { once: true });
      signal.addEventListener('abort', aborted, { once: true });
      video.preload = 'auto';
      video.src = source;
      video.load();
    });
  }

  async function loadScene() {
    const token = ++loadToken;
    controller?.abort();
    controller = new AbortController();
    const { signal } = controller;
    const scene = scenes[index];
    releaseMedia();
    let loaded = 0;
    showStatus('Loading videos (0 / 5)…');
    try {
      await Promise.all(videos.map(async video => {
        let source = `${scene.directory}/${video.dataset.kind}.mp4`;
        if (location.protocol !== 'file:') {
          // Blob sources allow seeking on static hosts without HTTP Range support.
          const response = await fetch(source, { signal });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const blob = await response.blob();
          if (token !== loadToken || signal.aborted) return;
          source = URL.createObjectURL(blob);
          objectUrls.add(source);
        }
        await prepareVideo(video, source, signal);
        if (token === loadToken && !signal.aborted) showStatus(`Loading videos (${++loaded} / 5)…`);
      }));
      if (token !== loadToken || signal.aborted) return;
      duration = Math.min(scene.duration, ...videos.map(video => video.duration));
      progress.max = duration;
      ready = true;
      controls.forEach(control => { control.disabled = false; });
      alignTo(0, true);
      showStatus();
      if (wantsPlayback) playAll();
    } catch (error) {
      if (token !== loadToken || signal.aborted) return;
      controller.abort();
      releaseMedia();
      showStatus('The videos could not be loaded. Select the scene again to retry.');
      console.error('PanDWM video loading failed:', error);
    }
  }

  function selectScene(next) {
    loadToken++;
    controller?.abort();
    releaseMedia();
    index = (next + scenes.length) % scenes.length;
    const scene = scenes[index];
    duration = scene.duration;
    progress.max = duration;
    progress.value = 0;
    updateTime(0);
    picker.querySelectorAll('button').forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
    $('scene-title').textContent = scene.title;
    $('scene-setting').textContent = `Scene ${scene.scene}. Student setting: autoregressive history.`;
    $('scene-caption').textContent = scene.caption;
    videos.forEach(video => {
      video.poster = `${scene.posterPrefix}_${video.dataset.kind}.jpg`;
    });
    showStatus();
    if (visible) loadScene();
  }

  scenes.forEach((scene, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'scene-thumb';
    button.setAttribute('aria-label', `Scene ${scene.scene}: ${scene.title}`);
    const thumb = document.createElement('img');
    thumb.src = `${scene.posterPrefix}_student.jpg`;
    thumb.alt = '';
    thumb.loading = 'lazy';
    const caption = document.createElement('span');
    caption.textContent = `Scene ${scene.scene} · ${scene.title}`;
    button.append(thumb, caption);
    button.addEventListener('click', () => selectScene(i));
    picker.append(button);
  });

  $('previous-scene').addEventListener('click', () => selectScene(index - 1));
  $('next-scene').addEventListener('click', () => selectScene(index + 1));
  playButton.addEventListener('click', () => {
    if (playing || starting) pauseAll();
    else playAll();
  });
  restartButton.addEventListener('click', restart);
  progress.addEventListener('pointerdown', () => {
    scrubbing = true;
    pauseAll(false);
  });
  progress.addEventListener('input', () => {
    if (!ready) return;
    pauseAll(false);
    alignTo(Number(progress.value), true);
  });
  const finishScrubbing = () => {
    scrubbing = false;
    if (wantsPlayback) playAll();
  };
  progress.addEventListener('change', finishScrubbing);
  window.addEventListener('pointerup', () => { if (scrubbing) finishScrubbing(); });
  window.addEventListener('pointercancel', () => { if (scrubbing) finishScrubbing(); });
  videos.forEach(video => {
    video.addEventListener('click', () => {
      if (playing || starting) pauseAll();
      else playAll();
    });
    video.addEventListener('waiting', () => {
      if (!ready || !playing || video.seeking) return;
      buffering = true;
      pauseAll(false);
      showStatus('Synchronizing videos…');
    });
    video.addEventListener('canplay', () => {
      if (!buffering || !videos.every(item => item.readyState >= 3)) return;
      buffering = false;
      showStatus();
      if (wantsPlayback) playAll();
    });
  });
  master.addEventListener('ended', () => { if (master.ended) advanceScene(); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pauseAll(false);
    else if (wantsPlayback) playAll();
  });

  selectScene(0);
  if (!visible) {
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      observer.disconnect();
      visible = true;
      loadScene();
    }, { rootMargin: '300px 0px' });
    observer.observe($('comparison-shell'));
  }
})();
