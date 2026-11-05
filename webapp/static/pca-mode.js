/**
 * PCA exploration mode: hear the artwork's reduced feature space.
 *
 * Author: Anjali Kolapalli
 * Method after Dalziel & Lieck, "Sonification of High-Dimensional Data Using
 * Harmonic Overtone Amplitudes" (ICAD 2025).
 *
 * The server reduces the artwork to a grid of patches, each described by seven
 * features -- mean R, G, B, the standard deviation of R, G, B, and a Laplacian
 * edge-density score -- standardized and projected onto five principal
 * components. This module draws that grid, tints it by PC1, and plays a patch's
 * overtone series when the pointer passes over it.
 *
 * Deliberately self-contained: it owns its own canvas overlay and audio graph
 * so it does not reach into the pointer or scan engines.
 */

const PCA = (() => {
  const COOLWARM_COLD = [59, 76, 192];
  const COOLWARM_MID = [221, 221, 221];
  const COOLWARM_WARM = [180, 4, 38];
  const TONE_SECONDS = 0.15;

  const state = {
    ready: false,
    active: false,
    data: null,
    overlay: null,
    context: null,
    node: null,
    master: null,
    hovered: -1,
    mapping: "pow2",
    gainScale: 0.9,
    muted: false,
  };

  let shell = null;
  let artCanvas = null;
  let readout = null;

  /** Diverging colour ramp, matching the notebook's "coolwarm" figures. */
  function coolwarm(t) {
    const clamped = Math.max(0, Math.min(1, t));
    const [from, to, mix] = clamped < 0.5
      ? [COOLWARM_COLD, COOLWARM_MID, clamped * 2]
      : [COOLWARM_MID, COOLWARM_WARM, (clamped - 0.5) * 2];
    return from.map((value, index) => Math.round(value + (to[index] - value) * mix));
  }

  function isAvailable() {
    return Boolean(state.data && state.data.available !== false && state.data.patches?.length);
  }

  /**
   * Accept the `pca` block from a manifest. Precomputes the global
   * normalization so one patch's loudness is meaningful relative to the rest
   * of this artwork, which is what the notebook's global scaling achieved.
   */
  function load(pca) {
    state.hovered = -1;
    if (!pca || pca.available === false || !Array.isArray(pca.patches) || !pca.patches.length) {
      state.data = null;
      state.ready = false;
      return false;
    }
    const absMax = Math.max(1e-6, Number(pca.pc_abs_max) || 1);
    const mapped = pca.patches.map((patch) => {
      const total = patch.pc.reduce((sum, value) => sum + Math.pow(2, Math.abs(value)), 0);
      return total;
    });
    const loudest = Math.max(...mapped, 1e-6);
    state.data = {
      ...pca,
      pc1Range: absMax,
      gainScale: 0.9 / loudest,
    };
    state.gainScale = state.data.gainScale;
    state.ready = true;
    configure();
    return true;
  }

  function attach(shellElement, canvasElement, readoutElement) {
    shell = shellElement;
    artCanvas = canvasElement;
    readout = readoutElement;
    if (!shell || state.overlay) return;
    const overlay = document.createElement("canvas");
    overlay.id = "pcaOverlay";
    overlay.className = "pca-overlay";
    overlay.hidden = true;
    shell.appendChild(overlay);
    state.overlay = overlay;
  }

  /** Map the patch grid, which is in source-image pixels, onto the canvas. */
  function layout() {
    if (!state.overlay || !artCanvas || !state.data) return null;
    const rect = artCanvas.getBoundingClientRect();
    const shellRect = shell.getBoundingClientRect();
    const image = state.data.geometry.image_size;
    const ratio = window.devicePixelRatio || 1;
    if (state.overlay.width !== Math.round(rect.width * ratio)
      || state.overlay.height !== Math.round(rect.height * ratio)) {
      state.overlay.width = Math.round(rect.width * ratio);
      state.overlay.height = Math.round(rect.height * ratio);
    }
    state.overlay.style.width = `${rect.width}px`;
    state.overlay.style.height = `${rect.height}px`;
    state.overlay.style.left = `${rect.left - shellRect.left}px`;
    state.overlay.style.top = `${rect.top - shellRect.top}px`;
    return {
      ratio,
      scaleX: rect.width / image.width,
      scaleY: rect.height / image.height,
      width: rect.width,
      height: rect.height,
    };
  }

  function draw() {
    if (!state.active || !isAvailable()) return;
    const box = layout();
    if (!box) return;
    const ctx = state.overlay.getContext("2d");
    const { patch_size: size } = state.data.geometry;
    ctx.setTransform(box.ratio, 0, 0, box.ratio, 0, 0);
    ctx.clearRect(0, 0, box.width, box.height);

    const range = Math.max(1e-6, state.data.pc1Range);
    state.data.patches.forEach((patch, index) => {
      const x = patch.x * box.scaleX;
      const y = patch.y * box.scaleY;
      const w = size * box.scaleX;
      const h = size * box.scaleY;
      const [r, g, b] = coolwarm((patch.pc[0] / range + 1) / 2);
      const hovered = index === state.hovered;
      // The tint has to read clearly over a busy mural, so it sits well above
      // half opacity; the hovered patch becomes near-opaque with a dark border.
      ctx.fillStyle = `rgba(${r},${g},${b},${hovered ? 0.92 : 0.62})`;
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = hovered ? "rgba(23,24,22,.95)" : "rgba(23,24,22,.22)";
      ctx.lineWidth = hovered ? 3 : 0.75;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    });
  }

  function patchAt(clientX, clientY) {
    if (!isAvailable() || !artCanvas) return -1;
    const rect = artCanvas.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
      return -1;
    }
    const image = state.data.geometry.image_size;
    const imageX = (clientX - rect.left) / rect.width * image.width;
    const imageY = (clientY - rect.top) / rect.height * image.height;
    const { patch_size: size } = state.data.geometry;
    return state.data.patches.findIndex((patch) => (
      imageX >= patch.x && imageX < patch.x + size
      && imageY >= patch.y && imageY < patch.y + size
    ));
  }

  /** Browsers refuse to start audio without a user gesture, so a suspended
   *  context is resumed on the next click anywhere on the page. */
  function armGestureResume() {
    if (state.gestureArmed) return;
    state.gestureArmed = true;
    const resume = async () => {
      try { await state.context?.resume(); } catch (_) { /* nothing to resume */ }
      if (state.context?.state === "running") {
        document.removeEventListener("pointerdown", resume);
        state.gestureArmed = false;
        describe(state.hovered);
      }
    };
    document.addEventListener("pointerdown", resume);
  }

  async function ensureAudio() {
    if (state.node) return true;
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      state.context = state.context || new AudioContextClass();
      if (state.context.state === "suspended") {
        await state.context.resume().catch(() => {});
        if (state.context.state !== "running") armGestureResume();
      }
      await state.context.audioWorklet.addModule("/static/pca-worklet.js?v=1");
      state.node = new AudioWorkletNode(state.context, "pca-additive", { outputChannelCount: [2] });
      state.master = state.context.createGain();
      state.master.gain.value = 0.5;
      state.node.connect(state.master).connect(state.context.destination);
      configure();
      return true;
    } catch (error) {
      console.warn("PCA audio unavailable:", error);
      return false;
    }
  }

  function configure() {
    state.node?.port.postMessage({
      type: "configure",
      mapping: state.mapping,
      gainScale: state.gainScale,
    });
  }

  const AMPLITUDE_FN = {
    pow10: (z) => Math.pow(10, Math.abs(z)),
    pow2: (z) => Math.pow(2, Math.abs(z)),
    linear: (z) => Math.abs(z),
    offset: (z) => 1 + Math.abs(z),
    clipped: (z) => Math.pow(10, Math.min(Math.abs(z), 2)),
  };

  /** Harmonic amplitudes for a patch under the active mapping, scaled so the
   *  loudest partial reads 1. This is what the ear actually receives. */
  function amplitudesFor(patch) {
    const fn = AMPLITUDE_FN[state.mapping] || AMPLITUDE_FN.pow2;
    const raw = patch.pc.map((value) => fn(value));
    const peak = Math.max(...raw, 1e-9);
    return raw.map((value) => value / peak);
  }

  function notifyPatch(index) {
    if (typeof state.listener !== "function") return;
    const patch = index >= 0 ? state.data?.patches[index] : null;
    state.listener(patch ? {
      row: patch.row,
      col: patch.col,
      pc: patch.pc,
      amplitudes: amplitudesFor(patch),
      baseFrequency: state.data.audio?.base_frequency || 220,
    } : null);
  }

  function setPatchListener(fn) {
    state.listener = fn;
  }

  function play(index) {
    if (state.muted || !state.node || !isAvailable()) return;
    const patch = state.data.patches[index];
    if (!patch) return;
    state.node.port.postMessage({
      type: "note",
      pc: patch.pc,
      duration: TONE_SECONDS,
      sustain: true,
    });
  }

  function describe(index) {
    if (!readout) return;
    const patch = index >= 0 ? state.data?.patches[index] : null;
    if (!patch) {
      readout.innerHTML = '<p class="pca-hint">Move across the grid to hear each patch.</p>';
      return;
    }
    const range = Math.max(1e-6, state.data.pc1Range);
    const variance = state.data.explained_variance_ratio || [];
    const bars = patch.pc.map((value, index2) => {
      const width = Math.min(100, Math.abs(value) / range * 100);
      const share = variance[index2] != null ? `${(variance[index2] * 100).toFixed(1)}%` : "";
      return `<div class="pca-bar-row">
          <span class="pca-bar-label">PC${index2 + 1}</span>
          <span class="pca-bar-track"><span class="pca-bar-fill${value < 0 ? " negative" : ""}" style="width:${width}%"></span></span>
          <span class="pca-bar-value">${value >= 0 ? "+" : ""}${value.toFixed(2)}</span>
          <span class="pca-bar-share">${share}</span>
        </div>`;
    }).join("");
    readout.innerHTML = `<div class="pca-patch-id">patch row ${patch.row}, column ${patch.col}</div>${bars}`;
  }

  function onPointerMove(event) {
    if (!state.active) return;
    const index = patchAt(event.clientX, event.clientY);
    if (index === state.hovered) return;
    state.hovered = index;
    if (index >= 0) play(index);
    else state.node?.port.postMessage({ type: "release" });
    describe(index);
    notifyPatch(index);
    draw();
  }

  function onPointerLeave() {
    if (!state.active) return;
    state.hovered = -1;
    state.node?.port.postMessage({ type: "release" });
    describe(-1);
    notifyPatch(-1);
    draw();
  }

  async function setActive(active) {
    state.active = Boolean(active) && isAvailable();
    if (state.overlay) state.overlay.hidden = !state.active;
    if (state.active) {
      await ensureAudio();
      describe(-1);
      draw();
    } else {
      state.hovered = -1;
      state.node?.port.postMessage({ type: "clear" });
    }
    return state.active;
  }

  function setMapping(mapping) {
    state.mapping = mapping;
    configure();
  }

  function setMuted(muted) {
    state.muted = Boolean(muted);
    if (state.muted) state.node?.port.postMessage({ type: "release" });
  }

  function summary() {
    if (!isAvailable()) return null;
    const geometry = state.data.geometry;
    return {
      patches: state.data.n_patches,
      cols: geometry.cols,
      rows: geometry.rows,
      patchSize: geometry.patch_size,
      variance: state.data.explained_variance_ratio,
      total: state.data.total_explained,
    };
  }

  window.addEventListener("resize", () => { if (state.active) draw(); });

  function audioState() {
    return {
      context: state.context?.state || "none",
      node: Boolean(state.node),
      mapping: state.mapping,
      gainScale: state.gainScale,
      hovered: state.hovered,
    };
  }

  return {
    load, attach, setActive, setMapping, setMuted, draw, setPatchListener,
    isAvailable, summary, audioState, onPointerMove, onPointerLeave,
  };
})();

window.PCA = PCA;
