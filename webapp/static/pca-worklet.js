/**
 * Additive overtone synthesis for PCA exploration mode.
 *
 * Author: Anjali Kolapalli
 * Method after Dalziel & Lieck, "Sonification of High-Dimensional Data Using
 * Harmonic Overtone Amplitudes" (ICAD 2025).
 *
 * One patch of the artwork becomes one tone. Each of the patch's principal
 * components sets the amplitude of one harmonic over a fixed 220 Hz
 * fundamental, so a patch that is unusual in the reduced feature space has an
 * unusual spectrum. This is the notebook's `patch_tone` rendered live rather
 * than concatenated into a WAV file.
 */

const BASE_FREQUENCY = 220.0;
const FADE_SECONDS = 0.01;
const MAX_VOICES = 6;

// The notebook measured each mapping's dynamic range over 195 patches.
// 10^|PC| put 185 of them below -40 dB, because a single flat background
// patch reaches |PC1| ~ 5.6 and is then ~442,000x louder than a typical
// patch; global normalization silences everything else. It is kept for
// comparison, but 2^|PC| is the default.
const MAPPINGS = {
  pow10: (z) => Math.pow(10, Math.abs(z)),
  pow2: (z) => Math.pow(2, Math.abs(z)),
  linear: (z) => Math.abs(z),
  offset: (z) => 1 + Math.abs(z),
  clipped: (z) => Math.pow(10, Math.min(Math.abs(z), 2)),
};

class PCAAdditiveProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.voices = [];
    this.mapping = "pow2";
    this.gainScale = 0.9;
    this.sustain = false;
    this.port.onmessage = (event) => this.handleMessage(event.data || {});
  }

  handleMessage(message) {
    if (message.type === "configure") {
      if (message.mapping && MAPPINGS[message.mapping]) this.mapping = message.mapping;
      if (Number.isFinite(message.gainScale)) this.gainScale = message.gainScale;
      if (typeof message.sustain === "boolean") this.sustain = message.sustain;
      return;
    }
    if (message.type === "note") {
      this.addVoice(message);
      return;
    }
    if (message.type === "release") {
      for (const voice of this.voices) voice.releasing = true;
      return;
    }
    if (message.type === "clear") {
      this.voices = [];
    }
  }

  addVoice(message) {
    const components = Array.isArray(message.pc) ? message.pc : [];
    if (!components.length) return;
    const map = MAPPINGS[message.mapping || this.mapping] || MAPPINGS.pow2;

    const amplitudes = components.map((value) => {
      const amplitude = map(Number(value) || 0);
      return Number.isFinite(amplitude) ? amplitude : 0;
    });
    const total = amplitudes.reduce((sum, value) => sum + value, 0);
    if (total <= 0) return;

    // Scale by the loudest patch in this artwork rather than by this patch,
    // so quiet patches stay quiet relative to busy ones. That global
    // normalization is what makes the latent space audible as contrast.
    const normalized = amplitudes.map((value) => value * this.gainScale);

    // A hover replaces the previous patch rather than stacking on it.
    if (this.voices.length >= MAX_VOICES) this.voices.shift();
    for (const voice of this.voices) voice.releasing = true;

    this.voices.push({
      amplitudes: normalized,
      phases: new Float64Array(normalized.length),
      age: 0,
      duration: Math.max(0.05, Number(message.duration) || 0.15),
      sustain: message.sustain ?? this.sustain,
      releasing: false,
      released: 0,
    });
  }

  /** Linear fade in and out, matching the notebook's 10 ms envelope. */
  envelope(voice) {
    const fade = FADE_SECONDS;
    if (voice.releasing) {
      const remaining = 1 - voice.released / fade;
      return Math.max(0, remaining);
    }
    if (voice.age < fade) return voice.age / fade;
    if (!voice.sustain) {
      const end = voice.duration;
      if (voice.age > end - fade) return Math.max(0, (end - voice.age) / fade);
      if (voice.age >= end) return 0;
    }
    return 1;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || !output.length) return true;
    const left = output[0];
    const right = output.length > 1 ? output[1] : null;
    const step = 1 / sampleRate;
    left.fill(0);
    if (right) right.fill(0);

    for (const voice of this.voices) {
      for (let frame = 0; frame < left.length; frame++) {
        const gain = this.envelope(voice);
        if (gain > 0) {
          let sample = 0;
          for (let partial = 0; partial < voice.amplitudes.length; partial++) {
            const frequency = BASE_FREQUENCY * (partial + 1);
            voice.phases[partial] += 2 * Math.PI * frequency * step;
            if (voice.phases[partial] > 2 * Math.PI) voice.phases[partial] -= 2 * Math.PI;
            sample += voice.amplitudes[partial] * Math.sin(voice.phases[partial]);
          }
          const value = sample * gain;
          left[frame] += value;
          if (right) right[frame] += value;
        }
        voice.age += step;
        if (voice.releasing) voice.released += step;
      }
    }

    this.voices = this.voices.filter((voice) => {
      if (voice.releasing) return voice.released < FADE_SECONDS;
      return voice.sustain || voice.age < voice.duration;
    });

    // A soft limiter keeps overlapping patch tones from clipping while the
    // pointer sweeps quickly across the grid.
    for (let frame = 0; frame < left.length; frame++) {
      left[frame] = Math.tanh(left[frame]);
      if (right) right[frame] = left[frame];
    }
    return true;
  }
}

registerProcessor("pca-additive", PCAAdditiveProcessor);
