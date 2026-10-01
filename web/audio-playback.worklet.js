class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.cap = 16000 * 2;
    this.buf = new Float32Array(this.cap);
    this.read = 0;
    this.write = 0;
    this.size = 0;
    this.pos = 0;
    this.step = 16000 / sampleRate;
    this.playing = false;
    this.idlePulls = 0;
    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.cmd === "cut") {
        this.read = 0;
        this.write = 0;
        this.size = 0;
        this.pos = 0;
        this.idlePulls = 0;
        this._setPlaying(false);
        return;
      }
      if (data.cmd !== "pcm" || !data.samples || !data.samples.length) return;
      const incoming = data.samples;
      for (let i = 0; i < incoming.length; i += 1) {
        if (this.size === this.cap) {
          this.read = (this.read + 1) % this.cap;
          this.size -= 1;
        }
        this.buf[this.write] = incoming[i];
        this.write = (this.write + 1) % this.cap;
        this.size += 1;
      }
      this.idlePulls = 0;
      this._setPlaying(true);
    };
  }

  _at(offset) {
    return this.buf[(this.read + offset) % this.cap];
  }

  _setPlaying(on) {
    if (this.playing === on) return;
    this.playing = on;
    this.port.postMessage({ type: "state", playing: on });
  }

  process(_inputs, outputs) {
    const out = outputs[0] && outputs[0][0];
    if (!out) return true;
    if (this.size < 2) {
      out.fill(0);
      this.idlePulls += 1;
      if (this.idlePulls > 30) this._setPlaying(false);
      return true;
    }
    this.idlePulls = 0;
    this._setPlaying(true);
    for (let i = 0; i < out.length; i += 1) {
      const idx = Math.floor(this.pos);
      if (idx >= this.size) {
        out[i] = 0;
        continue;
      }
      const frac = this.pos - idx;
      const a = this._at(idx);
      const next = Math.min(idx + 1, this.size - 1);
      const b = this._at(next);
      out[i] = a + (b - a) * frac;
      this.pos += this.step;
    }
    const drop = Math.floor(this.pos);
    if (drop > 0) {
      const n = Math.min(drop, this.size);
      this.read = (this.read + n) % this.cap;
      this.size -= n;
      this.pos -= n;
    }
    return true;
  }
}

registerProcessor("duplex-playback", PlaybackProcessor);
