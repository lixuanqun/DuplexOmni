class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0;
    this.buf = [];
    this.target = 320;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i += 1) {
      this.acc += 1;
      if (this.acc < this.ratio) continue;
      this.acc -= this.ratio;
      this.buf.push(channel[i]);
      if (this.buf.length < this.target) continue;
      const pcm = new Int16Array(this.buf.length);
      let energy = 0;
      for (let j = 0; j < this.buf.length; j += 1) {
        const sample = Math.max(-1, Math.min(1, this.buf[j]));
        pcm[j] = (sample * 32767) | 0;
        energy += sample * sample;
      }
      const rms = Math.sqrt(energy / this.buf.length);
      this.port.postMessage({ pcm, rms }, [pcm.buffer]);
      this.buf = [];
    }
    return true;
  }
}

registerProcessor("duplex-capture", CaptureProcessor);
