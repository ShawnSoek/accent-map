/* AudioWorklet: collects microphone samples into ~100 ms mono chunks for the page to send.
   Runs in the audio thread; `sampleRate` is a global there. */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    var opts = (options && options.processorOptions) || {};
    this.size = Math.max(128, Math.round(sampleRate * (opts.chunkSeconds || 0.1)));
    this.buffer = new Float32Array(this.size);
    this.filled = 0;
  }

  process(inputs) {
    var channels = inputs[0];
    if (!channels || channels.length === 0) return true;
    var n = channels[0].length;
    for (var i = 0; i < n; i++) {
      var sum = 0;
      for (var c = 0; c < channels.length; c++) sum += channels[c][i];
      this.buffer[this.filled++] = sum / channels.length;   // downmix to mono
      if (this.filled === this.size) this.flush();
    }
    return true;
  }

  flush() {
    var chunk = this.buffer;
    var energy = 0;
    for (var i = 0; i < chunk.length; i++) energy += chunk[i] * chunk[i];
    this.port.postMessage({ samples: chunk, rms: Math.sqrt(energy / chunk.length) }, [chunk.buffer]);
    this.buffer = new Float32Array(this.size);
    this.filled = 0;
  }
}

registerProcessor("capture", CaptureProcessor);
