"""Drum events -> MIDI file, and a tiny synthesizer so you can LISTEN to results.
Only numpy is needed for the synth (pretty_midi is imported only when writing MIDI)."""
import numpy as np

from config import DRUM_SR
from drum_core import CLASS_TO_PITCH

_GAIN = [1.0, 0.7, 0.35, 0.35, 0.8, 0.4, 0.3]     # kick, snare, hh closed/open, tom, crash, ride


def events_to_midi(events, path, velocity=100, note_len=0.1):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for t, k in events:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=CLASS_TO_PITCH[int(k)],
                                           start=float(t), end=float(t) + note_len))
    pm.instruments.append(inst)
    pm.write(str(path))


def _noise_hp(rng, n):
    return np.diff(rng.standard_normal(n), prepend=0.0)      # crude high-pass noise


def _hit(kind, sr, rng, vary):
    def v(lo, hi):
        return rng.uniform(lo, hi) if vary else (lo + hi) / 2

    if kind == 0:                                            # kick: falling sine
        n = int(0.4 * sr); t = np.arange(n) / sr
        f = v(45, 60) + v(60, 110) * np.exp(-t * 30)
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * v(8, 14))
    if kind == 1:                                            # snare: noise + tone
        n = int(0.3 * sr); t = np.arange(n) / sr
        return (0.6 * _noise_hp(rng, n) * np.exp(-t * v(18, 28))
                + 0.5 * np.sin(2 * np.pi * v(170, 230) * t) * np.exp(-t * v(25, 40)))
    if kind == 2:                                            # closed hi-hat: short noise
        n = int(0.12 * sr); t = np.arange(n) / sr
        return _noise_hp(rng, n) * np.exp(-t * v(60, 90))
    if kind == 3:                                            # open hi-hat: long noise
        n = int(0.6 * sr); t = np.arange(n) / sr
        return _noise_hp(rng, n) * np.exp(-t * v(7, 12))
    if kind == 4:                                            # tom: sine with small drop
        n = int(0.5 * sr); t = np.arange(n) / sr
        f = v(90, 160) * (1 + 0.3 * np.exp(-t * 20))
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * v(8, 14))
    if kind == 5:                                            # crash: long bright noise
        n = int(1.2 * sr); t = np.arange(n) / sr
        return _noise_hp(rng, n) * np.exp(-t * v(3, 5))
    n = int(0.8 * sr); t = np.arange(n) / sr                 # ride: noise + bell tone
    return (_noise_hp(rng, n) * np.exp(-t * v(6, 9))
            + 0.4 * np.sin(2 * np.pi * v(2500, 3500) * t) * np.exp(-t * 8))


def render_drums(events, sr=DRUM_SR, length_seconds=None, tail=1.0, seed=0, vary=True):
    """events (N, 2) [time, class] -> float32 audio. Not realistic, but enough to listen to."""
    events = np.asarray(events, dtype=np.float64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    if length_seconds is None:
        length_seconds = (events[:, 0].max() if len(events) else 0.0) + tail
    audio = np.zeros(int(length_seconds * sr) + 1)
    for t, k in events:
        i = int(t * sr)
        if i >= len(audio):
            continue
        h = _hit(int(k), sr, rng, vary) * _GAIN[int(k)]
        if vary:
            h = h * rng.uniform(0.7, 1.0)
        seg = min(len(h), len(audio) - i)
        audio[i:i + seg] += h[:seg]
    peak = np.abs(audio).max()
    if peak > 0:
        audio = 0.9 * audio / peak
    return audio.astype(np.float32)
