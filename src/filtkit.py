"""FiltKit engine: audio in -> stems, MIDI and sheet music out.

Command line:
  python src/filtkit.py song.mp3 --mode song --out output/jobs/demo                    full song
  python src/filtkit.py song.mp3 --mode song --instruments piano drums voice           only some instruments
  python src/filtkit.py take.wav --mode instrument --instruments piano                 one instrument, no Demucs

From Python (this is what the web backend calls):
  from filtkit import FiltKit
  engine = FiltKit()                         # reads models.json; models load the first time they are needed
  result = engine.analyze("uploads/abc.mp3", "jobs/abc", mode="song", instruments=["piano", "drums"],
                          on_progress=lambda step, fraction: print(step, fraction))
  # `result` is a dict and is also saved as jobs/abc/result.json

Modes
  song        Demucs splits the recording into stems, then each instrument's own model transcribes its stem.
  instrument  the upload is ONE instrument: only that instrument's model runs (no Demucs).

Output folder:  stems/...  midi/<instrument>.mid  midi/all_parts.mid  score.musicxml  result.json
The trained models and their thresholds are listed in models.json (project root). An instrument whose
weights file is missing is skipped with a warning, so the engine works with whatever is trained so far.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from config import ROOT

REGISTRY_DEFAULT = ROOT / "models.json"
INSTRUMENTS = ("piano", "guitar", "voice", "drums")
STEM_FOR = {"piano": "piano", "guitar": "guitar", "voice": "vocals", "drums": "drums"}   # Demucs stem names
MIDI_PROGRAM = {"piano": 0, "guitar": 25, "voice": 53}


class FiltKitError(Exception):
    """A problem the user can understand (bad file, missing model, ...). The message is safe to show."""


# ---------------------------------------------------------------- MIDI output
def write_midi(instrument, notes, path):
    if instrument == "drums":
        from drum_midi import events_to_midi
        events_to_midi(notes, path)
    else:
        from guitar_core import notes_to_guitar_midi
        notes_to_guitar_midi(notes, path, program=MIDI_PROGRAM[instrument])


def write_combined_midi(parts, path):
    """parts: [(instrument, notes)] -> one MIDI file with one track per instrument."""
    import pretty_midi
    from drum_core import CLASS_TO_PITCH
    pm = pretty_midi.PrettyMIDI()
    for kind, notes in parts:
        if kind == "drums":
            inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
            for t, k in notes:
                inst.notes.append(pretty_midi.Note(100, CLASS_TO_PITCH[int(k)], float(t), float(t) + 0.1))
        else:
            inst = pretty_midi.Instrument(program=MIDI_PROGRAM[kind], name=kind.capitalize())
            for s, e, p in notes:
                inst.notes.append(pretty_midi.Note(80, int(p), float(s), float(e)))
        pm.instruments.append(inst)
    pm.write(str(path))


# ---------------------------------------------------------------- the engine
class FiltKit:
    def __init__(self, registry=REGISTRY_DEFAULT, device=None):
        registry = Path(registry)
        if not registry.exists():
            raise FiltKitError(f"model registry not found: {registry}")
        self.registry = json.loads(registry.read_text(encoding="utf-8"))
        self.device = device
        self._models = {}
        self._separators = {}

    # -- helpers
    def _device(self):
        if not self.device:
            from guitar_evaluate import pick_device
            self.device = pick_device()
        return self.device

    def _weights(self, instrument):
        cfg = self.registry.get(instrument)
        if not cfg:
            return None
        w = Path(cfg["weights"])
        return w if w.is_absolute() else ROOT / w

    def available_instruments(self):
        """Instruments that have a trained weights file."""
        return [i for i in INSTRUMENTS if self._weights(i) is not None and self._weights(i).exists()]

    def _model(self, instrument):
        if instrument not in self._models:
            w = self._weights(instrument)
            if w is None or not w.exists():
                raise FiltKitError(f"no trained {instrument} model found (expected {w}); skipping {instrument}")
            from guitar_evaluate import load_weights
            if instrument == "drums":
                from drum_model import DrumCRNN as Net
            else:
                from model import PianoCNN as Net
            model = Net().to(self._device())
            load_weights(model, str(w), self._device())
            model.eval()
            self._models[instrument] = model
        return self._models[instrument]

    def _separate(self, audio_path, out_dir, model):
        """Demucs: file -> {stem name: wav path}. The Demucs model is loaded once and reused."""
        try:
            import demucs.api
            from separate import separate_file
        except ImportError as e:
            raise FiltKitError("Demucs is not installed. Run:  pip install -U demucs") from e
        if model not in self._separators:
            self._separators[model] = demucs.api.Separator(model=model, device=self._device(), progress=False)
        return separate_file(audio_path, out_dir, model, separator=self._separators[model])

    # -- one instrument: audio file -> notes
    def transcribe(self, instrument, audio_path):
        """Returns an (N, 3) array [start_sec, end_sec, midi_pitch], or for drums an (N, 2) array [time_sec, class]."""
        cfg, model, dev = self.registry[instrument], self._model(instrument), self._device()
        if instrument == "drums":
            from drum_core import pick_peaks
            from drum_evaluate import predict_full
            from drum_features import compute_logmel, load_audio
            prob = predict_full(model, compute_logmel(load_audio(str(audio_path))), dev)
            return pick_peaks(prob, cfg.get("thr", 0.5), cfg.get("window", 2))
        from features import compute_cqt, load_audio
        from guitar_evaluate import predict_full
        fp, op = predict_full(model, compute_cqt(load_audio(str(audio_path))), dev)
        ft, ot = cfg.get("frame_thr", 0.5), cfg.get("onset_thr", 0.5)
        if instrument == "voice":
            from voice_core import mono_decode
            notes = mono_decode(fp, op, ft, ot)
        elif instrument == "guitar":
            from guitar_core import decode_notes
            notes = decode_notes(fp, op, ft, ot)
        else:
            from guitar_core import decode_notes
            notes = decode_notes(fp, op, ft, ot, lo=21, hi=108)             # whole piano range
        return np.array(notes, dtype=np.float64).reshape(-1, 3)

    # -- the whole job
    def analyze(self, audio_path, out_dir, mode="song", instruments=None, separation=None, bpm=None,
                time_sig=(4, 4), title=None, stem_map=None, on_progress=None):
        """See the module docstring. Returns the result dict (also written to <out_dir>/result.json)."""
        from score import make_musicxml
        t_start = time.time()
        audio_path, out = Path(audio_path), Path(out_dir)
        if not audio_path.exists():
            raise FiltKitError(f"audio file not found: {audio_path}")
        if mode not in ("song", "instrument"):
            raise FiltKitError("mode must be 'song' or 'instrument'")
        instruments = list(instruments) if instruments else (self.available_instruments() if mode == "song" else [])
        bad = [i for i in instruments if i not in INSTRUMENTS]
        if bad:
            raise FiltKitError(f"unknown instrument {bad}; choose from {list(INSTRUMENTS)}")
        if mode == "instrument" and len(instruments) != 1:
            raise FiltKitError("instrument mode needs exactly one instrument")
        if not instruments:
            raise FiltKitError("no instrument selected and no trained model found in models.json")
        stem_map = dict(stem_map or {})
        progress = on_progress or (lambda step, fraction: None)
        warnings, timings = [], {}
        out.mkdir(parents=True, exist_ok=True)
        progress("start", 0.0)

        # 1) which audio goes to which instrument model
        sources, stems_out, sep_model = {}, {}, None
        if mode == "song":
            sep_model = separation or ("htdemucs_6s" if {"piano", "guitar"} & set(instruments) else "htdemucs")
            progress("separating the song into instruments", 0.05)
            t = time.time()
            stems = self._separate(audio_path, out / "stems", sep_model)
            timings["separation_s"] = round(time.time() - t, 1)
            stems_out = {k: str(v) for k, v in stems.items()}
            for ins in instruments:
                name = stem_map.get(ins, STEM_FOR[ins])
                if name not in stems:
                    if ins in ("piano", "guitar") and "other" in stems:
                        warnings.append(f"the {sep_model} separation has no '{name}' stem; the 'other' stem is used for {ins}")
                        name = "other"
                    else:
                        warnings.append(f"no '{name}' stem available for {ins}; skipped")
                        continue
                sources[ins] = (stems[name], name)
        else:
            sources[instruments[0]] = (audio_path, "upload")

        # 2) transcription with each instrument's own model
        parts, part_info = [], []
        order = list(sources.items())
        for k, (ins, (src, stem_name)) in enumerate(order):
            progress(f"transcribing {ins}", 0.5 + 0.4 * k / max(len(order), 1))
            t = time.time()
            try:
                notes = self.transcribe(ins, src)
            except FiltKitError as e:
                warnings.append(str(e))
                continue
            timings[f"transcribe_{ins}_s"] = round(time.time() - t, 1)
            n = len(notes)
            midi_path = None
            if n:
                (out / "midi").mkdir(exist_ok=True)
                midi_path = out / "midi" / f"{ins}.mid"
                write_midi(ins, notes, midi_path)
                parts.append(dict(name=ins.capitalize(), kind=ins, notes=notes))
            else:
                warnings.append(f"no {'hits' if ins == 'drums' else 'notes'} found for {ins}")
            part_info.append(dict(instrument=ins, items=int(n), unit="hits" if ins == "drums" else "notes",
                                  stem=stem_name, audio=str(src), midi=str(midi_path) if midi_path else None,
                                  thresholds={k2: v for k2, v in self.registry[ins].items() if k2 != "weights"}))

        # 3) combined MIDI and the score
        score_info, combined = None, None
        if parts:
            progress("writing MIDI and sheet music", 0.92)
            combined = out / "midi" / "all_parts.mid"
            write_combined_midi([(p["kind"], p["notes"]) for p in parts], combined)
            xml, info = make_musicxml(parts, bpm, tuple(time_sig), title or audio_path.stem)
            (out / "score.musicxml").write_text(xml, encoding="utf-8")
            score_info = dict(file=str(out / "score.musicxml"), bpm=round(info["bpm"], 1),
                              bpm_source="given" if bpm else "estimated", measures=int(info["measures"]),
                              time_signature=info["time_sig"],
                              mean_timing_error_ms=round(info["mean_timing_error_ms"], 1))
            if info["mean_timing_error_ms"] > 40:
                warnings.append("the notes do not sit well on a steady tempo, so the sheet music may look messy; "
                                "giving the real tempo (bpm) usually helps")
        else:
            warnings.append("nothing was transcribed, so no MIDI or score was written")

        # 4) result file
        def rel(p):
            return None if p is None else Path(os.path.relpath(p, out)).as_posix()
        result = dict(
            status="ok" if parts else "empty",
            input=dict(file=audio_path.name, mode=mode),
            separation=dict(model=sep_model, stems={k: rel(v) for k, v in stems_out.items()}) if mode == "song" else None,
            parts=[dict(p, audio=rel(p["audio"]), midi=rel(p["midi"])) for p in part_info],
            midi_all=rel(combined),
            score=dict(score_info, file=rel(score_info["file"])) if score_info else None,
            warnings=warnings,
            timings=dict(timings, total_s=round(time.time() - t_start, 1)),
        )
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress("done", 1.0)
        return result


def main():
    ap = argparse.ArgumentParser(description="FiltKit: audio -> stems, MIDI, sheet music")
    ap.add_argument("audio")
    ap.add_argument("--mode", choices=["song", "instrument"], default="song")
    ap.add_argument("--instruments", nargs="+", default=None, choices=list(INSTRUMENTS),
                    help="default: every instrument that has a trained model (song mode)")
    ap.add_argument("--out", default=None, help="job folder (default: output/jobs/<file name>)")
    ap.add_argument("--separation", choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"], default=None,
                    help="default: htdemucs_6s when piano or guitar is requested, else htdemucs")
    ap.add_argument("--stem", nargs="+", default=[], metavar="INSTRUMENT=STEM",
                    help="use another stem for an instrument, e.g. piano=other")
    ap.add_argument("--bpm", type=float, default=None, help="tempo of the song (default: estimated)")
    ap.add_argument("--time-sig", default="4/4")
    ap.add_argument("--title", default=None)
    ap.add_argument("--registry", default=str(REGISTRY_DEFAULT))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    beats, beat_type = (int(x) for x in args.time_sig.split("/"))
    stem_map = dict(s.split("=", 1) for s in args.stem)
    out = args.out or str(ROOT / "output" / "jobs" / Path(args.audio).stem)
    try:
        engine = FiltKit(args.registry, args.device)
        result = engine.analyze(args.audio, out, args.mode, args.instruments, args.separation, args.bpm,
                                (beats, beat_type), args.title, stem_map,
                                on_progress=lambda step, f: print(f"[{f * 100:3.0f}%] {step}"))
    except FiltKitError as e:
        raise SystemExit(f"FiltKit: {e}")
    for p in result["parts"]:
        print(f"  {p['instrument']}: {p['items']} {p['unit']}" + (f"  -> {p['midi']}" if p["midi"] else ""))
    if result["score"]:
        s = result["score"]
        print(f"score: {s['measures']} bars, {s['bpm']} bpm ({s['bpm_source']}), {s['time_signature']} -> {s['file']}")
    for w in result["warnings"]:
        print("warning:", w)
    print(f"done in {result['timings']['total_s']}s. Everything is in {out}  (see result.json)")


if __name__ == "__main__":
    main()