"""Sheet music: note lists / MIDI files -> MusicXML (opens in MuseScore, Finale, the web viewer, ...).

  python src/score.py --part output/clip_pred.mid:piano --out output/clip.musicxml
  python src/score.py --part a.mid:piano b.mid:guitar c.mid:drums d.mid:voice --bpm 100 --title "My song"

Part kinds: piano (2 staves, split at middle C), guitar (treble clef 8vb), voice, drums (percussion clef).

How it works: the tempo is estimated from the note onsets (or given with --bpm), every note is rounded to
the nearest 16th note of that tempo, long notes are cut at bar lines and joined with ties, gaps become rests.

Limits (on purpose, to keep the score readable):
  * 16th-note grid only: triplets and faster notes are rounded to the nearest 16th
  * one time signature and tempo for the whole piece (default 4/4); the first note starts bar 1
  * simultaneous notes become chords. A chord lasts until the next onset on the same staff
  * sharps only (no key detection); no dynamics, pedal or articulation
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DIV = 4                                   # divisions per quarter note: one 16th note = 1 division
SHARPS = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
          ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]
NOTE_TYPES = {16: ("whole", 0), 12: ("half", 1), 8: ("half", 0), 6: ("quarter", 1),
              4: ("quarter", 0), 3: ("eighth", 1), 2: ("eighth", 0), 1: ("16th", 0)}
ALIGN = {16: 16, 12: 8, 8: 8, 6: 4, 4: 4, 3: 2, 2: 2, 1: 1}       # a value must start on a multiple of this

# drum class index (see drum_core.DRUM_CLASSES) -> (display step, display octave, notehead)
DRUM_STAFF = {0: ("F", 4, None),        # kick
              1: ("C", 5, None),        # snare
              2: ("G", 5, "x"),         # closed hi-hat
              3: ("G", 5, "circle-x"),  # open hi-hat
              4: ("D", 5, None),        # tom
              5: ("A", 5, "x"),         # crash
              6: ("F", 5, "x")}         # ride


# ---------------------------------------------------------------- tempo and quantization
def estimate_tempo(onsets, lo=60.0, hi=180.0, prior_bpm=110.0, prior_sigma=0.8):
    """Tempo (bpm) at which the onsets line up best with a sixteenth-note grid (the grid the notation uses).
    Returns (bpm, phase_seconds, strength 0..1). A tempo and its double fit equally well, so a mild prior
    around 110 bpm decides; use --bpm when the guess is wrong."""
    t = np.unique(np.round(np.asarray(onsets, dtype=np.float64), 4))
    if len(t) < 4:
        return 120.0, float(t[0]) if len(t) else 0.0, 0.0
    best = None
    for bpm in np.arange(lo, hi + 0.25, 0.25):
        g = 60.0 / bpm / 4.0
        z = np.exp(2j * np.pi * t / g).mean()
        score = abs(z) * math.exp(-0.5 * (math.log2(bpm / prior_bpm) / prior_sigma) ** 2)
        if best is None or score > best[0]:
            best = (score, float(bpm), z, g)
    _, bpm, z, g = best
    return bpm, float((np.angle(z) / (2 * np.pi) * g) % g), float(abs(z))


def grid_phase(onsets, bpm):
    t = np.unique(np.round(np.asarray(onsets, dtype=np.float64), 4))
    g = 60.0 / bpm / 4.0
    z = np.exp(2j * np.pi * t / g).mean() if len(t) else 0
    return float((np.angle(z) / (2 * np.pi) * g) % g), float(abs(z))


# ---------------------------------------------------------------- rhythm helpers
def decompose(pos, d, compound=False):
    """Split a duration of d sixteenths that starts at position pos (within its bar) into note values that
    start on their natural beat boundaries (ties join them). Dotted values may not cross the middle of the
    bar (or, for dotted eighths, a beat), which keeps the rhythm readable. Returns values from NOTE_TYPES."""
    align = dict(ALIGN)
    nocross = {12: 8, 6: 8, 3: 4}
    if compound:                                  # 6/8, 9/8, 12/8: the dotted quarter is the beat
        align[6] = 6
        nocross[6] = 6
    out = []
    while d > 0:
        for v in (16, 12, 8, 6, 4, 3, 2, 1):
            if v > d or pos % align[v] != 0:
                continue
            n = nocross.get(v)
            if n and pos // n != (pos + v - 1) // n:
                continue
            out.append(v)
            pos += v
            d -= v
            break
    return out


def pitched_events(notes_q):
    """[(start, end, pitch)] in sixteenths -> [(start, dur, [pitches])], never overlapping."""
    groups = {}
    for s, e, p in notes_q:
        groups.setdefault(s, []).append((e, p))
    starts = sorted(groups)
    events = []
    for i, s in enumerate(starts):
        end = max(e for e, _ in groups[s])
        if i + 1 < len(starts):
            end = min(end, starts[i + 1])
        events.append((s, max(1, end - s), sorted({p for _, p in groups[s]})))
    return events


def drum_events(ev_q):
    """[(start, class)] in sixteenths -> [(start, dur, [classes])]; each hit lasts until the next one (max a quarter)."""
    groups = {}
    for s, k in ev_q:
        groups.setdefault(s, set()).add(int(k))
    starts = sorted(groups)
    out = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else s + 4
        out.append((s, max(1, min(4, nxt - s)), sorted(groups[s])))
    return out


def layout(events, n_measures, measure_div, compound=False):
    """Events -> per bar a list of items: dict(dur, pitches (None = rest), tie_start, tie_stop, whole).
    Gaps become rests, long notes are cut at bar lines and split into tied note values."""
    total = n_measures * measure_div
    timeline, pos = [], 0
    for s, d, p in events:
        if s >= total:
            break
        d = min(d, total - s)
        if s > pos:
            timeline.append((pos, s - pos, None))
        timeline.append((s, d, tuple(p)))
        pos = s + d
    if pos < total:
        timeline.append((pos, total - pos, None))

    measures = [[] for _ in range(n_measures)]
    for s, d, p in timeline:
        values = []                                       # (bar index, value) for every note value of this event
        while d > 0:
            m = s // measure_div
            seg = min(d, (m + 1) * measure_div - s)
            for v in decompose(s % measure_div, seg, compound):
                values.append((m, v))
            s += seg
            d -= seg
        for k, (m, v) in enumerate(values):
            measures[m].append(dict(dur=v, pitches=p, whole=False,
                                    tie_start=p is not None and k < len(values) - 1,
                                    tie_stop=p is not None and k > 0))
    for items in measures:
        if items and all(it["pitches"] is None for it in items):
            items[:] = [dict(dur=measure_div, pitches=None, tie_start=False, tie_stop=False, whole=True)]
    return measures


# ---------------------------------------------------------------- MusicXML writing
def _sub(parent, tag, text=None, **attrs):
    el = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
    if text is not None:
        el.text = str(text)
    return el


def _write_note(measure, item, staff, voice, drum=False):
    """One MusicXML <note> per pitch (chord notes carry <chord/>). Child order follows the MusicXML schema."""
    dur = item["dur"]
    pitches = item["pitches"] if item["pitches"] is not None else [None]
    for i, p in enumerate(pitches):
        n = _sub(measure, "note")
        if i > 0:
            _sub(n, "chord")
        if p is None:
            _sub(n, "rest", **({"measure": "yes"} if item.get("whole") else {}))
        elif drum:
            step, octave, head = DRUM_STAFF[p]
            u = _sub(n, "unpitched")
            _sub(u, "display-step", step)
            _sub(u, "display-octave", octave)
        else:
            step, alter = SHARPS[p % 12]
            pe = _sub(n, "pitch")
            _sub(pe, "step", step)
            if alter:
                _sub(pe, "alter", alter)
            _sub(pe, "octave", p // 12 - 1)
        _sub(n, "duration", dur)
        if p is not None:
            if item["tie_stop"]:
                _sub(n, "tie", type="stop")
            if item["tie_start"]:
                _sub(n, "tie", type="start")
        _sub(n, "voice", voice)
        if not item.get("whole"):
            tname, dots = NOTE_TYPES[dur]
            _sub(n, "type", tname)
            for _ in range(dots):
                _sub(n, "dot")
        if p is not None and drum and DRUM_STAFF[p][2]:
            _sub(n, "notehead", DRUM_STAFF[p][2])
        if staff is not None:
            _sub(n, "staff", staff)
        if p is not None and (item["tie_stop"] or item["tie_start"]):
            nots = _sub(n, "notations")
            if item["tie_stop"]:
                _sub(nots, "tied", type="stop")
            if item["tie_start"]:
                _sub(nots, "tied", type="start")


def _quantize_part(part, bpm, origin):
    """Part dict -> list of events in sixteenths, plus the timing errors (seconds) of its onsets."""
    spb = 60.0 / bpm
    arr = np.asarray(part["notes"], dtype=np.float64)
    errs = []
    if part["kind"] == "drums":
        arr = arr.reshape(-1, 2)
        ev = []
        for t, k in arr:
            q = max(0, int(round((t - origin) / spb * DIV)))
            errs.append(abs(t - (origin + q * spb / DIV)))
            ev.append((q, int(k)))
        return ev, errs
    arr = arr.reshape(-1, 3)
    notes = []
    for s, e, p in arr:
        qs = max(0, int(round((s - origin) / spb * DIV)))
        qe = max(qs + 1, int(round((e - origin) / spb * DIV)))
        errs.append(abs(s - (origin + qs * spb / DIV)))
        notes.append((qs, qe, int(round(p))))
    return notes, errs


def make_musicxml(parts, bpm=None, time_sig=(4, 4), title="Transcription", split_pitch=60):
    """parts: list of dict(name=..., kind='piano'|'guitar'|'voice'|'drums'|'generic', notes=...)
         notes for pitched kinds: (N, 3) [start_sec, end_sec, midi_pitch]
         notes for drums:         (N, 2) [time_sec, drum_class_index]
    Returns (musicxml_text, info dict)."""
    parts = [p for p in parts if len(p["notes"])]
    if not parts:
        raise ValueError("no notes to write")
    onsets = np.concatenate([np.asarray(p["notes"], dtype=np.float64).reshape(len(p["notes"]), -1)[:, 0] for p in parts])
    if bpm is None:
        bpm, phase, strength = estimate_tempo(onsets)
    else:
        phase, strength = grid_phase(onsets, bpm)
    g = 60.0 / bpm / 4.0
    origin = phase + round((onsets.min() - phase) / g) * g          # grid point of the first note

    beats, beat_type = time_sig
    measure_div = DIV * 4 * beats // beat_type
    compound = beat_type == 8 and beats % 3 == 0

    quantized, all_errs, last = [], [], 0
    for p in parts:
        q, errs = _quantize_part(p, bpm, origin)
        quantized.append(q)
        all_errs += errs
        last = max(last, max((e[1] if p["kind"] != "drums" else e[0] + 1) for e in q))
    n_measures = max(1, math.ceil(last / measure_div))

    root = ET.Element("score-partwise", version="3.1")
    work = _sub(root, "work")
    _sub(work, "work-title", title)
    ident = _sub(root, "identification")
    _sub(_sub(ident, "encoding"), "software", "music transcription project")
    plist = _sub(root, "part-list")
    for i, p in enumerate(parts, 1):
        sp = _sub(plist, "score-part", id=f"P{i}")
        _sub(sp, "part-name", p.get("name") or p["kind"].capitalize())

    for i, (p, q) in enumerate(zip(parts, quantized), 1):
        kind = p["kind"]
        if kind == "piano":
            staves = [pitched_events([n for n in q if n[2] >= split_pitch]),
                      pitched_events([n for n in q if n[2] < split_pitch])]
        elif kind == "drums":
            staves = [drum_events(q)]
        else:
            staves = [pitched_events(q)]
        laid = [layout(ev, n_measures, measure_div, compound) for ev in staves]

        part_el = _sub(root, "part", id=f"P{i}")
        for m in range(n_measures):
            me = _sub(part_el, "measure", number=m + 1)
            if m == 0:
                at = _sub(me, "attributes")
                _sub(at, "divisions", DIV)
                _sub(_sub(at, "key"), "fifths", 0)
                tm = _sub(at, "time")
                _sub(tm, "beats", beats)
                _sub(tm, "beat-type", beat_type)
                if len(staves) == 2:
                    _sub(at, "staves", 2)
                    c1 = _sub(at, "clef", number=1)
                    _sub(c1, "sign", "G"); _sub(c1, "line", 2)
                    c2 = _sub(at, "clef", number=2)
                    _sub(c2, "sign", "F"); _sub(c2, "line", 4)
                elif kind == "drums":
                    _sub(_sub(at, "clef"), "sign", "percussion")
                else:
                    c = _sub(at, "clef")
                    _sub(c, "sign", "G"); _sub(c, "line", 2)
                    if kind == "guitar":
                        _sub(c, "clef-octave-change", -1)
                if i == 1:
                    d = _sub(me, "direction", placement="above")
                    met = _sub(_sub(d, "direction-type"), "metronome")
                    _sub(met, "beat-unit", "quarter")
                    _sub(met, "per-minute", int(round(bpm)))
                    _sub(d, "sound", tempo=int(round(bpm)))
            for s_idx, bars in enumerate(laid):
                if s_idx == 1:
                    _sub(_sub(me, "backup"), "duration", measure_div)
                for item in bars[m]:
                    _write_note(me, item, staff=(s_idx + 1) if len(staves) == 2 else None,
                                voice=s_idx + 1, drum=(kind == "drums"))

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">\n' + body + "\n")
    info = dict(bpm=float(bpm), measures=n_measures, time_sig=f"{beats}/{beat_type}",
                tempo_fit=float(strength), mean_timing_error_ms=float(np.mean(all_errs) * 1000))
    return xml, info


# ---------------------------------------------------------------- MIDI input and command line
def read_part_from_midi(path, kind):
    """MIDI file -> notes array for make_musicxml (drums are converted to drum classes)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted((n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes)
    if kind == "drums":
        from drum_core import events_from_notes
        return events_from_notes((s, p) for s, e, p in notes)
    return np.array(notes, dtype=np.float64).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser(description="MIDI -> sheet music (MusicXML)")
    ap.add_argument("--part", nargs="+", required=True, help="FILE.mid:KIND   (KIND = piano, guitar, voice, drums)")
    ap.add_argument("--out", default="output/score.musicxml")
    ap.add_argument("--bpm", type=float, default=None, help="tempo; default: estimated from the notes")
    ap.add_argument("--time-sig", default="4/4")
    ap.add_argument("--title", default="Transcription")
    args = ap.parse_args()

    beats, beat_type = (int(x) for x in args.time_sig.split("/"))
    parts = []
    for spec in args.part:
        path, kind = spec.rsplit(":", 1)                     # rsplit: Windows paths contain ':'
        if kind not in ("piano", "guitar", "voice", "drums"):
            raise SystemExit(f"unknown part kind '{kind}' in {spec}")
        notes = read_part_from_midi(path, kind)
        print(f"{Path(path).name}: {len(notes)} {'hits' if kind == 'drums' else 'notes'} as {kind}")
        parts.append(dict(name=kind.capitalize(), kind=kind, notes=notes))

    xml, info = make_musicxml(parts, args.bpm, (beats, beat_type), args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    print(f"tempo {info['bpm']:.1f} bpm ({'given' if args.bpm else 'estimated'}), {info['measures']} bars of "
          f"{info['time_sig']}, average timing error after rounding {info['mean_timing_error_ms']:.0f} ms")
    if info["mean_timing_error_ms"] > 40:
        print("warning: the notes do not sit well on a steady tempo, so the rhythm may look messy. "
              "Try --bpm with the real tempo (or half/double it).")
    print("written:", out, " (open it in MuseScore, or the web viewer)")


if __name__ == "__main__":
    main()