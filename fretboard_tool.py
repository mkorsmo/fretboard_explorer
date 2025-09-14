#!/usr/bin/env python3
# fretboard_tool.py
# Readable fretboard with vertical fret lines, bold nut, fret markers.
# High strings on TOP, wound strings (low E) on BOTTOM.
# Supports: full diatonic scales, pentatonics, chord-tone highlighting by degree,
# approximate CAGED windows, and STRICT pentatonic CAGED (5 canonical boxes).
#
# New:
# --boxes pent-caged  → draws the 5 standard pentatonic boxes (minor/major oriented)
# --pent-orient minor|major (default minor) → which root to align Pattern 1 to

import argparse
from typing import List, Dict, Optional, Tuple

# ---------------- Config / constants ----------------

LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
LETTER_TO_SEMI = {"C":0, "D":2, "E":4, "F":5, "G":7, "A":9, "B":11}
MAJOR_STEPS = [2,2,1,2,2,2,1]
NMIN_STEPS  = [2,1,2,2,1,2,2]

# Pentatonic intervals from the root (semitones)
PENT_MAJOR_SEMIS = [0, 2, 4, 7, 9]    # 1 2 3 5 6
PENT_MINOR_SEMIS = [0, 3, 5, 7, 10]   # 1 b3 4 5 b7

# Standard tuning, low → high (6 → 1). We render reversed (1 on top).
TUNING = ["E", "A", "D", "G", "B", "E"]

# Typical fret marker frets
FRET_MARKERS = {3,5,7,9,12,15,17,19,21,24}

DEGREE_TO_ROMAN_MAJOR = ["I","ii","iii","IV","V","vi","vii°"]
DEGREE_TO_ROMAN_NMIN  = ["i","ii°","III","iv","v","VI","VII"]

# ---------------- Pitch/scale helpers ----------------

def parse_pitch(name: str):
    name = name.strip()
    letter = name[0].upper()
    acc = 0
    for ch in name[1:]:
        if ch == "#":
            acc += 1
        elif ch in "bB":
            acc -= 1
        else:
            raise ValueError(f"Bad accidental {ch} in {name}")
    if letter not in LETTERS:
        raise ValueError(f"Bad letter {letter}")
    return letter, acc

def pitch_to_pc(letter: str, acc: int) -> int:
    return (LETTER_TO_SEMI[letter] + acc) % 12

def semitone_for_letter(letter: str, target_pc: int):
    nat = LETTER_TO_SEMI[letter]
    for acc, sym in ((0,""), (1,"#"), (-1,"b")):
        if (nat + acc) % 12 == target_pc:
            return f"{letter}{sym}"
    for acc, sym in ((2,"##"), (-2,"bb")):
        if (nat + acc) % 12 == target_pc:
            return f"{letter}{sym}"
    raise ValueError(f"Cannot spell {letter} at pc {target_pc}")

def build_scale_spelled(root: str, mode: str) -> List[str]:
    """Return 7 spelled notes for major or natural minor from a root like 'C#' or 'Db'."""
    steps = MAJOR_STEPS if mode == "major" else NMIN_STEPS
    letter, acc = parse_pitch(root)
    root_pc = pitch_to_pc(letter, acc)
    idx = LETTERS.index(letter)

    letters_seq = [LETTERS[(idx+i)%7] for i in range(7)]
    pcs = [root_pc]
    total = 0
    for s in steps[:-1]:
        total += s
        pcs.append((root_pc + total) % 12)

    out = []
    out.append(letter + ("" if acc==0 else ("#"*acc if acc>0 else "b"*(-acc))))
    for i in range(1,7):
        out.append(semitone_for_letter(letters_seq[i], pcs[i]))
    return out  # 7 spelled notes

def degree_index_from_roman(roman: str, mode: str) -> Optional[int]:
    """Return 0..6 for the diatonic degree; supports '°' optional and case-insensitive."""
    roman = roman.strip()
    base = roman.replace("°", "")
    arr = DEGREE_TO_ROMAN_MAJOR if mode == "major" else DEGREE_TO_ROMAN_NMIN
    for i, r in enumerate(arr):
        if base.lower() == r.replace("°","").lower():
            return i
    return None

def diatonic_triads(scale7: List[str]) -> List[List[str]]:
    return [[ scale7[i], scale7[(i+2)%7], scale7[(i+4)%7] ] for i in range(7)]

def diatonic_sevenths(scale7: List[str]) -> List[List[str]]:
    return [[ scale7[i], scale7[(i+2)%7], scale7[(i+4)%7], scale7[(i+6)%7] ] for i in range(7)]

def pc_of_note_name(name: str) -> int:
    L, a = parse_pitch(name)
    return pitch_to_pc(L, a)

# ---------------- Label maps (what to show on grid) ----------------

def degree_map_diatonic(scale7: List[str]) -> Dict[int, str]:
    m = {}
    for i, note in enumerate(scale7, start=1):
        L, a = parse_pitch(note)
        pc = pitch_to_pc(L, a)
        m[pc] = "R" if i == 1 else str(i)
    return m

def label_map_pentatonic(root: str, mode: str, use_unicode_flat: bool) -> Dict[int, str]:
    """Auto flavor based on mode: major→major pent, natural_minor→minor pent."""
    letter, acc = parse_pitch(root)
    root_pc = pitch_to_pc(letter, acc)
    if mode == "major":
        semis = PENT_MAJOR_SEMIS
        labels = ["R", "2", "3", "5", "6"]
    else:
        semis = PENT_MINOR_SEMIS
        labels = ["R", "♭3" if use_unicode_flat else "b3", "4", "5",
                  "♭7" if use_unicode_flat else "b7"]
    return {(root_pc+s)%12: lab for s, lab in zip(semis, labels)}

def label_map_pent_forced(root: str, pent: str, use_unicode_flat: bool) -> Dict[int, str]:
    letter, acc = parse_pitch(root)
    root_pc = pitch_to_pc(letter, acc)
    if pent == "major":
        semis = PENT_MAJOR_SEMIS
        labels = ["R", "2", "3", "5", "6"]
    else:
        semis = PENT_MINOR_SEMIS
        labels = ["R", "♭3" if use_unicode_flat else "b3", "4", "5",
                  "♭7" if use_unicode_flat else "b7"]
    return {(root_pc+s)%12: lab for s, lab in zip(semis, labels)}

def label_map_chord_tones(scale7: List[str], degree_idx: int,
                          tones: str, use_unicode_flat: bool) -> Dict[int, str]:
    chord = diatonic_sevenths(scale7)[degree_idx] if tones == "seventh" else diatonic_triads(scale7)[degree_idx]
    root_pc = pc_of_note_name(chord[0])
    pcs, labels = [], []
    for note in chord:
        pc = pc_of_note_name(note)
        pcs.append(pc)
        semis = (pc - root_pc) % 12
        if semis == 0: labels.append("R")
        elif semis == 3: labels.append("♭3" if use_unicode_flat else "b3")
        elif semis == 4: labels.append("3")
        elif semis == 7: labels.append("5")
        elif semis == 10: labels.append("♭7" if use_unicode_flat else "b7")
        elif semis == 11: labels.append("7")
        else: labels.append("?")
    return {pc: lab for pc, lab in zip(pcs, labels)}

# ---------------- Grid construction ----------------

def build_grid(pc_to_label: Dict[int, str], frets: int):
    rows = []
    for s in TUNING:
        open_pc = pc_of_note_name(s)
        row = []
        for f in range(frets+1):
            pc = (open_pc + f) % 12
            row.append(pc_to_label.get(pc))
        rows.append(row)
    return rows

def cell_text(label: Optional[str], pad=2):
    if label is None: return " " * pad
    if len(label) >= pad: return label[:pad]
    return label + " " * (pad - len(label))

# ---------------- Rendering (single grid) ----------------

def _header_and_markers_unicode(frets: int):
    NUT = "║"
    header = ["    ", f"{NUT} "] + [f"{f:>2} " + (" " if f<frets else "") for f in range(1, frets+1)]
    marker = ["    ", f"{NUT} "] + [(f" • " if f in FRET_MARKERS else "   ") + (" " if f<frets else "") for f in range(1, frets+1)]
    return NUT, "".join(header), "".join(marker)

def _header_and_markers_ascii(frets: int):
    NUT = "||"
    header = ["    ", f"{NUT} "] + [f"{f:>2} " + (" " if f<frets else "") for f in range(1, frets+1)]
    marker = ["    ", f"{NUT} "] + [(f" * " if f in FRET_MARKERS else "   ") + (" " if f<frets else "") for f in range(1, frets+1)]
    return NUT, "".join(header), "".join(marker)

def render_single_unicode(scale7, grid, frets: int, legend_lines: List[str]) -> str:
    V = "│"; NUT, header_line, marker_line = _header_and_markers_unicode(frets)
    lines = []
    for si, row in enumerate(reversed(grid)):
        string_num = si + 1
        string_label = f"{string_num}({TUNING[::-1][si]})"
        label = f"{string_label:<4}"
        line = [label, NUT, cell_text(row[0]), V] + [cell_text(c)+(V if f<frets-1 else "") for f,c in enumerate(row[1:],1)]
        lines.append(" ".join(line))
    return "\n".join([header_line, marker_line] + lines + [""] + legend_lines)

def render_single_ascii(scale7, grid, frets: int, legend_lines: List[str]) -> str:
    V = "|"; NUT, header_line, marker_line = _header_and_markers_ascii(frets)
    lines = []
    for si, row in enumerate(reversed(grid)):
        string_num = si + 1
        string_label = f"{string_num}({TUNING[::-1][si]})"
        label = f"{string_label:<4}"
        line = [label, NUT, cell_text(row[0]), V] + [cell_text(c)+(V if f<frets-1 else "") for f,c in enumerate(row[1:],1)]
        lines.append(" ".join(line))
    return "\n".join([header_line, marker_line] + lines + [""] + legend_lines)

# ---------------- CAGED windows (approximate) ----------------

def first_low_e_root_fret(root_pc: int) -> int:
    """Find the lowest fret (0..12) on low E with this pc."""
    for f in range(0, 13):
        if (4 + f) % 12 == root_pc:
            return f
    return 0

def caged_windows(start_fret: int, frets: int, min_width: int = 5):
    centers = [start_fret + off for off in (0, 3, 5, 8, 10)]
    shapes = ["C", "A", "G", "E", "D"]
    wins = []
    for c, sh in zip(centers, shapes):
        half = max(1, min_width // 2)
        s, e = c-half, c+half
        if (e - s + 1) > min_width: e -= 1
        if s < 0: e += -s; s = 0
        if e > frets: s -= (e - frets); e = frets
        s, e = max(0,s), min(frets,e)
        if s > e: s = e
        wins.append((s,e,sh))
    return wins

def slice_grid(grid, s: int, e: int):
    return [row[s:e+1] for row in grid]

def render_caged_unicode(scale7, grid, frets: int, root_pc: int, legend_lines):
    NUT, V = "║", "│"
    start = first_low_e_root_fret(root_pc)
    wins = caged_windows(start, frets)
    blocks = []
    for (s,e,shape) in wins:
        header = ["    ", f"{NUT} "] + [f"{f:>2} " + (" " if f<e else "") for f in range(max(1,s), e+1)]
        marker = ["    ", f"{NUT} "] + [(f" • " if f in FRET_MARKERS else "   ") + (" " if f<e else "") for f in range(max(1,s), e+1)]
        sub = slice_grid(grid, s, e)
        lines = [f"   [{shape} shape]  frets {s}–{e}", "".join(header), "".join(marker)]
        for si,row in enumerate(reversed(sub)):
            string_num = si+1
            string_label = f"{string_num}({TUNING[::-1][si]})"
            line = [f"{string_label:<4}", NUT, cell_text(row[0]), V] + [cell_text(c)+(V if f<len(row)-2 else "") for f,c in enumerate(row[1:],1)]
            lines.append(" ".join(line))
        blocks.append("\n".join(lines)+"\n")
    return "\n".join(blocks+legend_lines)

def render_caged_ascii(scale7, grid, frets: int, root_pc: int, legend_lines):
    NUT, V = "||", "|"
    start = first_low_e_root_fret(root_pc)
    wins = caged_windows(start, frets)
    blocks = []
    for (s,e,shape) in wins:
        header = ["    ", f"{NUT} "] + [f"{f:>2} " + (" " if f<e else "") for f in range(max(1,s), e+1)]
        marker = ["    ", f"{NUT} "] + [(f" * " if f in FRET_MARKERS else "   ") + (" " if f<e else "") for f in range(max(1,s), e+1)]
        sub = slice_grid(grid, s, e)
        lines = [f"   [{shape} shape]  frets {s}–{e}", "".join(header), "".join(marker)]
        for si,row in enumerate(reversed(sub)):
            string_num = si+1
            string_label = f"{string_num}({TUNING[::-1][si]})"
            line = [f"{string_label:<4}", NUT, cell_text(row[0]), V] + [cell_text(c)+(V if f<len(row)-2 else "") for f,c in enumerate(row[1:],1)]
            lines.append(" ".join(line))
        blocks.append("\n".join(lines)+"\n")
    return "\n".join(blocks+legend_lines)

# ---------------- Strict PENTATONIC CAGED (5 canonical boxes) ----------------

def relative_minor_pc(major_pc: int) -> int:
    return (major_pc - 3) % 12  # major -> relative minor down 3 semitones

def relative_major_pc(minor_pc: int) -> int:
    return (minor_pc + 3) % 12  # minor -> relative major up 3 semitones

def pent_caged_windows(root_pc: int, frets: int, orient: str) -> List[Tuple[int,int,int]]:
    """
    Pattern 1 anchored to the low-E root of the chosen orientation:
      - orient='minor' → use minor-pent root on low E
      - orient='major' → use major-pent root on low E
    Returns list of (start,end,pattern_index 1..5).
    Uses canonical ~4-fret spans per box with practical overlaps.
    """
    start = first_low_e_root_fret(root_pc)

    # Canonical-ish spans (width 4 frets) relative to Pattern 1 start
    # P1: s..s+3, P2: s+2..s+5, P3: s+4..s+7, P4: s+5..s+8, P5: s+7..s+10
    offsets = [(0,3), (2,5), (4,7), (5,8), (7,10)]
    wins = []
    for i,(lo,hi) in enumerate(offsets, start=1):
        s = max(0, start + lo)
        e = min(frets, start + hi)
        if s > e: s = e
        wins.append((s,e,i))
    return wins

def render_pent_caged(scale7, grid, frets: int, root_pc: int, legend_lines, ascii_mode: bool):
    NUT, V = ("||","|") if ascii_mode else ("║","│")
    wins = pent_caged_windows(root_pc, frets, orient="minor")  # root_pc already set for chosen orient by caller
    blocks = []
    for (s,e,pat) in wins:
        header = ["    ", f"{NUT} "] + [f"{f:>2} " + (" " if f<e else "") for f in range(max(1,s), e+1)]
        marker = ["    ", f"{NUT} "] + [((f" * " if ascii_mode else f" • ") if f in FRET_MARKERS else "   ") + (" " if f<e else "") for f in range(max(1,s), e+1)]
        sub = slice_grid(grid, s, e)
        title = f"   [Pentatonic Pattern {pat}]  frets {s}–{e}"
        lines = [title, "".join(header), "".join(marker)]
        for si,row in enumerate(reversed(sub)):
            string_num = si+1
            string_label = f"{string_num}({TUNING[::-1][si]})"
            line = [f"{string_label:<4}", NUT, cell_text(row[0]), V] + [cell_text(c)+(V if f<len(row)-2 else "") for f,c in enumerate(row[1:],1)]
            lines.append(" ".join(line))
        blocks.append("\n".join(lines)+"\n")
    return "\n".join(blocks+legend_lines)

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Fretboard mapper with scales, pentatonics, chord tones & CAGED windows")
    ap.add_argument("--key", required=True, help="Key root, e.g. A, F#, Db, C#")
    ap.add_argument("--mode", choices=["major","natural_minor"], default="major")
    ap.add_argument("--show", choices=["scale","pent"], default="scale")
    ap.add_argument("--pent", choices=["auto","major","minor"], default="auto")
    ap.add_argument("--tones", choices=["all","triad","seventh"], default="all")
    ap.add_argument("--degree", type=str, default=None)
    ap.add_argument("--boxes", choices=["caged","pent-caged"], default=None)
    ap.add_argument("--pent-orient", choices=["minor","major"], default="minor",
                    help="For --boxes pent-caged: align Pattern 1 to this pentatonic root on the low E string")
    ap.add_argument("--frets", type=int, default=12)
    ap.add_argument("--ascii", action="store_true")
    args = ap.parse_args()

    frets = max(1, min(args.frets, 24))
    # Ensure boxed views have enough neck to avoid truncating later patterns
    if args.boxes in ('caged', 'pent-caged'):
        frets = max(frets, 24)

    mode = "major" if args.mode=="major" else "natural_minor"
    scale = build_scale_spelled(args.key, mode)
    ascii_mode = bool(args.ascii)

    # Decide WHAT to plot
    if args.degree:
        idx = degree_index_from_roman(args.degree, mode)
        if idx is None: raise SystemExit(f"Invalid --degree {args.degree}")
        tones = "seventh" if args.tones=="all" else args.tones
        pc_to_label = label_map_chord_tones(scale, idx, tones, not ascii_mode)
        legend_mode = f"View: Chord tones for {args.degree} ({tones})"
    else:
        # Pentatonic or full scale
        if args.show=="scale":
            pc_to_label = degree_map_diatonic(scale)
            legend_mode = f"View: Full scale ({mode})"
        else:
            # decide pent flavor
            if args.pent=="auto":
                pc_to_label = label_map_pentatonic(args.key, mode, not ascii_mode)
                pent_flavor = "major" if mode=="major" else "minor"
                legend_mode = f"View: Pentatonic (auto→{pent_flavor})"
            else:
                pc_to_label = label_map_pent_forced(args.key, args.pent, not ascii_mode)
                pent_flavor = args.pent
                legend_mode = f"View: Pentatonic ({pent_flavor})"

    grid = build_grid(pc_to_label, frets)
    legend_lines = [
        "Scale: " + " ".join(scale),
        legend_mode,
        "Legend: R=root, numbers=degrees; b3/♭3, b7/♭7 as applicable"
    ]

    # Determine root pitch-class for windowing
    key_root_pc = pc_of_note_name(scale[0])

    if args.boxes == "pent-caged":
        # We need to anchor Pattern 1 to requested orientation's root on low E.
        if args.pent == "auto":
            active_pent = "major" if mode=="major" else "minor"
        else:
            active_pent = args.pent

        if args.pent_orient == "minor":
            anchor_pc = relative_minor_pc(key_root_pc) if active_pent == "major" else key_root_pc
            out = render_pent_caged(scale, grid, frets, anchor_pc, legend_lines, ascii_mode)
        else:
            anchor_pc = relative_major_pc(key_root_pc) if active_pent == "minor" else key_root_pc
            out = render_pent_caged(scale, grid, frets, anchor_pc, legend_lines, ascii_mode)
    elif args.boxes == "caged":
        out = (render_caged_ascii if ascii_mode else render_caged_unicode)(
            scale, grid, frets, key_root_pc, legend_lines
        )
    else:
        out = (render_single_ascii if ascii_mode else render_single_unicode)(
            scale, grid, frets, legend_lines
        )
    print(out)

if __name__ == "__main__":
    main()
