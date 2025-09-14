#!/usr/bin/env python3
# fretboard_tool.py
# Readable fretboard with vertical fret lines, bold nut, fret markers.
# High strings on TOP, wound strings (low E) on BOTTOM.
# Supports: full diatonic scales, pentatonics, chord-tone highlighting by degree,
# and approximate CAGED windows.

import argparse
from typing import List, Dict, Tuple, Optional

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
    if mode == "major":
        arr = DEGREE_TO_ROMAN_MAJOR
    else:
        arr = DEGREE_TO_ROMAN_NMIN
    for i, r in enumerate(arr):
        if base.lower() == r.replace("°","").lower():
            return i
    return None

def diatonic_triads(scale7: List[str]) -> List[List[str]]:
    """Return seven diatonic triads (1-3-5) as spelled notes."""
    triads = []
    for i in range(7):
        triads.append([ scale7[i], scale7[(i+2)%7], scale7[(i+4)%7] ])
    return triads

def diatonic_sevenths(scale7: List[str]) -> List[List[str]]:
    """Return seven diatonic seventh chords (1-3-5-7) as spelled notes."""
    sevenths = []
    for i in range(7):
        sevenths.append([ scale7[i], scale7[(i+2)%7], scale7[(i+4)%7], scale7[(i+6)%7] ])
    return sevenths

def pc_of_note_name(name: str) -> int:
    L, a = parse_pitch(name)
    return pitch_to_pc(L, a)

# ---------------- Label maps (what to show on grid) ----------------

def degree_map_diatonic(scale7: List[str]) -> Dict[int, str]:
    """Full diatonic scale: {pc: 'R' or '2'..'7'}."""
    m = {}
    for i, note in enumerate(scale7, start=1):
        L, a = parse_pitch(note)
        pc = pitch_to_pc(L, a)
        m[pc] = "R" if i == 1 else str(i)
    return m

def label_map_pentatonic(root: str, mode: str, use_unicode_flat: bool) -> Dict[int, str]:
    """Pent chosen by mode: major→major pent, natural_minor→minor pent."""
    letter, acc = parse_pitch(root)
    root_pc = pitch_to_pc(letter, acc)
    if mode == "major":
        semis = PENT_MAJOR_SEMIS
        labels = ["R", "2", "3", "5", "6"]
    else:
        semis = PENT_MINOR_SEMIS
        labels = ["R", "♭3" if use_unicode_flat else "b3", "4", "5",
                  "♭7" if use_unicode_flat else "b7"]
    m = {}
    for s, lab in zip(semis, labels):
        pc = (root_pc + s) % 12
        m[pc] = lab
    return m

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
    m = {}
    for s, lab in zip(semis, labels):
        pc = (root_pc + s) % 12
        m[pc] = lab
    return m

def label_map_chord_tones(scale7: List[str], degree_idx: int,
                          tones: str, use_unicode_flat: bool) -> Dict[int, str]:
    """
    Build {pc: label} for a diatonic chord on degree_idx.
    tones: 'triad' or 'seventh'
    Labels show R, and 3 / b3, 5, and 7 / b7 based on semitone distances from the chord root.
    """
    chord = diatonic_sevenths(scale7)[degree_idx] if tones == "seventh" else diatonic_triads(scale7)[degree_idx]
    root_pc = pc_of_note_name(chord[0])
    pcs = []
    for note in chord:
        pcs.append(pc_of_note_name(note))
    # interval labels
    labels = []
    for pc in pcs:
        semis = (pc - root_pc) % 12
        if semis == 0:
            labels.append("R")
        elif semis == 3:
            labels.append("♭3" if use_unicode_flat else "b3")
        elif semis == 4:
            labels.append("3")
        elif semis == 7:
            labels.append("5")
        elif semis == 10:
            labels.append("♭7" if use_unicode_flat else "b7")
        elif semis == 11:
            labels.append("7")
        else:
            # rare (e.g., diminished fifth or something off-scale if presented),
            # but we won't encounter here for standard major/natural minor diatonic sevenths
            labels.append("?")
    return { pc: lab for pc, lab in zip(pcs, labels) }

# ---------------- Grid construction ----------------

def build_grid(pc_to_label: Dict[int, str], frets: int) -> List[List[Optional[str]]]:
    """Return 6 rows of length frets+1 with labels or None.
       Row 0 = string 6 (low E), Row 5 = string 1 (high E)."""
    rows = []
    for s in TUNING:
        open_pc = pc_of_note_name(s)
        row = []
        for f in range(frets+1):
            pc = (open_pc + f) % 12
            row.append(pc_to_label.get(pc))  # None or label like 'R','3','b7'
        rows.append(row)
    return rows

def cell_text(label: Optional[str], pad=2):
    if label is None:
        return " " * pad
    # Keep width ~2 chars (R, 2..7, b3/♭3, b7/♭7). ASCII mode avoids wide glyphs.
    if len(label) >= pad:
        return label[:pad]
    return label + " " * (pad - len(label))

# ---------------- Rendering (single grid) ----------------

def _header_and_markers_unicode(frets: int):
    NUT = "║"
    header = ["    ", f"{NUT} "]
    for f in range(1, frets+1):
        header.append(f"{f:>2} ")
        if f < frets:
            header.append(" ")
    header_line = "".join(header)
    marker = ["    ", f"{NUT} "]
    for f in range(1, frets+1):
        mark = "•" if f in FRET_MARKERS else " "
        marker.append(f" {mark} ")
        if f < frets:
            marker.append(" ")
    return NUT, header_line, "".join(marker)

def _header_and_markers_ascii(frets: int):
    NUT = "||"
    header = ["    ", f"{NUT} "]
    for f in range(1, frets+1):
        header.append(f"{f:>2} ")
        if f < frets:
            header.append(" ")
    marker = ["    ", f"{NUT} "]
    for f in range(1, frets+1):
        mark = "*" if f in FRET_MARKERS else " "
        marker.append(f" {mark} ")
        if f < frets:
            marker.append(" ")
    return NUT, "".join(header), "".join(marker)

def render_single_unicode(scale7: List[str], grid, frets: int, legend_lines: List[str]) -> str:
    V = "│"
    NUT, header_line, marker_line = _header_and_markers_unicode(frets)
    # Lines for each string: HIGH to LOW
    lines = []
    reversed_grid = list(reversed(grid))
    reversed_tuning = list(reversed(TUNING))
    for si, row in enumerate(reversed_grid):
        string_num = si + 1
        string_label = f"{string_num}({reversed_tuning[si]})"
        label = f"{string_label:<4}"
        line = [label, NUT, cell_text(row[0]), V]
        for f in range(1, frets+1):
            line.append(cell_text(row[f]))
            if f < frets:
                line.append(V)
        lines.append(" ".join(line))
    return "\n".join([header_line, marker_line] + lines + [""] + legend_lines)

def render_single_ascii(scale7: List[str], grid, frets: int, legend_lines: List[str]) -> str:
    V = "|"
    NUT, header_line, marker_line = _header_and_markers_ascii(frets)
    lines = []
    reversed_grid = list(reversed(grid))
    reversed_tuning = list(reversed(TUNING))
    for si, row in enumerate(reversed_grid):
        string_num = si + 1
        string_label = f"{string_num}({reversed_tuning[si]})"
        label = f"{string_label:<4}"
        line = [label, NUT, cell_text(row[0]), V]
        for f in range(1, frets+1):
            line.append(cell_text(row[f]))
            if f < frets:
                line.append(V)
        lines.append(" ".join(line))
    return "\n".join([header_line, marker_line] + lines + [""] + legend_lines)

# ---------------- CAGED windows (approximate) ----------------

def first_low_e_root_fret(root_pc: int) -> int:
    """Find the lowest fret (0..12) on low E with this pc."""
    # Low E open is pc 4. Find smallest f such that (4+f)%12 == root_pc
    for f in range(0, 13):
        if (4 + f) % 12 == root_pc:
            return f
    return 0

def caged_windows(start_fret: int, frets: int) -> List[Tuple[int,int,str]]:
    """
    Return five overlapping (start,end,label) windows across the neck.
    Heuristic: 5 windows of ~4-5 frets each, starting near the first low-E root.
    Labels follow C-A-G-E-D order.
    """
    # Keep windows within [0, frets]
    # Center points spaced to cover ~12 frets with overlap
    centers = [start_fret + off for off in (0, 3, 5, 8, 10)]
    windows = []
    shapes = ["C","A","G","E","D"]
    for c, sh in zip(centers, shapes):
        s = max(0, c - 2)
        e = min(frets, c + 2)
        if e - s < 4 and e < frets:
            e = min(frets, s + 4)
        windows.append((s, e, sh))
    return windows

def slice_grid(grid, s: int, e: int):
    """Slice columns [s..e] (inclusive) from a full grid."""
    return [row[s:(e+1)] for row in grid]

def render_caged_unicode(scale7: List[str], grid, frets: int, root_pc: int, legend_lines: List[str]) -> str:
    NUT_U, header_line_u, marker_line_u = _header_and_markers_unicode(frets)  # for styling lengths
    start = first_low_e_root_fret(root_pc)
    wins = caged_windows(start, frets)
    blocks = []
    V = "│"; NUT = "║"
    for (s,e,shape) in wins:
        # Header per window
        hdr = [f"   [{shape} shape]  frets {s}–{e}"]
        # Build mini header/marker for this slice
        header = ["    ", f"{NUT} "]
        for f in range(s if s>0 else 1, e+1):
            # show absolute fret numbers
            header.append(f"{f:>2} ")
            if f < e:
                header.append(" ")
        header_line = "".join(header)
        marker = ["    ", f"{NUT} "]
        for f in range(s if s>0 else 1, e+1):
            mark = "•" if f in FRET_MARKERS else " "
            marker.append(f" {mark} ")
            if f < e:
                marker.append(" ")
        marker_line = "".join(marker)

        # Slice grid and render lines (HIGH to LOW)
        sub = slice_grid(grid, s, e)
        lines = []
        reversed_sub = list(reversed(sub))
        reversed_tuning = list(reversed(TUNING))
        for si, row in enumerate(reversed_sub):
            string_num = si + 1
            string_label = f"{string_num}({reversed_tuning[si]})"
            label = f"{string_label:<4}"
            # nut only prints as column 0; within slices, we still print vertical bar
            line = [label, NUT, cell_text(row[0]), V]
            for col in range(1, len(row)):
                line.append(cell_text(row[col]))
                if col < len(row)-1:
                    line.append(V)
            lines.append(" ".join(line))

        blocks.append("\n".join(hdr + [header_line, marker_line] + lines + [""]))
    return "\n".join(blocks + legend_lines)

def render_caged_ascii(scale7: List[str], grid, frets: int, root_pc: int, legend_lines: List[str]) -> str:
    NUT_A, header_line_a, marker_line_a = _header_and_markers_ascii(frets)
    start = first_low_e_root_fret(root_pc)
    wins = caged_windows(start, frets)
    blocks = []
    V = "|"; NUT = "||"
    for (s,e,shape) in wins:
        hdr = [f"   [{shape} shape]  frets {s}–{e}"]
        header = ["    ", f"{NUT} "]
        for f in range(s if s>0 else 1, e+1):
            header.append(f"{f:>2} ")
            if f < e:
                header.append(" ")
        header_line = "".join(header)
        marker = ["    ", f"{NUT} "]
        for f in range(s if s>0 else 1, e+1):
            mark = "*" if f in FRET_MARKERS else " "
            marker.append(f" {mark} ")
            if f < e:
                marker.append(" ")
        marker_line = "".join(marker)

        sub = slice_grid(grid, s, e)
        lines = []
        reversed_sub = list(reversed(sub))
        reversed_tuning = list(reversed(TUNING))
        for si, row in enumerate(reversed_sub):
            string_num = si + 1
            string_label = f"{string_num}({reversed_tuning[si]})"
            label = f"{string_label:<4}"
            line = [label, NUT, cell_text(row[0]), V]
            for col in range(1, len(row)):
                line.append(cell_text(row[col]))
                if col < len(row)-1:
                    line.append(V)
            lines.append(" ".join(line))

        blocks.append("\n".join(hdr + [header_line, marker_line] + lines + [""]))
    return "\n".join(blocks + legend_lines)

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Readable fretboard mapper (high strings on top, wound at bottom) with scales, pentatonics, chord tones & CAGED windows"
    )
    ap.add_argument("--key", required=True, help="Key root, e.g. A, F#, Db, C#")
    ap.add_argument("--mode", choices=["major","natural_minor"], default="major",
                    help="For diatonic view OR to choose default pent flavor in auto mode")
    ap.add_argument("--show", choices=["scale","pent"], default="scale",
                    help="Show full diatonic scale or pentatonic set")
    ap.add_argument("--pent", choices=["auto","major","minor"], default="auto",
                    help="Which pentatonic to use (auto picks major for major keys, minor for natural minor)")
    ap.add_argument("--tones", choices=["all","triad","seventh"], default="all",
                    help="When --degree is set: show only triad (R/3/5) or seventh (R/3/5/7) tones; otherwise ignored")
    ap.add_argument("--degree", type=str, default=None,
                    help="Diatonic Roman numeral to highlight chord tones across the neck (e.g., I, ii, iii, IV, V, vi, vii°)")
    ap.add_argument("--boxes", choices=["caged"], default=None,
                    help="Show five approximate C-A-G-E-D windows instead of a single full-width grid")
    ap.add_argument("--frets", type=int, default=12, help="Number of frets to display (max 24)")
    ap.add_argument("--ascii", action="store_true", help="Use ASCII instead of Unicode box-drawing/flat symbols")
    args = ap.parse_args()

    frets = max(1, min(args.frets, 24))
    mode = "major" if args.mode == "major" else "natural_minor"
    show = args.show
    pent_choice = args.pent
    tones = args.tones
    use_unicode_flat = not args.ascii

    # Build baseline diatonic scale (always for legends, and for 'scale' view)
    scale = build_scale_spelled(args.key, mode)

    # Decide WHAT to label on the grid
    if args.degree:
        idx = degree_index_from_roman(args.degree, mode)
        if idx is None:
            raise SystemExit(f"Invalid --degree '{args.degree}' for {mode} mode. Try one of: "
                             f"{', '.join(DEGREE_TO_ROMAN_MAJOR if mode=='major' else DEGREE_TO_ROMAN_NMIN)}")
        if tones == "all":
            # default to seventh for clarity when a degree is provided
            tones = "seventh"
        pc_to_label = label_map_chord_tones(scale, idx, tones, use_unicode_flat)
        legend_mode = f"View: Chord tones for {args.degree} ({'triad' if tones=='triad' else 'seventh'})"
    else:
        if show == "scale":
            pc_to_label = degree_map_diatonic(scale)
            legend_mode = f"View: Full scale ({'major' if mode=='major' else 'natural minor'})"
        else:
            if pent_choice == "auto":
                pc_to_label = label_map_pentatonic(args.key, mode, use_unicode_flat)
                legend_mode = f"View: Pentatonic (auto → {'major' if mode=='major' else 'minor'})"
                pent_choice = "major" if mode == "major" else "minor"
            else:
                pc_to_label = label_map_pent_forced(args.key, pent_choice, use_unicode_flat)
                legend_mode = f"View: Pentatonic ({pent_choice})"

    grid = build_grid(pc_to_label, frets)

    legend_scale = "Scale: " + " ".join(scale)
    legend = "Legend: R = root, numbers = degrees; b3/♭3 and b7/♭7 as applicable; blank = not shown"
    legend_lines = [legend_scale, legend_mode, legend]

    root_pc = pc_of_note_name(scale[0])

    if args.boxes == "caged":
        out = render_caged_ascii(scale, grid, frets, root_pc, legend_lines) if args.ascii \
              else render_caged_unicode(scale, grid, frets, root_pc, legend_lines)
    else:
        out = render_single_ascii(scale, grid, frets, legend_lines) if args.ascii \
              else render_single_unicode(scale, grid, frets, legend_lines)
    print(out)

if __name__ == "__main__":
    main()
