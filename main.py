#!/usr/bin/env python3
# theory_quiz_gui.py
# A pure-Python Tkinter GUI for guitar/music theory drills.

import random
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

# ------------------------- Music spelling utilities -------------------------

LETTER_TO_NAT_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTERS = ["C", "D", "E", "F", "G", "A", "B"]

MAJOR_STEP_PATTERN = [2, 2, 1, 2, 2, 2, 1]        # W W H W W W H
NAT_MINOR_STEP_PATTERN = [2, 1, 2, 2, 1, 2, 2]    # W H W W H W W

DEGREE_TO_ROMAN_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
DEGREE_TO_QUALITY_MAJOR = ["M", "m", "m", "M", "M", "m", "dim"]

DEGREE_TO_ROMAN_NMIN = ["i", "ii°", "III", "iv", "v", "VI", "VII"]
DEGREE_TO_QUALITY_NMIN = ["m", "dim", "M", "m", "m", "M", "M"]

NOTE_POOL = [
    "C","G","D","A","E","B","F#","C#",
    "F","Bb","Eb","Ab","Db","Gb","Cb"
]

def parse_pitch_name(name: str):
    name = name.strip()
    if not name:
        raise ValueError("Empty pitch name")
    letter = name[0].upper()
    if letter not in LETTERS:
        raise ValueError(f"Invalid pitch letter: {letter}")
    rest = name[1:]
    acc = 0
    for ch in rest:
        if ch == "#":
            acc += 1
        elif ch in "bB":
            acc -= 1
        else:
            raise ValueError(f"Invalid accidental: {ch} in {name}")
    return letter, acc

def pitch_to_semitone(letter: str, acc: int) -> int:
    base = LETTER_TO_NAT_SEMITONE[letter]
    return (base + acc) % 12

def semitone_for_spelled_letter(letter: str, target_semi: int):
    nat = LETTER_TO_NAT_SEMITONE[letter]
    for acc, sym in [(0, ""), (1, "#"), (-1, "b")]:
        if (nat + acc) % 12 == target_semi:
            return acc, f"{letter}{sym}"
    for acc, sym in [(2, "##"), (-2, "bb")]:
        if (nat + acc) % 12 == target_semi:
            return acc, f"{letter}{sym}"
    raise ValueError(f"Cannot spell letter {letter} at semitone {target_semi}")

def build_scale_spelled(root: str, mode: str = "major"):
    letter, acc = parse_pitch_name(root)
    steps = MAJOR_STEP_PATTERN if mode == "major" else NAT_MINOR_STEP_PATTERN
    root_semi = pitch_to_semitone(letter, acc)

    idx = LETTERS.index(letter)
    letters_seq = [LETTERS[(idx + i) % 7] for i in range(7)]

    semis = [root_semi]
    total = 0
    for step in steps[:-1]:
        total += step
        semis.append((root_semi + total) % 12)

    spelled = []
    if acc == 0:
        spelled.append(letter)
    elif acc > 0:
        spelled.append(letter + "#" * acc)
    else:
        spelled.append(letter + "b" * (-acc))

    for i in range(1, 7):
        L = letters_seq[i]
        target = semis[i]
        _, name = semitone_for_spelled_letter(L, target)
        spelled.append(name)
    return spelled

def diatonic_triads(scale7):
    triads = []
    for i in range(7):
        n1 = scale7[i]
        n3 = scale7[(i + 2) % 7]
        n5 = scale7[(i + 4) % 7]
        triads.append([n1, n3, n5])
    return triads

def triad_qualities(mode="major"):
    return DEGREE_TO_QUALITY_MAJOR if mode == "major" else DEGREE_TO_QUALITY_NMIN

def roman_numerals(mode="major"):
    return DEGREE_TO_ROMAN_MAJOR if mode == "major" else DEGREE_TO_ROMAN_NMIN

# ------------------------- Quiz generators -------------------------

def ask_scale_degree():
    root = random.choice(["C","G","D","A","E","B","F","Bb","Eb","Ab","Db","F#","C#"])
    deg = random.choice([2,3,4,5,6,7])
    scale = build_scale_spelled(root, "major")
    answer = scale[(deg - 1) % 7]
    prompt = f"Scale degree: What is the {deg} of {root}?"
    return prompt, answer, {"type": "degree", "root": root, "deg": deg}

def ask_diatonic_chord_in_key():
    key = random.choice(NOTE_POOL)
    mode = random.choice(["major", "natural_minor"])
    romans = roman_numerals(mode)
    qualities = triad_qualities(mode)
    idx = random.randrange(7)
    roman = romans[idx]
    scale = build_scale_spelled(key, "major" if mode == "major" else "natural_minor")
    triads = diatonic_triads(scale)
    chord_root = triads[idx][0]
    qual = qualities[idx]
    qual_name = {"M": "major", "m": "minor", "dim": "diminished"}[qual]
    prompt = f"In {key} {('major' if mode=='major' else 'natural minor')}, what is the {roman} chord?"
    answer_primary = f"{chord_root} {qual_name}"
    answer_alt = chord_root
    return prompt, (answer_primary, answer_alt), {"type": "diatonic_chord", "key": key, "mode": mode, "degree": roman}

def ask_scale_spelling():
    key = random.choice(NOTE_POOL)
    mode = random.choice(["major", "natural_minor"])
    spelled = build_scale_spelled(key, mode)
    prompt = f"Spell the {key} {('major' if mode=='major' else 'natural minor')} scale (7 notes):"
    answer = " ".join(spelled)
    return prompt, answer, {"type": "scale_spelling", "key": key, "mode": mode}

def ask_quality_by_degree():
    key = random.choice(NOTE_POOL)
    mode = random.choice(["major", "natural_minor"])
    romans = roman_numerals(mode)
    qualities = triad_qualities(mode)
    idx = random.randrange(7)
    roman = romans[idx]
    qual = qualities[idx]
    qual_name = {"M":"major","m":"minor","dim":"diminished"}[qual]
    prompt = f"In {key} {('major' if mode=='major' else 'natural minor')}, what is the chord QUALITY of {roman}?"
    answer = qual_name
    return prompt, answer, {"type":"quality", "key": key, "mode": mode, "degree": roman}

def ask_nashville_to_chord():
    key = random.choice(NOTE_POOL)
    romans = DEGREE_TO_ROMAN_MAJOR
    idx = random.randrange(7)
    roman = romans[idx]
    scale = build_scale_spelled(key, "major")
    triads = diatonic_triads(scale)
    root = triads[idx][0]
    qual = DEGREE_TO_QUALITY_MAJOR[idx]
    qual_name = {"M":"major","m":"minor","dim":"diminished"}[qual]
    prompt = f"Nashville → Chord: In {key} major, what is {roman}?"
    answer_primary = f"{root} {qual_name}"
    answer_alt = root
    return prompt, (answer_primary, answer_alt), {"type":"nashville_to_chord", "key": key, "roman": roman}

def ask_chord_to_nashville():
    key = random.choice(NOTE_POOL)
    scale = build_scale_spelled(key, "major")
    triads = diatonic_triads(scale)
    idx = random.randrange(7)
    chord_root = triads[idx][0]
    qual = DEGREE_TO_QUALITY_MAJOR[idx]
    qual_name = {"M":"major","m":"minor","dim":"diminished"}[qual]
    roman = DEGREE_TO_ROMAN_MAJOR[idx]
    prompt = f"Chord → Nashville: In {key} major, what Nashville number is {chord_root} {qual_name}?"
    answer = roman
    return prompt, answer, {"type":"chord_to_nashville", "key": key, "roman": roman}

QUESTION_TYPES = {
    "Scale degree (e.g., 5th of D)": ask_scale_degree,
    "Diatonic chord in key (name & quality)": ask_diatonic_chord_in_key,
    "Scale spelling (7 notes)": ask_scale_spelling,
    "Chord quality by degree": ask_quality_by_degree,
    "Nashville → Chord": ask_nashville_to_chord,
    "Chord → Nashville": ask_chord_to_nashville,
    "Random mix": None,  # handled specially
}

def normalize_user_answer(s: str) -> str:
    return " ".join(s.strip().replace("maj","major").replace("min","minor").split())

def check_answer(user, answer):
    u = normalize_user_answer(user).lower()
    if isinstance(answer, tuple):
        primary, alt = answer
        return (
            u == normalize_user_answer(primary).lower()
            or u == normalize_user_answer(alt).lower()
        )
    else:
        return u == normalize_user_answer(answer).lower()

# ------------------------- Tkinter GUI -------------------------

class QuizApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Guitar Theory Drills")
        self.geometry("760x520")
        self.minsize(680, 480)

        self.total = 0
        self.correct = 0
        self.recent = deque([], maxlen=5)

        self.current_prompt = ""
        self.current_answer = ""
        self.current_meta = {}

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 12, "pady": 8}

        # Header
        header = ttk.Frame(self)
        header.pack(fill="x", **pad)
        ttk.Label(header, text="Guitar Theory Drills", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.mode_label = ttk.Label(header, text="Mode: Random mix", foreground="#0d6efd")
        self.mode_label.pack(side="right")

        # Mode chooser + buttons
        row1 = ttk.Frame(self)
        row1.pack(fill="x", **pad)

        ttk.Label(row1, text="Choose a drill:").pack(side="left")
        self.mode_var = tk.StringVar(value="Random mix")
        self.mode_combo = ttk.Combobox(
            row1,
            textvariable=self.mode_var,
            state="readonly",
            values=list(QUESTION_TYPES.keys()),
            width=38,
        )
        self.mode_combo.pack(side="left", padx=8)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.btn_new = ttk.Button(row1, text="New Question", command=self.new_question)
        self.btn_new.pack(side="left", padx=4)
        self.btn_reveal = ttk.Button(row1, text="Show Answer", command=self.reveal_answer)
        self.btn_reveal.pack(side="left", padx=4)
        self.btn_reset = ttk.Button(row1, text="Reset Score", command=self.reset_score)
        self.btn_reset.pack(side="right", padx=4)

        # Prompt
        prompt_frame = ttk.LabelFrame(self, text="Prompt")
        prompt_frame.pack(fill="x", **pad)
        self.prompt_lbl = ttk.Label(prompt_frame, text="Press “New Question” to start.", font=("Segoe UI", 12))
        self.prompt_lbl.pack(anchor="w", padx=10, pady=10)

        # Answer entry
        ans_frame = ttk.LabelFrame(self, text="Your Answer")
        ans_frame.pack(fill="x", **pad)
        self.answer_var = tk.StringVar()
        ans_row = ttk.Frame(ans_frame)
        ans_row.pack(fill="x", padx=10, pady=10)
        self.answer_entry = ttk.Entry(ans_row, textvariable=self.answer_var, font=("Segoe UI", 12))
        self.answer_entry.pack(side="left", fill="x", expand=True)
        self.btn_submit = ttk.Button(ans_row, text="Submit ↵", command=self.submit_answer)
        self.btn_submit.pack(side="left", padx=6)
        self.btn_next = ttk.Button(ans_row, text="Next", command=self.new_question)
        self.btn_next.pack(side="left")

        # Feedback
        self.feedback_lbl = ttk.Label(self, text="", font=("Segoe UI", 11))
        self.feedback_lbl.pack(fill="x", padx=16, pady=4)

        # Score & help
        stat_frame = ttk.Frame(self)
        stat_frame.pack(fill="x", **pad)

        self.score_lbl = ttk.Label(stat_frame, text="Score: 0/0 (0.0%)")
        self.score_lbl.pack(side="left")

        self.recent_lbl = ttk.Label(stat_frame, text="Recent: [· · · · ·]")
        self.recent_lbl.pack(side="right")

        help_frame = ttk.LabelFrame(self, text="Answer format tips")
        help_frame.pack(fill="both", expand=True, **pad)

        tips = (
            "• Use proper spelling where it belongs (E#, B#, Cb, Fb when required by the key).\n"
            "• For diatonic chord questions: either the root only (e.g., “F#”) or root + quality (e.g., “F# minor”) is accepted.\n"
            "• For Nashville questions, answer as the prompt requests (Roman numeral or chord name).\n"
            "• Keyboard shortcuts: N = New, Enter = Submit, R = Reveal, ] = Next, Esc = Reset score."
        )
        ttk.Label(help_frame, text=tips, justify="left").pack(anchor="w", padx=10, pady=8)

        # Keyboard shortcuts
        self.bind("<Return>", lambda e: self.submit_answer())
        self.bind("<n>", lambda e: self.new_question())
        self.bind("<N>", lambda e: self.new_question())
        self.bind("<r>", lambda e: self.reveal_answer())
        self.bind("<R>", lambda e: self.reveal_answer())
        self.bind("]", lambda e: self.new_question())
        self.bind("<Escape>", lambda e: self.reset_score())

    # ------------------ UI actions ------------------

    def _on_mode_change(self, _event=None):
        sel = self.mode_var.get()
        self.mode_label.config(text=f"Mode: {sel}")
        self.answer_var.set("")
        self.feedback_lbl.config(text="", foreground="black")

    def pick_question(self):
        sel = self.mode_var.get()
        if sel == "Random mix":
            fn = random.choice([
                ask_scale_degree,
                ask_diatonic_chord_in_key,
                ask_scale_spelling,
                ask_quality_by_degree,
                ask_nashville_to_chord,
                ask_chord_to_nashville,
            ])
        else:
            fn = QUESTION_TYPES[sel]
        return fn()

    def new_question(self):
        try:
            prompt, answer, meta = self.pick_question()
        except Exception as e:
            messagebox.showerror("Error generating question", str(e))
            return
        self.current_prompt = prompt
        self.current_answer = answer
        self.current_meta = meta
        self.prompt_lbl.config(text=prompt)
        self.answer_var.set("")
        self.answer_entry.focus_set()
        self.feedback_lbl.config(text="", foreground="black")

    def submit_answer(self):
        user = self.answer_var.get()
        if not user.strip():
            self.feedback_lbl.config(text="Enter an answer (or click Show Answer).", foreground="#b58900")
            self.answer_entry.focus_set()
            return

        self.total += 1
        ok = False
        try:
            ok = check_answer(user, self.current_answer)
        except Exception:
            ok = False

        if ok:
            self.correct += 1
            self.recent.append(True)
            self.feedback_lbl.config(text="Correct ✅", foreground="#0a7d28")
        else:
            self.recent.append(False)
            if isinstance(self.current_answer, tuple):
                primary, alt = self.current_answer
                msg = f"Not quite. Answer: {primary}  (also accepted: {alt})"
            else:
                msg = f"Not quite. Answer: {self.current_answer}"
            self.feedback_lbl.config(text=msg, foreground="#b00020")

        self._update_scoreline()

    def reveal_answer(self):
        if not self.current_prompt:
            self.feedback_lbl.config(text="No question yet. Click New Question.", foreground="#b58900")
            return
        if isinstance(self.current_answer, tuple):
            primary, alt = self.current_answer
            txt = f"Answer: {primary}  (also accepted: {alt})"
        else:
            txt = f"Answer: {self.current_answer}"
        self.feedback_lbl.config(text=txt, foreground="#0d6efd")

    def reset_score(self):
        self.total = 0
        self.correct = 0
        self.recent.clear()
        self._update_scoreline()
        self.feedback_lbl.config(text="Score reset.", foreground="#0d6efd")

    def _update_scoreline(self):
        pct = (self.correct / self.total) * 100 if self.total else 0.0
        self.score_lbl.config(text=f"Score: {self.correct}/{self.total} ({pct:.1f}%)")
        recent = "".join("✔" if x else "·" for x in self.recent)
        if not recent:
            recent = "· · · · ·"
        self.recent_lbl.config(text=f"Recent: [{recent}]")

def main():
    try:
        app = QuizApp()
        app.mainloop()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
