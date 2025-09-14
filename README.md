# Fretboard Tool

A zero-dependency Python CLI that renders guitar fretboards with scales, pentatonics, chord tones, and CAGED windows. High E is at the top, low E (wound strings) at the bottom. Useful for visualizing theory, practicing chord tones, and connecting scales to the neck.

---

## Features

- Major and natural minor scales (properly spelled, e.g. C# major → E#, B#).
- Pentatonics (major/minor, auto-selects from mode or force with flag).
- Chord tones by degree (`--degree I … vii°`) with triad or 7th overlays.
- CAGED windows (`--boxes caged`) for five overlapping practice positions.
- Clean output: bold nut, fret numbers, standard fret markers, Unicode or ASCII.
- Pure Python 3, no dependencies.

---

## Installation

Clone or copy the script. Requires Python 3.7+.

```bash
python fretboard_tool.py --key C --mode major --show scale
```

On Windows PowerShell:

```powershell
python .\fretboard_tool.py --key C --mode major --show scale
```

---

## Usage

```bash
python fretboard_tool.py --key <KEY> [options]
```

### Options

| Flag | Description | Values / Defaults |
|------|-------------|-------------------|
| `--key` | **Required.** Key root (e.g. C, F#, Bb, Db, C#). | — |
| `--mode` | Scale mode. | `major` (default), `natural_minor` |
| `--show` | What to display. | `scale` (default), `pent` |
| `--pent` | Pentatonic flavor (if `--show pent`). | `auto` (default), `major`, `minor` |
| `--degree` | Highlight diatonic chord tones (Roman numerals). | e.g. `I`, `ii`, `IV`, `vii°` |
| `--tones` | Chord tones to show (with `--degree`). | `all` (default = seventh), `triad`, `seventh` |
| `--boxes` | Show five CAGED windows. | `caged` |
| `--frets` | Number of frets to show. | Default `12`, max `24` |
| `--ascii` | Use plain ASCII instead of Unicode. | Flag only |

---

## Examples

**Full C major scale (12 frets)**  
```bash
python fretboard_tool.py --key C --mode major --show scale
```

**C major pentatonic (auto)**  
```bash
python fretboard_tool.py --key C --mode major --show pent
```

**A natural minor pentatonic (explicit)**  
```bash
python fretboard_tool.py --key A --mode natural_minor --show pent --pent minor
```

**IV triad tones in E major**  
```bash
python fretboard_tool.py --key E --mode major --degree IV --tones triad
```

**V7 tones in G major**  
```bash
python fretboard_tool.py --key G --mode major --degree V --tones seventh
```

**Five CAGED windows for D major**  
```bash
python fretboard_tool.py --key D --mode major --show scale --boxes caged
```

**ASCII safe output**  
```bash
python fretboard_tool.py --key G --mode natural_minor --show pent --pent minor --ascii
```

---

## Legend

- **R** = root  
- **2, 3, 4, 5, 6, 7** = scale degrees  
- **b3 / ♭3, b7 / ♭7** = flattened 3rd / 7th  
- Bold nut (`║` or `||`), fret markers (3, 5, 7, 9, 12, …)  
- Top row = string 1 (high E), bottom row = string 6 (low E)

---

## Roadmap

- Strict CAGED shape windows (canonical fret spans).  
- Arpeggio overlays (`--arpeggio DEGREE`).  
- Harmonic/melodic minor and other scales.  
- Alternate tunings.  
- Export to PNG/PDF.  
- Interactive UI.

---

## License

MIT
