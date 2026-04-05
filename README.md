# Sports CV Commentary System

CSE AI/ML Computer Vision Assignment — automatic player tracking + game commentary overlay.

## Project Structure

```
Sports/
├── input/                  ← Drop raw campus videos here
│   ├── 20260402_175953.mp4   (volleyball)
│   ├── 20260402_180711.mp4   (basketball)
│   └── input.mp4             (football)
├── output/                 ← Generated annotated videos appear here
├── src/
│   ├── detector.py         YOLOv8 person + ball detection
│   ├── tracker.py          ByteTrack multi-player tracking (labels A, B, C…)
│   ├── commentary.py       Rule-based NLP commentary templates
│   ├── overlay.py          Bounding-box + animated subtitle renderer
│   ├── techslide.py        Animated tech-stack outro slide
│   ├── pipeline.py         Main entry point (single video)
│   └── event_engine/
│       ├── basketball.py   Pass / Shot / Dribble / Intercept
│       ├── volleyball.py   Serve / Spike / Set / Block / Rally
│       └── football.py     Pass / Shoot / Tackle / Dribble / Header
├── run_all.py              ← Run this to process all 3 videos at once
└── requirements.txt
```

## Quick Start

### 1. Install dependencies (once)

```powershell
py -m pip install -r requirements.txt
```

### 2. Process all 3 videos

```powershell
py run_all.py
```

Outputs will be saved as:
- `output/output_volleyball.mp4`
- `output/output_basketball.mp4`
- `output/output_football.mp4`

### 3. Process a single video

```powershell
py src/pipeline.py --input input/input.mp4 --output output/football_out.mp4 --sport football
py src/pipeline.py --input input/20260402_175953.mp4 --output output/vball.mp4 --sport volleyball
py src/pipeline.py --input input/20260402_180711.mp4 --output output/bball.mp4 --sport basketball
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| CV Framework | OpenCV 4.x |
| Detection | YOLOv8 (Ultralytics) |
| Tracking | ByteTrack (via Supervision) |
| Rendering | PIL / Pillow |
| Video I/O | OpenCV VideoCapture / VideoWriter |
| Commentary | Rule-based NLP Templates |
| Event Logic | Geometric Heuristics (sport-specific) |

## How It Works

1. **Detection** — YOLOv8 detects every `person` and `sports ball` in each frame
2. **Tracking** — ByteTrack assigns stable letter IDs (A, B, C…) across frames
3. **Event Engine** — Sport-specific geometric rules detect game events in real-time
4. **Commentary** — Events are mapped to randomized natural-language sentences
5. **Overlay** — Commentary fades in/holds/fades out as animated subtitles
6. **Outro** — A 5-second animated tech-stack slide is appended to each video

## Mapping Input Files → Sports

Edit the `SPORT_MAP` in `run_all.py` if your filenames differ:

```python
SPORT_MAP = {
    "20260402_175953.mp4": "volleyball",
    "20260402_180711.mp4": "basketball",
    "input.mp4":           "football",
}
```
