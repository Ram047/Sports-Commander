"""
run_all.py — Batch-process all 3 sports videos automatically.

Maps each input filename → sport type automatically. Edit SPORT_MAP below
if your filenames differ from what's listed.
"""

import os
import sys
import subprocess

# ── EDIT THIS IF YOUR FILENAMES ARE DIFFERENT ──────────────────────────────
# Map: filename (without path) → sport name
SPORT_MAP = {
    "20260402_175953.mp4": "volleyball",
    "20260402_180711.mp4": "basketball",
    "input.mp4":           "football",
}
# ───────────────────────────────────────────────────────────────────────────

INPUT_DIR  = os.path.join(os.path.dirname(__file__), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("  Sports CV Commentary System -- Batch Runner")
print("=" * 60)

success, failed = [], []

for filename, sport in SPORT_MAP.items():
    input_path  = os.path.join(INPUT_DIR, filename)
    output_name = f"output_{sport}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    if not os.path.isfile(input_path):
        print(f"\n  [SKIP] Skipping {filename} - file not found.")
        failed.append(filename)
        continue

    print(f"\n  [START] Processing: {filename} -> [{sport}]")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "src", "pipeline.py"),
        "--input",  input_path,
        "--output", output_path,
        "--sport",  sport,
        "--conf",   "0.25",
    ]

    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode == 0:
        success.append(output_name)
    else:
        failed.append(filename)

print("\n" + "=" * 60)
print(f"  [DONE] Completed: {len(success)} video(s)")
for f in success:
    print(f"      -> output/{f}")
if failed:
    print(f"  [FAIL] Failed  : {len(failed)} video(s)")
    for f in failed:
        print(f"      -> {f}")
print("=" * 60 + "\n")
