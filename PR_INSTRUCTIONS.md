# How to open the parameter-golf PR

This standalone repo (`jayyvk/parameter-golf-study`) is the canonical artifact. To submit it as a `Non-record` PR to `openai/parameter-golf`, mirror the contents into a fork of that repo under `records/track_non_record_16mb/2026-04-29_Energy_Instrumentation_5Configs/` and open the PR from there.

The convention in `track_non_record_16mb/` is exactly the same folder shape as a normal record (README.md, submission.json, plus whatever support files), and the PR title format is `Non-Record:` or `Non-record:`. Examples to read for tone/format:

- [PR #1556 — JEPA-NTP Auxiliary Losses (Negative Result)](https://github.com/openai/parameter-golf/pull/1556) — multi-file analysis, README + submission.json + scripts
- [PR #1418 — Autoresearch-Guided Optimization (100+ Experiments)](https://github.com/openai/parameter-golf/pull/1418) — methodology + negative results
- [PR #1369 — Non-record: Negative results from gated multi-order hash n-grams](https://github.com/openai/parameter-golf/pull/1369)

## Step-by-step

### 1. Fork `openai/parameter-golf` on GitHub

Go to https://github.com/openai/parameter-golf and click **Fork** (top-right). This creates `jayyvk/parameter-golf` (your personal fork).

### 2. Clone the fork locally

```bash
cd ~
git clone https://github.com/jayyvk/parameter-golf.git parameter-golf-fork
cd parameter-golf-fork
git checkout -b non-record/energy-instrumentation-2026-04-29
```

### 3. Create the submission folder and copy files

```bash
DEST="records/track_non_record_16mb/2026-04-29_Energy_Instrumentation_5Configs"
mkdir -p "$DEST"

# Copy the standalone study's content (everything except .git, PR_INSTRUCTIONS.md, .venv)
SRC="$HOME/Desktop/parameter-golf-energy-study"
cp -R "$SRC/README.md" "$DEST/"
cp -R "$SRC/submission.json" "$DEST/"
cp -R "$SRC/data" "$DEST/"
cp -R "$SRC/figures" "$DEST/"
cp -R "$SRC/scripts" "$DEST/"
cp -R "$SRC/notebook" "$DEST/"

# Sanity check
ls -R "$DEST" | head -30
du -sh "$DEST"   # should be ~1.5 MB
```

### 4. Commit and push to your fork

```bash
git add records/track_non_record_16mb/2026-04-29_Energy_Instrumentation_5Configs/
git status   # double-check only new files under that path are staged

git commit -m "Non-record: Energy instrumentation across 5 leaderboard configs (8xH100 SXM5)

Reproduced 5 published submissions and added NVML hardware-counter energy
measurement (matcha 0.3.1). Found 14x spread in Wh per 0.001 BPB-drop
between most/least efficient techniques at identical wallclock budget;
post-training overhead 5-31% of total energy invisible to leaderboard.

Standalone study repo: https://github.com/jayyvk/parameter-golf-study"

git push -u origin non-record/energy-instrumentation-2026-04-29
```

### 5. Open the PR on GitHub

Either:

**Via the link printed in the `git push` output** — GitHub usually prints a one-click PR-create URL. Click it.

**Or via the `gh` CLI:**

```bash
gh pr create \
  --repo openai/parameter-golf \
  --base main \
  --head jayyvk:non-record/energy-instrumentation-2026-04-29 \
  --title "Non-record: Energy as the missing leaderboard axis — Wh per BPB-drop across 5 configs" \
  --body "$(cat <<'EOF'
## Summary

This PR adds a non-record submission under:

\`records/track_non_record_16mb/2026-04-29_Energy_Instrumentation_5Configs/\`

Reproduced 5 published parameter-golf submissions on 8xH100 SXM5 and added NVML hardware-counter energy measurement (via [matcha](https://github.com/keeyalabs/usematcha) 0.3.1, but the methodology works with DCGM, codecarbon, or any NVML-based tool). The val_bpb leaderboard ranks on training quality — this PR adds the energy axis the leaderboard does not currently track.

## Results (5 SP1024 configs, 8xH100 SXM5)

| # | Config | val_bpb | kWh | Wh per 0.001 BPB-drop vs baseline |
|---|---|---:|---:|---:|
| 1 | NaiveBaseline | 1.2255 | 1.090 | — (anchor) |
| 2 | SlidingWindow eval | 1.1926 | 1.035 | **−1,658 Wh** *(free win)* |
| 3 | LoRA TTT | 1.1923 | 0.996 | **−2,808 Wh** *(free win)* |
| 4 | 11L EMA + GPTQ-lite | 1.1233 | 1.105 | **+147 Wh** |
| 5 | ParResid + Mini Depth Recurrence | 1.1055 | 1.346 | **+2,137 Wh** *(14× less efficient)* |

Reproduced \`val_bpb\` matches each submission's published number within ±0.0011 BPB (well below the 0.005-nat leaderboard significance threshold).

## Three things only visible with energy data

1. **The wallclock cap normalizes total training energy** (training-phase Wh spans only 919–1033 across all 5 configs — 12% spread). What varies is **post-training overhead**, ranging 5–31% of total run energy. The published "training time" metric understates true compute cost by 5–117%.

2. **Two configs hit the same loss (~1.193) at near-identical total energy via completely different mechanisms** — sliding-window eval (richer context per scored token) vs LoRA TTT (per-document fine-tuning). The intuition that "TTT is expensive" did not hold at parameter-golf scale.

3. **The frontier is 14× less energy-efficient than the engineering tier** per unit of BPB-improvement. Going from 1.123 → 1.105 (the SP1024 ceiling) costs 14× more per unit of improvement than getting from 1.226 → 1.123. Diminishing returns are real and quantifiable.

## Submission contents

- \`README.md\` — full writeup with figures
- \`submission.json\` — structured metadata (configs measured, energy/loss ranges, hardware, caveats)
- \`data/matcha/*.jsonl\` — raw matcha records (5 files; per-step + session_end)
- \`data/pg_logs/*.txt\` — each submission's own training log (5 files; includes post-quant val_bpb)
- \`data/hardware.txt\` — \`nvidia-smi -L\` + matcha --version + per-GPU TDPs
- \`scripts/analyze.py\` — regenerates the merged comparison table from raw data
- \`scripts/plot.py\` — regenerates all 5 figures from raw data
- \`notebook/report.ipynb\` — Jupyter notebook used for chart iteration
- \`figures/*.png\` — 5 PNGs (hero, trajectories, efficiency bars, train vs post-train stacked, per-GPU deviation)

Total artifact size: ~1.5 MB. No model weights — methodology + measurements only. Standalone repo for verification: https://github.com/jayyvk/parameter-golf-study

## Test plan

- [x] Reproduced val_bpb within ±0.0011 of each submission's published number on all 5 configs
- [x] Energy measurement: NVML hardware counter (no sampling artifact on totals); per-GPU breakdown verified
- [x] Single-seed runs (caveat noted; 3-seed re-validation would be ~$75 of additional compute)
- [x] Scripts (\`analyze.py\`, \`plot.py\`) regenerate table + figures from raw JSONL/.txt
- [x] No code changes to any submission's \`train_gpt.py\` — wrap-only instrumentation

## Caveats (also in README)

- Single-seed (vs published 3-seed means)
- Config #4 artifact 17.05 MB > 16 MB cap (zlib vs zstd-22 mismatch; loss measurement unaffected)
- No SP4096/SP8192 configs (HF dataset mirrors had been depopulated; SP1024 ceiling ~1.10 is the lowest reproducible point here)
- GPU 2 was a hardware-level laggard on this specific Runpod allocation
EOF
)"
```

### 6. After the PR is open

- Watch for maintainer feedback. They may ask for restructuring (e.g. moving things into a different subdir), tightening the README, or splitting the PR.
- Keep the standalone repo as the source of truth — link to it from the PR body so reviewers can see the data outside the parameter-golf monorepo.
- If the PR is rejected or sits unreviewed, the standalone repo still stands on its own. The standalone publication is the primary artifact; the PR is a way to surface it to the parameter-golf community.

---

## Common gotchas

- **Don't forget the `2026-04-29_` prefix** — folder naming convention in `track_non_record_16mb/` is `YYYY-MM-DD_Descriptive_Name`. Don't deviate.
- **Don't include `.venv`, `.git`, `__pycache__`, or `PR_INSTRUCTIONS.md`** in the copied folder. Only the artifact files.
- **Don't change anyone else's files** in the parameter-golf repo. The PR should only add files under the new folder. If `git status` shows changes outside that folder, undo them.
- **The PR title MUST start with `Non-record:` or `Non-Record:`** — that's how reviewers filter records vs non-records.
- **Total folder size should be ~1–2 MB.** If it balloons, check whether you accidentally included raw model weights or notebook checkpoints.
