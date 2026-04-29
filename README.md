# Energy as the missing leaderboard axis

**A study of 5 parameter-golf leaderboard submissions, instrumented with [matcha](https://github.com/keeyalabs/usematcha) on 8×H100 SXM5.**

[parameter-golf](https://github.com/openai/parameter-golf) is OpenAI's compute-capped LLM-training challenge: train a small model under a 16 MB compressed-artifact budget, in 600 seconds wallclock on 8×H100 SXM, and beat the previous `val_bpb` score by ≥0.005 nats. The leaderboard ranks on `val_bpb` alone.

This study replicates 5 published submissions and adds the energy axis the leaderboard does not track. We measure with NVML hardware counters via matcha; loss numbers come from each submission's own training script.

**Headline finding:** when the wallclock is capped, total energy is essentially a constant (~1.0–1.35 kWh per run on 8×H100 SXM5). The interesting variation is what techniques *do* with that energy budget — and the spread between "free win" eval-only changes and "frontier" architectural work is **14× in energy per unit of `val_bpb` improvement**.

![Hero chart — val_bpb vs total kWh](figures/01_hero_loss_vs_energy.png)

---

## The data, in one table

| # | Config | params | steps | `val_bpb` | Δbpb | kWh | duration | Wh/Mtok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [NaiveBaseline](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-17_NaiveBaseline) | 17.1 M | 13,793 | **1.2255** | anchor | 1.090 | 807 s | 0.151 |
| 2 | [SlidingWindowEval](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_SlidingWindowEval) | 17.1 M | 13,766 | **1.1926** | −0.0329 | 1.035 | 773 s | 0.143 |
| 3 | [LoRA TTT](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-17_LoRA_TTT) | 17.1 M | 13,339 | **1.1923** | −0.0332 | 0.996 | 734 s | 0.143 |
| 4 | [11L EMA + GPTQ-lite + warmdown3500](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233) | 27.0 M | 7,034 | **1.1233** | −0.1022 | 1.105 | 888 s | 0.200 |
| 5 | [ParResid + Mini Depth Recurrence](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence) | 30.2 M | 6,310 | **1.1055** | −0.1200 | 1.346 | 1300 s | 0.271 |

All 5 trained for the full 600 s wallclock cap on 8×H100 SXM5 (700 W TDP per GPU). Reproduced `val_bpb` matches each submission's published number within run-to-run noise.

---

## Five findings

### 1. The wallclock cap is a hard equalizer for *training* energy.

Every submission hits the 600-second cap, so per-step energy and total *training* energy converge regardless of model size. Training-phase energy spans only **919–1033 Wh** (12% spread) across all 5 configs.

The dynamic range lives elsewhere — in **post-training overhead**, which varies from **5% to 31%** of total run energy.

![Training vs post-training energy](figures/04_train_vs_post_train_energy.png)

The published "training time" metric (600 s) understates the true compute cost by **5% (baseline) to 117% (config 5)**. For the SP1024 frontier config (#5), nearly half the GPU-time happens *after* "training" formally ends — quantization calibration, sliding-window eval, serialization.

### 2. Two free wins: SlidingWindow eval and LoRA TTT both reduce loss AND energy vs baseline.

The improvement from `1.2255 → 1.193` (configs #2 and #3) cost **less** total energy than the baseline — because both run their post-quantization eval more efficiently than the baseline does (and both fit slightly fewer training steps within the 600 s cap, balancing out).

![Energy efficiency vs baseline](figures/03_energy_efficiency.png)

These are the only two "free wins" in the dataset. Everything below `1.193` requires real architectural work and burns more total energy.

### 3. Matched-loss A/B: SlidingWindow vs LoRA TTT.

Configs #2 and #3 land at essentially identical `val_bpb` (1.1926 vs 1.1923) via completely different mechanisms — sliding-window eval scoring tokens with richer context vs. per-document LoRA test-time training. Total energy is also nearly identical (1.035 vs 0.996 kWh). The intuition that "TTT is expensive" did not hold at parameter-golf scale.

The mechanisms reveal themselves in the eval-time energy split:
- **SlidingWindow eval**: stride=64 means each token is forward-passed ~16× → **72.6 s** of full-power eval
- **LoRA TTT**: per-document rank-8 LoRA updates are cheap → **63.6 s** of eval

Both routes to the same loss cost roughly the same energy, but they look very different in fine-grained per-step traces.

### 4. The frontier is 14× less energy-efficient than the engineering tier.

| | Config | Wh per 0.001 BPB-drop vs baseline |
|---|---|---:|
| | 4. 11L EMA+GPTQ | **147 Wh** |
| | 5. ParResid+MiniDR | **2137 Wh** |

Going from `1.123 → 1.105` (the last 0.018 BPB at the SP1024 ceiling) costs **14× more per unit of improvement** than going from `1.226 → 1.123`. Diminishing returns are real and quantifiable. The SP1024 leaderboard ended around `1.10` for exactly this reason — every entry below that switched to SP4096+ vocabularies, where the energy/loss curve resets.

### 5. The post-training tail is dominated by AR self-generated GPTQ calibration.

Config 5's 577-second "other" phase (between training-end and exit) is mostly autoregressive GPTQ calibration. From `data/pg_logs/lb_par_resid_dr.txt`:

```
gptq:generating autoregressive calibration data (64 seqs x 2048 tokens, temp=0.80)...
gptq:generated 64 sequences in 245.9s
gptq:collecting hessians from autoregressive data...
gptq:done in 247.0s
wallclock:post_gptq total_elapsed:916.1s train_budget:600.0s
```

That's **~8 minutes of the model autoregressively generating its own calibration data** — a real compute cost invisible to any "training time" metric. matcha sees it because NVML keeps integrating energy until the wrapped process exits.

---

## Methodology sidebar: GPU 2 is the laggard in 5/5 runs.

![Per-GPU energy deviation from per-config median](figures/05_per_gpu_deviation.png)

Across all 5 configs, GPU index 2 consistently consumes 0.5–3.2% less energy than the per-config median — a hardware-level pattern (likely binned/throttled silicon on this Runpod allocation), not algorithmic.

The straggle is most pronounced in **config 3 (LoRA TTT)** at 4.4% spread — TTT's per-document workload has more variance than uniform training-step work, so existing GPU asymmetries are amplified. Config 4 has the *tightest* spread (0.74%) — its larger model + larger batch better saturates per-GPU work, so per-GPU asymmetries are amortized.

This is observability matcha gives you "for free" via per-GPU NVML counters. Without it, the only signal you have is `nvidia-smi` snapshots — which can hide sustained 30 W per-GPU gaps perfectly well.

---

## Hardware and software

- **8× NVIDIA H100 80GB HBM3 SXM5** (700 W TDP per GPU; confirmed via `nvidia-smi --query-gpu=name,power.max_limit`)
- **Runpod pod 064ce247e933**, NVIDIA driver 580.126.09, parameter-golf official template
- **matcha 0.3.1** ([keeyalabs/usematcha](https://github.com/keeyalabs/usematcha)) — adds two regex patterns over 0.3.0 for parameter-golf's bare `N/total train_loss:` step format and TTT per-chunk progress lines (`tttg: cN/total`)
- **NVML energy source: hardware counter** (`nvmlDeviceGetTotalEnergyConsumption`), 100 ms polling for power, no sampling artifact on the energy total

Full hardware metadata: [`data/hardware.txt`](data/hardware.txt).

---

## Reproducing

```bash
# Run from this repo's root
python3 scripts/analyze.py     # prints the merged table
python3 scripts/plot.py        # writes the 5 PNGs to figures/
```

Both scripts read from `data/matcha/*.jsonl` (matcha's session_end + per-step records) and `data/pg_logs/*.txt` (the parameter-golf training scripts' own log files; matcha doesn't capture those because they're written directly to disk, not stdout).

To regenerate the data on a fresh pod, the parameter-golf submission directories under `records/track_10min_16mb/` ship their own `train_gpt.py`. Wrap each with:

```bash
matcha wrap \
  --run-id <run_id> \
  --label config=<config_name> \
  --output runs/<run_id>.jsonl \
  -- torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Specific env-var overrides per submission are captured in each submission's README and re-quoted in [`notebook/report.ipynb`](notebook/report.ipynb).

---

## Caveats and known limitations

- **Single-seed runs.** Each config was reproduced with one seed (matching the seed each submission's README defaults to). Published leaderboard numbers are 3-seed means. Our `val_bpb` matches published values within ±0.0011 BPB — well below the leaderboard's 0.005-nat significance threshold. Energy figures are reproducible to ~5–10% on identical hardware (Runpod allocation variance dominates).
- **Config 4 artifact size: 17.05 MB (over the 16 MB cap).** The published submission used `zstd-22` compression; our run defaulted to `zlib`. The `val_bpb` measurement is unaffected (compression is post-training; only changes disk size). Methodology footnote, not a measurement issue. Loss matches published value to 5 decimal places.
- **No SP4096 / SP8192 configs.** The HuggingFace data mirrors that hosted larger-vocabulary FineWeb shards (per the published submission READMEs) had been depopulated by the time of this study. SP1024 ceiling is ~1.10 BPB; reaching the current SOTA (~1.06) requires SP8192 datasets we couldn't materialize. The SP1024 frontier config (#5) is the lowest reproducible point in this study.
- **GPU 2 straggle is hardware-level**, observed across all 5 runs on the same Runpod pod. A different allocation likely gives a different straggle pattern (or none).
- **matcha `wrap --output FILE` silenced training stdout** in 0.3.1. The fix is patched in `examples/`-adjacent worktree code but unbuilt — slated for matcha 0.3.2. Did not affect data quality (parameter-golf logs to its own files in parallel), just made live monitoring impossible during the runs.

---

## What's next

- **Add SP8192 SOTA** when the dataset becomes accessible from a public mirror, or via local raw-FineWeb regeneration through `prepare_caseops_data.py` from the SOTA submission folder. That would extend the curve from `1.10 → 1.06` and triple the loss-axis dynamic range without changing the energy axis much (still ~1.3 kWh per run).
- **Multi-seed runs** for tighter error bars on energy (3 seeds × ~$5/run ≈ $75 for full re-validation).
- **Add the SOTA-frontier energy efficiency point.** The hypothesis is that the slope continues — frontier configs cost an order of magnitude more energy per unit of BPB improvement than mid-tier engineering work.
- **Propose energy as a tracked leaderboard metric** (issue, not PR) on `openai/parameter-golf`.

---

## Repo layout

```
parameter-golf-energy-study/
├── README.md                    ← this file
├── data/
│   ├── matcha/                  ← 5 matcha JSONLs (per-step + session_end records)
│   ├── pg_logs/                 ← 5 parameter-golf training log .txt files
│   └── hardware.txt             ← nvidia-smi -L + matcha --version + power limits
├── notebook/
│   └── report.ipynb             ← Jupyter notebook used to iterate on charts
├── scripts/
│   ├── analyze.py               ← prints the merged comparison table
│   └── plot.py                  ← regenerates all 5 figures from JSONL + .txt
└── figures/
    ├── 01_hero_loss_vs_energy.png           ← article opener
    ├── 02_loss_vs_cumulative_energy.png     ← training trajectories
    ├── 03_energy_efficiency.png             ← Wh per 0.001 BPB-drop bars
    ├── 04_train_vs_post_train_energy.png    ← stacked energy share
    └── 05_per_gpu_deviation.png             ← GPU 2 straggle sidebar
```

---

## License & attribution

This study reproduces published [parameter-golf](https://github.com/openai/parameter-golf) submissions; full credit to the original authors of each submission (links in the data table above). The matcha instrumentation tool: [keeyalabs/usematcha](https://github.com/keeyalabs/usematcha) (Apache-2.0). This repository's own contributions (analysis scripts, README, figures) are released under the same terms.
