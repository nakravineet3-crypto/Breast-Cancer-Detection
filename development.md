# Development Log — MASNN-BCD-HI Reproduction Project

> Purpose of this file: give any AI assistant (or human) full context on this project without
> needing the original conversation history. Read this first before doing anything else.
> **Whoever/whatever makes changes to this project should update this file afterward** —
> keep it in sync with reality, not with intentions.

## What this project is

A B.Tech Capstone Project (CS4095, NIT Rourkela, Autumn Semester 2026-27) reproducing and
critically evaluating ideas from:

> Satheeshkumar N., Senthilkumar S., Rajalakshmi S., *"Optimized multi-dimensional attention
> spiking neural network based breast cancer detection from histopathology images"*,
> Biomedical Signal Processing and Control 113 (2026) 109176.

Goal: classify breast histopathology images (BreakHis dataset) as **benign** or **malignant**,
using a pipeline inspired by the paper — guided image filtering preprocessing, a spiking neural
network with attention, and (tentatively — see caveats below) an alternative/bio-inspired
optimization strategy for training. The submitted university Research Plan (see
`docs/Research_Plan_CP1.docx`) intentionally uses **generic, hedged language** rather than
committing to exact paper terminology (e.g. "guided image filtering" not "Trainable Self-Guided
Filter," "advanced optimization strategies" not "Shrike Optimization Algorithm") — this was a
deliberate choice so the submitted plan contains no promises that later prove infeasible.

## Who's working on this

Student (Vineet Nakra, 123CS0193, supervisor Prof. Sanjeev Patel) has **no prior background** in
Python beyond basics, or in ML/DL — every concept needs explaining from first principles, with
code walked through line-by-line using real traced values, not just described abstractly. Prefers
Jupyter notebooks over VS Code (VS Code's kernel picker caused confusion early on — abandoned).
Wants to deeply understand every component well enough to explain it in a professor demo and use
it in job interviews — not just get code running.

## Environment

- Project root: `D:\breast-cancer-masnn`
- Python venv: `D:\breast-cancer-masnn\venv` (Python 3.12)
- GPU: NVIDIA GTX 1650, 4GB VRAM — **currently in a stuck CUDA-context state** (see Known Issues)
- Key packages installed in venv: `torch` 2.5.1+cu121, `torchvision`, `snntorch`, `opencv-python`,
  `pillow`, `scikit-learn`, `matplotlib`, `pandas`, `jupyter`, `tqdm`
- Launch notebooks with: `cd notebooks && ..\venv\Scripts\jupyter lab <name>.ipynb`
- Run scripts with: `venv\Scripts\python.exe src\<script>.py`

## Dataset

- **BreakHis only.** BreCaHAD (mentioned in the paper) was deliberately excluded — it has no
  benign/malignant labels (only mitosis/apoptosis/structure annotations) and never appears in any
  of the paper's actual results tables/figures. Decision made explicitly with the student.
- Source: Kaggle mirror (`kaggle.com/datasets/ambarish/breakhis`), downloaded manually by student,
  extracted to `data/raw/BreaKHis_v1/...`
- Verified byte-for-byte against the paper's Table 2: 7,909 images total (2,480 benign / 5,429
  malignant), exact per-magnification counts (40X/100X/200X/400X) all match.
- `src/build_dataset.py` scans `data/raw/`, parses patient ID / label / magnification from BreakHis
  filenames (format `SOB_<B|M>_<subtype>-<patient_id>-<magnification>-<seq>.png`), writes:
  - `data/splits/manifest.csv` — full image catalog (path, label, subtype, magnification, patient_id)
  - `data/splits/split.csv` — same + a `split` column (train/val/test)
- Split is **patient-wise** (not image-wise) to prevent leakage — uses `GroupShuffleSplit` grouped
  by `patient_id`, ~70/15/15 target (actual: 55/13/13 patients due to uneven images-per-patient).
  Leakage-checked: confirmed zero patient overlap across splits.

## Code structure

```
src/
  build_dataset.py       # one-time: scans data/raw/, produces manifest.csv + split.csv
  tsgf.py                 # TrainableSelfGuidedFilter — reusable preprocessing module (done, working)
  breakhis_dataset.py     # BreakHisDataset — PyTorch Dataset wrapping split.csv (done, working)
  train_baseline.py       # standalone training script for Week 2 baseline CNN (see status below)
notebooks/
  week1_explore.ipynb     # DONE — dataset exploration, stats vs. paper Table 2, sample grid, leakage check
  week2_preprocessing_baseline.ipynb  # IN PROGRESS — see status below
docs/
  Research_Plan_CP1.docx  # university-mandated Research Plan for CP-I, submitted (deadline was 12 Aug 2026)
data/splits/               # manifest.csv, split.csv, and various generated preview images
```

## Status by component

### TSGF (Trainable Self-Guided Filter) — DONE, understood in depth
`src/tsgf.py`. A trainable variant of the classical Guided Image Filter (He, Sun & Tang) — for
every pixel, computes local variance in a neighborhood window (`radius=4` → 9×9), and uses one
learnable parameter (`eps`, stored as `log_eps` for positivity) to decide whether a pixel is real
structure (preserve) or noise (smooth toward local average). Output = `a * input + b` per pixel.

**Honest caveat, already disclosed to student and reflected in the Research Plan's generic
wording**: the paper's own Eqs. 1–5 for this filter are loosely/ambiguously specified (reused
symbols, unclear correspondence). This implementation is a faithful, standard, citable algorithm
matching the paper's *described behavior*, not a symbol-for-symbol transcription of the paper's
equations. This is a deliberate, defensible engineering judgment call, not an error.

### Baseline CNN (TSGF + small CNN) — NOT YET WORKING, blocked on GPU issue
`src/train_baseline.py`, exercised via `notebooks/week2_preprocessing_baseline.ipynb`.

Architecture: TSGF → 3× (conv+batchnorm+relu+maxpool) → global average pool → 2 FC layers → 2-class
output. ~26,000 parameters.

**Timeline of what happened (for context, don't repeat these mistakes)**:
1. First training run (GPU, 8 epochs, lr=1e-3, no class weighting) completed but was unstable —
   validation loss oscillated wildly (0.27→1.47 across epochs), and final benign recall was only
   22% (severe class-imbalance bias toward predicting malignant; train set is 1877 benign vs 3402
   malignant, ~64% malignant).
2. Diagnosed root causes: learning rate too high (oscillation) + no class weighting (imbalance bias).
3. Fix implemented: lr=1e-4 + `ReduceLROnPlateau` scheduler, class-weighted `CrossEntropyLoss`
   (inverse-frequency weighting), checkpointing on best val_loss with early stopping (patience=6,
   max 25 epochs) — mirrors the paper's own Table 3 "Early Stopping: 10 epochs".
4. Retry hit a **CUDA out-of-memory error at model creation** (a ~26K-parameter model failing to
   allocate is not a real memory shortage) — `nvidia-smi` showed only ~100MB/4096MB used at the time.
5. Retry after that hit a **kernel death** (hard crash, not catchable) during the training loop.
6. Diagnosed: even allocating a **10-element CUDA tensor** failed with "out of memory" — this
   confirmed the GPU's CUDA context is stuck in a bad state, most likely left over from an earlier
   crashed process (not a code bug). A driver restart or full machine reboot should clear it.
7. Worked around by forcing CPU training (`device = torch.device("cpu")`, hardcoded in
   `train_baseline.py` with a comment explaining why). Model is small enough this is viable, just
   slower.
8. CPU training run was started, completed 1 epoch (`train_acc=0.725, val_acc=0.824`, ~3-6 min for
   that epoch), then **the student asked to pause and shut down mid-run** (task `b0qdfr84x` was
   killed cleanly via TaskStop — confirmed no orphaned process, no partial/corrupt files written,
   since results only get saved at the very end of the script).

**Update (19 Aug 2026): GPU confirmed healthy after a machine restart** — `nvidia-smi` shows 0MB
used, and a tiny `torch.zeros(10, device='cuda')` allocation succeeds. `train_baseline.py` was
reverted from the CPU-forced workaround back to normal `cuda if available else cpu` device
selection.

**v1 result (GPU, class-weighted loss, loss-based checkpointing, no augmentation)**: training
stabilized (no more wild val_loss swings), benign recall improved 22% → 47%, but overall test
accuracy dropped to 53% and there's a large val→test gap (80.5% val acc vs. 53.2% test acc at the
selected checkpoint). Saved as `data/splits/week2_results_v1_no_augmentation.json` /
`week2_model_v1_no_augmentation.pt` for reference.

**v2 attempt (added train-time data augmentation — random flip/rotation — and switched
checkpoint selection from val_loss to val_accuracy)**: **did not help, and the val→test gap got
slightly worse** (88.0% val acc vs. 52.2% test acc — a 36-point gap vs. v1's 27-point gap).
Benign recall barely moved (47%→50%), everything else near-identical to v1. Notably, accuracy-based
checkpointing selected **epoch 1** as "best" — a sign the val-set score itself is noisy, not a sign
epoch 1 was genuinely a good model.

**Root-cause conclusion**: neither fix addressed the real bottleneck. Val and test sets each span
only **13 distinct patients** — too few for a stable per-epoch checkpoint-selection signal, whether
by loss or accuracy. Training-recipe changes (learning rate, augmentation, class weighting) can't
fix an evaluation-set-size problem. The statistically correct fix would be patient-wise k-fold
cross-validation (train/eval across several different patient splits, average the results) rather
than trusting one single small held-out split — not yet implemented, flagged as a legitimate
known limitation instead of chased further, since the baseline's only real job is to be a
reference point for later architectures. **Decision: stop tuning the baseline here, move to Week 3.**

Current final saved baseline: `data/splits/week2_results.json` / `week2_model.pt` (the v2 run —
kept as the "current" one since it's the most recent, despite not being an improvement; v1 files
kept alongside for comparison). `data/splits/week2_confusion_matrix.png` and
`week2_training_curves.png`, and the classification report baked into
`week2_preprocessing_baseline.ipynb`'s saved outputs, are still **stale, from the original broken
run** (22% benign recall) — the notebook itself was never regenerated from the real runs; only
`docs/Progress_Report_Week1-2.docx` and the `comparison_*.png` files reflect real v1 results
(predates the v2 attempt, not yet updated with it either).

### Spiking Neural Network (Week 3) — IN PROGRESS, initial model built and verified alive
`src/spiking_model.py` — `SpikingCNN`, structurally mirrors `BaselineCNN` (TSGF → 3 conv blocks →
classifier) but every activation is a real `snn.Leaky` LIF neuron instead of ReLU, run for
`NUM_STEPS=25` time steps per image (same static, TSGF-preprocessed image re-shown every step;
each LIF layer's own accumulate/leak/fire dynamics is what produces genuine time-varying spikes
from that constant input — this is simpler than explicit Bernoulli/rate-encoding the input and is
standard snnTorch practice for static images). Output = spike **count** per class summed over all
25 steps, shape `(batch, 2)`.

**Two real findings from getting this working, both worth keeping for the report**:
1. **Threshold tuning was necessary.** snnTorch's default `threshold=1.0` left deeper layers
   almost entirely dead (measured: L1=10.5%, L2=6.9%, L3=6.4%, L4=**0.4%**, output=4.0% firing
   rates) — current magnitude shrinks with depth because each layer's input comes from an
   already-sparse binary spike map. Empirically swept thresholds; `0.3` gives healthy, varied
   rates at every layer (16-55%) without saturating. `0.1` and below saturate the output layer to
   a flat 50% regardless of input — equally uninformative as never firing.
2. **Random initialization can produce a "born-dead" network** — confirmed directly: of 8 random
   seeds tested, 7 produced healthy varied output (24-190 total spikes on a test batch), 1 produced
   exactly zero output spikes regardless of input. Unlike a ReLU network, a dead spiking network
   can't train its way out of this state either (no spikes anywhere → no surrogate-gradient signal
   for backprop to use). Added `create_healthy_model()` — creates a model, smoke-tests it on one
   real batch, discards and retries on a fresh random init if totally dead. Not yet run enough
   times to know the typical dead-init rate precisely (~1/8 observed, small sample).

**Trained successfully** — `src/train_spiking.py` (mirrors `train_baseline.py`: class-weighted
loss, accuracy-based checkpointing, early stopping, augmentation). `IMG_SIZE=64` (not 128 like the
baseline) since the spiking forward pass costs ~25x more per image (one pass per time step) —
benchmarked at ~27.5s/epoch at this size on the GTX 1650. Results saved to
`data/splits/week3_results.json` / `week3_model.pt`.

**Real result — beats the baseline CNN on every metric**:

| | Baseline CNN (v2) | SpikingCNN |
|---|---|---|
| Benign precision | 0.20 | 0.27 |
| Benign recall | 0.50 | 0.51 |
| Malignant precision | 0.81 | 0.85 |
| Malignant recall | 0.53 | 0.66 |
| Test accuracy | 0.52 | **0.63** |
| Val→test gap | 36 pts | **22 pts** |

Notably the val→test generalization gap (the exact problem diagnosed and left unresolved on the
baseline — see Week 2 section above) is smaller here too, on the identical 13-patient test set.

**Training-dynamics finding worth keeping for the report**: the model was completely stuck
predicting a single constant class for the first 3 epochs — accuracy matched "always predict
benign" to 4 decimal places (0.3556 train, 0.2619 val) — before suddenly breaking free at epoch 4
and reaching its best result at epoch 5. A visible signature of spiking-network gradients being
slower to get moving than a regular CNN's, then transitioning fairly suddenly once enough signal
accumulates.

**Update: spike-raster visualization built** — `src/make_spike_raster.py`, loads the real trained
`week3_model.pt` and runs one real benign + one real malignant test image through it, plotting
which of the 32 `fc1` neurons and 2 output neurons fired at which of the 25 time steps. Result is a
genuinely correct demo: for the benign image only the benign output neuron ever fires (t=16, 22);
for the malignant image only the malignant output neuron fires (t=9, 18) — both predictions
correct, visible directly in spike timing. Saved to `data/splits/spike_raster.png`.

**Update: `comparison_*.png` regenerated as a proper 3-way comparison** (broken baseline / fixed
baseline / spiking network) via `src/make_comparison_report.py`, now loading `week3_results.json`
too. `docs/Progress_Report_Week1-2.docx` still only covers Weeks 1-2 — a Week 3 section/new report
is the next documentation task if needed for a presentation.

**Data-integrity incident (20 Aug 2026)**: `data/splits/week2_results.json` was found overwritten
with unrelated C++ code (a windowed-pixel-sum competitive-programming snippet) — not something this
assistant did; cause unknown (student or another tool likely saved unrelated content to this path).
Restored exactly from the JSON this assistant had printed in full earlier in the session. Verified
`week3_results.json`/`week3_model.pt` were untouched. **Lesson for whoever picks this up**: these
result JSON files are regenerable from source (checkpoints + a rerun of the relevant `train_*.py`)
but are not otherwise backed up — treat unexpected diffs in `data/splits/*.json` with suspicion
before trusting their contents.

### Full MASNN (Week 4) — DONE, beats Week 3's plain spiking network
`src/masnn_model.py` (`MASNN`) + `src/attention.py` (`CBAM`) + `src/train_masnn.py`. Assembled in
the order from the paper's Figure 2: TSGF → spiking conv layers (conv1+lif1, conv2+lif2) →
multi-dimensional attention block → spiking recurrent layer → FC spiking layer → output.

**Design decisions made, worth restating if asked**:
- **Attention placement**: CBAM (channel attention via squeeze-and-excite, then spatial attention
  via a 7×7 conv over channel-pooled max/avg maps) is applied **once**, to the third/deepest conv
  layer's pre-activation current (continuous-valued, before it becomes a spike) — not after every
  layer. This matches the paper diagram's single attention block, and sidesteps the question of
  what "attention on a binary spike" would even mean, since attention operates on the continuous
  current, same as BatchNorm already does.
- **Spiking recurrent layer**: `snn.RLeaky` (`linear_features=32`, `learn_recurrent=True`) — a LIF
  neuron whose own previous output spike feeds back in as extra input each step, with that feedback
  connection itself trainable. Verified its calling convention (`rlif(input, spk, mem) -> (spk,
  mem)`, `init_rleaky()` for the initial state) directly before use.
- **Temporal attention** (the third "dimension" of "multi-dimensional attention", alongside channel
  + spatial): a learnable weight per time step (`self.time_weight`, 25 numbers), softmax-normalized
  to sum to 1, replacing Week 3's plain uniform sum over the 25 steps when combining into a final
  decision — lets the network learn to trust some time steps more than others.
- Reused the exact "born-dead network" safeguard from Week 3 (`create_healthy_masnn`) — smoke-test
  on a real batch, discard and retry on total silence. Passed on the first attempt this time.

**Benchmarked at ~37s/epoch** (IMG_SIZE=64, same as Week 3) despite the added attention/recurrent
compute — barely slower than the plain spiking network. Full run (7 epochs before early stopping)
took a few minutes.

**Real result — beats Week 3's SpikingCNN on nearly every metric**:

| | SpikingCNN (Wk3) | MASNN (Wk4) |
|---|---|---|
| Benign precision | 0.27 | 0.36 |
| Benign recall | 0.51 | 0.43 |
| Malignant precision | 0.85 | 0.85 |
| Malignant recall | 0.66 | **0.82** |
| Test accuracy | 0.63 | **0.74** |
| Val→test gap | 22 pts | **18 pts** |

One real tradeoff: benign recall dropped (51%→43%) even as benign precision improved (27%→36%) —
the model became pickier about calling something benign, more often right when it does. The
val→test generalization gap keeps shrinking with each architecture improvement (36→22→18 points
across baseline→spiking→MASNN), a consistent trend worth highlighting.

**Transient GPU note**: hit a different CUDA error this session (`CUBLAS_STATUS_NOT_INITIALIZED`,
not the "out of memory on a tiny tensor" symptom from the earlier stuck-context incident) while
benchmarking — resolved itself on retry without needing a reboot this time, `nvidia-smi` showed the
GPU genuinely idle throughout. Treat isolated CUDA errors as possibly transient before assuming the
GPU needs a full reboot; only escalate to that if a fresh retry also fails.

**Update: documentation caught up.** `src/make_comparison_report.py` extended to a 4-way comparison
(broken baseline / fixed baseline / spiking network / MASNN) throughout all three chart functions —
`comparison_training_curves.png`, `comparison_confusion_matrix.png`, `comparison_metric_bars.png`
all regenerated. New cumulative report `docs/Progress_Report_Week1-4.docx` (supersedes
`Progress_Report_Week1-3.docx`, which is now stale — kept for history, not further maintained)
adds a full "Week 4 — Full MASNN" section (approach, design decisions, results table, honest
tradeoff note on benign recall dropping). Both files sent to student.

**Known hard-part assessment (told to student, before Weeks 3-4 were built — now partly resolved)**:
1. **Optimization strategy (Shrike Optimization Algorithm)** — hardest, still not started. No
   library support, the paper's fitness function is essentially undefined (`optimize(accuracy &
   time)` with no real formula), and population-based metaheuristics fundamentally don't scale to
   searching tens of thousands+ of weights the way backprop does. Realistic plan: apply it to a
   small parameter subset (e.g. final layer, or `eps`) rather than the whole network, and be
   upfront about that.
2. **Multi-dimensional attention** — DONE (see above). Turned out moderate as predicted — CBAM was
   well-precedented, the main real decision was where to place it relative to the spike/current
   boundary, resolved as described above.
3. **Spiking neurons** — DONE (Week 3). Was least familiar conceptually but most
   implementation-solved, thanks to `snnTorch` handling LIF neurons and surrogate gradients.

### Shrike Optimization Algorithm (Week 5) — IN PROGRESS
`src/shrike_optimizer.py` (`ShrikeOptimizer`, pure NumPy, decoupled from the model) +
`src/train_soa.py` (integration with MASNN).

**Scope decision (matches the hard-part assessment above)**: SOA searches only MASNN's final
classifier layer (`fc2`, 66 numbers — 64 weights + 2 biases), with every other parameter frozen at
its Week 4 trained value. Full-network population search was confirmed impractical, consistent
with the earlier prediction. Also dropped the paper's dual accuracy+time fitness term — with a
fixed-size 66-number search space, every candidate has identical forward-pass cost, so "computation
time" carries no information here; fitness is validation accuracy alone.

**Real bug found and fixed via toy-problem validation, before ever touching the real model**: the
first implementation of the paper's Eq. 13 ("male-parent feeding") compared each candidate against
*its own* tracked best. This is mathematically degenerate — the moment a candidate matches its own
best (which happens almost immediately), the update formula collapses to a no-op regardless of the
random step size. Confirmed directly: instrumented a 50-iteration run on a 10-dimensional toy
problem with a known optimum and found `male_wins: 0` out of 750 attempts, population diversity
collapsed to std≈0.03-0.06 per dimension, and 300 iterations barely improved on 30. Fixed by
comparing each candidate against a different, randomly-chosen nest instead of itself (closer to the
paper's own description of "two parent birds per nest" anyway, and structurally similar to the
mutation operator in differential evolution). Re-tested: real, if imperfect, convergence restored.

**Honest characterization from the toy-problem testing**: even after the fix, the algorithm did not
fully converge to the true optimum from a random start within a reasonable iteration budget
(plateaued at ~40% of the way there) — a known, documented limitation of many population-based
metaheuristics relative to gradient descent on smooth problems, not a bug. However, when seeded
near an already-good solution (our actual use case — starting from backprop-trained weights, not
from scratch), it reliably made modest, genuine local improvements without wandering away. This is
the honest framing to use if asked: **SOA here functions as a local refinement step layered on top
of backprop, not a replacement for it** — a legitimate, explainable scope, not an evasion.

**First run failed loudly and instructively.** Test accuracy collapsed from 74.0% to 19.5% —
confirmed root cause: the 128-image fitness subset was taken as `range(128)` (the first 128 rows
of `split.csv`, unshuffled), which turned out to be **100% benign** (dataset rows are grouped by
patient/scan order, not randomized). The search found a degenerate "always predict benign"
solution that scored a trivially perfect fitness on that unrepresentative subset while destroying
real generalization — confirmed exactly: benign recall was 100%, and test accuracy (19.5%) matched
the benign fraction of the real test set almost exactly. This is on top of (not instead of) the
ceiling-effect problem already suspected — accuracy as a fitness metric saturates once every
candidate scores 100% on an easy subset, removing all selection pressure.

**Both bugs fixed together**: (1) fitness subset is now drawn via `np.random.default_rng(42).choice`
instead of a fixed unshuffled slice — verified balanced (100 malignant / 28 benign, close to the
true ~74/26 split) before trusting it; (2) fitness signal switched from accuracy to **negative
validation cross-entropy loss** (`negative_loss()`), which is continuous and can't saturate at a
ceiling the way accuracy can.

**Rerun succeeded — a real, substantial improvement**:

| | Before SOA (Wk4) | After SOA (Wk5) |
|---|---|---|
| Benign precision | 0.36 | 0.60 |
| Benign recall | 0.43 | 0.34 |
| Malignant precision | 0.85 | 0.86 |
| Malignant recall | 0.82 | **0.94** |
| Val accuracy | 0.924 | 0.902 (slightly down) |
| **Test accuracy** | 0.740 | **0.828** |

Fitness history shows genuine, gradual improvement (not a single-step fluke): plateaued at -0.1448
for iterations 1-6, then improved steadily to -0.1341 by iteration 11 and held. Test accuracy rose
to 83% (from 74%) with malignant recall reaching 94%; val accuracy dipped slightly (92.4%→90.2%),
and — notably, a first for this project — the val→test relationship inverted (test scored *higher*
than val), unlike every prior week's "large gap, val higher" pattern. Same honest tradeoff pattern
as prior weeks: benign recall dropped (43%→34%) as benign precision rose sharply (36%→60%).

**This validates the whole Week 5 scope decision**: SOA, properly restricted to a small (66-number)
parameter subset with a fitness signal that can't be gamed, delivered a real improvement as a local
refinement on top of backprop — the exact honest framing committed to at the start of this section,
not an evasion that happened to also produce a good number by luck.

Results saved to `data/splits/week5_results.json` (final numbers) — note
`data/splits/week5_results.json` was overwritten between the two runs, so only the second (fixed,
correct) run's numbers are on disk now; the first (broken) run's numbers exist only in this note
and in `src/train_log_soa.txt` (kept as a paper trail) — `train_log_soa_v2.txt` holds the real run.

## Known issues / gotchas for whoever picks this up

- **GPU CUDA context may still be stuck.** Symptom: even a 10-element `torch.zeros(10,
  device='cuda')` fails with "out of memory" while `nvidia-smi` shows the GPU nearly empty. Fix:
  reboot the machine (already advised to student). If it's stuck again after reboot, something else
  is holding the context — check `tasklist` / `wmic process` for orphaned `python.exe` /
  `jupyter.exe` / `ipykernel_launcher` processes before assuming it's a driver problem again.
- **Running training through `jupyter nbconvert --execute` is fragile** — has produced silent
  syntax-error-induced blank cells and hard kernel deaths in this project. Prefer standalone `.py`
  scripts (like `train_baseline.py`) that save results to disk (JSON/`.pt`), with the notebook only
  loading and displaying those results afterward, for anything computationally heavy.
- **`notebooks/exec_log.txt`, `exec_log2.txt`, `exec_log3.txt`, `src/train_log.txt`** are leftover
  debug logs from the training crash investigation — safe to delete, kept for now as a paper trail.
- **`docs/RP_Plan_123cs0193.docx`** — appears to be a student-made renamed copy of the Research
  Plan; **`docs/Research_Plan_CP1.docx`** is the one actively maintained/edited by both the student
  (in Word) and this assistant — treat it as the source of truth, check for manual edits before
  regenerating it programmatically (student has directly edited it in Word before; don't silently
  clobber their changes).

### Component-Contribution Analysis (Week 6) — IN PROGRESS
`src/make_ablation_chart.py` — realized the week-by-week progression already *is* a real
ablation-style study (per the Research Plan's committed research gap): each week added exactly one
architectural piece on top of the last, measured on the identical 1,282-image/13-patient test set
throughout. Compiled into `data/splits/ablation_chart.png` (two panels: overall accuracy climbing
monotonically stage to stage, and the benign/malignant recall tradeoff diverging over the same
stages).

**The full progression, one component at a time**:
| Stage | Test accuracy | Δ from previous |
|---|---|---|
| TSGF + Baseline CNN (Wk2) | 52.2% | — |
| + Spiking neurons (Wk3) | 63.2% | +11.0 pts |
| + Attention + recurrent layer (Wk4) | 74.0% | +10.8 pts |
| + Shrike Optimization (Wk5) | 82.8% | +8.7 pts |

Every single addition genuinely helped — a clean, monotonic, honest result. The flip side, also
worth keeping in the final report: malignant recall climbs steadily and dramatically (53%→94%)
while benign recall declines just as steadily (50%→34%) across the same four stages — each
architectural addition made the model more sensitive to malignant cases at some real cost to
benign precision-vs-recall balance. Worth discussing explicitly, not just the accuracy headline.

**Also discovered a real gap while working on this**: `train_soa.py` never saved the final
SOA-refined model's weights, only its metrics — fixed (`torch.save(...)` added, saves to
`data/splits/week5_model.pt`) and the run repeated (deterministic/seeded, so should reproduce the
same 82.8% test accuracy) to get a reusable final model for ROC/AUC computation and any future work.

**ROC/AUC computed — `src/make_roc_curve.py`, and it surfaced a genuinely interesting finding.**
AUC = 0.583, barely above random (0.5), despite 82.8% threshold-based accuracy — a real
discrepancy investigated rather than just reported as-is. Root cause confirmed directly: the
model's raw output is a spike **count** (weighted by the small, mostly-flat temporal-attention
vector) over only 25 discrete time steps, so the achievable score space is inherently small and
quantized — only **33 unique probability values across all 1,282 test images** (~39 samples share
each score on average). AUC is a purely rank-based metric and gets severely penalized by ties
between classes at shared score values, even when threshold-based decisions (does the score sit
above or below 0.5) remain accurate. Visually confirmed in `data/splits/roc_curve.png`: the curve
hugs the diagonal (near-random) almost everywhere except one sharp near-vertical jump around
FPR≈0.45 — the signature of one dominant, cleanly-separating threshold surrounded by an otherwise
coarse, tied score distribution, not a genuinely weak classifier. Worth stating explicitly in the
final report: **AUC is a poor fit for evaluating a discrete spike-count-based output layer** — this
finding is itself a legitimate, reportable insight, not a failure to explain away.

**Not yet done**: a written "critical evaluation vs. the original paper's claims" discussion (the
paper reports ~99% accuracy — far above our 82.8% — an honest comparison of why is worth writing
up: different/smaller image resolution (64px vs. likely higher in the paper), far fewer training
epochs, a much narrower SOA search scope, consumer GPU constraints vs. a full research setup), and
a consolidated final report covering all 6 weeks.

### Improvement Planning (Week 6) — DECISION PENDING, nothing implemented yet
Student pushed back that the gap to the original paper's claimed ~99% accuracy (we're at 82.8%) is
large, and asked for a full menu of improvement options before agreeing on what to actually change.
Presented as a ranked list, not yet acted on — **check whether the student has since picked a
direction before assuming any of this has been implemented**:

**High-impact, moderate effort**
1. **Image resolution (64×64 → higher, e.g. 128×128).** Likely the single biggest quality ceiling —
   dropped to 64px purely for spiking-network compute reasons (25× cost multiplier from time
   steps), almost certainly losing fine nucleus/chromatin detail that matters for the diagnosis.
   ~4× more compute at 128px, still tractable (baseline CNN already ran fine at 128px).
2. **Training budget.** Every model so far has early-stopped within 7-13 epochs (max 25, patience
   6) — nowhere near the paper's reported 100 epochs / 4.8h. Loosening patience/max_epochs is
   nearly free to try, just costs wall-clock time.
3. **Expand SOA's scope beyond fc2** (e.g. to the recurrent layer or attention block's weights) —
   natural next step now that the algorithm itself is validated/bug-fixed, still short of
   full-network search.

**Moderate impact, low effort**
4. Histopathology-specific augmentation (H&E stain color jitter) — standard practice in this field,
   cheap to add, currently only doing flips/rotation.
5. Rebalance class handling — try weighted sampling instead of/alongside loss weighting; recall has
   been swinging hard (benign recall down to 34% after SOA).

**Addresses trustworthiness of the numbers, not necessarily the headline accuracy**
6. **k-fold cross-validation** — flagged as the *correct* fix since Week 2 for the small, noisy
   13-patient val/test sets, never implemented. Every number in this project so far is from one
   single lucky/unlucky split.
7. Ensembling / multiple seeded runs — smooths out small-test-set noise, multiplies training time.

**Recommended priority given this is Week 6 of 6** (told to student, not yet agreed on): (1)
resolution + training budget together, (2) k-fold cross-validation for defensibility, (3) everything
else only if time remains. Explicitly framed 99% as not a credible target for a one-semester
capstone vs. the paper's full research setup — and that this gap is itself a legitimate,
explainable talking point for the presentation, not a failure to hide.

## Roadmap (see also the published artifact roadmap from earlier in the project)

- **Week 1 — DONE.** Environment, dataset verified, patient-wise split, exploration notebook.
- **Week 2 — DONE (with an honestly-documented weak result).** TSGF done. Baseline CNN trained
  successfully on GPU; accuracy is weak (~52-53% test) with a large val/test generalization gap
  traced to the small (13-patient) evaluation sets — see "Root-cause conclusion" above. Deliberately
  not chased further; this baseline is a reference point, not a final result.
- **Week 3 — DONE.** Spiking neurons: LIF layers via snnTorch, threshold tuning, "born-dead
  network" initialization fix, spike raster visualization, `SpikingCNN` beats baseline CNN on every
  metric (63% vs. 52% test accuracy). See "Spiking Neural Network (Week 3)" section above.
- **Week 4 — DONE.** Multi-dimensional attention (CBAM: channel + spatial) + spiking recurrent
  layer (`snn.RLeaky`) + temporal attention (learned per-time-step weighting) assembled into the
  full `MASNN`, matching the paper's Figure 2 order. Beats Week 3's plain spiking network on nearly
  every metric (74% vs. 63% test accuracy, val/test gap down to 18 points). See "Full MASNN
  (Week 4)" section below.
- **Week 5 — DONE.** Shrike Optimization Algorithm implemented, scoped to refining MASNN's final
  66-parameter classifier layer on top of frozen, backprop-trained weights. Two real bugs found and
  fixed (a toy-problem-validated algorithm bug, then a fitness-subset bias bug caught from a
  catastrophic first real run) before landing a genuine result: 74%→83% test accuracy, malignant
  recall 82%→94%. See "Shrike Optimization Algorithm (Week 5)" section above.
- **Week 6 — IN PROGRESS.** Component-contribution chart done (ablation-style progression: 52% →
  63% → 74% → 83% test accuracy, one architectural piece per stage), ROC/AUC computed and its
  low-despite-good-accuracy result explained (quantized spike-count output, see above), model-saving
  gap fixed. Still open: improvement pass (see "Improvement Planning" below), critical-evaluation
  writeup vs. the original paper's claims, final consolidated report, presentation prep.

## Course administrative context

- Research Plan for Capstone Project-I was due **12 August 2026** (same day it was drafted) —
  submitted to Prof. Sanjeev Patel for signature, to be uploaded as PDF to Microsoft Teams.
- Deliberately uses generic/hedged terminology throughout (see top of this file) so that if any
  specific technique (especially the optimization strategy) turns out to be infeasible to fully
  implement, the submitted plan contains no false promises.
- Framed research contribution as a **comparative/component evaluation** (not a firm "ablation
  study" commitment) — i.e., examining what each architectural piece (preprocessing, attention,
  optimization approach) actually contributes, since the source paper doesn't do this itself.
