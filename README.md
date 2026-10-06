# Face Recognition with Local Binary Patterns — reproducible implementation

Implementation of Ahonen, Hadid & Pietikäinen, *Face Recognition with Local Binary Patterns*
(ECCV 2004): LBP<sup>u2</sup> histograms over a grid of windows, χ² / weighted-χ² nearest-neighbour
classification, learned window weights, and the permutation-based evaluation used in the paper.

**Status in one paragraph.** The ORL experiment of the paper runs end to end on real data and
reproduces the paper's number (0.983 ± 0.010 vs. the paper's 0.98 ± 0.012). The FERET pipeline
(preprocessing → weights → fb/fc/dup I/dup II rates → permutation test) is fully implemented and
tested on synthetic faces, but **FERET itself cannot be redistributed, so the FERET numbers of
Table 4 have not been reproduced** — see [What is and is not verified](#what-is-and-is-not-verified).

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate          # Python 3.11
pip install -r requirements-dev.txt                    # exact pinned versions
# put ORL in data/ORL/s1/1.pgm ... data/ORL/s40/10.pgm  (see "Getting the data")
make test          # 88 tests, ~30 s
make orl           # the paper's ORL experiment, ~8 s
make reproduce     # re-run and fail unless the result digest equals expected/orl_seed0.json
make sweep         # Table 1 / Fig. 4 analogues on ORL, a few minutes
make feret-demo    # whole FERET pipeline on synthetic faces (no FERET needed)
```

## Results

**ORL protocol of the paper** — LBP<sup>u2</sup><sub>16,2</sub>, 30×37 windows (3×3 grid on 92×112), χ², no
weights, no preprocessing; 5 random images per subject as gallery and the other 5 as probes;
100 permutations. Feature length 2187.

| run | mean rate | std (100 perms) | 95 % interval |
|---|---|---|---|
| **paper** | **0.98** | **0.012** | – |
| this code, seed 0 (`expected/orl_seed0.json`) | 0.9826 | 0.0096 | 0.962 – 1.000 |
| seeds 1, 2, 3, 4 | 0.9822, 0.9835, 0.9845, 0.9823 | 0.0114, 0.0114, 0.0094, 0.0108 | |

The mean agrees with the paper to within seed-to-seed variation. The std is 0.009–0.011 here against
0.012 in the paper; the paper calls its ORL numbers "preliminary" and does not give the seed, the ORL
copy or the exact split procedure, so this is as close as the text allows.

**Operator / window / dissimilarity sweep on ORL** (`make sweep`; χ² mean rates, k × k windows,
`expected/sweep_orl_seed0.json` has everything including the log-likelihood and intersection columns):

| k (window w×h) | LBP<sub>8,1</sub> | LBP<sub>8,2</sub> | LBP<sub>16,2</sub> |
|---|---|---|---|
| 2 (46×56) | 0.947 | 0.964 | 0.970 |
| 3 (30×37) | 0.970 | 0.976 | **0.983** |
| 4 (23×28) | 0.970 | 0.975 | 0.979 |
| 5 (18×22) | 0.960 | 0.970 | 0.970 |
| 6 (15×18) | 0.960 | 0.968 | 0.967 |

What carries over from the paper's FERET findings: with χ² the representation is quite insensitive to
operator and window size (0.95–0.98 everywhere); log-likelihood is clearly the worst measure once
histograms get sparse (e.g. LBP<sub>8,1</sub>, k=4: 0.829 vs 0.970 for χ²; LBP<sub>16,2</sub>, k=2: 0.759) but is
competitive when there are few bins per pixel (8-neighbour operators with k=2, where P(HI > LL) is only
0.28 and 0.18) — the paper's "average number of labels per histogram bin" argument. What does **not** carry over: the paper finds
χ² slightly ahead of histogram intersection; on ORL the two are statistically indistinguishable
(P(χ² > HI) between 0.33 and 0.79, direction varies with the operator). The paper itself notes that none
of the measures beats the others with probability > 0.95.

## Layout and mapping to the paper

| paper | code |
|---|---|
| §2, Fig. 1 basic 3×3 operator | `lbpface/lbp.py::lbp_3x3` (test: Fig. 1 gives 211) |
| §2, Fig. 2 circular (P,R) neighbourhood, bilinear interpolation | `lbp.py::lbp_codes`, `sampling_offsets` |
| §2 uniform patterns, LBP<sup>u2</sup> (59 / 243 labels) | `lbp.py::lbp_u2`, `uniform_mapping`, `n_labels` |
| §2 Eq. 1–2 spatially enhanced histogram | `features.py::spatial_histogram`, `LBPDescriptor` |
| §2 Eq. 3–6 histogram intersection, log-likelihood, χ², weighted χ² | `distances.py` |
| §2 nearest-neighbour classifier, §3 rank curves | `classify.py` |
| §3 preprocessing: eye registration, elliptical mask, histogram equalisation | `preprocess.py` |
| §3 permutation tool: mean, 95 % interval, P(A > B) | `stats.py` |
| §4 window weights (0 / 1 / 2 / 4 from percentiles, Fig. 5b) | `weights.py` |
| §4 FERET experiment (Table 4, Fig. 6) | `experiments/feret.py` |
| §4 Table 1 / Fig. 4 | `experiments/sweep.py` (on ORL) |
| §4 ORL experiment | `experiments/orl.py` |

```
lbpface/        the library (lbp, features, distances, classify, weights, stats, preprocess,
                datasets, synthetic, repro)
experiments/    orl.py  feret.py  sweep.py        (argparse CLIs, every one writes a JSON with a digest)
tests/          88 tests
expected/       stored results (ORL seed 0, sweep seed 0) for bit-exact comparison
```

Numbers from the paper that are asserted in the tests: 59 and 243 labels; 11907 = 49 × 243 (LBP<sub>16,2</sub>,
18×21 windows); 2124 = 36 × 59 (LBP<sub>8,2</sub>, 21×25 windows); 2301 = 39 × 59 (49 windows minus the 10 that get
weight 0); the ten window sizes 8×9 … 32×37 of Fig. 4 and 43×50 of Table 1; 92×112 with 30×37 windows gives
a 3×3 grid.

### Minimal use

```python
from lbpface.features import LBPDescriptor
from lbpface.distances import pairwise_distances
from lbpface.classify import recognition_rate

desc = LBPDescriptor.feret()                      # LBP^u2_{8,2}, 18x21 windows, 59 bins each
g = desc.transform(gallery_images)                # (n, 7*7*59)
p = desc.transform(probe_images)
d = pairwise_distances(p, g, "chi2")              # add region_weights=w, n_bins=59 for weighted chi2
print(recognition_rate(d, gallery_labels, probe_labels))
```

## Reproducibility — what is guaranteed and how to check it

* **Pinned environment.** `requirements.txt` has exact versions (Python 3.11, NumPy 2.4.4, SciPy 1.17.1,
  Pillow 12.2.0). A fresh virtualenv built from it passes all tests and reproduces the stored ORL digest.
* **Explicit seeds, no global state.** Every random draw comes from `Generator(PCG64(seed))`. Only
  `Generator.random` + `argsort` are used (NumPy guarantees that stream; it does not guarantee
  `choice`/`permutation`). `tests/test_stats.py::test_split_stream_is_pinned` fails loudly if a NumPy
  release ever changes the stream.
* **Same permutations for every algorithm**, so P(A > B) compares like with like.
* **Deterministic arithmetic.** No BLAS / matrix products anywhere (element-wise ops and NumPy pairwise
  sums only), so results do not depend on thread count — verified by re-running with
  `OMP_NUM_THREADS=1` and getting the identical digest.
* **Deterministic data order.** Files are loaded sorted by (subject, image) numerically.
* **Result digest.** Every experiment writes `{config, results, result_digest, meta}`. The digest is the
  SHA-256 of the canonical JSON of `results`, which contains only rank-derived quantities (rates, curves,
  counts), so it is insensitive to last-bit floating-point differences. `meta` records the dataset
  fingerprint, a hash of the library source, library versions and a timestamp (none of which enter the
  digest). `--expect FILE` compares digests and exits non-zero on mismatch.
* **Dataset fingerprint.** `python -m lbpface.datasets data/ORL` prints a SHA-256 over all file
  names + contents. The stored results were made with
  `9972801133ec8ba8b06221a0f2c7c087c2e2a8d99a5fdd0ddfbf90beb82d5ac3`; if your ORL copy differs, the bit-exact
  test is skipped (with a message) rather than failing misleadingly.
* **Independent cross-check of the operator.** The raw LBP codes are compared with
  `skimage.feature.local_binary_pattern` and match exactly for (P,R) ∈ {(8,1), (8,2), (16,2), (4,1), (12,3)}.

What is *not* guaranteed: a different NumPy major version could change floating-point rounding (the
digest is designed to survive that, but I have only tested the pinned versions), and the ORL bit-exact
check needs the same 400 files.

## Getting the data

**ORL** (AT&T / Olivetti Research Laboratory, 40 subjects × 10 images, 92×112 PGM). The original
Cambridge host is no longer reliably online; it is widely mirrored. Put it under `data/ORL` as
`s1/1.pgm … s40/10.pgm`. The copy used here came from a public GitHub mirror of the dataset and has the
fingerprint above. Please credit the Olivetti Research Laboratory (Samaria & Harter 1994) when using it.
It is not bundled here.

**FERET** requires a NIST licence. Build a manifest CSV from your copy (header required):

```
path,subject,sets,left_eye_x,left_eye_y,right_eye_x,right_eye_y
00001_fa.tif,1,fa|perm,251.0,310.5,330.2,305.9
```

`sets` is a `|`-separated list of tags: `fa fb fc dup1 dup2` (standard sets; `dup2` images are also tagged
`dup1`), `perm` (the 160 subjects × 4 images of the CSU `list640.srt` permutation list), `wgal` / `wprobe`
(gallery / probe of the weight-training subset — the paper's `subfc`: the `fa` and `fc` images of subjects
1013–1109). Then `python -m experiments.feret --manifest manifest.csv --root /path/to/images`.
I did not write a converter from the CSU list/eye files because I could not check their exact format.
`python -m experiments.feret --demo demo_data` shows the whole flow on synthetic faces.

## What is and is not verified

**Verified**
* LBP operator: Fig. 1 example; uniform-pattern counts and the examples in the text; exact agreement with
  scikit-image; invariance properties; bilinear interpolation on a ramp.
* All feature-vector lengths and window sizes quoted in the paper.
* χ², weighted χ², intersection and log-likelihood against naive loop implementations; rank curves and the
  permutation test against brute force; weight rule on crafted inputs.
* ORL experiment against the paper's reported number; bit-exact rerun; thread-count independence.
* FERET code path on synthetic faces: registration puts the eyes within 0.5 px of the target; wrong eye
  coordinates measurably hurt accuracy, so registration is really used.

**Not verified / not done**
* **FERET results (Table 2–4, Fig. 6, weights of Fig. 5b).** No FERET data here. Even with FERET, exact
  agreement should not be expected because the paper does not give the CSU normalisation constants
  (target eye positions, ellipse size), which are assumptions in `preprocess.py`.
* The comparison algorithms (PCA, Bayesian, EBGM — from the CSU system) are not implemented; Table 4's
  non-LBP rows are therefore only quoted, not recomputed.
* Table 3 (timings on a 1.8 GHz Athlon) is hardware-specific and not reproduced.
* The held-out validation of the weights (paper: subjects 1110–1206, 0.52 → 0.77) is not scripted; you can
  do it by tagging a second subset.
* The claim that 79.3 % of LBP<sub>16,2</sub> patterns on preprocessed FERET are uniform can be checked with
  `features.uniform_fraction` once you have FERET.

## Choices the paper leaves open (all are parameters or documented in the module docstrings)

| topic | what I did |
|---|---|
| bit order / which uniform pattern gets which label | counter-clockwise from east, uniform labels ascending in code, non-uniform last (a bin permutation: no effect on any distance) |
| image border for the LBP operator | edge replication, label image keeps the full image size (needed for ORL's 3×3 window grid) |
| leftover pixels when the image is not a multiple of the window | split evenly on both sides (keeps left/right mirror windows exact) |
| histograms | raw counts, as in Eq. 2; masked-out pixels are not excluded from the histograms |
| log-likelihood with empty bins | `log(M + 1e-10)`; histogram intersection is returned negated so smaller = more similar |
| percentile rule for weights | `numpy.percentile` (linear), strict inequalities, left/right symmetrised first |
| rank ties | pessimistic: a tie with another subject counts as a miss |
| permutation tool | 95 % interval = empirical 2.5/97.5 percentiles; P(A > B) = fraction of permutations with strictly higher rate; std uses ddof = 1 |
| FERET preprocessing constants | 130×150 output, eyes at (43.3, 60) and (86.7, 60), inscribed ellipse — **my assumptions** |
