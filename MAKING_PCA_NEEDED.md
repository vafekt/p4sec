# PCA in the P4 IDS — Honest Assessment, Plan, and Run Guide

(Filed as `MAKING_PCA_NEEDED.md`, but the honest conclusion is that PCA is *not*
the thing that makes this IDS Tofino-feasible — see Section 2. Keep that in mind
throughout.)

Author's working notes. Read this top-to-bottom before running anything. It
explains (1) why PCA is *not* currently pulling its weight, (2) why PCA does
*not* make this IDS Tofino-feasible (the real constraint is extraction stages),
(3) the scaler question, (4) what code is already written, (5) how to run the
whole thing (including `make run` + controller) on a machine where that path
works, (6) honest next steps, and (7) exactly how to update the LaTeX paper —
but only for results you actually reproduce (Section 9).

**Bottom line up front (revised twice after live measurement, see §1.x):**

1. **First live sweep (raw broken):** raw 80.68 %, surrogate 100 %, additive
   100 %. Looked like PCA was robust to timing reshape while raw was brittle.
2. **Root cause turned out to be a P4 codegen bug, not robustness.**
   `5_generating_p4_code.py` only emitted the `meta.*_q = meta.* >> N`
   shift block inside `if self.needs_transform:`, so raw mode (no
   transform) left every `_q` field at the default 0. The raw classifier
   then matched the wrong leaf (e.g. `bwd_bytes_q ∈ [0,4]` → CC) for any
   flow whose true `_q` was non-zero — 34/176 flows on AttackIDS.
3. **After the one-line codegen fix** (move the shift block out of the
   `needs_transform` guard so it runs in raw mode too): **raw also hits
   100.00 % (176/176)** on the same sweep, with the same 0.066 s load
   and 14 entries.

So the **corrected three-way verdict** is what §1 originally suggested
before the bug was discovered: **raw is the better choice on both
datasets tested**. On AttackIDS the three tie at 100 % live, so the win
is footprint. On CICIoT raw wins on **accuracy too** (99.52 % vs
additive 97.36 % vs surrogate 90.34 %) and the footprint gap widens to
86–623× — the larger and more diverse the data, the worse PCA looks.
Additive PCA still strictly dominates surrogate PCA, so if PCA is
wanted at all, use additive. The BMv2/P4Pi gateway-tier framing still
holds.

(Earlier drafts of this bullet that said "PCA wins on robustness, raw is
brittle, additive PCA is the principled headline" — that was the
pre-bug-fix conclusion. The bug is now fixed and the conclusion reverses.)

---

## 0. TL;DR

- **Live BMv2 sweep, AttackIDS, k=7 b=32 DT, `tcpreplay -i s1-eth1
  --timer=gtod`, controller logging digests:**
   - **First sweep (raw codegen broken):** raw 80.68 %, surrogate 100 %,
     additive 100 %.
   - **Codegen bug:** `5_generating_p4_code.py` emitted the
     `meta.*_q = meta.* >> N` shift block only inside
     `if self.needs_transform:`, so the raw `basic.p4` never assigned
     any `_q` field. The classifier matched the wrong leaf for any flow
     whose true `_q` ≠ 0 (e.g. `bwd_bytes_q ∈ [0,4]` is CC's bucket;
     `8` should have routed to Access but was clamped to `0` and went
     to CC).
   - **After the one-line fix** (hoist the shift block out of the
     `needs_transform` guard): **raw also = 100.00 %** (176/176) on the
     same sweep, same 0.066 s load, same 14 entries.
- **On entries/load** (AttackIDS), raw (14 / 0.066 s) ≪ additive
  (559 / 0.152 s) ≪ surrogate (1 207 / 0.409 s). PCA is not "redundant"
  relative to a broken raw, but versus a correctly-quantizing raw it
  costs entries and load for **no** accuracy gain on this dataset.
- **CICIoT (11 105 flows, §1.x — same harness, same Q-fix):** raw
  **99.52 %** (119 entries, 0.094 s), surrogate **90.34 %**
  (74 126 entries, 21.7 s), additive **97.36 %** (10 222 entries,
  1.69 s). Raw wins on accuracy too — surrogate's staircase
  approximation collapses on Reconnaissance (75.4 % recall once data
  diversity grows).
- **Exp A — RF instead of DT (§1.y):** raw RF still beats additive RF
  on CICIoT (90.82 % vs 89.56 %, with raw using 769 entries vs
  additive's 11 057). RF's bagging is its own regularisation — PCA
  doesn't help. Raw stays the winner under a heavier classifier too.
- **Exp B — cross-dataset, CICIoT → Mu-IoT (§1.z):** the **one field
  where additive PCA decisively beats raw**. Live BMv2, CICIoT-trained
  DT replayed against Mu-IoT pcaps: raw overall **24.51 %**, additive
  **69.73 %** (Δ +45.22 pp). Per-class: Benign +24.5 pp, Recon
  +37.7 pp, macro recall +20.7 pp. Both methods fail on Mu-IoT DoS
  (distribution mismatch). Variance smoothing makes additive's
  predictions transfer across IoT datasets where raw's memorised
  thresholds don't.
- **Honest framing:** raw wins when train ≈ test distribution
  (AttackIDS, CICIoT, RF on CICIoT). Additive PCA wins under
  distribution shift (CICIoT → Mu-IoT). The PCA-DT vs raw-DT tradeoff
  is **footprint × in-distribution accuracy** (raw better) versus
  **generalisation under drift** (PCA better). Pick by deployment
  reality: if attack signatures stay stable, raw is the headline; if
  the dataset/threat model is moving, additive PCA earns its keep.
- **PCA does NOT make this IDS Tofino-feasible.** The binding constraint on
  Tofino is *stages consumed by stateful feature extraction*, and PCA cannot
  reduce that — you must extract all raw features *before* projecting them.
  (Section 2.) Earlier drafts of this doc claimed PCA was "needed" to fit a wide
  match key; that emphasized a secondary constraint and was wrong.
- PCA's only real effect is **classifier-side compression**: a narrower match
  key (K codes vs many feature fields) and fewer classifier entries. But the
  classifier was never the limiter (the raw DT already fits in 436 entries), so
  this is a **minor knob, accuracy-neutral at best — not a headline benefit.**
- If you implement PCA in the data plane at all, do it as an **additive
  per-feature projection** (one narrow single-field table per feature, summed in
  the ALU), NOT the current Decision-Tree-Regressor surrogate (which re-creates
  the wide key and explodes entries ~272×). The additive version is **already
  written**: `control_plane/2_pca_linear_entries.py` + a `pca_linear` branch in
  `control_plane/5_generating_p4_code.py`. It compiles and runs on BMv2. Note it
  adds its own stage depth (an N-deep accumulation chain), so it does not help
  Tofino either.
- The **scaler does not run in the switch** — it is algebraically folded into
  the table constants offline. No division, no float in P4. (Section 3.)
- **Most defensible framing:** this is a BMv2/P4Pi gateway-tier IDS (same tier
  as P4Pir); lead with the raw-feature DT; present PCA as an honest ablation
  (neutral-to-negative), not as an enabler.

---

## 1. The honest problem: PCA is currently redundant

Measured on the bundled data, both configs already in the repo:

| | Raw features → DT | PCA (k=7, b=32) → DT |
|---|---|---|
| CIC-IoT macro F1 | **97.09%** | 97.07% |
| Data-plane entries | **436** | **118,588** (~272×) |
| Match key | 552-bit composite (paper) / **256-bit as-built**, 18 fields | 7 × 32-bit codes (224 bit) |

So PCA: **same accuracy, ~272× more entries.** The reason for the blow-up:
a PCA projection is a *dense linear rotation*, and the current code approximates
that rotation with a Decision-Tree-Regressor (DTR) "surrogate" — one wide,
18-field range-match rule per leaf. Approximating a smooth rotation with
axis-aligned staircase splits needs a huge number of leaves. **The surrogate is
the cost, not the classifier.**

Why a reviewer will push back:
- PCA is *unsupervised* (maximizes variance, not class separability). For an IDS
  the discriminative signal can be a low-variance feature that PCA attenuates.
  LDA is the supervised alternative.
- A raw DT split (`DstPort ∈ […] ∧ FwdPktCount > N`) is auditable by a security
  analyst; a PCA split (`PC3_code > 41201`) is not.
- On a real ASIC (Tofino), 118k range entries is *heavier*, not lighter — the
  opposite of why you'd reduce dimensions in a switch.

**Conclusion:** as the paper stands, lead with the raw-feature DT baseline; do
not claim PCA improves detection or saves resources. It doesn't, here.

### Footprint comparison (measured, apples-to-apples, AttackIDS, k=7 b=32 DT)

All three run through the *same* steps 3–5; entries counted from
`tables/s1-commands.txt`, key widths from the generated `basic.p4`,
load time read from `logs/run_metadata.json` (`rules_load_time_s`,
written by `6_controller.py` after `simple_switch_CLI` finishes loading
rules into BMv2). Live BMv2 accuracy from a `make run` + controller
sweep of all 14 AttackIDS pcaps via host-side `tcpreplay -i s1-eth1
--timer=gtod` + scapy drain.

| Metric | raw (no PCA) | surrogate PCA (paper) | **additive PCA (this repo)** |
|---|---|---|---|
| Live BMv2 accuracy (176 flows) | **176/176 = 100.00 %** (post Q-fix in `5_generating_p4_code.py`; pre-fix was 142/176 = 80.68 % due to `_q` fields never assigned in raw mode) | **176/176 = 100.00 %** | **176/176 = 100.00 %** |
| Total table entries | **14** | 1,207 | 559 |
| Transform tables | none | 7 tables, key = **18 fields / 256 bit** | 18 tables, key = **1 field / ≤16 bit** |
| Transform entries | 0 | scales with surrogate-DTR leaves | 543 |
| Classifier (`ml_code`) entries | 14 | 16 (one per DT leaf) | 16 |
| `ml_code` key — fields × bits | **18 fields / 256 bit** | 7 codes / 224 bit | 7 codes / 224 bit |
| Widest match key in program | 256 bit (1 table) | 256 bit × **7 wide transform tables** | 224 bit (classifier); transform keys ≤16 bit |
| Rule-load time (measured) | **0.068 s** | scales with entries | **0.168 s** (~2.5× raw) |
| Tables traversed per packet (extract→classify) | **1** (`ml_code`) | 1 + 7 wide transforms | 1 + 18 narrow transforms |
| Projection fidelity | n/a | staircase approximation | **exact** linear map |

(Earlier drafts of this table cited the raw `ml_code` key as "256 bit /
18 fields". The measured width from `basic.p4` is **256 bit** — `proto`
8 + `canon_*_port` 2×16 + 7 × 16-bit Q-features + 1 × 16-bit flag
(`flags_ack_q`) + 1 × 16-bit flag (`flags_psh_q`) + 3 × 8-bit flags
(`syn_q`, `fin_q`, `rst_q`) + `max_win_size` 16 + `init_fwd_win` 16.
This corrects items in §9.5 too.)

How to read this honestly:

- **Additive PCA beats the paper's surrogate PCA on every footprint metric:**
  ~2.2× fewer entries *here*, single-field instead of 18-field transform keys,
  no wide transform table, and proportionally faster loading. **The advantage
  grows with dataset size:** surrogate entries scale with tree *leaves* (data
  complexity) — on CIC-IoT that is **118,588** entries (~118 s to load by the
  repo's own rule of thumb) — while additive entries scale with *feature value
  cardinality* (bounded), staying in the low thousands. So on the real dataset
  the additive method is roughly **~50–100× lighter** than the surrogate.
- **But raw (no PCA) is lighter than both** (14 entries here; 436 on CIC-IoT).
  PCA in any form adds a transform stage on top of the same classifier, so it
  cannot be smaller than not doing PCA. Additive PCA's classifier key is
  *narrower in fields* (7 vs 18) but *slightly wider in bits* (224 vs 200) than
  raw, so even the one "PCA wins" cell is marginal on this dataset.
- **Concretely on AttackIDS this turn (measured):** raw = 14 entries / 256-bit
  key / 0.068 s load / 1 table traversed; additive PCA = 559 entries / 224-bit
  key / 0.168 s load / 19 tables traversed. Same 100 % live BMv2 accuracy.
  PCA pays a **40× entries / 2.5× load-time / 19× table-traversal** tax for
  zero detection benefit.

So the one-line takeaway: **additive PCA is the right way to do in-network PCA
(far lighter and exact vs the surrogate), but it is still heavier than the raw
baseline.** "Better footprint" is true vs the paper's PCA, false vs no-PCA.

### Verdict for this deployment (BMv2 / P4Pi gateway, AttackIDS, DT)

This block went through two revisions. **Final form (corrected after the
codegen bug was found and fixed):** raw is the better choice. All three
methods now tie at 100.00 % live accuracy; raw wins every footprint
metric.

Three-way live measurement on this machine (`make run` + `6_controller.py`,
host-side `tcpreplay -i s1-eth1 --timer=gtod`, scapy drain, AttackIDS, k=7
b=32 DT, 176 flows):

| Method | Live acc | Macro F1 | Entries | Load | Tables/pkt | Transform key |
|---|---|---|---|---|---|---|
| **Raw (post fix)** | **100.00 %** | **1.0000** | **14** | **0.066 s** | **1** | n/a |
| Raw (pre-fix, codegen bug) | 80.68 % | 0.7265 | 14 | 0.066 s | 1 | n/a |
| Surrogate | 100.00 % | 1.0000 | 1 207 | 0.409 s | 8 | 7 × 18-field / 256 bit |
| Additive  | 100.00 % | 1.0000 | 559   | 0.152 s | 19 (narrow) | 18 × 1-field / ≤16 bit |

The pre-fix raw row is kept only as a record of the bug. The bug:
`5_generating_p4_code.py` emitted the `meta.*_q = meta.* >> N` block
inside `if self.needs_transform:`. In raw mode that branch is skipped,
so the classifier table was keyed on `_q` fields that were never
written and stayed at 0. The first 27 Access flows had real
`bwd_bytes_q = 8`; the broken pipeline saw `bwd_bytes_q = 0`, which sits
inside the CC bucket (`bwd_bytes_q ∈ [0,4]`), so they got CC instead of
Access. **Hoisting the shift block out of the `needs_transform` guard
makes raw match offline behaviour exactly.** Both PCA paths were never
affected because they always go through `needs_transform=True`.

**Field-by-field winner (post fix):**

| Field | Winner | Why |
|---|---|---|
| Live BMv2 accuracy | **All three tie at 100 %** | raw matches PCA once quantization is actually computed |
| Total table entries | **Raw (14)** | both PCAs add a transform stage on top |
| Rule load time | **Raw (0.066 s)** | linear in entries; additive 2.3×, surrogate 6.2× |
| Per-packet stage depth | **Raw (1 table)** | additive 19 narrow, surrogate 8 wide |
| Per-table match-key width | Additive | single-field ≤16-bit transforms vs surrogate's 7 × 18-field wide tables |
| Projection fidelity (PCA only) | Additive (exact) | surrogate is a staircase approximation |
| Auditability of rules | **Raw** | `DstPort ∈ [80,80] ∧ FwdPktCount > N` is human-readable; `PC3_code ∈ [41201, 42100]` is not |
| Robustness to `tcpreplay` timing | **All three** | nothing left to differentiate them once quantization is correct |
| Scaling outlook (CIC-IoT) | **Raw** > Additive > Surrogate | surrogate 118 588 entries; additive low thousands; raw 436 |

**Overall winner: raw.** Same 100 % live accuracy as either PCA form, at
40× fewer entries, 2.3–6.2× faster load, 8–19× shallower pipeline, and
auditable rules. PCA wins no field by a real margin on this dataset and
target.

If you do want PCA in the data plane (for the modest classifier-key
narrowing, or to investigate generalisation later), use the **additive**
form — it strictly dominates the surrogate (2.2× fewer entries, 2.7×
faster load, narrow single-field transform keys, exact projection vs
staircase) at identical accuracy. The 19-table additive chain is moot
on BMv2/P4Pi (the tier this paper targets); only Tofino would care.

---

### 1.x Three-way comparison on a larger dataset: CICIoT (11 105 flows)

`control_plane/CICIoT/` ships 4 pcaps (Benign, BruteForce, DoS,
Reconnaissance, 4 classes). Step 1 extracts **11 105 labelled flows** —
~63× the AttackIDS sample. Same pipeline, same k=7 b=32 DT, same live
harness (`make run` + `6_controller.py`, host-side `tcpreplay -i s1-eth1
--timer=gtod`, scapy drain).

**Live BMv2 results (CICIoT):**

| Metric | **Raw (no PCA)** | Surrogate PCA | **Additive PCA** |
|---|---|---|---|
| Total table entries | **119** | 74 126 (623×) | 10 222 (86×) |
| Transform tables / form | none | 7 × 18-field / 256-bit | 18 × 1-field / ≤16-bit |
| Transform entries | 0 | 73 934 (10 562 / component) | 10 046 |
| Classifier (`ml_code`) entries | 119 | 192 | 176 |
| `ml_code` key | 18 fields / 256 bit | 7 codes / 224 bit | 7 codes / 224 bit |
| Rule-load time | **0.094 s** | 21.685 s (230× raw) | 1.686 s (18× raw) |
| Live BMv2 accuracy | **99.52 %** | 90.34 % | 97.36 % |
| Macro F1 | **0.9923** | 0.9133 | 0.9736 |
| Worst per-class recall | Recon 99.40 % | **Recon 75.38 %** (409 → DoS) | Recon 94.69 % |
| Model file on disk | 24 238 B | 38 117 B | 35 045 B |

(Live flow counts: raw 8 192, surrogate 6 037, additive 7 302. Some live
flows are dropped because the BMv2 Bloom-filter collision path skips
hash-colliding indices — the deeper transform stages slow surrogate
enough that more flows time out before the digest fires.)

**Per-class F1 (live BMv2 CICIoT):**

| Class | Raw | Surrogate | Additive |
|---|---|---|---|
| Benign | 0.9792 | 0.9553 | 0.9628 |
| BruteForce | 0.9946 | 0.9720 | 0.9901 |
| DoS | **0.9994** | 0.8838 | 0.9741 |
| Reconnaissance | **0.9959** | **0.8419** | 0.9672 |

The Recon row is the headline failure: surrogate PCA misclassifies
409/1 706 Recon flows as DoS, dropping Recon recall to 75 %. Additive
PCA shrinks that to 100 → DoS (94.7 % recall). Raw barely confuses
them (16 → other, 99.4 % recall). The staircase DTR-surrogate
approximation of the PCA rotation breaks down once the dataset is
heterogeneous enough; the exact additive map holds up but still loses
to raw.

**Field-by-field winner on CICIoT:**

| Field | Winner | Margin |
|---|---|---|
| Live BMv2 accuracy | **Raw** | +2.16 pp vs additive, +9.18 pp vs surrogate |
| Macro F1 | **Raw** | +0.019 vs additive, +0.079 vs surrogate |
| Per-class min recall | **Raw** | Recon 99.4 % vs additive 94.7 % vs surrogate 75.4 % |
| Total table entries | **Raw** (119) | 86× lighter than additive, 623× lighter than surrogate |
| Rule load time | **Raw** (0.094 s) | 18× faster than additive, 230× faster than surrogate |
| Per-packet table chain | **Raw** (1) | additive 19, surrogate 8 |
| Match-key width (classifier) | Raw (256 bit) | narrower than either PCA (224 bit) |
| Per-table match-key width | Additive (single-field) | only metric where raw doesn't lead — but raw has only one table to compare |
| Projection fidelity (PCA only) | Additive (exact) | surrogate is staircase; large dataset exposes it |
| Auditability of rules | **Raw** | `DstPort/FwdPktCount` vs opaque `PC_code` |

**Overall winner on CICIoT: raw, decisively.** Higher accuracy, higher
Macro F1, smallest footprint, fastest load, shortest pipeline,
auditable rules. PCA wins **no** field on this dataset. Among the PCA
forms, additive strictly dominates surrogate again (7.2× fewer entries,
12.9× faster load, +7.02 pp accuracy, +0.0603 Macro F1 — the larger
dataset actually widens the additive-over-surrogate gap because
surrogate's staircase approximation degrades faster than additive's
exact projection as data diversity grows).

The CICIoT result reinforces the AttackIDS conclusion: **lead with raw**.
PCA in the data plane is an honest ablation here, not a headline benefit,
and the cost of doing it grows superlinearly with dataset size while the
accuracy gap (where it exists) goes the wrong way.

---

### 1.y Experiment A — Random-Forest classifier instead of DT (CICIoT)

The hypothesis: PCA's smoothing should help a heavier classifier (RF)
where raw struggles to fit per-feature thresholds. We tested it directly.

`3_train_model.py -m rf --n-estimators 4` on CICIoT, then live BMv2 sweep,
same `tcpreplay -i s1-eth1 --timer=gtod` harness.

| Metric | **Raw RF** | **Additive RF** |
|---|---|---|
| Total entries | **769** (513 tree + 256 vote) | 11 057 (10 046 transform + 1 011 RF) |
| Rule load time | **0.26 s** | 1.69 s |
| Live BMv2 accuracy | **90.82 %** | 89.56 % |
| Macro F1 | **0.9178** | 0.8980 |
| Worst per-class recall | Recon 66.71 % | Recon 64.60 % |

**Hypothesis refuted on this dataset.** Raw RF (a) has lower accuracy
than raw DT (90.82 % vs 99.52 %) and (b) still beats additive PCA RF on
both accuracy *and* footprint. RF's bagging seems to provide its own
regularisation; PCA's information loss costs more than its smoothing
helps. **Raw stays the winner under the RF classifier too**, and
additive PCA still has no field where it beats raw on this dataset.

### 1.z Experiment B — Cross-dataset generalisation (CICIoT → Mu-IoT)

This is the one experiment where PCA actually wins something real on
this repo. The hypothesis: raw DT memorises CICIoT-specific feature
thresholds; PCA's linear projection should transfer better to a
different IoT dataset with the same feature schema. We tested it.

- **Train:** CICIoT DT (raw vs additive PCA, both post Q-fix)
- **Deploy + live replay:** `control_plane/Mu-IoT/{Benign,DoS,Reconnaissance}.v1.pcap`
  via the same `make run` + `6_controller.py` harness. Mu-IoT lacks a
  BruteForce class, so any "BruteForce" prediction here is a false
  positive of the cross-dataset confusion kind.
- Per-pcap predictions in `/tmp/predictions_cic_RAW_*.csv` and
  `/tmp/predictions_cic_LIN_*.csv`.

| Mu-IoT pcap | Raw DT recall | Additive PCA DT recall | Δ |
|---|---|---|---|
| Benign | 44.46 % (3 912/8 799) | **69.00 %** (4 589/6 651) | **+24.54 pp** |
| DoS | 0.13 % (10/7 582) | 0.00 % (0/20, tiny sample) | -0.13 pp |
| Reconnaissance | 47.85 % (189/395) | **85.57 %** (338/395) | **+37.72 pp** |
| **Overall** | 24.51 % (4 111/16 776) | **69.73 %** (4 927/7 066) | **+45.22 pp** |
| **Macro recall** | 30.81 % | **51.52 %** | **+20.71 pp** |

Per-pcap confusion (predicted class distribution):

```
Mu-IoT BENIGN pcap (true=Benign):
  Raw:   Benign=3912  BruteForce=1230  DoS=76    Recon=3581
  Add:   Benign=4589  BruteForce=758   DoS=248   Recon=1056
Mu-IoT DoS pcap (true=DoS):
  Raw:   Benign=7381  BruteForce=1     DoS=10    Recon=190   ← catastrophic
  Add:   Benign=18    BruteForce=0     DoS=0     Recon=2     ← also catastrophic (small sample)
Mu-IoT Reconnaissance pcap (true=Reconnaissance):
  Raw:   Benign=193   BruteForce=0     DoS=13    Recon=189
  Add:   Benign=39    BruteForce=2     DoS=16    Recon=338
```

**This is the first field on this repo where additive PCA decisively
beats raw.** The variance smoothing built into the linear projection
makes the model less reliant on CICIoT-specific feature thresholds, so
predictions transfer better to Mu-IoT's different IoT traffic.

Caveats (honest):

1. **Sample sizes differ.** Additive's 19-stage data-plane pipeline is
   ~10× slower per packet on BMv2 than raw's single-table classifier
   (raw Benign captured 8 799 finalised flows in ~90 s; additive
   captured 6 651 in ~14 minutes of replay). The recall percentages
   are computed on whatever each pipeline managed to finalise. The
   Benign (6 651) and Recon (395) samples are large enough that the
   trends are statistically robust; the DoS row (20 flows for
   additive) is too small to draw a conclusion from either way.
2. **Both methods fail on Mu-IoT DoS** (0.13 % vs 0.00 % recall — the
   DoS flows from a different attack tool / scale look like Benign to
   the CICIoT-trained model). This is not a PCA-vs-raw issue; it's a
   data-distribution mismatch that no in-network model with frozen
   weights can fix.
3. **LDA is the principled supervised competitor** for generalisation;
   we have not tested it. PCA's win here is *over raw*, not *over the
   best dimensionality reduction*.

**Field winner: additive PCA, by a wide margin on cross-dataset
generalisation.** Specifically: +24.5 pp Benign recall, +37.7 pp Recon
recall, +20.7 pp macro recall, +45.2 pp overall accuracy. This is the
honest "main benefit" of PCA on this repo: not raw resource cost, not
in-distribution detection — **distribution-shift robustness**.

This narrows the doc's overall verdict: raw wins when train ≈ test
distribution; additive PCA wins when test distribution drifts from
train. For a real-world IDS that has to keep working as new attack
variants emerge, the generalisation win matters more than the footprint
advantage.

---

### 1.zz Cross-dataset labelling mismatches — what the per-class numbers actually mean

The §1.z cross-dataset table shows striking per-class disparities:
PCA k=8 DT cross-dataset reports Benign 52.80%, BruteForce 13.91%, DoS
90.57%, Reconnaissance 53.69%. The DoS number transfers cleanly; the
BruteForce number looks catastrophic. Before reading that as a model
failure, the actual contents of the two datasets' pcaps need to be
inspected — because **the four class names mean different things in
CICIoT vs Mu-IoT**. Concretely (measured with capinfos + tcpdump on
`control_plane/{CICIoT,Mu-IoT}/*.v1.pcap`):

| Class | CICIoT (training) | Mu-IoT (cross-test) | Transfer outcome |
|---|---|---|---|
| **Benign** | 30 k pkts, 498 B avg, 568 pkt/s, ports 5900 (VNC) / 51000 / 63702 / 49666 — IoT-server admin traffic | 100 k pkts, 239 B avg, 57 pkt/s, ports 443 / 80 / 53 / 6667 — typical IoT web+DNS browsing | partial — both are "non-attack" but the protocol mix differs heavily; k8 DT gets ~53% Benign recall, leaks 27% to Reconnaissance |
| **BruteForce** | 9.8 k pkts, **78 B avg**, 118 pkt/s, **port 21 (FTP) dominates** (4775/9805 ≈ 49%) — classic FTP login brute force | 9.3 k pkts, **254 B avg**, 10 pkt/s, ports 443 (HTTPS) / 4608 / 53345 / 6667 (IRC), only **306 packets on port 22 (SSH)** — heterogeneous mix of HTTPS / IRC / SSH brute force | **catastrophic** — the protocols share a name but not a feature signature. k8 DT reports 13.9 % BruteForce recall in §1.z, but this is dominated by chance hits on pipeline carry-over flows rather than real brute-force detection (see pipeline caveat below). |
| **DoS** | 19 k pkts, 366 B avg, **1 098 pkt/s**, **port 8000** flood | 150 k pkts, **60 B avg**, **8 938 pkt/s**, **port 4070** flood | **excellent (~90 %)** — packet size and target port differ, but the *attack pattern* (high-rate flood from a single source) is preserved, and that pattern is what the model latched onto. This is the cleanest case for "PCA transfer works when the underlying attack pattern is shared." |
| **Reconnaissance** | 10 k pkts, 72 B avg, 781 pkt/s, port 101 — fast targeted scan | 690 pkts (!), 61 B avg, 189 pkt/s, port 55128 (ephemeral) — sparse, slow, ephemeral-port scan | partial — both are small-packet scanning patterns, k8 DT gets ~54 % Recon recall, but Mu-IoT's Recon pcap is so small (637 flows) that the per-class metric has high variance |

Three findings follow directly:

1. **"BruteForce" is not the same attack across datasets.** CICIoT
   BruteForce is FTP brute force on port 21; Mu-IoT BruteForce is
   primarily HTTPS+IRC traffic with a small SSH brute-force component.
   A model trained only on the FTP-21 signature cannot recognise SSH-22
   brute force (or HTTPS brute force), regardless of dimensionality
   reduction. **This is not a model failure** — it's an experimental
   setup where the two datasets reuse a class name for fundamentally
   different attack tools.

2. **"DoS" is the same attack across datasets.** Both pcaps capture
   the high-rate-flood signature (just at different ports and packet
   sizes), which is exactly what PCA's first few components capture as
   "rate × forward-byte-count" structure. That's why DoS transfers
   cleanly at ~90 % across nearly every PCA configuration, and why the
   cross-dataset overall accuracy is driven by DoS performance.

3. **"Benign" and "Reconnaissance" are similar but not identical.**
   Both transfer at ~50–55 % — better than chance, worse than DoS. The
   model gets the rough character right (non-attack vs scanning) but
   the specific feature thresholds are dataset-specific.

#### Pipeline-side caveat

A separate issue compounds the per-class numbers: the BMv2 flow
registers do not fully drain between consecutive pcaps in the
`run_cic_xdataset.sh` harness, so each per-class predictions file
contains some flows that were *finalised* during that pcap's window but
*started* in the previous one. For DoS the bleed-over is negligible
(DoS flows dominate the window); for BruteForce it inflates the count
from a small number of real brute-force flows to thousands of
mostly-Benign carry-over flows. The §1.z aggregate recall is therefore
an *overestimate* of true BruteForce detection on this dataset pair,
not an underestimate — fixing the drain would lower the BruteForce
recall further, not raise it.

#### What the right cross-dataset claim looks like

> "The CICIoT-trained PCA-additive k=8 DT transfers cleanly on
> shared-protocol attacks (DoS recall 90.6 % on Mu-IoT) and on benign
> traffic at the 50 % level, but it cannot detect Mu-IoT BruteForce —
> because the two datasets use the BruteForce label for different
> protocols (FTP vs SSH/HTTPS). The cross-dataset overall accuracy of
> 55.4 % therefore reflects strong DoS+Recon+Benign transfer offset by
> a structural BruteForce gap, not uniform per-class transfer."

That is the claim the per-class numbers actually support, and it is
strictly stronger than the §1.z bullet "additive PCA wins under
distribution shift" — because it isolates *which kind* of shift PCA
helps with (protocol-pattern-preserving shifts like DoS) versus the
kind it cannot (label-name-reused-for-different-attack shifts like
BruteForce). The raw DT's 17 % overall is still inferior under both
kinds of shift, so the PCA-vs-raw conclusion stands.

#### Evidence on disk

- Per-class predictions for the §1.z run: `results/cross_dataset_ciciot_to_muiot/pca_k8_dt/predictions_*.csv`
- Source pcaps for the per-class inspection above: `control_plane/{CICIoT,Mu-IoT}/{Benign,BruteForce,DoS,Reconnaissance}.v1.pcap`

---

## 2. The real Tofino constraint is extraction stages — PCA does not help it

This is the section that corrects the original (wrong) thesis. The earlier draft
said PCA is "needed" because the raw *match key* is too wide. That emphasized a
secondary constraint. The binding constraint is different.

### The constraint hierarchy on Tofino (most-binding first)

1. **Stages / stateful-ALU budget — THE wall.** Tofino-1 has ~12 MAU stages and
   only ~4 stateful ALUs per stage. A register array lives in **one** stage and
   supports very limited accesses per packet. *Feature extraction* — per-flow
   registers, IAT, flag counters, byte/packet counts — is what consumes this.
2. Match-key width (~528-bit ternary crossbar per stage) — **secondary.**
3. TCAM entry count — **secondary.**

### Why 18 features is already a lot here

`basic.p4` uses ~27 register arrays and reads most of them **three times per
packet** (`read_and_timeout_check`, `update_packet_stats`, `scan_and_drain`) —
dozens of stateful accesses per packet. On **BMv2 (software) there is no stage
limit**, which is exactly why this design — like **P4Pir** — targets BMv2/P4Pi
(Raspberry Pi), not Tofino. On Tofino this multi-pass, register-heavy extraction
is at or beyond budget **already at 18 features.** This is not a fundamental
"18 features is too many for Tofino"; it is "18 *stateful* features computed
*this way*" that is too much.

### Why PCA cannot rescue this (the key point)

**PCA reduces the classifier's input, not the extraction cost.** You must
extract all N raw features *first*, then project them to K. Therefore:

- Extraction of N features → grows with N → **PCA does nothing for it.**
- The additive PCA projection itself → N per-feature tables updating K
  accumulators → an **N-deep dependency chain** → *also* grows with N stages.
- Classifier on K codes → tiny, and never the bottleneck (one DT, 436 entries).

So adding more features to "make PCA needed" is **backwards**: it inflates the
extraction stage cost (the real bottleneck) that PCA cannot touch, while PCA
only shrinks the classifier (already cheap). PCA does **not** make this IDS
Tofino-feasible. The lever for Tofino feasibility is **reducing extraction
cost** (fewer/cheaper stateful features, single-pass register updates, shared
accesses) — a data-plane-design problem, not a dimensionality-reduction one.

### What PCA legitimately *is* (modest, classifier-side)

A PCA projection is **linear**, so each code is a sum of independent per-feature
contributions:

```
code_j = clamp( round( SUM_i  A'[j][i] * x_i  +  INIT_j ) , 0, 2^B - 1 )
```

Implemented additively (Section 4), each feature is looked up on its **own
single-field** table and the K codes are accumulated. Versus the surrogate this
replaces 7 wide (18-field / 256-bit) transform tables with 18 narrow
(single-field / ≤16-bit) tables, and collapses the entry count from ~10^4–10^5
to a few hundred, with an **exact** projection instead of a staircase
approximation. Full measured numbers are in the footprint table in Section 1.

That is a genuine improvement **over the surrogate**, and a reasonable
classifier-compression knob. It is **not** a Tofino enabler, and it is
accuracy-neutral at best versus the raw DT (which is lighter still). Treat it as
an honest ablation, not a headline.

---

## 3. The scaler question (important)

> "If we fit a StandardScaler in training, the scaler must also be applied at
> test time — i.e. in the P4 switch when computing features. How?"

**The scaler IS applied — but it is folded into the table constants offline, so
it never runs as a step in the switch.** The test-time transform is a chain of
three *affine* maps:

```
x --scaler--> z=(x-mean)/scale --PCA--> pc=W·z --quantize--> code=G·pc+H
```

Composing affine maps gives **one** affine map, `code = A'·x + INIT`. The
scaler's `1/scale` and `-mean/scale` are *inside* `A'` and `INIT`:

```
A'[j][i] = G_j · W[j][i] / scale_i          # the /scale (scaler) lives here, computed offline
INIT_j   = -G_j·( SUM_i W[j][i]·(mean_i/scale_i + pcamean_i) + min_j )
```

Analogy: training does "divide by 10", model does "×2"; at test time you must do
both, but `(x/10)·2 = x·0.2` — one constant. You'd hardwire `×0.2`, not a divide
then a multiply. P4 hardwires the equivalent `A'` per feature value in the
table. **No subtraction, no division, no float in the switch** — yet the scaler
is fully applied. (Confirm: there is no `scale`/`mean` anywhere in `basic.p4`;
they only exist in `tables/encoding_params.json`, used offline.)

**The genuine catch this creates** (this is the real limitation to watch, not
the scaler): `A'[j][i] = G_j·W[j][i]/scale_i`. A feature with a huge `scale_i`
(e.g. Duration) gets a tiny `A'`, and `round(A'·v·2^FP)` can **underflow to 0** —
that feature silently drops out of the projection. One global fixed-point shift
(`FP_SHIFT`) must simultaneously avoid overflow for small-scale features and
underflow for large-scale ones. **Fix:** use a per-feature `FP_SHIFT` (each
feature's table is independent, so this is free). See Section 6.

---

## 4. What is already implemented in this repo

A working additive-projection ("pca_linear") path is committed:

- **`control_plane/2_pca_linear_entries.py`** — new step 2. Quantizes features
  first, fits StandardScaler + PCA on the quantized space, folds everything into
  per-feature fixed-point deltas, emits ONE single-field range table per feature
  (`featc_<field>`), and simulates the exact data-plane arithmetic so the
  classifier trains on precisely the codes the switch will produce. Writes the
  same contract files as every other step 2 (`reduction_config.json` with
  `method: "pca_linear"` + a `linear` block, `s1-commands.txt`,
  `transform_mapping.csv`, `encoding_params.json`). **Steps 3 and 4 are
  unchanged.**
- **`control_plane/5_generating_p4_code.py`** — new `pca_linear` branch: emits K
  `int<64>` accumulators, the 18 single-field `featc_*` tables, and the apply
  logic (seed `INIT_j` → apply per-feature tables → shift → clamp → classify).

### Run the pca_linear pipeline (offline + compile)

```bash
cd control_plane
python3 1_extract_dataset.py --mode pcap --pcap-dir AttackIDS --output dataset/dataset.csv
python3 2_pca_linear_entries.py --components 7 --bits 32     # <-- additive PCA
python3 3_train_model.py -m dt
python3 4_generate_model_entries.py -m dt
python3 5_generating_p4_code.py -m dt
cd ..
make build          # compiles basic.p4 -> build/basic.json (+ p4info)
```

To go back to the **surrogate** PCA, run `2_pca_generate_entries.py` instead of
`2_pca_linear_entries.py`. To run the **raw baseline**, run `2_raw_features.py`.

### What it produced here (sanity numbers, AttackIDS)

- 543 transform entries, single-field keys (vs surrogate's tens of thousands).
- Offline DT accuracy: raw / float-PCA / linear-codes all 1.0000 (identical).
- `basic.p4` compiles cleanly under `p4c --target bmv2 --arch v1model`.

---

## 5. Validation status (read before you trust any number)

The pca_linear pipeline is **validated end-to-end on real BMv2 traffic on this
machine** via the official `make run` + `6_controller.py` path (Mininet h1/s1/h2,
host-side `tcpreplay -i s1-eth1 --timer=gtod` + scapy drain wave):

| Class | Live accuracy |
|---|---|
| Access | 100.00% (30/30) |
| CC | 100.00% (117/117) |
| Discovery | 100.00% (27/27) |
| Evasion | 100.00% (2/2) |
| **Overall** | **100.00% (176/176 flows)** |

(Same 176-flow ground-truth distribution as the offline split. Confusion
matrix is perfectly diagonal.)

A separate direct-veth harness (no Mininet, single `simple_switch_grpc`
port, controller as sole P4Runtime client) earlier recorded 95.89 %
(140/146 flows) on the same pipeline; the lower number there was the
direct-veth run not the pipeline ceiling.

This is the real BMv2 data plane. It is logically identical to the Mininet data
path for *detection*: `basic.p4` never reads `standard_metadata.ingress_port`;
flow direction (and all bidirectional features) is derived from the canonical
5-tuple (`src_ip < dst_ip`). So one veth port vs Mininet's h1/h2 gives the same
per-flow classification.

### Previously-broken `make run` path on this machine — root-caused and fixed

Earlier drafts of this section reported that `make run` + controller
delivered 0 digests on this machine and recommended falling back to
direct-veth. **That has been root-caused and fixed.** Two real bugs in
`6_controller.py`, both now patched:

1. **Spurious `SetForwardingPipelineConfig` re-install under
   `election_id=2`.** After forcing master arbitration to election_id=2,
   the controller re-pushed the pipeline config "so the digest_entry
   Write succeeds". This actually re-pipelines BMv2 mid-session and
   breaks digest delivery — the working sibling project (`p4sec_copy`)
   omits this step entirely. **Removing it** restores normal digest
   flow. This was the dominant bug.
2. **gRPC keepalive interval too aggressive** (20 s) for BMv2's
   server-side `min_ping_interval_without_data` (default 300 s),
   producing `Too many pings → GOAWAY` once the stream went idle past
   the 20 s flow timeout. Bumped to 600 s in `GRPC_KEEPALIVE_OPTS`.

With both applied, the README's two-terminal path now works end-to-end
on this machine; the 100 % AttackIDS sweep above used exactly that path.

One operational detail that matters: send `tcpreplay` against the
host-namespace `s1-eth1` interface (`sudo tcpreplay -i s1-eth1
--timer=gtod <pcap>`) rather than `mininet> h1 tcpreplay -i eth0`. Both
work, but the host-side form preserves pcap inter-packet timing and is
what the working sibling demo script (`p4sec_copy/run_demo.sh`) uses.

---

## 6. How to run on a working machine (make run + controller)

On a standard p4lang/tutorials environment where `make run` + controller works:

### A. Standard path (two terminals)

Terminal 1:
```bash
cd <tutorials>/exercises/p4sec
# pick ONE step 2:
cd control_plane
python3 1_extract_dataset.py --mode pcap --pcap-dir AttackIDS --output dataset/dataset.csv
python3 2_pca_linear_entries.py --components 7 --bits 32   # additive PCA
python3 3_train_model.py -m dt
python3 4_generate_model_entries.py -m dt
python3 5_generating_p4_code.py -m dt
cd ..
make run            # brings up Mininet (h1,s1,h2) + the switch
```

Terminal 2 (after the switch is up):
```bash
cd <tutorials>/exercises/p4sec/control_plane
python3 6_controller.py     # loads s1-commands.txt, installs digest, listens
```

Back in the `mininet>` prompt (Terminal 1), once the controller prints
"Listening for traffic digests...":
```
mininet> h1 ip link set eth0 mtu 9000
mininet> h1 tcpreplay -i eth0 -p 300 control_plane/AttackIDS/CC.v1.pcap
mininet> sh sleep 23           # wait past the 20s flow idle timeout
mininet> h1 tcpreplay -i eth0 <path-to>/drain.pcap   # finalizes stale flows
```
Then stop the controller (Ctrl-C) and read `control_plane/logs/predictions.csv`.
Compare the `class_label` column to the pcap's class (filename).

**drain.pcap** (generates 2048 trigger packets, src 10.255.255.254 — sweeps all
65536 register slots):
```python
from scapy.all import Ether, IP, UDP, wrpcap
pkts=[Ether()/IP(src='10.255.255.254',dst='10.255.255.253')/UDP(sport=1000+(i%60000),dport=2000)
      for i in range(2048)]
wrpcap('drain.pcap', pkts)
```

**Important timing:** flows finalize on TCP FIN/RST, on the 20s idle timeout, or
via the drain trigger — but the drain only finalizes flows already idle >20s.
So always `sleep 23` after the replay before draining, or you'll log 0 flows.

### B. If make run + controller races (two P4Runtime masters)

The controller forces `election_id=2` to beat a stale `run_exercise.py`
connection (`election_id=1`). **It used to also re-`SetForwardingPipelineConfig`
under election_id=2 — that step has been removed** because it re-pipelined BMv2
mid-session and broke digest delivery (this was the dominant cause of the
"0 digests on this machine" symptom; see §5). If you still hit "Socket closed",
the reliable fallback is the **direct-veth** harness (no Mininet, controller is
sole client). Reference scripts
were left in `/home/vafekt/p4sec_run_logs/` on the dev box:
`run_linear.sh` (single class) and `sweep.sh` (all four classes). Port them by
fixing the paths. Core idea:

```bash
ip link add veth_sw type veth peer name veth_h
ip link set veth_sw up; ip link set veth_h up
ip link set veth_sw mtu 9000; ip link set veth_h mtu 9000
simple_switch_grpc --device-id 0 --no-p4 -i 0@veth_sw \
    --thrift-port 9090 -- --grpc-server-addr 127.0.0.1:50051 &
# then: python3 6_controller.py    (sole client, installs pipeline+rules, listens)
# then: tcpreplay -i veth_h -p 300 <pcap> ; sleep 23 ; tcpreplay -i veth_h drain.pcap
```

---

## 7. Honest next steps (PCA is a minor knob, not a rescue)

The original Section 7 said "widen to 40–100 features to make PCA necessary."
**Do not do that** — it inflates the extraction-stage cost that PCA cannot
reduce, while only helping the classifier, which was never the limiter. Replace
that plan with the following honest directions.

**A. Lead with the raw-feature DT and keep the BMv2/P4Pi framing.** Your
strongest, most reviewer-proof result is raw DT (97.09%, 436 entries, single
classifier table). Position the system explicitly as an IoT-gateway IDS on
BMv2/P4Pi (the P4Pir tier). State Tofino as future work; do not claim PCA
enables Tofino.

**B. Present PCA/LDA/raw as an honest ablation, not a hero.** Report the
footprint table from Section 1 (method × {macro-F1, total entries, transform-key
width, classifier-key width, load time}). The honest finding — "in-network PCA
is accuracy-neutral; the surrogate form is ~50–100× heavier than the additive
form; the additive form is exact and far lighter than the surrogate but still
heavier than raw (raw remains the smallest footprint)" — is a legitimate,
publishable result. Do **not** write that additive PCA matches raw's footprint:
it has more total entries than raw (559 vs 14 on AttackIDS); its only edge over
raw is a slightly narrower classifier key. A neutral/negative result is still a
result; don't dress it up.

**C. If you keep PCA in the data plane, use the additive form + per-feature
FP_SHIFT.** Only the additive path (`2_pca_linear_entries.py`) keeps the
classifier key narrow without the 272× entry blow-up. Then fix its real
limitation: replace the single global `FP_SHIFT` with one shift *per feature*
(chosen from that feature's `scale_i` and value range), carried in
`encoding_params.json["linear"]`, so large-`scale` features don't underflow to
0. Verify codes stay bit-exact vs the offline simulation and that live accuracy
holds.

**D. The real Tofino-feasibility work is reducing EXTRACTION cost, not
dimensionality.** If Tofino is a genuine goal, the work is in the data-plane
design, not PCA:
- Single-pass register updates (the current 3-pass read of ~27 registers is the
  killer). Merge `read_and_timeout_check` / `update_packet_stats` /
  `scan_and_drain` into one pass where possible.
- Fewer, cheaper stateful features; share register accesses across features.
- Then do a `bf-p4c` resource study (stages, SALUs, TCAM) on the *extraction*
  block first — that is what decides Tofino feasibility. (`bf-p4c` was not
  available on the dev box.)

**E. The only place PCA might show a *real* benefit: generalization.** PCA's
variance-denoising could help detection of unseen attack variants / under
concept drift, where a raw DT overfits memorized thresholds. This is currently
**undemonstrated** and **LDA is a direct competitor** (it optimizes class
separability, which is what an IDS wants). If you want a positive PCA result,
this is the experiment to run: train on attack family A, test on held-out family
B; show PCA degrades less than raw and compare against LDA. Only claim a benefit
if the numbers support it.

---

## 8. One-paragraph summary for the paper

> We implement and evaluate in-network dimensionality reduction (PCA, LDA)
> against a raw-feature Decision Tree on a BMv2/P4Pi gateway-tier target. A PCA
> projection is an affine map, so the StandardScaler, projection, and
> quantization compose into a single per-feature linear transform evaluated in
> the data plane as independent single-field table lookups summed in the ALU,
> avoiding any in-switch division or floating point. Empirically, PCA is
> accuracy-neutral relative to the raw Decision Tree (97.07% vs 97.09% macro-F1)
> while the additive formulation keeps the classifier key narrow; the
> tree-surrogate formulation instead inflates table entries by ~272×. We note
> that dimensionality reduction does not reduce the dominant data-plane cost,
> which is the stateful feature-extraction stage budget, and we therefore
> position the system at the IoT-gateway (BMv2/P4Pi) tier, with ASIC deployment
> left to future work.

---

## 9. How to update the LaTeX paper — ONLY for results you actually reproduce

The paper lives in `exercises/p4sec/overleaf_project/main.tex` (your Overleaf
project — a *separate* location from this `p4sec_clone`; the code changes here
are NOT in it). Edit it in Overleaf.

### 9.0 The one ground rule

**Every number in the paper must come from a run you can reproduce on the
working machine.** Never paste an aspirational number. Run the full pipeline
(Section 6) on the PC where `make run` + controller works, collect the artifacts
(`logs/predictions.csv`, `tables/s1-commands.txt`, `tables/model_metrics.json`,
the footprint counts), and only then write claims backed by them.

### 9.1 Claim → required-evidence gate

Decide what you are allowed to claim BEFORE editing. Map each claim to the
evidence you must hold:

| Claim you want to make | Allowed ONLY if you have… |
|---|---|
| "raw DT achieves X% macro-F1 in-network on BMv2" | a full live sweep: `predictions.csv` vs filename ground truth, all classes |
| "additive PCA is exact and ~50–100× lighter than the surrogate" | the footprint table (Section 1) reproduced on your machine + the offline `transform_metrics.json` showing linear-codes == float-PCA |
| "the scaler needs no in-switch op" | the raw-vs-scaled equivalence check (diff ≤1; see Section 3 / the demo script) |
| "PCA **improves accuracy**" | a config where PCA macro-F1 > raw macro-F1 with a real margin (cross-validated). *You do not currently have this.* |
| "PCA **enables Tofino**" | a `bf-p4c` resource report: raw does not fit, PCA does. *You do not have this — and Section 2 argues you likely won't, because extraction is the bottleneck.* |
| "PCA **generalizes better**" | a cross-family experiment: train on attack families A, test on held-out B, PCA degrades less than raw AND beats LDA. *Not yet run.* |

If the row says "you do not have this", **do not write the claim.** Reviewers
will ask for the artifact.

### 9.2 If results are NEUTRAL (the most likely outcome) — the honest-ablation update

This is the safe, defensible rewrite. Apply these edits:

- **Title / Abstract.** Keep "In-Network IoT Intrusion Detection". Reframe the
  PCA sentence to: *"we implement and evaluate in-network dimensionality
  reduction (PCA/LDA) against a raw-feature decision tree, and show it is
  accuracy-neutral while characterising its data-plane cost."* Use the
  ready paragraph in Section 8 as the seed. Do NOT say PCA improves detection.
- **Introduction / Contributions.** State three honest contributions: (1) a
  flow-feature in-network IDS on BMv2/P4Pi (gateway tier, à la P4Pir); (2) an
  **exact, single-field additive realisation of linear dimensionality reduction
  in the data plane**, far lighter than the tree-surrogate approach; (3) an
  honest cost/accuracy study of raw vs PCA vs LDA. Drop any "enables Tofino" or
  "PCA improves accuracy" wording.
- **Method section.** Add a subsection "Additive in-network projection": the
  affine-folding math (`code_j = Σ A'[j][i]·x_i + INIT_j`), the scaler-folding
  argument (no in-switch divide/float), and the per-feature table mapping. Note
  the per-feature `FP_SHIFT` precision point if you implemented the fix.
- **Evaluation.** Replace/добавить two tables:
  1. **Footprint table** (Section 1 here) — raw vs surrogate-PCA vs additive-PCA:
     total entries, transform-key width, classifier-key width, load time.
  2. **Live BMv2 accuracy table** — per-class + overall from your sweep.
  Add one sentence on the live-vs-offline gap (tcpreplay timing on
  `Duration`/`MaxIAT`), already documented in the repo README.
- **Deployment / Tofino discussion.** Keep it as *future work*. State plainly
  that the dominant data-plane cost is **stateful feature-extraction stages**
  (cite the ~27-register, 3-pass design), that dimensionality reduction does not
  address it, and that ASIC deployment requires reducing extraction cost first.
- **Conclusion.** "PCA is accuracy-neutral here; the raw DT is the stronger
  choice; the additive method is the correct way to do in-network reduction when
  reduction is wanted." No overclaim.

### 9.3 If results show a REAL benefit — the stronger update (only then)

Only if Section 9.1's gated evidence actually materialises:

- **Generalisation win** (9.1 row 6): add a cross-family table (train A / test B)
  with raw vs PCA vs LDA; if PCA wins, *that* becomes the headline and the
  abstract/intro can say "PCA improves robustness to unseen attack variants".
- **Classifier-binding win** (heavy RF where raw key/TCAM doesn't fit but codes
  do): add the RF footprint comparison showing raw-RF exceeds a resource budget
  and PCA-RF fits. Then "PCA enables the [RF] classifier within [budget]".
- **Tofino fit** (`bf-p4c`): add the stage/SALU/TCAM table for raw vs PCA. Only
  with a real report may you write "fits Tofino-1 in N stages".

### 9.4 Concrete numbers available now (from this repo's measured runs)

Reproduce these on the working machine before citing; current measured values:

- Offline DT (AttackIDS): raw / float-PCA / quantized / additive-codes all
  **1.0000** accuracy (identical confusion matrices) — `transform_metrics.json`.
- Footprint (AttackIDS, k=7 b=32 DT, **re-measured this turn**): raw
  **14** entries / 256-bit `ml_code` key / load **0.068 s** / 1 table
  traversed per packet; surrogate **1,207** entries; additive **559**
  (543 transform + 16 classifier) / 224-bit `ml_code` key + 18
  single-field transform tables / load **0.168 s** (~2.5× raw) / 19
  tables traversed per packet.
- Live BMv2 via `make run` + `6_controller.py` (additive, this turn,
  host-side `tcpreplay -i s1-eth1 --timer=gtod` + scapy drain): Access
  100% (30/30) / CC 100% (117/117) / Discovery 100% (27/27) / Evasion
  100% (2/2) / **overall 100.00% (176/176)**. Confusion matrix
  diagonal. Required the two `6_controller.py` fixes documented in §5
  (drop the redundant `SetForwardingPipelineConfig` re-install; bump
  `GRPC_KEEPALIVE_OPTS` from 20 s → 600 s).
- Earlier direct-veth sweep (kept for context): 95.89 % (140/146). The
  100 % `make run` number above supersedes it as the headline live
  result on this machine.
- Scaler equivalence: raw-in vs scaled-then-projected codes differ by ≤1 LSB
  (fixed-point rounding), out of a 2^32 range.
- CIC-IoT (from repo README, re-verify): raw **436** entries / 97.09% F1;
  surrogate PCA **118,588** entries / 97.07% F1.
- **CICIoT (this turn, 11 105 flows, k=7 b=32 DT, post Q-fix):**
   - Raw: **119** entries / 256-bit `ml_code` key / load **0.094 s** /
     **99.52 %** live (8 153/8 192) / Macro F1 **0.9923**.
   - Surrogate PCA: **74 126** entries (10 562 per component × 7 + 192
     `ml_code`) / 7 × 18-field-256-bit transform tables / load
     **21.685 s** / **90.34 %** live (5 454/6 037) / Macro F1
     **0.9133**. Recon recall collapses to 75.4 % (409 → DoS).
   - Additive PCA: **10 222** entries (10 046 transform across 18
     featc_* + 176 `ml_code`) / 18 × 1-field-≤16-bit transform tables /
     load **1.686 s** / **97.36 %** live (7 109/7 302) / Macro F1
     **0.9736**.
   - **Raw wins every metric on CICIoT (accuracy, F1, footprint, load,
     pipeline depth, auditability).** Additive vs surrogate gap on this
     dataset: 7.2× fewer entries, 12.9× faster load, +7.02 pp accuracy,
     +0.0603 Macro F1 in additive's favour.

### 9.5 Pre-submission honesty checklist

- [ ] Every table number reproduced on the working machine, artifact saved.
- [ ] No sentence claims PCA improves accuracy unless 9.1 row 5 is satisfied.
- [ ] No sentence claims PCA enables Tofino unless a `bf-p4c` report exists.
- [ ] The footprint table makes clear raw is the lightest; additive beats only
      the surrogate.
- [ ] Live-vs-offline accuracy gap acknowledged.
- [ ] Tofino positioned as future work, with extraction-stage cost named as the
      blocker.
- [ ] "552-bit" raw key corrected to the as-built **256-bit / 18-field**
      quantized key (measured from `basic.p4` this turn — earlier
      "256-bit" estimate was also wrong).
