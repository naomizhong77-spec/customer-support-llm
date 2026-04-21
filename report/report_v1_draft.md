# Customer Support Intent Classification: Data Handling, Cleaning, and Statistics

**Author**: ZHONG QI
**Course**: CA6000
**Date**: 2026-04-21

---

## 1. Dataset Source, Import, and Error Detection

This project uses two English customer-support corpora to compare intent-classification
approaches across both templated synthetic text and real user-written queries.
Dataset 1 (Bitext) is a large synthetic-augmented corpus used as the primary
training dataset. Dataset 2 (BANKING77) is a smaller, real-user corpus used as a
noisier complement so that the downstream model comparison is not judged on
templated text alone. The same preprocessing audit (null check, duplicate
detection, encoding inspection, label-hierarchy validation) was applied to both.

### 1.1 Primary dataset: Bitext Customer Support LLM Chatbot Training Dataset

**Source.** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` on
HuggingFace Hub (also mirrored on Kaggle). License: CDLA-Sharing-1.0
(Community Data License Agreement — Sharing, v1.0).

**Purpose.** English customer-service intent detection. Each row pairs a
customer message (`instruction`) with a canonical agent reply (`response`)
and is annotated with a coarse `category` and a fine-grained `intent` in a
hierarchical label scheme.

**Import method.** The dataset was pulled once from HuggingFace using
`datasets.load_dataset` and serialised to a local parquet file
(`data/raw/bitext_customer_support.parquet`, 5.96 MB). Subsequent notebook
runs load the immutable local copy via `pandas.read_parquet` so the raw
inputs cannot drift between sessions.

**Shape and schema.** After load, the DataFrame has shape `(26872, 5)` and
the following columns, all of dtype `object`:

| Column | Description |
|---|---|
| `flags` | Augmentation tags (letters B, L, Q, Z, etc.) indicating colloquial / polite / typo variants |
| `instruction` | Customer message — the classification input |
| `category` | Coarse label (11 values, upper-case) |
| `intent` | Fine-grained label (27 values) |
| `response` | Canonical agent reply |

**Error detection performed.** Before any cleaning, the raw DataFrame was
audited for the defect classes below (notebook `01_data_exploration.ipynb`,
Section 4).

| Check | Raw Bitext result |
|---|---|
| Null / NaN values (any column) | 0 rows |
| Empty or whitespace-only strings | 0 rows (all 5 object columns) |
| Exact duplicate rows | 0 |
| Duplicates on (`instruction`, `response`) | 0 |
| Duplicates on `instruction` alone | 2,237 (paraphrase re-use — retained as legitimate signal) |
| Leading / trailing whitespace in `instruction` | 0 |
| Internal multi-space in `instruction` | 551 |
| Tabs in `instruction` | 0 |
| Non-ASCII in `instruction` | 0 |
| Non-ASCII in `response` | 111 |
| Mojibake signatures in `instruction` / `response` | 0 / 0 |
| Intents mapped to more than one category (hierarchy violation) | 0 |
| `instruction` length > 500 chars | 0 |
| `response` length > 2000 chars | 112 |

The raw Bitext dump is effectively pristine: no nulls, no exact duplicates, no
mojibake, and the 11-category / 27-intent hierarchy is invariant (every intent
maps to exactly one category). The only non-trivial finding is 551 rows with
internal multi-space in `instruction`, which is cosmetic. This cleanliness
is consistent with the dataset being programmatically generated from templates
rather than collected from real support logs.

### 1.2 Second dataset: BANKING77

**Source.** `PolyAI/banking77` on HuggingFace Hub. License: CC-BY-4.0.
Citation: Casanueva et al., *Efficient Intent Detection with Dual Sentence
Encoders*, NLP4ConvAI 2020.

**Purpose.** A real-user complement to the templated Bitext data. The contrast
enables a cross-domain comparison of the same three classifier architectures
against both a tidy synthetic distribution and a noisy real-user distribution.

**Import method.** Pulled directly via `datasets.load_dataset("PolyAI/banking77")`
and converted to pandas with `ds["train"].to_pandas()` and
`ds["test"].to_pandas()`. The integer `label` column carries a HuggingFace
`ClassLabel` feature, so the 77 intent names were materialised once via
`label_names = ds["train"].features["label"].names` and mapped onto a new
`label_name` string column in both splits.

**Shape and schema.** Two native splits, no validation set:

| Split | Rows | Columns |
|---|---|---|
| `train` | 10,003 | `text` (object), `label` (int64), `label_name` (object, derived) |
| `test` | 3,080 | same |

Total raw rows: 13,083. No category hierarchy and no agent reply — the task is
flat 77-way classification on user text.

**Error detection performed.** The same audit function (`audit()` in
notebook `07_banking77_exploration.ipynb`, Section 4) was applied to both
splits. Unlike Bitext, the raw issues discovered here are genuine artefacts
of the upstream release and were not injected.

| Check | Train | Test |
|---|---|---|
| Null `text` / Null `label` | 0 / 0 | 0 / 0 |
| Empty or whitespace-only `text` | 0 | 0 |
| Exact duplicate rows | 0 | 0 |
| Duplicates on (`text`, `label`) (raw) | 0 | 0 |
| Leading / trailing whitespace | 9 | 3 |
| Internal multi-space | 454 | 100 |
| Embedded newlines | 10 | 3 |
| Non-ASCII characters | 52 | 9 |
| Mojibake signatures | 0 | 0 |
| Text > 300 chars | 20 | 3 |
| Raw-text overlap between train and test | 0 | — |

The BANKING77 audit is noisier than Bitext's in a structurally different way.
There are no nulls or exact duplicates, but there are 564 rows across the two
splits carrying whitespace artefacts (leading, trailing, internal, newlines),
61 rows with non-ASCII characters (dominated by currency symbols such as `£`
and `€`), and 23 rows exceeding 300 chars. One of the 77 label names carries
non-lowercase casing (`Refund_not_showing_up`) and another carries a trailing
punctuation character (`reverted_card_payment?`); both are kept verbatim
because downstream models must predict the exact string released by
PolyAI.

A second-order issue surfaces only after whitespace normalisation (reported
in Section 2.2): five duplicate `(text, label)` pairs within splits and seven
cross-split leakage pairs that were masked in the raw strings by stray
newlines or double-spaces.

---

## 2. Error Fixing with Pandas

CA6000 Spec #2 requires demonstrating error-fixing with pandas techniques and
permits deliberately injecting errors when a dataset is already clean. The
Bitext pipeline (Section 2.1) follows the injection pathway because the raw
dump is pristine; the BANKING77 pipeline (Section 2.2) uses only genuine
upstream issues. Both datasets end with stratified train / val / test parquet
files written to `data/processed/` (Bitext) and `data/banking77/processed/`
(BANKING77).

### 2.1 Bitext cleaning operations

Because the raw Bitext frame contained zero targetable defects, notebook
`02_data_cleaning.ipynb` works on a deep-copied working frame with six
classes of synthetic errors deliberately injected at controlled rates. All
injection uses `numpy.random.default_rng(seed=42)` so the process is
reproducible. Counts below are taken from the injection log.

| Injected defect class | Rate | Count injected | Mechanism |
|---|---|---|---|
| `nan_intent` | ~5% | 1,344 | `df.loc[idx, "intent"] = np.nan` on random indices |
| `dup_rows` | ~2% | 537 | Sample random rows and concatenate copies |
| `mojibake` | ~1% | 269 | Prepend `Ã©` and substitute `'` with `â€™` in `response` |
| `length_outlier` | ~1% | 269 | Repeat `instruction` 10× to exceed the 500-char cap |
| `case_inconsistency` | ~1% | 269 | Alternate `.lower()` / `.title()` on `category` |
| `ws_artefact` | ~2% | 537 | Inject leading spaces, triple-spaces, trailing tab |

Injection grew the working frame from 26,872 to 27,409 rows (the 537
duplicate-row copies are concatenated; the other classes overwrite values
in place).

**Cleaning pipeline.** A deterministic nine-step pipeline (notebook 02,
Section 5) was then applied. Each step used a standard pandas or pandas-string
primitive and recorded before / after row counts.

```python
# 1. Drop rows with NaN in the classification target
df = df.dropna(subset=["intent"]).reset_index(drop=True)

# 2. Deduplicate on (instruction, intent)
df = df.drop_duplicates(subset=["instruction", "intent"], keep="first")

# 3. Collapse internal whitespace on instruction + response
df["instruction"] = df["instruction"].map(lambda s: " ".join(s.split()))
df["response"]    = df["response"].map(lambda s: " ".join(s.split()))

# 4. Repair mojibake byte signatures BEFORE NFKC (ordering matters)
df["response"] = (
    df["response"]
      .str.replace("â€™", "'", regex=False)
      .str.replace("â€œ", '"', regex=False)
      .str.replace("Ã©",  "",  regex=False)
)

# 5. Unicode NFKC normalise
df["instruction"] = df["instruction"].map(lambda s: unicodedata.normalize("NFKC", s))
df["response"]    = df["response"].map(lambda s: unicodedata.normalize("NFKC", s))

# 6. Upper-case category to the canonical ACCOUNT / ORDER / ... form
df["category"] = df["category"].str.upper()

# 7. Drop length outliers (instruction > 500 chars or response > 3000 chars)
mask_ok = (df["instruction"].str.len() <= 500) & (df["response"].str.len() <= 3000)
df = df[mask_ok].reset_index(drop=True)

# 8. Re-dedup after normalisation (whitespace collapse can expose new dupes)
df = df.drop_duplicates(subset=["instruction", "intent"], keep="first")

# 9. Assert hierarchy invariant: every intent maps to exactly one category
assert df.groupby("intent")["category"].nunique().max() == 1
```

The ordering of step 4 (byte-level mojibake repair) before step 5 (NFKC) is
load-bearing: NFKC decomposes the `â€™` sequence into a different,
non-invertible form, so the raw mojibake pattern must be replaced first.

**Cleaning log — row-level accounting.** The four row-count-changing steps
produce the trail in Table 2.1, drawn from
`outputs/metrics/data_stats.json` (`cleaning_log`):

| Step | Rows before | Rows after | Dropped | Note |
|---|---|---|---|---|
| `drop_nan_intent` | 27,409 | 26,048 | 1,361 | target column must be present |
| `dedup_instr_intent` | 26,048 | 23,588 | 2,460 | keep=first |
| `drop_length_outliers` | 23,588 | 23,477 | 111 | instruction<=500ch, response<=3000ch |
| `dedup_post_normalise` | 23,477 | 23,326 | 151 | remove dupes exposed by normalisation |

The 1,361 dropped NaN rows include the 1,344 injected values plus 17 rows
whose `intent` became NaN as a collateral effect of other in-place edits.
The 2,460 `(instruction, intent)` dedup drops include the 537 injected
duplicates plus the ~1,923 natural paraphrase-level repeats already present
in the raw frame (the audit in Section 1.1 recorded 2,237 `instruction`-only
duplicates; pair-level dedup is slightly more permissive than that).

**Post-cleaning audit.** Every targeted defect count drops to zero
(notebook 02, Section 6): 0 NaN intents, 0 duplicates on (`instruction`,
`intent`), 0 exact duplicate rows, 0 mojibake in `response`, 0 length
outliers, 0 non-upper-case categories, 0 leading/trailing whitespace, 0
multi-space instances, 0 tab characters. All 11 categories and all 27
intents are still present. Final cleaned row count: 23,326.

### 2.2 BANKING77 cleaning operations

BANKING77 is real data and required no synthetic injection. Notebook
`08_banking77_cleaning.ipynb` applies a smaller, targeted pipeline:

```python
# 1. Collapse internal whitespace + strip leading/trailing on text
df["text"] = df["text"].map(lambda s: " ".join(s.split()))

# 2. Unicode NFKC normalise
df["text"] = df["text"].map(lambda s: unicodedata.normalize("NFKC", s))

# 3. Drop rows where text became empty after normalisation (no such rows arose)
df = df[df["text"].str.len() > 0].reset_index(drop=True)

# 4. Deduplicate on (text, label), keep first
df = df.drop_duplicates(subset=["text", "label"], keep="first")
```

Length outliers (> 300 chars) and non-ASCII characters (currency symbols and
accented names) were flagged but retained because both are legitimate features
of real user queries. Label names were preserved verbatim including the one
mixed-case and one trailing-question-mark entries, since the downstream
classifiers must predict the canonical string released by PolyAI.

**Discovery by normalisation: within-split and cross-split leakage.**
The most substantive pandas-driven finding for BANKING77 is that several
duplicate and leakage pairs only become visible after whitespace
normalisation. On the raw strings, the pair-level duplicate and
train-vs-test overlap audits (Section 1.2) both returned zero. After
applying `" ".join(s.split())` followed by `unicodedata.normalize("NFKC", ...)`,
a second round of `.duplicated(subset=["text", "label"])` and of set-based
intersection across splits revealed the following:

| Defect | Count | Action |
|---|---|---|
| Within-train `(text, label)` duplicates (post-NFKC) | 4 | Dropped (10,003 → 9,999; `keep="first"`) |
| Within-test `(text, label)` duplicates (post-NFKC) | 1 | Dropped (3,080 → 3,079; `keep="first"`) |
| Cross-split `(text, label)` leakage: train vs test | 6 | Dropped from train side |
| Cross-split `(text, label)` leakage: val vs test | 1 | Dropped from val side |
| Cross-split `(text, label)` leakage: train vs val | 0 | — |
| Text > 300 chars (length outliers), train | 20 | Flagged, retained |
| Text > 300 chars (length outliers), test | 3 | Flagged, retained |
| Non-ASCII characters retained, train | 49 | Kept (currency / accents) |
| Non-ASCII characters retained, test | 9 | Kept |

The seven cross-split leakage pairs were masked in raw form by stray
leading newlines and internal double-spaces. For example,
`"\nHow do I unblock my PIN?"` (raw train) and `"How do I unblock my PIN?"`
(raw test) are distinct as bytes but identical after NFKC + whitespace
collapse. The decision rule, applied uniformly, is to drop from the
train / val side and preserve the canonical test set intact: the test
parquet is sha256-hashed and shared across all downstream models, so
modifying it retrospectively would break cross-run comparability. Dropping
the contaminating rows from train / val removes the leakage without
disturbing the pinned evaluation contract. The numerical effect of the
leakage removal, had it been left in, is approximately 7 / 3,079 ≈ 0.23
percentage points of upward accuracy inflation — small in aggregate but
unevenly distributed across the 77 classes and therefore distortionary for
per-class F1.

Final BANKING77 cleaned row count: 13,071 (9,999 train-pool + 3,079 test
less the 7 cross-split drops and 1 within-test drop, reconciled against
the pre-cleaning total of 13,083).

### 2.3 Split creation

**Bitext splits.** Stratified on `intent`, using two successive calls to
`sklearn.model_selection.train_test_split` with `random_state=42`. Round 1
allocates 75 % to `train` and 25 % to a "rest" pool. Round 2 splits the
25 % pool into `val` (40 % of rest = 10 % of total) and `test` (60 % of
rest = 15 % of total), again stratified on `intent`. Final split sizes
(from `outputs/metrics/data_stats.json.split_sizes`):

| Split | Rows | Fraction |
|---|---|---|
| `train` | 17,494 | 75.00% |
| `val` | 2,332 | 10.00% |
| `test` | 3,500 | 15.00% |
| Total cleaned | 23,326 | — |

The maximum per-intent distribution deviation between any split and the
overall cleaned frame is 0.0003 (≤ 0.03 percentage points), and
`(instruction, intent)` pairs are disjoint across splits (zero leakage).

**BANKING77 splits.** PolyAI's native `test` split (3,080 rows cleaned to
3,079) is preserved as-is and functions as the canonical evaluation set.
The cleaned `train` pool (9,999 rows) is split 88 / 12 on `label` via a
single stratified `train_test_split` call with `random_state=42`. Six
rows are subsequently removed from train (leakage vs test) and one row
from val (leakage vs test), producing the final sizes (from
`outputs/metrics/banking77_stats.json.cleaning.split_sizes`):

| Split | Rows | Note |
|---|---|---|
| `train` | 8,793 | 88 % stratified carve-out, minus 6 leakage drops |
| `val` | 1,199 | 12 % stratified carve-out, minus 1 leakage drop |
| `test` | 3,079 | HF native test, less 1 within-test duplicate |
| Cleaned HF train pool (pre-split) | 9,999 | — |
| Cleaned HF test pool | 3,079 | Pinned for evaluation |

Every one of the 77 classes appears in every split (maximum per-class
distribution deviation between the stratified train split and the overall
cleaned train pool is 0.0011 ≈ 0.11 percentage points).

**Test-set reproducibility.** Both test parquet files are sha256-hashed
and the hashes are recorded in the project's results JSON so any
downstream model can verify it is evaluating on bit-identical bytes:

- Bitext test: `sha256:5641a8ab0fb4814b` (from
  `outputs/metrics/data_stats.json` and every Bitext `results.json`;
  truncated form used by the Bitext evaluation scripts).
- BANKING77 test: `sha256:6b7f43ccbe394d73310fa8d23ac97cebf9ce1292e989bca5f6001c52d8e33ddc`
  (from `outputs/metrics/banking77_test_hash.txt` and every BANKING77
  `results.json`).

---

## 3. Dataset Statistics

This section summarises structural and distributional properties of the two
cleaned datasets. All values are aggregated over the combined
train + val + test parquet files written in Section 2 unless otherwise
noted.

### 3.1 Summary table

Table 3.1 consolidates the core descriptors side by side, with all values
traced to `outputs/consolidated/datasets_summary.json`.

| Metric | Bitext | BANKING77 |
|---|---|---|
| Total rows after cleaning | 23,326 | 13,071 |
| Number of intents | 27 | 77 |
| Number of categories | 11 | N/A (flat) |
| Train split (rows) | 17,494 | 8,793 |
| Val split (rows) | 2,332 | 1,199 |
| Test split (rows) | 3,500 | 3,079 |
| Class imbalance ratio (max/min, combined cleaned) | 1.05 (intent) / 6.30 (category) | 3.03 |
| Class imbalance ratio (max/min, raw train) | — | 5.34 (min 35, max 187) |
| Mean text length (chars) | 49.26 | 58.19 |
| Std text length (chars) | 30.68 | 39.47 |
| Median text length (chars) | 48.0 | 46.0 |
| p95 text length (chars) | 61.0 | 145.0 |
| Max text length (chars) | 499.0 | 429.0 |
| Mean text length (words) | 9.25 | 11.72 |
| Median text length (words) | 9.0 | 10.0 |
| p95 text length (words) | 13.0 | 28.0 |
| Max text length (words) | 120.0 | 79.0 |
| Test-set sha256 prefix (reproducibility pin) | `5641a8ab0fb4814b` (sorted-CSV method) | `6b7f43ccbe394d73` (file-bytes method) |

*Table 3.1. Side-by-side descriptive statistics. Text-length values are
computed over the classification input column (`instruction` for Bitext,
`text` for BANKING77).*

### 3.2 Class distribution

**Bitext intent distribution (27 classes).** Using raw-dump counts from
notebook 01 (the distribution is preserved by stratified splitting), intents
range from 950 (`check_cancellation_fee`) to 1,000 (multiple intents including
`contact_customer_service`, `complaint`, `check_invoice`). The imbalance
ratio max/min is 1.05, i.e. near-uniform. At the category level the ratio is
6.30 (min 950 for `CANCEL`, max 5,986 for `ACCOUNT`) because several intents
collapse under a shared category.

![Bitext intent distribution.](../outputs/figures/01_intent_distribution.png)

*Figure 3.1. Bitext intent distribution. 27 classes are near-uniform
(imbalance ratio 1.05).*

![Bitext category distribution.](../outputs/figures/01_category_distribution.png)

*Figure 3.2. Bitext category distribution. 11 categories, imbalance
ratio 6.30 driven by `ACCOUNT` (5,986 rows) versus `CANCEL` (950 rows).*

**BANKING77 class distribution (77 classes).** On the raw 10,003-row train
split, counts range from 35 (`contactless_not_working`) to 187
(`card_payment_fee_charged`), giving an imbalance ratio of 5.34. On the
combined cleaned train + val + test totals (n = 13,071), the range widens in
absolute terms but the ratio narrows to 3.03 (min 75, max 227). Every one
of the 77 classes has at least 35 samples in raw train (≥ 9 after a 12 %
val carve-out), so stratified splitting on `label` is feasible without
merging or oversampling.

![BANKING77 class distribution.](../outputs/figures/07_banking77_class_distribution.png)

*Figure 3.3. BANKING77 class distribution on the raw 10,003-row train
split. Imbalance ratio max/min is 5.34 (min 35, max 187).*

### 3.3 Text length distribution

**Bitext.** The `instruction` column has mean 49.26 chars, standard deviation
30.68 chars, median 48.0 chars, p95 61.0 chars, and max 499.0 chars.
Whitespace-token counts follow the same shape: mean 9.25, median 9.0,
p95 13.0, max 120.0. The distribution is tightly concentrated around the
median with a short right tail (Figure 3.4, left panel) — a signature of
templated text generation. The right panel of Figure 3.4 shows the
`response` column, which is much longer (median 540 chars, p95 1,295 chars)
and is not used for classification in this project.

![Bitext text-length distributions (chars): instruction vs response.](../outputs/figures/01_text_length_chars.png)

*Figure 3.4. Bitext text-length distributions in characters. The left
panel (instruction) is the classification input; the right panel (response)
is included for completeness. Median lines are overlaid.*

**BANKING77.** The `text` column has mean 58.19 chars, standard deviation
39.47 chars, median 46.0 chars, p95 145.0 chars, and max 429.0 chars. In
whitespace tokens, the mean is 11.72, median 10.0, p95 28.0, max 79.0.
The median is very close to Bitext's (46 vs 48 chars), but the distribution
is much more heavy-tailed — the BANKING77 p95 of 145 chars is more than
double Bitext's 61, and the p99 reaches 215 chars (Figure 3.5).

![BANKING77 text-length distribution.](../outputs/figures/07_banking77_text_length.png)

*Figure 3.5. BANKING77 text-length distribution on the raw train split.
Characters (left) and whitespace tokens (right); median and p95 lines are
overlaid.*

**Side-by-side contrast.** Figure 3.6 overlays the two distributions in
density form. Near the median the two corpora occupy a similar range,
but Bitext is concentrated and near-unimodal (consistent with templated
generation) whereas BANKING77 has a longer right tail reflecting the
variable length of real user queries.

![Text-length density comparison between Bitext and BANKING77.](../outputs/figures/07_banking77_vs_bitext_length.png)

*Figure 3.6. Density comparison of text lengths between Bitext and
BANKING77. Both distributions peak around the same median but BANKING77
has a noticeably longer right tail in both character and
whitespace-token space.*

### 3.4 Qualitative contrast

Ten random samples (seed 42) were drawn from each dataset in
`07_banking77_exploration.ipynb`, Section 7. Five from each are reproduced
in Table 3.2 for descriptive comparison.

| Source | Intent / label | Text |
|---|---|---|
| BANKING77 | `change_pin` | Is it possible for me to change my PIN number? |
| BANKING77 | `declined_card_payment` | I'm not sure why my card didn't work |
| BANKING77 | `top_up_failed` | I don't think my top up worked |
| BANKING77 | `card_payment_fee_charged` | Can you explain why my payment was charged a fee? |
| BANKING77 | `balance_not_updated_after_bank_transfer` | How long does a transfer from a UK account take? I just made one and it doesn't seem to be working, wondering if everything is okay |
| Bitext | `check_cancellation_fee` | assistance seeing the termination penalty |
| Bitext | `track_refund` | where could I check the current status of the compensation? |
| Bitext | `create_account` | I need information about opening a `{{Account Category}}` account |
| Bitext | `recover_password` | I don't know what I have to do to reset my pass |
| Bitext | `payment_issue` | assistancesolving a trouble with payment |

*Table 3.2. Random samples (seed 42) from each cleaned dataset.*

The Bitext rows are short (typically fewer than ten words), often lower-case,
include unresolved template placeholders such as `{{Account Category}}` and
`{{Order Number}}`, and occasionally contain deliberate typographical
artefacts (e.g. `assistancesolving`, `pass` for `password`) associated with
the `Z` augmentation flag. The BANKING77 rows are formed as natural
questions ("Is it possible for me…", "Can you explain why…", "How long does
a transfer…"), use standard punctuation, and carry no placeholder tokens.
They also span a wider length range: the last example in Table 3.2 is 143
characters, which exceeds Bitext's p95 length of 61 characters. This
syntactic and lexical contrast underpins the decision to evaluate all three
downstream classifiers on both datasets rather than on one alone.

---

## 4. Neural Network Architectures

### 4.1 Model overview

This project trains a 2 x 3 experiment grid: three classifier families are
trained independently on the Bitext corpus (27-intent templated support
dialogue) and on the BANKING77 corpus (77-intent real-user banking queries),
yielding six fully-trained runs in total. The CA6000 spec requires a neural
predictor; two of the three classifier families are therefore neural and
are treated as the primary architectures, while the third (TF-IDF + logistic
regression) is included as a non-neural lexical baseline used to establish a
lower bound on the 2 x 3 comparison. The two neural architectures are
DistilBERT fine-tuned with a classification head (Section 4.3) and
Qwen2.5-7B fine-tuned with QLoRA in a generative-classification formulation
(Section 4.4).

### 4.2 Baseline (non-neural): TF-IDF + Logistic Regression

The lexical baseline uses `sklearn.feature_extraction.text.TfidfVectorizer`
with `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`,
and `lowercase=True` (see `configs/tfidf_config.yaml`). The unigram/bigram
TF-IDF matrix is fed to a multinomial
`sklearn.linear_model.LogisticRegression` configured with `solver="saga"`,
`multi_class="multinomial"`, `C=1.0`, `max_iter=1000`, `n_jobs=-1`, and
`random_state=42`. Total parameter counts (from the `resources.parameters_total`
field in each run's `results.json`) are 127,845 for Bitext
(4,734-token vocabulary x 27 intent classes plus biases) and 726,033 for
BANKING77 (larger vocabulary and 77 classes). The model is included as a
lexical-features baseline establishing the lower bound against which the
two neural models are compared; it is not itself a neural predictor.

### 4.3 Primary neural architecture 1: DistilBERT fine-tuning

The first neural architecture is `distilbert-base-uncased`, a 66M-parameter
distilled-BERT encoder with 6 transformer layers, 768 hidden dimensions, and
12 attention heads per layer. A single linear classification head is
appended to the `[CLS]` pooled output: 27 output units for Bitext and 77
output units for BANKING77. All parameters are fine-tuned (no layer
freezing): the `resources.parameters_trainable` field reports 66,974,235
trainable parameters for the Bitext run and 67,012,685 for the BANKING77 run
(the small delta is the extra classification-head weights for the larger
label space). Inputs are tokenised with the WordPiece tokenizer at
`max_length=128`; Bitext p99 text length is ~21 whitespace tokens and
BANKING77 p95 is 28 tokens, so 128 provides comfortable headroom.

Training uses the standard cross-entropy objective (`problem_type:
single_label_classification`) with AdamW, peak learning rate `2e-5`, linear
schedule with `warmup_ratio=0.1`, weight decay 0.01, per-device train
batch size 32, bf16 mixed precision (A40 does not support fp16 stably),
and `seed=42`. The Bitext run trains for 3 epochs (`configs/distilbert_config.yaml`);
the BANKING77 run trains for 10 epochs (`configs/distilbert_banking77_config.yaml`).
Both runs specify `early_stopping_patience=2` on `eval_macro_f1` with
`load_best_model_at_end=true`. The 10-epoch BANKING77 setting is the
fixed run: an earlier 5-epoch run experienced silent early-stopping (no
callback fired) because `metric_for_best_model` was not aligned with the
metric key actually emitted by `compute_metrics`, so the run effectively
never saved a non-initial best checkpoint and terminated at 86.00% test
accuracy. After aligning the metric name and raising the budget to 10
epochs, the retry reached 91.78% test accuracy. Both the buggy and fixed
outcomes are recorded in `all_runs.json.known_retros`, and the retrospective
is discussed further in Section 5.4.

### 4.4 Primary neural architecture 2: Qwen2.5-7B with QLoRA fine-tuning

The second neural architecture is `unsloth/Qwen2.5-7B-Instruct`, a 7B
decoder-only transformer used in a generative-classification formulation:
the model is prompted with an Alpaca-style instruction template and is
trained to emit the intent label as text. The base weights are loaded in
4-bit NF4 quantisation with double-quantisation enabled and bf16 compute
dtype (`bnb_4bit_quant_type: nf4`, `bnb_4bit_use_double_quant: true`,
`bnb_4bit_compute_dtype: bfloat16`). LoRA adapters are attached with
`r=16`, `lora_alpha=32`, `lora_dropout=0.0`, `bias="none"`,
`task_type="CAUSAL_LM"`, and `target_modules=[q_proj, k_proj, v_proj,
o_proj, gate_proj, up_proj, down_proj]` — i.e. adapters on all four
self-attention projections and all three MLP projections at every
decoder layer. The result is 40,370,176 trainable parameters sitting on
top of a 4,393,342,464-parameter frozen base, i.e. 0.919% of total
parameters are trainable (figures from the Bitext Qwen run's
`resources.parameters_trainable` / `resources.parameters_total`).

The training objective is the standard causal-LM next-token loss, but
restricted to the response span via TRL's
`DataCollatorForCompletionOnlyLM`. The collator masks out the loss on
every token preceding a `response_template` anchor string; only tokens
after the anchor contribute to the gradient. The anchor differs by
dataset: `"Category:"` for Bitext (where the target is the two-line
"Category: X\nIntent: Y" output) and `"Intent:"` for BANKING77 (where the
target is the single-line "Intent: <label>" output). Inputs are formatted
as Alpaca-style JSONL (`{instruction, input, output}` per line) rendered
through Qwen's chat template (`<|im_start|>` / `<|im_end|>` turn
delimiters) with `max_seq_length=256`. Training runs for 3 epochs at
effective batch size 16 (`per_device_train_batch_size=4`,
`gradient_accumulation_steps=4`), peak learning rate `2e-4`, cosine
schedule with `warmup_ratio=0.03`, `paged_adamw_8bit` optimiser, bf16
mixed precision, gradient checkpointing on, and `seed=42`. At inference
time generation is greedy (`do_sample=false`, `temperature=0.0`) with
`max_new_tokens=32` for Bitext and `max_new_tokens=24` for BANKING77,
and the resulting text is parsed back into a label via a regex cascade
(strict-anchor, then bare-label fallback, then fuzzy match).

### 4.5 Hyperparameter summary

Table 4.1 consolidates the key training hyperparameters for all six runs.
Learning rate, batch size, epochs, and trainable-parameter counts are read
from the relevant config YAML and from each run's `results.json`.

| Model | Dataset | Epochs | Batch (effective) | Learning rate | Trainable params | Key config notes |
|---|---|---|---|---|---|---|
| TF-IDF + LR | Bitext | n/a (convex) | n/a | n/a (saga, `C=1.0`) | 127,845 | 1-2gram, `min_df=2`, `max_df=0.95`, sublinear TF |
| TF-IDF + LR | BANKING77 | n/a (convex) | n/a | n/a (saga, `C=1.0`) | 726,033 | Same vectoriser; 77 output classes |
| DistilBERT | Bitext | 3 | 32 | 2.0e-5 (linear, warmup=0.1) | 66,974,235 | bf16, max_length=128, early stop patience 2 on `eval_macro_f1` |
| DistilBERT | BANKING77 | 10 | 32 | 2.0e-5 (linear, warmup=0.1) | 67,012,685 | bf16; epochs raised from 5 after metric-name retro (Section 5.4) |
| Qwen2.5-7B QLoRA | Bitext | 3 | 16 (4 x 4 grad accum) | 2.0e-4 (cosine, warmup=0.03) | 40,370,176 trainable / 4,393,342,464 total | 4-bit NF4 base, LoRA r=16 alpha=32 dropout=0 on {q,k,v,o,gate,up,down}_proj, response anchor `"Category:"` |
| Qwen2.5-7B QLoRA | BANKING77 | 3 | 16 (4 x 4 grad accum) | 2.0e-4 (cosine, warmup=0.03) | 40,370,176 trainable / 4,393,342,464 total | Same LoRA config; response anchor `"Intent:"` |

*Table 4.1. Training hyperparameters for all six runs in the 2 x 3 grid.
"Effective batch" reports per-device batch size x gradient accumulation.
TF-IDF + LR rows show "n/a" for epoch-based fields because the logistic
regression is fit by saga to convergence rather than by mini-batch SGD.*

---

## 5. Training and Evaluation

### 5.1 Training infrastructure

All neural training (DistilBERT and Qwen QLoRA, both datasets) was
submitted via SLURM to the CCDS-TC2 cluster's `MGPU-TC2` partition and ran
on a single NVIDIA A40 (48GB VRAM, compute capability sm_86, Ampere). bf16
mixed precision was used throughout; fp16 was avoided because it is
known to be unstable on A40 for 7B-scale generative fine-tuning. The two
TF-IDF + LR runs ran on CPU (`device: CPU` in `results.json.latency`)
because the dataset is small and saga converges in under two seconds.
Reproducibility is enforced by `seed=42` seeded across `torch`, `numpy`,
Python `random`, and `sklearn.model_selection.train_test_split`, together
with the sha256 test-set hash pinned in every `results.json`
(`sha256:5641a8ab0fb4814b` for Bitext, full
`sha256:6b7f43ccbe394d73310fa8d23ac97cebf9ce1292e989bca5f6001c52d8e33ddc`
for BANKING77).

### 5.2 Training protocol

Each model was trained in a three-stage protocol. First, a smoke test was
run on ~100 training samples for at most 50 optimisation steps to verify
that the loss decreased, the tokenizer / collator / parser pipeline
executed end-to-end without CUDA OOM, and that at least one checkpoint
was written to disk. The Qwen config exposes the smoke-mode overrides
directly (`runtime.smoke_train_samples=100`,
`runtime.smoke_eval_samples=50`, `runtime.smoke_max_steps=50`,
`runtime.smoke_save_steps=25`). Second, the full training run was
submitted with periodic validation evaluation at the config's
`eval_strategy` cadence (per-epoch for DistilBERT,
every 200 steps for Qwen). Third, a dedicated evaluation pass was run
over the pinned test parquet, measuring top-1 predictions, per-class
precision/recall/F1, a confusion matrix, and inference latency
percentiles (p50, p95, p99) computed over a fixed sample (500 for
TF-IDF, 300 for DistilBERT and Qwen; see `results.json.latency.n_samples_timed`).

### 5.3 Evaluation metrics

Primary test-set metrics are top-1 accuracy and macro-F1 (equal-weighted
across classes; appropriate for BANKING77 because of its mild class
imbalance, ratio 3.03). Weighted-F1 is reported secondarily. Inference
latency is reported at p50, p95, and p99 (`results.json.latency.p50_ms`
etc.), all measured with batch size 1 so the numbers correspond to a
single-request interactive-use profile rather than to batch throughput.
For the Qwen generative classifier, an additional parse-error rate is
tracked: the fraction of generated outputs that fail the regex /
bare-label / fuzzy-match cascade and map to no known label. The
BANKING77 Qwen run is the subject of a significant retrospective on
this metric: its initial evaluation reported 3079/3079 parse errors and
0.00% test accuracy before the parser was patched, after which
accuracy settled at 91.65% with only 7/3079 residual parse errors
(`all_runs.json.known_retros[1]`). The retrospective is discussed in
Section 5.4.

### 5.4 Training process observations

**TF-IDF + LR.** Both TF-IDF runs complete in under two seconds on CPU:
0.782 s on Bitext and 1.203 s on BANKING77 (`summary_table.rows[0,3]
training_time_sec`). Saga reaches its convergence tolerance well within
the `max_iter=1000` budget on both datasets and no convergence warnings
are emitted.

**DistilBERT.** On Bitext, 3 epochs at batch size 32 over the 17,494-row
train split complete in 57.697 s wall-clock (`summary_table.rows[1]`).
Validation loss plateaus early (both val and test macro-F1 exceed 0.996
at epoch 3) and early stopping does not fire. On BANKING77 the initial
training run (5 epochs) terminated with 86.00% test accuracy because a
misspelled `metric_for_best_model` key silently disabled the early-stopping
/ best-checkpoint callbacks: the callback was evaluating an unknown metric
name against the `compute_metrics` output, which returned a different key,
so it was unable to detect improvement and the run loaded the epoch-1
checkpoint at end. After aligning the metric name and raising
`num_train_epochs` from 5 to 10 in `configs/distilbert_banking77_config.yaml`,
a retrained run converged to 91.78% test accuracy in 105.373 s wall-clock
(`summary_table.rows[4]`; `known_retros[0]`). Both numbers are recorded
verbatim in `all_runs.json.known_retros` and the fixed run is the one
used for all Section 6 comparisons.

**Qwen2.5-7B QLoRA.** The Bitext Qwen run trained in 5,813.164 s
(~97 minutes; `summary_table.rows[2]`) for 3 epochs at effective batch
size 16 over 17,494 training examples; peak training GPU memory was
7,166 MB, well below the A40's 48 GB budget. The BANKING77 Qwen run
trained in 2,781.031 s (~46 minutes; `summary_table.rows[5]`), roughly
half the Bitext wall-clock because the BANKING77 train split has 8,793
rows (approximately half as many) and the target string is shorter
("Intent: <label>" vs Bitext's two-line "Category: X\nIntent: Y").
Neither Qwen run suffered a training-side failure: loss curves for both
decreased monotonically past the early-stopping eval points. The
BANKING77 Qwen run did, however, suffer a significant evaluation-side
failure. The initial eval pass on the final adapter returned 0.00%
accuracy with 3,079 / 3,079 outputs flagged as parse errors. On
investigation the cause was a dual-role mismatch between the
`response_template` (`"Intent:"`) used by
`DataCollatorForCompletionOnlyLM` as a loss-masking anchor during
training, and the same string as a parser contract at evaluation time.
The collator's masking caused the model to stop emitting the literal
`"Intent:"` prefix, so the strict-anchor parser regex saw no match on
any output; but the model was correctly emitting bare label strings
because the supervised label text came immediately after the
masked anchor. No retraining was performed; instead the parser was
patched to add a bare-label match stage followed by a fuzzy Levenshtein
fallback, and a re-evaluation pass produced 91.65% test accuracy with
7 residual parse errors (`reval_generation_wallclock_seconds: 1351.398`
in the Qwen BANKING77 run; `known_retros[1]`). Of the 3,079 test
outputs, the fixed parser resolved 2,972 via the bare-label path,
100 via fuzzy match, and left only 7 unresolved
(`parse_path_counts: {regex: 0, bare: 2972, fuzzy: 100, none: 7}`).
This retrospective is referenced again in Section 7.

### 5.5 Error analysis (BANKING77 per-intent)

Aggregate test accuracy is a lossy descriptor of classifier quality on
BANKING77 because per-intent F1 varies substantially across the 77
classes even when aggregate macro-F1 converges. Using
`outputs/consolidated/per_intent_comparison.csv`, Table 5.1 lists the
five hardest BANKING77 intents ranked by DistilBERT F1 ascending, with
the corresponding TF-IDF and Qwen F1 values and the winning model per
intent.

| Intent | Support | TF-IDF F1 | DistilBERT F1 | Qwen F1 | Best model |
|---|---|---|---|---|---|
| balance_not_updated_after_bank_transfer | 40 | 0.6905 | 0.7901 | 0.7229 | distilbert |
| topping_up_by_card | 40 | 0.7838 | 0.7949 | 0.8649 | qwen |
| why_verify_identity | 40 | 0.8354 | 0.8000 | 0.9750 | qwen |
| declined_transfer | 40 | 0.9067 | 0.8169 | 0.8571 | tfidf |
| pending_transfer | 40 | 0.7353 | 0.8250 | 0.7895 | distilbert |

*Table 5.1. Five hardest BANKING77 intents, sorted by DistilBERT F1
ascending. Values are read verbatim from `per_intent_comparison.csv`.
Each intent has 40 test-set examples except `atm_support` (39), which
does not appear in this top-5.*

Beyond the tail of Table 5.1, the hardest confusions cluster around
near-synonym card-problem intents: `virtual_card_not_working`,
`card_not_working`, `compromised_card`, `lost_or_stolen_card`, and
`card_about_to_expire` all describe a broadly similar customer concern
("something is wrong with my card") and are separated by fine-grained
distinctions that lexical surface forms do not reliably encode. The
TF-IDF baseline effectively collapses on
`virtual_card_not_working` with F1 = 0.5185 (precision 1.0, recall
0.35) — it only ever predicts this intent when the exact phrase appears
and otherwise assigns probability mass to other card-problem classes.
DistilBERT recovers sharply on this same intent to F1 = 0.9067
(precision 0.9714, recall 0.85), and Qwen reaches F1 = 0.9630
(precision 0.9512, recall 0.975), a further +5.63 pp over DistilBERT
(per-intent values from `per_intent_comparison.csv`, rows for
`virtual_card_not_working`).

The inverse pattern appears on `compromised_card`, where Qwen regresses
relative to DistilBERT: its precision is 1.0 but recall is 0.375, for
F1 = 0.5455, while DistilBERT holds F1 = 0.8312 (precision 0.8649,
recall 0.8) on the same intent. Inspection of the BANKING77 Qwen
confusion matrix (Figure 5.1, lower panel) shows that Qwen's errors on
`compromised_card` are concentrated on two neighbouring labels:
`lost_or_stolen_card` and `card_about_to_expire` — i.e. Qwen tends to
re-route genuine "compromised card" queries into other card-replacement
intents. This is consistent with the LoRA adapter being insufficient
to displace the Qwen2.5 base model's general-English prior, which
lexically and semantically clusters "compromised", "lost", "stolen",
and "expiring" cards under a single "card needs replacing" concept.
Only 0.919% of the 4.39B-parameter model is trainable under the
chosen LoRA configuration (r=16, alpha=32 on seven projections per
layer); this budget is sufficient to learn the 77-way label
vocabulary and the majority of fine-grained distinctions, but
evidently not sufficient to overwrite this particular prior within
three epochs of training.

![DistilBERT BANKING77 confusion matrix.](../outputs/figures/09_distilbert_banking77_confusion_matrix.png)

*Figure 5.1a. DistilBERT BANKING77 test-set confusion matrix, row-normalised,
77 x 77, n = 3,079. Diagonal structure dominates; off-diagonal mass concentrates
along the card-problem intent cluster.*

![Qwen2.5-7B QLoRA BANKING77 confusion matrix (reval).](../outputs/figures/10_qwen_lora_banking77_confusion_matrix_reval.png)

*Figure 5.1b. Qwen2.5-7B QLoRA BANKING77 test-set confusion matrix
(post-parser-fix re-evaluation), row-normalised, 77 x 77, n = 3,079.
The diagonal is comparable to DistilBERT's but the off-diagonal
pattern differs — notably a visible row-wise leakage from
`compromised_card` to `lost_or_stolen_card` / `card_about_to_expire`.*

The two Bitext confusion matrices are essentially diagonal across all
three models (test macro-F1 at 0.9922 / 0.9981 / 0.9999 respectively),
which is consistent with the dataset's templated-generation origin
(Section 3). They are not embedded here because the near-diagonal
structure adds little descriptive information beyond the aggregate
metrics; the PNG files remain available in `outputs/figures/` for
reference (files
`03_tfidf_confusion_matrix.png`, `04_distilbert_confusion_matrix.png`,
`06_qwen_lora_confusion_matrix.png`).

---

## 6. Final Accuracy

### 6.1 Results summary

Table 6.1 consolidates the complete 2 x 3 result grid. Every value is
read verbatim from `outputs/consolidated/all_runs.json.summary_table.rows`;
accuracy and macro-F1 are reported to four decimal places, latency to two
decimal places, and training time to three decimal places — matching the
precision stored in source. For the Qwen rows, the
`summary_table.model_params` field stores the total (base + adapter)
parameter count of 4,393,342,464; the table below shows the trainable
count (40,370,176 = 40.37M) with the total in the `note`. Bitext Qwen's
`val_acc` and `val_macro_f1` are `null` in source (the Bitext Qwen run
used `eval_loss` as the best-checkpoint metric and did not write
`val_acc` / `val_macro_f1` to `results.json`) and are displayed as "—".

| Dataset | Model | val_acc | val_macro_f1 | test_acc | test_macro_f1 | p50 (ms) | p95 (ms) | Trainable params | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Bitext | TF-IDF + LR | 0.9889 | 0.9882 | 0.9926 | 0.9922 | 0.76 | 0.79 | 127,845 | 0.782 |
| Bitext | DistilBERT | 0.9961 | 0.9960 | 0.9980 | 0.9981 | 4.95 | 5.22 | 66,974,235 | 57.697 |
| Bitext | Qwen2.5-7B QLoRA | — | — | 0.9997 | 0.9999 | 551.14 | 648.80 | 40,370,176 (of 4.39B total) | 5813.164 |
| BANKING77 | TF-IDF + LR | 0.8499 | 0.8422 | 0.8535 | 0.8527 | 1.47 | 1.51 | 726,033 | 1.203 |
| BANKING77 | DistilBERT | 0.9099 | 0.9081 | 0.9178 | 0.9179 | 4.89 | 5.23 | 67,012,685 | 105.373 |
| BANKING77 | Qwen2.5-7B QLoRA | 0.9124 | 0.9012 | 0.9165 | 0.9050 | 273.13 | 511.27 | 40,370,176 (of 4.39B total) | 2781.031 |

*Table 6.1. Full 2 x 3 results grid across three models and two datasets.
Values from `all_runs.json.summary_table.rows`; Bitext Qwen val fields
are null in source. p50/p95 latency is single-request (batch size 1)
on CPU for TF-IDF and on A40 for DistilBERT / Qwen. Qwen trainable
parameter count is the LoRA adapter size (40.37M); the 4.39B total
includes the frozen 4-bit NF4 base weights.*

### 6.2 Cross-dataset observation

On Bitext, all three models achieve test accuracy above 99% (TF-IDF 99.26%,
DistilBERT 99.80%, Qwen QLoRA 99.97%), and macro-F1 values track accuracy
closely (0.9922 / 0.9981 / 0.9999). In this sense the Bitext dataset is
saturated from a classification standpoint — the remaining
between-model gap of roughly 0.7 percentage points of accuracy is within
the noise floor introduced by templated-text regularity and by the
deduplication choices discussed in Section 2.1. BANKING77 produces a
meaningful spread: TF-IDF reaches 0.8535 test accuracy, DistilBERT 0.9178,
and Qwen QLoRA 0.9165, so the two neural models exceed the lexical
baseline by approximately 6 percentage points on the real-user corpus but
are tied within noise with each other in aggregate test accuracy
(difference 0.13 pp). The per-intent decomposition in Section 5.5 shows
that this aggregate tie masks non-negligible per-class disagreement
between DistilBERT and Qwen, with the two models making errors on
different subsets of the 77 intents. The two neural models also differ
substantially on the inference-cost axis: Qwen's p50 inference latency is
273.13 ms versus DistilBERT's 4.89 ms on the same BANKING77 test set (a
factor of roughly 55x), and the trained Qwen artefact is a 170.0 MB LoRA
adapter loaded on top of a 4.39B-parameter 4-bit quantised base, while
the DistilBERT artefact is a 256.56 MB full-parameter checkpoint.
