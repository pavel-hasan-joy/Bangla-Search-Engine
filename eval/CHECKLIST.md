# Evaluation Query Set Checklist & Guidelines

Per project rules, **evaluation queries and relevance labels must NEVER be synthetically generated or fabricated**. They must be provided by the human evaluator.

This document provides the category checklist and structural requirements for constructing `eval/queries.tsv`.

---

## 1. Schema Specification (`eval/queries.tsv`)

Tab-separated values (`.tsv`) with UTF-8 encoding without BOM:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | String | Unique query identifier (`dev_01`..`dev_20`, `test_01`..`test_30`) | `dev_01` |
| `query` | String | The actual input query (Banglish Romanized or mixed) | `shikkha bebostha` |
| `category` | String | Standard category tag (see checklist below) | `spelling_variant` |
| `gold_bangla_query` | String | Ground-truth native Bengali query intended by user | `শিক্ষা ব্যবস্থা` |
| `intent_note` | String | Brief description of search intent and relevance criteria | `Overview of Bangladesh education system` |

---

## 2. Dev / Test Split Requirements

- **Total Queries**: 50 queries minimum
- **Development Split (`dev_`)**: Exactly **20 queries** (`dev_01` through `dev_20`). Used for parameter tuning ($k_1, b$, expansion weights, thresholds).
- **Test Split (`test_`)**: Exactly **30 queries** (`test_01` through `test_30`). Held out strictly for final reporting and ablation comparison.

---

## 3. Category Checklist

Every query must be tagged with one of the following official category labels:

### 1. `single_word`
- Single standalone Banglish concept or entity.
- *Examples*: `shikkha` (শিক্ষা), `dhaka` (ঢাকা), `itihas` (ইতিহাস), `shadhinota` (স্বাধীনতা).

### 2. `multi_word`
- Multi-token compositional search query expressing a compound topic.
- *Examples*: `dhaka bisshobiddaloy` (ঢাকা বিশ্ববিদ্যালয়), `bangladesh shorkar` (বাংলাদেশ সরকার).

### 3. `spelling_variant`
- **Mandatory**: Minimum **10 distinct intents** $\times$ **2 to 3 spelling variants** each.
- Tests phonetic robustness against idiosyncratic romanizations.
- *Examples for Intent 1 (শিক্ষা)*:
  - Variant A: `shikkha`
  - Variant B: `sikkha`
  - Variant C: `shikha`
- *Examples for Intent 2 (বিজ্ঞান)*:
  - Variant A: `biggan`
  - Variant B: `bijñan`
  - Variant C: `bijnan`
- *Examples for Intent 3 (স্বাধীনতা)*:
  - Variant A: `shadhinota`
  - Variant B: `sadinota`
  - Variant C: `shadhinata`

### 4. `loanword`
- English or foreign loanwords transliterated into Bengali culture.
- *Examples*: `komputar` (কম্পিউটার), `daktar` (ডাক্তার), `unibarsiti` (বিশ্ববিদ্যালয়/ইউনিভার্সিটি).

### 5. `name`
- Proper nouns, geographical places, historical figures, or institutions.
- *Examples*: `sheikh mujib` (শেখ মুজিব), `rabindranath` (রবীন্দ্রনাথ), `chittagong` (চট্টগ্রাম).

### 6. `typo`
- Real-world keyboard typographical errors (transpositions, omitted vowels, adjacent keys).
- *Examples*: `shikhka` (শিক্ষা with transposed letters), `dhka` (ঢাকা with dropped vowel).

### 7. `mixed_script`
- Mixed script queries containing both Bengali letters and Roman/Banglish tokens.
- *Examples*: `বাংলাদেশ er shikkha`, `dhaka বিশ্ববিদ্যালয়`.

---

## 4. Relevance Judgment Scale (`eval/qrels.tsv`)

Documents are graded using a 3-point scale via `python -m bsearch.eval.label_tool`:

- **Grade 2 (Highly Relevant)**: Directly answers or focuses on the specific search intent.
- **Grade 1 (Partially Relevant / Marginal)**: Discusses the broader topic or related context, but does not fully satisfy the specific search intent.
- **Grade 0 (Not Relevant)**: Completely off-topic or irrelevant to the query intent.
