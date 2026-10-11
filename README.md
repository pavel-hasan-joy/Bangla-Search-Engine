# banglish-search

A high-performance search engine for Bengali text supporting **Banglish** (phonetic romanization) and native Bengali queries.

Built entirely with the **Python standard library** for its core engine (Unicode normalization, tokenization, inverted indexing, BM25 ranking, phonetic keys, edit distance, and retrieval evaluation).

---

## Key Principles & Rules

1. **Pure Python Standard Library Core**: No external search libraries (Lucene, Elasticsearch, Whoosh, Tantivy), no tokenization/NLP libraries, no embedding models, and no rank_bm25.
2. **Allowed Non-Core Packages**:
   - `FastAPI` & `uvicorn` (Search API service)
   - `pytest` (Automated testing)
   - `numpy` & `matplotlib` (Offline evaluation and plotting only)
3. **Reproducibility**: No fabricated metrics or relevance labels. All evaluation metrics are generated from reproducible scripts using human-provided ground truth.
4. **Editable Rules**: All phonetic tables, character normalizations, stopwords, and suffixes reside in editable configuration files under [`rules/`](rules/).
5. **Attribution**: Content extracted from Wikipedia dumps is credited under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

---

## Repository Structure

```
banglish-search/
├── README.md                      # Project documentation and quickstart
├── PLAN.md                        # Architecture, file responsibilities, and risks
├── Makefile                       # Common automation targets
├── pyproject.toml                 # Package definition and dependencies
├── rules/                         # Human-editable linguistic and phonetic rule tables
│   ├── unicode_normalization.json # Unicode mappings and normalizations
│   ├── phonetic_rules.json        # Banglish-to-Bangla phonetic mapping rules
│   ├── bangla_stopwords.txt       # Bangla stopword list
│   └── bangla_suffixes.txt        # Inflectional suffixes for stemming
├── src/
│   └── bsearch/
│       ├── unicode/               # Unicode canonicalization and script detection
│       ├── phonetic/              # Romanization normalizer and candidate generator
│       ├── index/                 # Inverted index, tokenizer, and document store
│       ├── rank/                  # BM25 ranking and candidate score aggregator
│       ├── data/                  # Wiki dump stream-parser, markup stripper, chunker
│       ├── eval/                  # IR evaluation metrics (MRR, MAP, NDCG, P@K)
│       ├── bench/                 # Latency and memory benchmarking
│       └── api/                   # FastAPI search service
├── eval/                          # Ground-truth queries and relevance judgments
├── results/                       # Verifiable benchmark and evaluation logs
├── tests/                         # Comprehensive pytest test suite
└── web/                           # Search web application
```

---

## Quickstart

### 1. Requirements & Setup

Requires Python 3.11+.

```powershell
# Using uv (recommended)
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Or using standard `pip`:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Wikipedia Dump Ingestion

Download the latest Bangla Wikipedia XML dump (`bnwiki-latest-pages-articles.xml.bz2`) from [dumps.wikimedia.org](https://dumps.wikimedia.org/bnwiki/latest/):

```powershell
python -m bsearch.data.dump_parser \
    --input bnwiki-latest-pages-articles.xml.bz2 \
    --output data/bnwiki_articles.jsonl \
    --chunk-output data/bnwiki_passages.jsonl \
    --chunk-words 150
```

To inspect official live Bangla Wikipedia statistics without downloading:
```powershell
python -m bsearch.data.dump_parser --live-stats
```

### 3. Running Automated Tests

```powershell
pytest tests/ -v
```

---

## Implemented Milestones & Modules

### Task 0: Foundation & Corpus Pipeline
- Project structure initialized under Python 3.11+ using purely standard library for core modules.
- Streaming MediaWiki dump parser ([`dump_parser.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/data/dump_parser.py)) using `xml.etree.ElementTree.iterparse` with $O(1)$ memory consumption.
- Custom wikitext markup cleaner ([`markup_cleaner.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/data/markup_cleaner.py)) handling nested templates, tables, HTML tags, and wikilinks.
- ~150-word passage chunker ([`chunker.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/data/chunker.py)) respecting Bengali sentence boundaries (Dari `।`, `?`, `!`, newlines).
- Verified live article count via Wikimedia API (191,980+ articles, exceeding 100k).

### Task 1: Unicode Normalizer, Tokenizer & Stemmer (`bsearch.unicode`)
- **Normalizer** ([`normalizer.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/unicode/normalizer.py)):
  - Canonical NFC composition.
  - Canonical atomic forms for `য়` (`\u09DF`), `ড়` (`\u09DC`), `ঢ়` (`\u09DD`).
  - Two-part vowel signs composition (`ে+া` -> `ো`, `ে+ৗ` -> `ৌ`).
  - Repeated hasanta collapse (`্+` -> `্`).
  - Danda standardization (`॥` -> `।`).
  - Pre-base vowel sign repair for typing errors (e.g., `িশক্ষা` -> `শিক্ষা`, `েখলা` -> `খেলা`, `ৈতরি` -> `তৈরি`, `েপ্রম` -> `প্রেম`).
  - Bengali numerals mapped to ASCII (`০-৯` -> `0-9`).
  - Documented ZWJ (`\u200D`) and ZWNJ (`\u200C`) policy (stripped for robust retrieval by default; preservable via flag).
- **Grapheme Segmenter** ([`grapheme.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/unicode/grapheme.py)):
  - Preserves complex consonant conjuncts (`ক্ষ`, `স্ক্র`, `র্ক`, `ক্য`, `ত্ব`, `ক্ষ্মী`).
  - Attaches vowel signs (matras) and modifiers (`ং`, `ঃ`, `ঁ`) to base syllables.
- **Tokenizer** ([`tokenizer.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/unicode/tokenizer.py)):
  - Never splits Bengali conjuncts.
  - Lowercases Latin tokens; preserves decimal numbers (e.g. `3.11`).
  - Handles intra-word hyphens (`ঢাকা-সিলেট`, `state-of-the-art`, `covid-19`) and apostrophes (`user's`).
- **Stemmer** ([`stemmer.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/unicode/stemmer.py)):
  - Driven by editable [`rules/suffixes.tsv`](file:///d:/projects/bangla%20search%20engine/rules/suffixes.tsv).
  - Protects base roots via minimum stem length constraints (`min_stem_len >= 2`).
  - Configurable behind the `stem=True` flag in the tokenizer.

### Task 2: Inverted Index, DocStore & BM25 / WAND Retrieval (`bsearch.index`, `bsearch.rank`)
- **Varint Compression** ([`vbyte.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/index/vbyte.py)): Fast variable-byte encoding/decoding for doc-ID deltas and frequencies.
- **Block-Based Postings** ([`postings.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/index/postings.py)):
  - Postings grouped into blocks of 128.
  - Block skip index containing `max_doc_id`, `offset`, `count`, and `max_score` for Block-Max WAND.
  - $O(1)$ block skipping via `advance(target_doc_id)`.
- **Separate DocStore** ([`docstore.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/index/docstore.py)):
  - Binary offset index (`docstore.doci`) and data payload (`docstore.docd`).
  - Zero-copy `mmap` random access; instant snippet extraction.
- **Inverted Index Reader** ([`inverted_index.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/index/inverted_index.py)):
  - Binary term dictionary with `df`, postings offset/length, and upper-bound term score.
  - Memory-mapped postings with sub-millisecond to sub-50ms load time.
- **BM25 Scoring** ([`bm25.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/rank/bm25.py)):
  - Configurable $k_1$ (default 1.2), $b$ (default 0.75), and title boost (default 2.0).
  - Robertson-Spärck Jones IDF with smoothed positive floor.
  - Verified against exact hand-computed mathematical values in unit tests.
- **Top-K Retrieval** ([`topk.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/rank/topk.py)):
  - Exhaustive Document-At-A-Time (DAAT) top-$k$ using min-heap.
  - Weak AND (WAND) dynamic pruning with block skips.
  - **1000-query validation**: Automated test proves WAND and DAAT produce 100% identical top-$k$ documents and scores.
- **Index Build Telemetry (1,500 Wikipedia Articles)**:
  - **Build Time**: 0.93 seconds
  - **Total Disk Size**: 1.96 MB
  - **Process RSS**: 29.40 MB
  - **mmap Load Time**: 52.46 ms

### Task 3: Phonetic Engine & Banglish-to-Bangla Candidates (`bsearch.phonetic`)
- **Greedy Roman Pattern Parser** ([`parser.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/phonetic/parser.py)):
  - Driven by editable [`rules/roman_patterns.tsv`](file:///d:/projects/bangla%20search%20engine/rules/roman_patterns.tsv).
  - Greedy longest-match parser covering all required patterns: `sh`, `s`, `ch`, `chh`, `kh`, `gh`, `ng`, `kkh`, `gg`, `ee`, `oo`, `oi`, `ou`, `w`, `y`, `z`, `j`, `f`, `ph`.
  - Supports capital letters (`T`, `D`, `R`, `S`, `N`) as optional retroflex hints.
  - Expands ambiguous chunks up to 8 candidate phonetic spellings.
- **Grapheme-Aware G2P-lite** ([`g2p.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/phonetic/g2p.py)):
  - Syllable/grapheme-aware phoneme generator.
  - Word-initial vs. medial conjunct rules:
    - `ক্ষ` -> `kh` initial (`ক্ষতি` -> `khoti`), `kkh` medial (`শিক্ষা` -> `shikkha`).
    - `জ্ঞ` -> `g` initial (`জ্ঞান` -> `gan`), `gg` medial (`বিজ্ঞান` -> `biggan`).
  - Handles sibilants (`শ, ষ, স`), conjuncts (`ত্র, ত্ব, স্ব, স্ম`), matras, and modifiers (`ং` -> `ng`, `ঃ` -> `h`).
- **Three-Level Phonetic Keys** ([`keys.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/phonetic/keys.py)):
  - **K1 (Fine)**: Exact phonetics, normalized lowercase, retroflex hints mapped.
  - **K2 (Medium)**: Collapses aspiration (`kh->k`, `gh->g`, `chh/ch->c`, `th->t`, `dh->d`, `ph->p`, `bh->b`), gemination (`kk->k`, `kkh->k`, `tt->t`), merges `s/sh -> s`, `j/z -> j`, and `a/o -> a`.
  - **K3 (Skeleton)**: Consonant skeleton with vowels stripped and adjacent identical consonants collapsed.
- **Phonetic Index & Fuzzy Retrieval** ([`index.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/phonetic/index.py)):
  - Inverted mappings for K1, K2, and K3 keys to Bangla vocabulary terms.
  - Padded character-trigram inverted index over K2 keys.
  - Pure Python hand-written **Damerau-Levenshtein** distance ([`distance.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/phonetic/distance.py)) supporting insertions, deletions, substitutions, and transpositions.
  - Adaptive cutoff: distance $\le 1$ for keys $\le 5$ chars, else $\le 2$.
  - **Ambiguity Validation**: `shikkha`, `sikkha`, and `shikha` all reach `শিক্ষা` (top rank 1.0 or 0.85); `shikha` also reaches `শিখা` (score 1.0).
- **Rules Report CLI** ([`report.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/phonetic/report.py)):
  - Generates comprehensive tables of Roman patterns, G2P mappings, K1/K2/K3 transforms, and ambiguity resolutions:
    ```powershell
    python -m bsearch.phonetic.report
    ```

### Task 4: Query Expander, Combiners & Search CLI (`bsearch.rank.expand`, `bsearch.rank.searcher`, `bsearch.search`)
- **Query Expander** ([`expand.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/rank/expand.py)):
  - Detects native Bengali script tokens vs. Roman/Banglish tokens.
  - Native Bangla tokens pass through with weight `1.0` (`EXACT_BANGLA`).
  - Banglish tokens expand into weighted candidates:
    - `K1_EXACT`: ~1.0
    - `K2_EXACT`: ~0.7
    - `K2_FUZZY`: ~0.4 scaled by normalized edit similarity ($1.0 - \text{dist} / \text{len}$)
    - `K3_SKELETON`: ~0.2
  - Configurable weights via `ExpansionConfig`.
  - Caps candidates per query token (default 20).
  - Prunes and re-ranks candidates using a **Document-Frequency (DF) prior**: $P(t) \propto \log(DF + 1)$ to prefer common corpus terms over rare typos.
- **Searcher & Combiners** ([`searcher.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/rank/searcher.py)):
  - Coordinates multi-token query expansion, posting list traversal, and BM25 scoring.
  - **Two Selectable Combiners**:
    1. **`max`**: Takes the best candidate score for each token position: $\max_t (w_t \cdot BM25(t, D))$.
    2. **`saturating_sum`** (default): Sums candidate contributions with diminishing returns per token position: $\frac{\sum S}{1 + \lambda \sum S} \cdot (1 + \lambda)$.
  - **Explain Mode**: Provides a complete audit trail per hit: token candidate expansions, key levels, matched terms, frequencies, IDF, and weighted BM25 breakdown.
- **Search CLI** ([`search.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/search.py)):
  - Fast command-line search interface:
    ```powershell
    # Search with default saturating_sum combiner
    python -m bsearch.search "shikkha"

    # Search with max combiner and explain audit trail
    python -m bsearch.search "shikkha" --combiner max --explain -k 5
    ```

### Task 5: Evaluation Tooling & Benchmarking Suite (`bsearch.eval`)
- **Queries Template & Checklist** ([`queries.tsv`](file:///d:/projects/bangla%20search%20engine/eval/queries.tsv), [`CHECKLIST.md`](file:///d:/projects/bangla%20search%20engine/eval/CHECKLIST.md)):
  - Strictly user-provided ground-truth queries and judgments (no synthetic labels).
  - Schema: `id`, `query`, `category`, `gold_bangla_query`, `intent_note`.
  - Required categories: `single_word`, `multi_word`, `spelling_variant` ($\ge 10$ intents $\times$ 2-3 spellings), `loanword`, `name`, `typo`, `mixed_script`.
  - Dev/Test split: 20 development (`dev_01`..`dev_20`) / 30 test (`test_01`..`test_30`).
- **Five Evaluation Systems** ([`systems.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/eval/systems.py)):
  - **`B0`**: Raw Roman string directly searched with plain BM25.
  - **`B1`**: Static 1-to-1 transliteration + plain BM25 (naive deterministic baseline).
  - **`B2`**: Ours without fuzzy (K1, K2 exact, K3 skeleton, candidate combiner).
  - **`B3`**: Full system (K1, K2, Fuzzy K2 with similarity scaling, K3, combiners).
  - **`Oracle`**: BM25 using human gold native Bengali query.
- **Relevance Judgment Labeling Tool** ([`label_tool.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/eval/label_tool.py), [`eval/label_tool.py`](file:///d:/projects/bangla%20search%20engine/eval/label_tool.py)):
  - Pools top-20 documents across all systems per query.
  - Presents documents system-blind in random order to prevent bias.
  - Human grades: `0` (Not Relevant), `1` (Partially Relevant), `2` (Highly Relevant).
  - Resumable: writes directly to [`eval/qrels.tsv`](file:///d:/projects/bangla%20search%20engine/eval/qrels.tsv).
    ```powershell
    python eval/label_tool.py
    ```
- **Information Retrieval Metrics** ([`metrics.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/eval/metrics.py)):
  - **MRR@10**: Mean Reciprocal Rank at cutoff 10.
  - **nDCG@10**: Graded Discounted Cumulative Gain normalized by IDCG@10.
  - **Success@1** & **Success@5**: Binary success at rank 1 and within top-5.
  - **Recall@20**: Proportion of all known relevant documents retrieved in top-20.
  - Validated mathematically against hand-worked reference calculations.
  - **Paired Bootstrap Hypothesis Testing**: Computes 95% Confidence Intervals and $p$-values over 1,000 resamples.
  - Markdown table formatters for system ablation and per-category analysis.
- **Evaluation Runner CLI** ([`runner.py`](file:///d:/projects/bangla%20search%20engine/src/bsearch/eval/runner.py)):
  ```powershell
  python -m bsearch.eval.runner --queries eval/queries.tsv --qrels eval/qrels.tsv
  ```

---

## License & Attribution

- **Code**: MIT License.
- **Corpus Data**: Bengali Wikipedia articles extracted by the data pipeline are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and GFDL.
