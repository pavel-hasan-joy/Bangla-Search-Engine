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

---

## License & Attribution

- **Code**: MIT License.
- **Corpus Data**: Bengali Wikipedia articles extracted by the data pipeline are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and GFDL.
