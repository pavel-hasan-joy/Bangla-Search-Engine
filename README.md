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

---

## License & Attribution

- **Code**: MIT License.
- **Corpus Data**: Bengali Wikipedia articles extracted by the data pipeline are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and GFDL.
