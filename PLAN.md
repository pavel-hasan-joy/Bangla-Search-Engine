# Architectural Plan: Banglish Search Engine (`banglish-search`)

## 1. Executive Summary & Objective

`banglish-search` is an information retrieval engine designed to index Bengali (Bangla) text corpora and support search queries written in **Banglish** (Bengali phonetic romanization) as well as native Bengali script. 

In strict adherence to repository rules:
- Built for Python 3.11+.
- Core modules (Unicode normalization, tokenization, inverted indexing, BM25 ranking, phonetic keys, edit distance, evaluation metrics) are built **exclusively using the Python standard library**.
- No external search engines or NLP libraries (no Lucene, Elasticsearch, Whoosh, Tantivy, rank_bm25, scikit-learn, transliteration packages, or neural embeddings).
- Permitted non-core packages: `numpy` and `matplotlib` (offline eval/plotting only), `FastAPI` and `uvicorn` (serving API), `pytest` (automated testing).
- All mapping rules, phonetic tables, stopwords, and suffixes reside in editable configuration files under `rules/`.
- Zero synthetic metrics or labels; all evaluations rely exclusively on reproducible scripts and human-provided ground truth.

---

## 2. Core Architecture

The search engine operates in two principal phases: **Index Construction (Offline)** and **Phonetic Query Retrieval (Online)**.

```
+-----------------------------------------------------------------------------------+
|                            OFFLINE INDEXING PIPELINE                              |
|                                                                                   |
|  [Wikipedia XML Dump / Raw Corpus]                                                |
|                   |                                                               |
|                   v                                                               |
|  [Stream Parser (bz2 + iterparse) & Wiki Markup Cleaner]                          |
|                   |                                                               |
|                   v                                                               |
|  [Passage Chunker (~150 words) / Document Collector]                              |
|                   |                                                               |
|                   v                                                               |
|  [Bangla Unicode Normalizer & Tokenizer] (rules/unicode_normalization.json)       |
|                   |                                                               |
|                   v                                                               |
|  [Stemmer / Suffix Normalizer] (rules/bangla_suffixes.txt)                        |
|                   |                                                               |
|                   +-------------------------------+                               |
|                   |                               |                               |
|                   v                               v                               |
|         [Bangla Inverted Index]         [Phonetic Lexicon Mapping]                |
|       (Postings: DocID, TF, Pos)       (Bangla Term -> Phonetic Keys)             |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                            ONLINE RETRIEVAL PIPELINE                              |
|                                                                                   |
|  User Query (e.g. "dhakar itihas" or "ঢাকার ইতিহাস")                               |
|                   |                                                               |
|                   v                                                               |
|  [Script Detector & Query Tokenizer]                                              |
|         /                                   \                                     |
|    (Bangla Script)                     (Roman / Banglish)                         |
|         |                                   |                                     |
|         |                       [Roman Text Normalizer]                           |
|         |                                   |                                     |
|         |                       [Phonetic Key Extractor]                          |
|         |                       (rules/phonetic_rules.json)                       |
|         |                                   |                                     |
|         |                       [Bangla Candidate Generator]                      |
|         |                       (Generates candidates c_i with weight w_i)        |
|         \                                   /                                     |
|          +---------------------------------+                                      |
|                            |                                                      |
|                            v                                                      |
|        [Weighted Multi-Term BM25 Scoring Engine]                                  |
|                            |                                                      |
|                            v                                                      |
|        [Top-K Document Accumulator & Snippet Extractor]                           |
|                            |                                                      |
|                            v                                                      |
|                   Ranked JSON Results                                             |
+-----------------------------------------------------------------------------------+
```

### Retrieval Scoring Formulation

When a query $Q = \{q_1, q_2, \dots, q_m\}$ contains Roman/Banglish tokens:
1. Each token $q_i$ is mapped to candidate Bengali terms $C(q_i) = \{(t_{i,1}, w_{i,1}), (t_{i,2}, w_{i,2}), \dots\}$ using phonetic key equivalence and weighted edit distance.
2. If $q_i$ is already in Bengali script, $C(q_i) = \{(q_i, 1.0)\}$.
3. For document $D$, the total relevance score is computed by weighted BM25 summation:

$$\text{Score}(D, Q) = \sum_{q_i \in Q} \sum_{(t, w) \in C(q_i)} w \cdot \text{BM25}(t, D)$$

Where standard BM25 is defined as:

$$\text{BM25}(t, D) = \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

$$\text{IDF}(t) = \ln \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)$$

- $N$: total document count in corpus.
- $n(t)$: number of documents containing term $t$.
- $f(t, D)$: term frequency of $t$ in $D$.
- $|D|$: document length, and $\text{avgdl}$: average document length across corpus.
- Free parameters: $k_1 \in [1.2, 2.0]$ and $b \in [0.6, 0.85]$.

---

## 3. Directory Layout & Module Responsibilities

```
banglish-search/
├── README.md                          # Project overview, installation, usage
├── PLAN.md                            # Architecture, file responsibilities, risks
├── Makefile                           # Development targets (install, test, parse-dump)
├── pyproject.toml                     # Python packaging & project specification
├── rules/                             # User-editable rule tables (no code modifications needed)
│   ├── README.md                      # Documentation on rule table formats
│   ├── unicode_normalization.json     # Dari, hasant, nukta, and visual look-alikes
│   ├── phonetic_rules.json            # Digraphs, consonants, vowels, phonetic classes
│   ├── bangla_stopwords.txt           # High-frequency Bangla grammatical stopwords
│   └── bangla_suffixes.txt            # Inflectional suffixes for rule-based stemming
├── src/
│   └── bsearch/
│       ├── __init__.py                # Package root
│       ├── unicode/                   # Unicode normalization & script classification
│       │   ├── __init__.py
│       │   ├── normalizer.py          # Character canonicalization & NFC/NFD handling
│       │   └── detector.py            # Script identification (Bangla, Latin, Numeric)
│       ├── phonetic/                  # Banglish phonetic translation & candidate expansion
│       │   ├── __init__.py
│       │   ├── roman_normalizer.py    # Lowercasing, repeating letter squashing, cleanups
│       │   ├── encoder.py             # Rule-based Bengali phonetic key derivation
│       │   ├── distance.py            # Standard library Levenshtein / weighted edit distance
│       │   └── generator.py           # Candidate Bangla term generator with probability weights
│       ├── index/                     # Core inverted index & persistence
│       │   ├── __init__.py
│       │   ├── tokenizer.py           # Script-aware tokenization & punctuation handling
│       │   ├── stemmer.py             # Suffix-stripper driven by rules/bangla_suffixes.txt
│       │   ├── inverted_index.py      # Postings list (DocID, TF, positions) & term index
│       │   ├── document_store.py      # Title, raw/clean text, document lengths storage
│       │   └── serializer.py          # Fast binary/JSONL serialization & streaming loader
│       ├── rank/                      # Information retrieval scoring
│       │   ├── __init__.py
│       │   ├── bm25.py                # Pure standard-library BM25 scoring implementation
│       │   ├── aggregator.py          # Weighted candidate term scoring & top-K heap
│       │   └── searcher.py            # End-to-end search query coordinator & snippet generator
│       ├── data/                      # Ingestion, dump parsing, and corpus chunking
│       │   ├── __init__.py
│       │   ├── dump_parser.py         # Streaming bz2 XML parser with iterparse
│       │   ├── markup_cleaner.py      # Regex-based pure standard library wikitext stripper
│       │   └── chunker.py             # Passage chunker (~150 words) with title prefix
│       ├── eval/                      # Evaluation suite (standard library metrics)
│       │   ├── __init__.py
│       │   ├── metrics.py             # MRR, MAP, NDCG@K, Precision@K, Recall@K
│       │   └── runner.py              # Evaluator loading user-provided QREL ground truth
│       ├── bench/                     # Benchmarking & performance profiling
│       │   ├── __init__.py
│       │   ├── latency.py             # Latency measurement (p50, p95, p99)
│       │   └── memory.py              # RAM consumption & index size profiler
│       └── api/                       # API serving
│           ├── __init__.py
│           ├── main.py                # FastAPI application entrypoint
│           └── schemas.py             # Request & response Pydantic models
├── eval/                              # User-provided evaluation files & ground truth
├── results/                           # Verifiable run outputs & benchmark reports
├── tests/                             # Comprehensive automated test suite (pytest)
└── web/                               # Search web client
```

---

## 4. Key Risks and Mitigation Strategies

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **1. Banglish Orthographic Ambiguity** | One Banglish word can have multiple valid phonetic spellings (e.g., *kemon* vs *kamon*, *achhi* vs *asi*, *bhalo* vs *valo*), causing combinatorial explosion in candidate generation. | 1. Canonical phonetic key mapping (cluster similar sounds together).<br>2. Restrict candidates to terms actually present in the indexed Bengali vocabulary.<br>3. Candidate pruning: cap candidates to top-K (e.g. $K=5$) with a minimum probability/weight threshold ($\ge 0.15$). |
| **2. Pure Python Retrieval Latency** | Evaluating BM25 scores across tens of thousands of postings in pure Python could result in high latency (>500ms). | 1. Use Python standard library `array` module for compact integer docIDs and frequencies.<br>2. Implement conjunctive (AND) retrieval for multi-token queries as default, falling back to disjunctive (OR) only when necessary.<br>3. Term-at-a-time (TAAT) score accumulation with early pruning (WAND / MaxScore thresholding). |
| **3. Memory Exhaustion with 150k+ Documents** | Keeping full raw text, postings, and vocabulary in memory simultaneously could exceed typical memory budgets. | 1. Separation of concerns: inverted index contains only term IDs and compressed posting arrays.<br>2. Document texts stored in a line-indexed JSONL document store loaded on-demand for snippet extraction.<br>3. Single-Pass In-Memory Indexing (SPIMI) with segment merges for scalable index building. |
| **4. Over-Aggressive Suffix Stripping** | Suffix stripping on short words can corrupt base stems (e.g., removing `র` from `কর` or `এর` from `সের`). | 1. Minimum stem length constraint (e.g., stem length must be $\ge 3$ characters).<br>2. Rule ordering from longest suffix to shortest suffix.<br>3. Configurable suffix exclusion list in `rules/bangla_suffixes.txt`. |
| **5. Wikipedia Markup Extraction Fidelity** | Complex MediaWiki markup (nested templates, parser functions, tables, citations) can leave noise in cleaned text. | 1. Streamlined iterative regex cleaner targeting templates `{{...}}`, tables `{\|...\|}`, parser functions `#...`, and references `<ref>...</ref>`.<br>2. Automated regression tests on representative markup edge cases.<br>3. Filtering stubs (< 100 characters or < 20 words). |
| **6. Zero Synthetic Metrics Integrity** | Violating reproducibility rule by fabricating evaluation numbers. | 1. Evaluation runner fails fast if ground-truth file is absent.<br>2. Evaluation scripts emit deterministic logs with checksums of the evaluation datasets. |

---

## 5. Next Steps & Implementation Milestones

- **Milestone 0 (Current)**: Repository skeleton, architecture plan, rules setup, and Wikipedia dump stream parser + chunker with verified test suite.
- **Milestone 1**: Unicode normalization (`bsearch.unicode`) and script-aware tokenization.
- **Milestone 2**: Inverted index construction, posting lists, and document store (`bsearch.index`).
- **Milestone 3**: Banglish romanization normalizer, phonetic encoder, and candidate generator (`bsearch.phonetic`).
- **Milestone 4**: Weighted BM25 scoring and ranking engine (`bsearch.rank`).
- **Milestone 5**: Evaluation metrics suite (`bsearch.eval`) and performance benchmarks (`bsearch.bench`).
- **Milestone 6**: FastAPI endpoint (`bsearch.api`) and web interface (`web/`).
