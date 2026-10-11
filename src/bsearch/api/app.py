"""FastAPI search service supporting Banglish and native Bengali queries.

Endpoints:
- GET /api/search?q=...&k=10&combiner=saturating_sum
- GET /api/explain?q=...&k=10&combiner=saturating_sum
- GET /api/stats
- GET / (serves web/index.html static frontend)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bsearch.rank.bm25 import BM25Params
from bsearch.rank.expand import ExpansionConfig
from bsearch.rank.searcher import (
    CombinerType,
    SearchConfig,
    SearchResponse,
    Searcher,
)

logger = logging.getLogger("bsearch.api")

# Global searcher instance
_searcher: Searcher | None = None
INDEX_DIR = Path(os.environ.get("BSEARCH_INDEX_DIR", "data/index"))
WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index and warm searcher on application startup."""
    global _searcher
    logger.info("Initializing searcher from index directory: %s", INDEX_DIR)
    if not (INDEX_DIR / "index.meta").exists():
        logger.warning(
            "Index not found at %s. Ensure data/index or data/index_100k exists.",
            INDEX_DIR,
        )
    else:
        try:
            cfg = SearchConfig(
                combiner=CombinerType.SATURATING_SUM,
                bm25=BM25Params(k1=1.2, b=0.75, title_boost=2.0),
                expansion=ExpansionConfig(),
            )
            _searcher = Searcher.from_index_dir(INDEX_DIR, config=cfg)
            logger.info(
                "Searcher ready! Loaded %d documents, %d terms.",
                _searcher.index_reader.num_docs,
                _searcher.index_reader.vocab_size,
            )
        except Exception as e:
            logger.error("Failed to load searcher: %s", e)

    yield
    logger.info("Shutting down search service.")


app = FastAPI(
    title="Bangla Search Engine API",
    description="Search engine for Bengali text supporting Banglish (phonetic romanization) and native Bengali queries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_searcher() -> Searcher:
    """Retrieve initialized searcher or raise 503 Service Unavailable."""
    global _searcher
    if _searcher is None:
        if (INDEX_DIR / "index.meta").exists():
            _searcher = Searcher.from_index_dir(INDEX_DIR)
            return _searcher
        raise HTTPException(
            status_code=503,
            detail=f"Index not loaded or not found at {INDEX_DIR}. Please build the index first.",
        )
    return _searcher


@app.get("/api/search")
def search_api(
    q: str = Query(..., min_length=1, description="Query string in Banglish or native Bengali"),
    k: int = Query(10, ge=1, le=100, description="Maximum results to return"),
    combiner: str = Query("saturating_sum", regex="^(saturating_sum|max)$", description="Candidate combiner"),
) -> dict[str, Any]:
    """Standard search endpoint returning ranked documents and highlights."""
    searcher = get_searcher()
    # Apply requested combiner
    comb_enum = CombinerType(combiner)
    searcher.config = SearchConfig(
        combiner=comb_enum,
        bm25=searcher.config.bm25,
        expansion=searcher.config.expansion,
        default_top_k=k,
    )

    t0 = time.perf_counter()
    resp: SearchResponse = searcher.search(q, top_k=k, explain=False)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    results_data = []
    # Collect all expanded candidate terms for frontend highlighting
    all_candidate_terms = set()
    for exp in resp.expansions:
        for c in exp.candidates:
            if c.weight >= 0.5:
                all_candidate_terms.add(c.term)

    for r in resp.results:
        results_data.append({
            "rank": r.rank,
            "doc_id": r.doc_id,
            "score": r.score,
            "title": r.title,
            "snippet": r.snippet,
        })

    return {
        "query": resp.query,
        "tokens": resp.tokens,
        "total_hits": resp.total_hits,
        "execution_time_ms": elapsed_ms,
        "combiner": combiner,
        "highlight_terms": sorted(list(all_candidate_terms)),
        "results": results_data,
    }


@app.get("/api/explain")
def explain_api(
    q: str = Query(..., min_length=1, description="Query string in Banglish or native Bengali"),
    k: int = Query(10, ge=1, le=100, description="Maximum results to return"),
    combiner: str = Query("saturating_sum", regex="^(saturating_sum|max)$", description="Candidate combiner"),
) -> dict[str, Any]:
    """Explain endpoint returning full candidate expansion audit trail and scoring details."""
    searcher = get_searcher()
    comb_enum = CombinerType(combiner)
    searcher.config = SearchConfig(
        combiner=comb_enum,
        bm25=searcher.config.bm25,
        expansion=searcher.config.expansion,
        default_top_k=k,
    )

    t0 = time.perf_counter()
    resp: SearchResponse = searcher.search(q, top_k=k, explain=True)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Serialize expansions
    expansions_data = []
    for exp in resp.expansions:
        expansions_data.append({
            "token": exp.token,
            "is_bangla": exp.is_bangla,
            "candidates": [
                {
                    "term": c.term,
                    "weight": c.weight,
                    "base_weight": c.base_weight,
                    "level": c.level,
                    "df": c.df,
                    "similarity": c.similarity,
                    "edit_distance": c.edit_distance,
                }
                for c in exp.candidates
            ],
        })

    results_data = []
    for r in resp.results:
        expl_data = None
        if r.explanation:
            expl_data = {
                "token_scores": r.explanation.token_scores,
                "combiner": r.explanation.combiner,
                "matched_terms": [
                    {
                        "term": m.term,
                        "body_tf": m.body_tf,
                        "title_tf": m.title_tf,
                        "idf": m.idf,
                        "raw_bm25": m.raw_bm25,
                        "candidate_weight": m.candidate_weight,
                        "weighted_bm25": m.weighted_bm25,
                    }
                    for m in r.explanation.matched_terms
                ],
            }

        results_data.append({
            "rank": r.rank,
            "doc_id": r.doc_id,
            "score": r.score,
            "title": r.title,
            "snippet": r.snippet,
            "explanation": expl_data,
        })

    return {
        "query": resp.query,
        "tokens": resp.tokens,
        "total_hits": resp.total_hits,
        "execution_time_ms": elapsed_ms,
        "combiner": combiner,
        "expansions": expansions_data,
        "results": results_data,
    }


@app.get("/api/stats")
def stats_api() -> dict[str, Any]:
    """Corpus and index statistics."""
    searcher = get_searcher()
    return {
        "status": "online",
        "num_docs": searcher.index_reader.num_docs,
        "vocab_size": searcher.index_reader.vocab_size,
        "avgdl": round(searcher.index_reader.avgdl, 2),
        "total_tokens": searcher.index_reader.total_tokens,
    }


# Mount web directory if index.html exists
if (WEB_DIR / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def serve_frontend() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
