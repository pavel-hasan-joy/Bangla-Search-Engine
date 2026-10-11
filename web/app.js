/**
 * Bangla Search Engine - Interactive Web UI Client
 * Handles search requests, dynamic expansion visualization,
 * matched-term highlighting, and the "How we matched" explanation panel.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const searchForm = document.getElementById('search-form');
  const queryInput = document.getElementById('query-input');
  const clearBtn = document.getElementById('clear-btn');
  const combinerSelect = document.getElementById('combiner-select');
  const topkSelect = document.getElementById('topk-select');
  const tagButtons = document.querySelectorAll('.tag-btn');

  const expansionSection = document.getElementById('expansion-section');
  const expansionTokens = document.getElementById('expansion-tokens');
  const timingBadge = document.getElementById('timing-badge');

  const resultsHeader = document.getElementById('results-header');
  const resultsCount = document.getElementById('results-count');
  const resultsList = document.getElementById('results-list');
  const loadingSpinner = document.getElementById('loading-spinner');
  const statsBadge = document.getElementById('stats-badge');

  // Load index stats on page load
  fetchStats();

  // Clear button visibility
  queryInput.addEventListener('input', () => {
    clearBtn.style.display = queryInput.value.trim() ? 'block' : 'none';
  });

  clearBtn.addEventListener('click', () => {
    queryInput.value = '';
    clearBtn.style.display = 'none';
    queryInput.focus();
    resetResults();
  });

  // Quick tag buttons
  tagButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-query');
      if (q) {
        queryInput.value = q;
        clearBtn.style.display = 'block';
        performSearch(q);
      }
    });
  });

  // Form submit
  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (query) {
      performSearch(query);
    }
  });

  // Re-run search if combiner or top-k changes and query is present
  combinerSelect.addEventListener('change', () => {
    const query = queryInput.value.trim();
    if (query) performSearch(query);
  });

  topkSelect.addEventListener('change', () => {
    const query = queryInput.value.trim();
    if (query) performSearch(query);
  });

  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = await res.json();
        statsBadge.textContent = `${data.total_docs.toLocaleString()} docs • ${data.total_terms.toLocaleString()} terms • BM25 + WAND`;
      } else {
        statsBadge.textContent = 'API Ready';
      }
    } catch {
      statsBadge.textContent = 'Engine Online';
    }
  }

  function resetResults() {
    expansionSection.style.display = 'none';
    resultsHeader.style.display = 'none';
    resultsList.innerHTML = `
      <div class="initial-state" id="initial-state">
        <div class="initial-icon">🔍</div>
        <h3>Type a Banglish query to begin</h3>
        <p>The phonetic engine maps your romanized input to authentic Bengali orthography using multi-level phonetic keys.</p>
      </div>
    `;
  }

  async function performSearch(query) {
    loadingSpinner.style.display = 'flex';
    resultsList.style.display = 'none';
    expansionSection.style.display = 'none';
    resultsHeader.style.display = 'none';

    const combiner = combinerSelect.value;
    const topK = topkSelect.value;
    const url = `/api/explain?q=${encodeURIComponent(query)}&k=${encodeURIComponent(topK)}&combiner=${encodeURIComponent(combiner)}`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }
      const data = await response.json();
      renderExplanationAndResults(data);
    } catch (err) {
      console.error('Search failed:', err);
      resultsList.innerHTML = `
        <div class="initial-state" style="border-color: rgba(239, 68, 68, 0.4);">
          <div class="initial-icon">⚠️</div>
          <h3 style="color: #ef4444;">Search Request Failed</h3>
          <p>${escapeHtml(err.message || 'Unable to retrieve search results.')}</p>
        </div>
      `;
      resultsList.style.display = 'block';
    } finally {
      loadingSpinner.style.display = 'none';
    }
  }

  function renderExplanationAndResults(data) {
    timingBadge.textContent = `${data.execution_time_ms.toFixed(1)} ms`;

    // Render candidate expansions
    renderExpansions(data.expansions || []);
    expansionSection.style.display = 'block';

    // Collect all candidate terms for client-side highlighting
    const highlightTerms = new Set();
    (data.expansions || []).forEach(exp => {
      (exp.candidates || []).forEach(cand => {
        if (cand.weight >= 0.35) {
          highlightTerms.add(cand.term);
        }
      });
    });

    // Render result cards
    renderResults(data.results || [], highlightTerms, data.execution_time_ms, data.combiner);
    resultsList.style.display = 'flex';
    resultsHeader.style.display = 'flex';
    resultsCount.textContent = `Found ${data.total_hits.toLocaleString()} matching documents (${data.execution_time_ms.toFixed(1)} ms)`;
  }

  function renderExpansions(expansions) {
    expansionTokens.innerHTML = '';
    if (!expansions || expansions.length === 0) {
      expansionTokens.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No expansion tokens</span>';
      return;
    }

    expansions.forEach(exp => {
      const card = document.createElement('div');
      card.className = 'token-card';

      const header = document.createElement('div');
      header.className = 'token-header';
      header.innerHTML = `
        <span class="token-name">${escapeHtml(exp.token)}</span>
        <span class="token-type">${exp.is_bangla ? 'Native Bangla' : 'Banglish'}</span>
      `;
      card.appendChild(header);

      const candidateList = document.createElement('div');
      candidateList.className = 'candidate-list';

      if (!exp.candidates || exp.candidates.length === 0) {
        candidateList.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem;">No vocabulary candidates</span>';
      } else {
        exp.candidates.slice(0, 10).forEach(cand => {
          const pill = document.createElement('div');
          pill.className = 'candidate-pill';
          const levelClass = getLevelBadgeClass(cand.level);
          pill.innerHTML = `
            <span class="cand-term">${escapeHtml(cand.term)}</span>
            <span class="cand-weight">${(cand.weight).toFixed(2)}</span>
            <span class="cand-badge ${levelClass}">${cand.level}</span>
          `;
          pill.title = `Weight: ${cand.weight.toFixed(3)} | Base: ${cand.base_weight.toFixed(2)} | DF: ${cand.df}`;
          candidateList.appendChild(pill);
        });
      }

      card.appendChild(candidateList);
      expansionTokens.appendChild(card);
    });
  }

  function renderResults(results, highlightTerms, timeMs, combiner) {
    resultsList.innerHTML = '';

    if (!results || results.length === 0) {
      resultsList.innerHTML = `
        <div class="initial-state">
          <div class="initial-icon">📭</div>
          <h3>No matching documents found</h3>
          <p>Try searching with another spelling variation or fewer terms.</p>
        </div>
      `;
      return;
    }

    results.forEach((r, idx) => {
      const card = document.createElement('article');
      card.className = 'result-card';

      const highlightedTitle = highlightSnippet(r.title, highlightTerms);
      const highlightedSnippet = highlightSnippet(r.snippet, highlightTerms);

      const expl = r.explanation;
      const matchedTermsCount = expl && expl.matched_terms ? expl.matched_terms.length : 0;

      card.innerHTML = `
        <div class="result-header">
          <span class="result-rank">#${r.rank}</span>
          <h2 class="result-title">${highlightedTitle}</h2>
          <span class="result-score" title="BM25 composite score">BM25: ${r.score.toFixed(3)}</span>
        </div>
        <p class="result-snippet">${highlightedSnippet}</p>
        <div class="result-meta">
          <span class="meta-tag">Doc ID: ${r.doc_id}</span>
          ${matchedTermsCount > 0 ? `
            <button type="button" class="how-matched-toggle" data-idx="${idx}">
              <span>💡 How we matched (${matchedTermsCount} term${matchedTermsCount > 1 ? 's' : ''})</span>
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </button>
          ` : ''}
        </div>
        ${matchedTermsCount > 0 ? buildExplanationPanel(expl, idx, combiner) : ''}
      `;

      resultsList.appendChild(card);
    });

    // Attach toggle listeners
    document.querySelectorAll('.how-matched-toggle').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = btn.getAttribute('data-idx');
        const panel = document.getElementById(`expl-panel-${idx}`);
        if (panel) {
          const isOpen = panel.classList.contains('open');
          panel.classList.toggle('open');
          btn.classList.toggle('active');
          const chevron = btn.querySelector('svg');
          if (chevron) {
            chevron.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
          }
        }
      });
    });
  }

  function buildExplanationPanel(expl, idx, combiner) {
    if (!expl || !expl.matched_terms) return '';

    let rowsHtml = '';
    expl.matched_terms.forEach(m => {
      rowsHtml += `
        <tr>
          <td><strong class="font-bangla">${escapeHtml(m.term)}</strong></td>
          <td>${(m.candidate_weight || 1.0).toFixed(2)}</td>
          <td>${m.title_tf}</td>
          <td>${m.body_tf}</td>
          <td>${m.idf.toFixed(3)}</td>
          <td>${m.raw_bm25.toFixed(3)}</td>
          <td><span class="score-highlight">${m.weighted_bm25.toFixed(3)}</span></td>
        </tr>
      `;
    });

    let tokenScoresHtml = '';
    if (expl.token_scores) {
      const entries = Object.entries(expl.token_scores);
      if (entries.length > 0) {
        tokenScoresHtml = `
          <div class="token-score-breakdown">
            <span class="breakdown-label">Per-Token BM25 (${combiner}):</span>
            ${entries.map(([tok, sc]) => `
              <span class="token-score-pill">
                <code>${escapeHtml(tok)}</code>: <strong>${Number(sc).toFixed(3)}</strong>
              </span>
            `).join('')}
          </div>
        `;
      }
    }

    return `
      <div class="how-matched-panel" id="expl-panel-${idx}">
        <div class="panel-inner">
          <div class="panel-header">
            <h4>Scoring & Orthographic Alignment Audit Trail</h4>
            <span class="panel-sub">Combiner: <code>${escapeHtml(expl.combiner || combiner)}</code></span>
          </div>
          ${tokenScoresHtml}
          <div class="table-container">
            <table class="match-table">
              <thead>
                <tr>
                  <th>Term</th>
                  <th>Weight</th>
                  <th>Title TF</th>
                  <th>Body TF</th>
                  <th>IDF</th>
                  <th>Raw BM25</th>
                  <th>Weighted Score</th>
                </tr>
              </thead>
              <tbody>
                ${rowsHtml}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  }

  function highlightSnippet(text, terms) {
    if (!text) return '';
    if (!terms || terms.size === 0) return escapeHtml(text);

    // Escape HTML first to prevent XSS
    let escaped = escapeHtml(text);

    // Sort terms longest first to avoid partial substring collisions
    const sortedTerms = Array.from(terms).filter(t => t && t.length > 0).sort((a, b) => b.length - a.length);

    // Build regex
    const pattern = new RegExp(`(${sortedTerms.map(escapeRegex).join('|')})`, 'gu');
    return escaped.replace(pattern, '<mark class="highlight">$1</mark>');
  }

  function getLevelBadgeClass(level) {
    switch (level) {
      case 'K1': return 'badge-k1';
      case 'K2': return 'badge-k2';
      case 'K3': return 'badge-k3';
      case 'fuzzy': return 'badge-fuzzy';
      default: return 'badge-native';
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
});
