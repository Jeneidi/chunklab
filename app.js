"use strict";

const ENDPOINT = "https://api.openai.com/v1/embeddings";
const MODEL = "text-embedding-3-small";
const SESSION_KEY = "chunklab_openai_key";
const MAX_REQUESTS = 8;
const MAX_INPUTS = 1200;
const BATCH_SIZE = 256;
const REQUEST_TIMEOUT_MS = 30000;
const KS = [1, 3, 5, 10];
const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => typeof document === "undefined" ? null : document.querySelector(id);

const STRATEGIES = [
  { id: "fixed-500", name: "Fixed 500", description: "500 characters, no overlap" },
  { id: "fixed-10", name: "Fixed 500 + 10%", description: "500 characters, 50-character overlap" },
  { id: "fixed-25", name: "Fixed 500 + 25%", description: "500 characters, 125-character overlap" },
  { id: "recursive", name: "Recursive", description: "Paragraph, sentence, whitespace, then hard cut" },
  { id: "sentence-window", name: "Sentence window", description: "Index one sentence, return it with two neighbors" },
  { id: "parent-document", name: "Parent document", description: "Index 250-character children, return 1,000-character parents" },
  { id: "semantic", name: "Semantic (embeddings)", description: "80th-percentile adjacent-sentence embedding distance" },
  { id: "fixed-1000", name: "Fixed 1000", description: "1,000 characters, no overlap" },
];

let benchmark = null;
let examples = null;
let meta = null;
let sortKey = "ndcg_at_10";
let sortDirection = -1;
let activeController = null;
let requestCount = 0;

function fixedChunks(text, size, overlap = 0) {
  if (!text) return [];
  const chunks = [];
  for (let start = 0; start < text.length; start += size - overlap) {
    const end = Math.min(text.length, start + size);
    chunks.push(makeChunk(text, start, end));
  }
  return chunks;
}

function makeChunk(text, start, end, indexStart = start, indexEnd = end) {
  return { start, end, text: text.slice(start, end), indexStart, indexEnd, indexText: text.slice(indexStart, indexEnd) };
}

function sentenceRanges(text) {
  const ranges = [...text.matchAll(/\s*[^.!?]+(?:[.!?]+(?=\s|$)|$)/g)].map((match) => [match.index, match.index + match[0].length]);
  if (!ranges.length) return text ? [[0, text.length]] : [];
  if (ranges.at(-1)[1] < text.length) ranges.push([ranges.at(-1)[1], text.length]);
  return ranges;
}

function structuralChunks(text, size = 500) {
  if (text.length <= size) return text ? [makeChunk(text, 0, text.length)] : [];
  const chunks = [];
  const separators = ["\n\n", ". ", "? ", "! ", "\n", " "];
  for (let start = 0; start < text.length;) {
    const hardEnd = Math.min(text.length, start + size);
    let end = hardEnd;
    if (hardEnd < text.length) {
      const floor = start + Math.floor(size / 2);
      const candidates = separators
        .map((separator) => {
          const position = text.lastIndexOf(separator, hardEnd);
          return position >= floor ? position + separator.length : -1;
        })
        .filter((position) => position >= 0);
      if (candidates.length) end = Math.max(...candidates);
    }
    chunks.push(makeChunk(text, start, end));
    start = end;
  }
  return chunks;
}

function sentenceWindowChunks(text, neighbors = 2) {
  const sentences = sentenceRanges(text);
  return sentences.map(([indexStart, indexEnd], index) => {
    const start = sentences[Math.max(0, index - neighbors)][0];
    const end = sentences[Math.min(sentences.length - 1, index + neighbors)][1];
    return makeChunk(text, start, end, indexStart, indexEnd);
  });
}

function parentDocumentChunks(text, childSize = 250, parentSize = 1000) {
  const chunks = [];
  for (let parentStart = 0; parentStart < text.length; parentStart += parentSize) {
    const parentEnd = Math.min(text.length, parentStart + parentSize);
    for (let childStart = parentStart; childStart < parentEnd; childStart += childSize) {
      chunks.push(makeChunk(text, parentStart, parentEnd, childStart, Math.min(parentEnd, childStart + childSize)));
    }
  }
  return chunks;
}

function cosine(a, b) {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa && bb ? dot / Math.sqrt(aa * bb) : 0;
}

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  return low === high ? sorted[low] : sorted[low] * (high - position) + sorted[high] * (position - low);
}

function semanticChunks(text, sentenceVectors, thresholdQ = 0.8, maxSize = 500) {
  const sentences = sentenceRanges(text);
  if (sentences.length <= 1) return fixedChunks(text, maxSize);
  const distances = sentenceVectors.slice(0, -1).map((vector, index) => 1 - cosine(vector, sentenceVectors[index + 1]));
  const threshold = percentile(distances, thresholdQ);
  const breakAfter = new Set(distances.map((distance, index) => distance >= threshold ? index : -1).filter((index) => index >= 0));
  const chunks = [];
  let groupStart = sentences[0][0];
  sentences.forEach(([, sentenceEnd], index) => {
    const split = breakAfter.has(index) || index === sentences.length - 1 || sentenceEnd - groupStart >= maxSize;
    if (!split) return;
    while (sentenceEnd - groupStart > maxSize) {
      chunks.push(makeChunk(text, groupStart, groupStart + maxSize));
      groupStart += maxSize;
    }
    if (sentenceEnd > groupStart) chunks.push(makeChunk(text, groupStart, sentenceEnd));
    groupStart = sentenceEnd;
  });
  return chunks;
}

function normalize(text) {
  return text.toLocaleLowerCase().trim().split(/\s+/).join(" ");
}

function isHit(text, answers) {
  const haystack = normalize(text);
  return answers.some((answer) => haystack.includes(normalize(answer)));
}

function reciprocalRank(relevances, k = 10) {
  const index = relevances.slice(0, k).findIndex(Boolean);
  return index < 0 ? 0 : 1 / (index + 1);
}

function ndcg(relevances, totalRelevant, k = 10) {
  const dcg = relevances.slice(0, k).reduce((sum, relevance, index) => sum + relevance / Math.log2(index + 2), 0);
  let ideal = 0;
  for (let rank = 1; rank <= Math.min(totalRelevant, k); rank += 1) ideal += 1 / Math.log2(rank + 1);
  return ideal ? dcg / ideal : 0;
}

function aggregateMetrics(rankings) {
  const recalls = Object.fromEntries(KS.map((k) => [k, rankings.filter((row) => row.relevances.slice(0, k).some(Boolean)).length / rankings.length]));
  const metrics = Object.fromEntries(KS.map((k) => [`recall_at_${k}`, recalls[k]]));
  metrics.mrr_at_10 = mean(rankings.map((row) => reciprocalRank(row.relevances)));
  metrics.ndcg_at_10 = mean(rankings.map((row) => ndcg(row.relevances, row.totalRelevant)));
  metrics.context_k_to_90 = KS.find((k) => recalls[k] > 0.9) || null;
  metrics.context_tokens_to_90 = metrics.context_k_to_90
    ? mean(rankings.map((row) => row.contextTokens.slice(0, metrics.context_k_to_90).reduce((sum, value) => sum + value, 0)))
    : null;
  return metrics;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function estimateTokens(text) {
  return Math.ceil(new TextEncoder().encode(text).length / 4);
}

function tokenize(text) {
  return text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
}

function tfidfVectors(texts) {
  const counts = texts.map((text) => {
    const row = new Map();
    tokenize(text).forEach((term) => row.set(term, (row.get(term) || 0) + 1));
    return row;
  });
  const documentFrequency = new Map();
  counts.forEach((row) => row.forEach((_, term) => documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1)));
  const vocabulary = [...documentFrequency.keys()];
  return counts.map((row) => vocabulary.map((term) => (
    (row.get(term) || 0) * (Math.log((1 + texts.length) / (1 + documentFrequency.get(term))) + 1)
  )));
}

function buildBm25Ranker(texts, k1 = 1.2, b = 0.75) {
  const rows = texts.map((text) => {
    const row = new Map();
    tokenize(text).forEach((term) => row.set(term, (row.get(term) || 0) + 1));
    return row;
  });
  const lengths = rows.map((row) => [...row.values()].reduce((sum, value) => sum + value, 0));
  const averageLength = lengths.length ? mean(lengths) : 1;
  const postings = new Map();
  rows.forEach((row, index) => row.forEach((frequency, term) => {
    if (!postings.has(term)) postings.set(term, []);
    postings.get(term).push([index, frequency]);
  }));
  return (query) => {
    const scores = new Map();
    new Set(tokenize(query)).forEach((term) => {
      const matches = postings.get(term) || [];
      const idf = Math.log1p((rows.length - matches.length + 0.5) / (matches.length + 0.5));
      matches.forEach(([index, frequency]) => {
        const normalizer = frequency + k1 * (1 - b + b * lengths[index] / averageLength);
        scores.set(index, (scores.get(index) || 0) + idf * frequency * (k1 + 1) / normalizer);
      });
    });
    const ordered = [...scores].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([index]) => index);
    for (let index = 0; index < rows.length; index += 1) if (!scores.has(index)) ordered.push(index);
    return ordered;
  };
}

function buildSyncChunks(strategyId, text) {
  if (strategyId === "fixed-500") return fixedChunks(text, 500);
  if (strategyId === "fixed-10") return fixedChunks(text, 500, 50);
  if (strategyId === "fixed-25") return fixedChunks(text, 500, 125);
  if (strategyId === "recursive") return structuralChunks(text);
  if (strategyId === "sentence-window") return sentenceWindowChunks(text);
  if (strategyId === "parent-document") return parentDocumentChunks(text);
  if (strategyId === "fixed-1000") return fixedChunks(text, 1000);
  return null;
}

function formatScore(value) {
  return Number(value).toFixed(3);
}

function formatCount(value) {
  return value == null ? "n/a" : Math.round(value).toLocaleString();
}

function strategyValue(strategy, key) {
  return key === "name" ? strategy.name : strategy.metrics[key];
}

function renderLeaderboard() {
  const tbody = $("#leaderboard-table").tBodies[0];
  tbody.textContent = "";
  const rows = [...benchmark.strategies].sort((a, b) => {
    const av = strategyValue(a, sortKey);
    const bv = strategyValue(b, sortKey);
    if (av == null) return 1;
    if (bv == null) return -1;
    return (typeof av === "string" ? av.localeCompare(bv) : av - bv) * sortDirection;
  });
  for (const strategy of rows) {
    const tr = document.createElement("tr");
    if (strategy.id === benchmark.winner) tr.className = "winner";
    const values = [
      strategy.name,
      ...KS.map((k) => formatScore(strategy.metrics[`recall_at_${k}`])),
      formatScore(strategy.metrics.mrr_at_10),
      formatScore(strategy.metrics.ndcg_at_10),
      formatCount(strategy.metrics.mean_chunk_tokens),
      formatCount(strategy.metrics.context_tokens_to_90),
    ];
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  document.querySelectorAll("#leaderboard-table th").forEach((th) => th.removeAttribute("aria-sort"));
  const active = document.querySelector(`#leaderboard-table button[data-sort="${sortKey}"]`);
  if (active) active.closest("th").setAttribute("aria-sort", sortDirection === 1 ? "ascending" : "descending");
}

function syncUrl() {
  const url = new URL(location.href);
  url.searchParams.set("sort", sortKey);
  url.searchParams.set("dir", sortDirection === 1 ? "asc" : "desc");
  const fields = {
    inspect: $("#inspector-strategy").value,
    doc: $("#inspector-document").value,
    failureStrategy: $("#failure-strategy").value,
    failure: $("#failure-example").value,
  };
  Object.entries(fields).forEach(([key, value]) => {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  });
  history.replaceState(null, "", url);
}

function svgNode(name, attributes = {}, text = "") {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text) node.textContent = text;
  return node;
}

function seriesColors() {
  const styles = getComputedStyle(document.documentElement);
  return [
    styles.getPropertyValue("--series-1"),
    styles.getPropertyValue("--series-2"),
    styles.getPropertyValue("--series-3"),
    styles.getPropertyValue("--series-4"),
    styles.getPropertyValue("--series-5"),
    styles.getPropertyValue("--series-6"),
    styles.getPropertyValue("--series-7"),
  ];
}

function colorFor(strategy, index) {
  if (strategy.id === benchmark.winner) return getComputedStyle(document.documentElement).getPropertyValue("--accent");
  return seriesColors()[index % 7];
}

function renderRecallChart() {
  const chart = $("#recall-chart");
  chart.textContent = "";
  chart.appendChild(svgNode("title", {}, "Recall at ranks 1, 3, 5, and 10 for every chunking strategy"));
  const plot = { left: 64, right: 860, top: 24, bottom: 260 };
  for (let tick = 0; tick <= 10; tick += 2) {
    const y = plot.bottom - (tick / 10) * (plot.bottom - plot.top);
    chart.append(svgNode("line", { x1: plot.left, y1: y, x2: plot.right, y2: y, class: "axis" }));
    chart.append(svgNode("text", { x: plot.left - 12, y: y + 4, "text-anchor": "end" }, (tick / 10).toFixed(1)));
  }
  KS.forEach((k, index) => {
    const x = plot.left + index * ((plot.right - plot.left) / (KS.length - 1));
    chart.append(svgNode("text", { x, y: plot.bottom + 24, "text-anchor": "middle" }, String(k)));
  });
  chart.append(svgNode("text", { x: (plot.left + plot.right) / 2, y: 306, "text-anchor": "middle", class: "axis-label" }, "retrieval rank k"));
  chart.append(svgNode("text", { x: 16, y: (plot.top + plot.bottom) / 2, transform: `rotate(-90 16 ${(plot.top + plot.bottom) / 2})`, "text-anchor": "middle", class: "axis-label" }, "query recall"));

  benchmark.strategies.forEach((strategy, strategyIndex) => {
    const points = KS.map((k, index) => {
      const x = plot.left + index * ((plot.right - plot.left) / (KS.length - 1));
      const y = plot.bottom - strategy.metrics[`recall_at_${k}`] * (plot.bottom - plot.top);
      return [x, y];
    });
    const color = colorFor(strategy, strategyIndex);
    chart.append(svgNode("polyline", {
      points: points.map((point) => point.join(",")).join(" "),
      fill: "none",
      stroke: color,
      "stroke-width": strategy.id === benchmark.winner ? 3 : 1.7,
      "stroke-dasharray": strategy.id === benchmark.winner ? "none" : `${3 + strategyIndex} ${2 + strategyIndex % 3}`,
    }));
    points.forEach(([cx, cy]) => chart.append(svgNode("circle", { cx, cy, r: 3, fill: color })));
    const legendX = 64 + (strategyIndex % 4) * 200;
    const legendY = 330 + Math.floor(strategyIndex / 4) * 24;
    chart.append(svgNode("line", { x1: legendX, y1: legendY - 4, x2: legendX + 22, y2: legendY - 4, stroke: color, "stroke-width": 3 }));
    chart.append(svgNode("text", { x: legendX + 30, y: legendY }, strategy.name));
  });
}

function renderCostChart() {
  const chart = $("#cost-chart");
  chart.textContent = "";
  const tokenLabel = meta.token_counting.estimated ? "estimated context tokens" : "context tokens";
  chart.appendChild(svgNode("title", {}, `nDCG at 10 against ${tokenLabel} required to exceed 90 percent recall`));
  const plot = { left: 72, right: 850, top: 24, bottom: 282 };
  const defined = benchmark.strategies.map((strategy) => strategy.metrics.context_tokens_to_90).filter((value) => value != null);
  const maxCost = Math.max(...defined, 1);
  const maxX = maxCost * 1.18;
  const scores = benchmark.strategies.map((strategy) => strategy.metrics.ndcg_at_10);
  const minY = Math.max(0, Math.min(...scores) - 0.05);
  const maxY = Math.min(1, Math.max(...scores) + 0.05);
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = plot.bottom - (tick / 4) * (plot.bottom - plot.top);
    const value = minY + (tick / 4) * (maxY - minY);
    chart.append(svgNode("line", { x1: plot.left, y1: y, x2: plot.right, y2: y, class: "axis" }));
    chart.append(svgNode("text", { x: plot.left - 10, y: y + 4, "text-anchor": "end" }, value.toFixed(2)));
    const x = plot.left + (tick / 4) * (plot.right - plot.left);
    chart.append(svgNode("text", { x, y: plot.bottom + 22, "text-anchor": "middle" }, Math.round((tick / 4) * maxX).toLocaleString()));
  }
  chart.append(svgNode("text", { x: (plot.left + plot.right) / 2, y: 330, "text-anchor": "middle", class: "axis-label" }, `${tokenLabel} at threshold (right edge = threshold not reached)`));
  chart.append(svgNode("text", { x: 16, y: (plot.top + plot.bottom) / 2, transform: `rotate(-90 16 ${(plot.top + plot.bottom) / 2})`, "text-anchor": "middle", class: "axis-label" }, "nDCG@10"));
  benchmark.strategies.forEach((strategy, index) => {
    const cost = strategy.metrics.context_tokens_to_90 ?? maxX;
    const x = plot.left + (cost / maxX) * (plot.right - plot.left);
    const y = plot.bottom - ((strategy.metrics.ndcg_at_10 - minY) / (maxY - minY || 1)) * (plot.bottom - plot.top);
    const color = colorFor(strategy, index);
    chart.append(svgNode("circle", { cx: x, cy: y, r: strategy.id === benchmark.winner ? 7 : 5, fill: color }));
    chart.append(svgNode("text", { x: x + (x > 700 ? -9 : 9), y: y - 8, "text-anchor": x > 700 ? "end" : "start" }, strategy.name));
  });
}

function populateExplorerControls() {
  for (const select of [$("#inspector-strategy"), $("#failure-strategy")]) {
    benchmark.strategies.forEach((strategy) => {
      const option = document.createElement("option");
      option.value = strategy.id;
      option.textContent = strategy.name;
      select.appendChild(option);
    });
  }
  examples.documents.forEach((documentData, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = documentData.title;
    $("#inspector-document").appendChild(option);
  });
  const params = new URLSearchParams(location.search);
  $("#inspector-strategy").value = benchmark.strategies.some((item) => item.id === params.get("inspect"))
    ? params.get("inspect")
    : benchmark.winner;
  $("#inspector-document").value = examples.documents[Number(params.get("doc"))] ? params.get("doc") : "0";
  $("#failure-strategy").value = benchmark.strategies.some((item) => item.id === params.get("failureStrategy"))
    ? params.get("failureStrategy")
    : benchmark.winner;
  renderInspector();
  populateFailures(params.get("failure"));
}

function renderInspector() {
  const strategyId = $("#inspector-strategy").value;
  const documentIndex = Number($("#inspector-document").value);
  const output = $("#chunk-inspector");
  output.textContent = "";
  if (!strategyId || !Number.isInteger(documentIndex) || !examples.documents[documentIndex]) return;
  const documentData = examples.documents[documentIndex];
  const chunks = documentData.strategies[strategyId] || [];
  const boundaries = [...new Set([0, documentData.text.length, ...chunks.flatMap((item) => [item.start, item.end])])].sort((a, b) => a - b);
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < boundaries.length - 1; index += 1) {
    const start = boundaries[index];
    const end = boundaries[index + 1];
    const coverage = chunks.map((item, chunkIndex) => item.start < end && item.end > start ? chunkIndex + 1 : 0).filter(Boolean);
    const node = document.createElement(coverage.length ? "mark" : "span");
    if (coverage.length) {
      node.className = "chunk-span";
      node.dataset.overlap = String(coverage.length > 1);
      node.title = `Returned chunk${coverage.length > 1 ? "s" : ""}: ${coverage.join(", ")}`;
    }
    node.textContent = documentData.text.slice(start, end);
    fragment.appendChild(node);
  }
  output.appendChild(fragment);
  const indexedDifferent = chunks.some((item) => item.index_start !== item.start || item.index_end !== item.end);
  $("#inspector-summary").textContent = `${chunks.length} returned contexts${indexedDifferent ? "; indexed spans are smaller than returned context" : ""}.`;
  syncUrl();
}

function populateFailures(selected = "") {
  const strategyId = $("#failure-strategy").value;
  const select = $("#failure-example");
  select.textContent = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select an example";
  select.appendChild(placeholder);
  (examples.failures[strategyId] || []).forEach((failure, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1}. ${failure.question}`;
    select.appendChild(option);
  });
  if (select.options.length > 1) {
    select.value = (examples.failures[strategyId] || [])[Number(selected)] ? selected : "0";
    renderFailure();
  }
}

function renderFailure() {
  const strategyId = $("#failure-strategy").value;
  const failure = (examples.failures[strategyId] || [])[Number($("#failure-example").value)];
  if (!failure) return;
  $("#failure-question").textContent = failure.question;
  $("#gold-document").textContent = failure.gold_document;
  $("#gold-span").textContent = failure.gold_answer;
  $("#retrieved-document").textContent = failure.retrieved_document;
  $("#retrieved-chunk").textContent = failure.retrieved_chunk;
  syncUrl();
}

function storedKey() {
  try {
    return sessionStorage.getItem(SESSION_KEY) || "";
  } catch {
    return "";
  }
}

function saveKey(value) {
  try {
    if (value) sessionStorage.setItem(SESSION_KEY, value);
    else sessionStorage.removeItem(SESSION_KEY);
  } catch {
    // ponytail: private browsing can reject storage; the input still works for this run.
  }
}

function parsePairs(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2 || lines.length > 20) throw new Error("Add between 2 and 20 question and answer lines.");
  return lines.map((line, index) => {
    const tab = line.indexOf("\t");
    if (tab < 1 || tab === line.length - 1) throw new Error(`Line ${index + 1} needs a tab between question and answer.`);
    return {
      question: line.slice(0, tab).trim(),
      answers: line.slice(tab + 1).split("||").map((answer) => answer.trim()).filter(Boolean),
    };
  });
}

function setLiveStatus(message, state = "") {
  $("#live-status").textContent = message;
  $("#live-status").dataset.state = state;
}

async function fetchEmbeddingBatch(inputs, key, signal) {
  if (requestCount >= MAX_REQUESTS) throw new Error("The 8-request cap was reached.");
  requestCount += 1;
  $("#request-count").textContent = `${requestCount} / ${MAX_REQUESTS}`;
  const timeoutController = new AbortController();
  const timer = setTimeout(() => timeoutController.abort(), REQUEST_TIMEOUT_MS);
  const abort = () => timeoutController.abort();
  signal.addEventListener("abort", abort, { once: true });
  try {
    const response = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify({ model: MODEL, input: inputs, encoding_format: "float" }),
      signal: timeoutController.signal,
    });
    if (!response.ok) throw new Error(`OpenAI embeddings request failed with HTTP ${response.status}.`);
    const payload = await response.json();
    return payload.data.sort((a, b) => a.index - b.index).map((item) => item.embedding);
  } catch (error) {
    if (timeoutController.signal.aborted) throw new Error(signal.aborted ? "Run aborted." : "Embedding request timed out after 30 seconds.");
    throw error;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", abort);
  }
}

async function embedTexts(texts, key, signal, map = new Map()) {
  const unique = [...new Set(texts)].filter((text) => !map.has(text));
  if (map.size + unique.length > MAX_INPUTS) {
    throw new Error(`This run needs ${map.size + unique.length} embedding inputs. The cap is ${MAX_INPUTS}.`);
  }
  for (let offset = 0; offset < unique.length; offset += BATCH_SIZE) {
    setLiveStatus(`Embedding batch ${Math.floor(offset / BATCH_SIZE) + 1}.`);
    const batch = unique.slice(offset, offset + BATCH_SIZE);
    const vectors = await fetchEmbeddingBatch(batch, key, signal);
    batch.forEach((text, index) => map.set(text, vectors[index]));
  }
  return map;
}

function rankChunks(chunks, queryVector, vectorMap) {
  const ordered = chunks
    .map((item) => ({ item, score: cosine(queryVector, vectorMap.get(item.indexText)) }))
    .sort((a, b) => b.score - a.score);
  const seen = new Set();
  return ordered.filter(({ item }) => {
    const identity = `${item.start}:${item.end}`;
    if (seen.has(identity)) return false;
    seen.add(identity);
    return true;
  }).slice(0, 10).map(({ item }) => item);
}

function scoreLiveStrategy(chunks, pairs, vectorMap) {
  const rankings = pairs.map((pair) => {
    const ranked = rankChunks(chunks, vectorMap.get(pair.question), vectorMap);
    const uniqueContexts = [...new Map(chunks.map((item) => [`${item.start}:${item.end}`, item])).values()];
    return {
      relevances: ranked.map((item) => Number(isHit(item.text, pair.answers))),
      totalRelevant: uniqueContexts.filter((item) => isHit(item.text, pair.answers)).length,
      contextTokens: ranked.map((item) => estimateTokens(item.text)),
    };
  });
  return aggregateMetrics(rankings);
}

async function runLive(event) {
  event.preventDefault();
  const key = ($("#api-key").value || storedKey()).trim();
  const text = $("#document-text").value;
  let pairs;
  if (!key) {
    setLiveStatus("Add an OpenAI API key.", "error");
    $("#api-key").focus();
    return;
  }
  if (!key.startsWith("sk-")) {
    setLiveStatus("The OpenAI key should start with sk-.", "error");
    $("#api-key").focus();
    return;
  }
  if (text.length < 100) {
    setLiveStatus("Add at least 100 characters of document text.", "error");
    $("#document-text").focus();
    return;
  }
  try {
    pairs = parsePairs($("#qa-pairs").value);
  } catch (error) {
    setLiveStatus(error.message, "error");
    $("#qa-pairs").focus();
    return;
  }

  saveKey(key);
  requestCount = 0;
  $("#request-count").textContent = `0 / ${MAX_REQUESTS}`;
  activeController = new AbortController();
  $("#run-live").disabled = true;
  $("#run-live").textContent = "Running…";
  $("#abort-live").disabled = false;
  $("#live-results").hidden = true;

  try {
    const sentences = sentenceRanges(text);
    if (sentences.length > 500) throw new Error("The document has more than 500 sentences. Shorten it before running.");
    setLiveStatus("Embedding sentences for semantic breakpoints.");
    const sentenceTexts = sentences.map(([start, end]) => text.slice(start, end));
    const vectorMap = await embedTexts(sentenceTexts, key, activeController.signal);
    const sentenceVectors = sentenceTexts.map((sentence) => vectorMap.get(sentence));
    const chunksByStrategy = Object.fromEntries(STRATEGIES.map((strategy) => [
      strategy.id,
      strategy.id === "semantic" ? semanticChunks(text, sentenceVectors) : buildSyncChunks(strategy.id, text),
    ]));
    const allInputs = [
      ...pairs.map((pair) => pair.question),
      ...Object.values(chunksByStrategy).flatMap((chunks) => chunks.map((item) => item.indexText)),
    ];
    await embedTexts(allInputs, key, activeController.signal, vectorMap);
    const results = STRATEGIES.map((strategy) => ({
      ...strategy,
      metrics: scoreLiveStrategy(chunksByStrategy[strategy.id], pairs, vectorMap),
    }));
    renderLiveResults(results);
    setLiveStatus(`Complete. ${pairs.length} labeled queries scored across 8 strategies.`, "ok");
  } catch (error) {
    setLiveStatus(error.message, "error");
  } finally {
    activeController = null;
    $("#run-live").disabled = false;
    $("#run-live").textContent = "Run Benchmark";
    $("#abort-live").disabled = true;
  }
}

function renderLiveResults(results) {
  const shell = $("#live-results");
  const tbody = shell.querySelector("tbody");
  tbody.textContent = "";
  results.sort((a, b) => b.metrics.ndcg_at_10 - a.metrics.ndcg_at_10).forEach((strategy) => {
    const row = document.createElement("tr");
    const values = [
      strategy.name,
      ...KS.map((k) => formatScore(strategy.metrics[`recall_at_${k}`])),
      formatScore(strategy.metrics.mrr_at_10),
      formatScore(strategy.metrics.ndcg_at_10),
      formatCount(strategy.metrics.context_tokens_to_90),
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    tbody.appendChild(row);
  });
  shell.hidden = false;
}

function runSelftest() {
  let passed = 0;
  let failed = 0;
  const check = (condition, label) => {
    console.assert(condition, label);
    if (condition) passed += 1;
    else failed += 1;
  };
  const rankings = [
    { relevances: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], totalRelevant: 1, contextTokens: Array(10).fill(1) },
    { relevances: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], totalRelevant: 1, contextTokens: Array(10).fill(1) },
  ];
  const metrics = aggregateMetrics(rankings);
  const recalls = KS.map((k) => metrics[`recall_at_${k}`]);
  check(recalls.every((value, index) => !index || value >= recalls[index - 1]), "recall@k is monotone");
  check(reciprocalRank([0, 0, 1]) === 1 / 3, "MRR first hit at r equals 1/r");
  check(ndcg([1, 1, 0], 2) === 1, "ideal nDCG equals 1");
  check(Math.abs(cosine([1, 2], [1, 2]) - 1) < 1e-12, "cosine vector with itself equals 1");
  check(buildBm25Ranker(["alpha alpha", "beta"])("alpha")[0] === 0, "BM25 ranks a matching chunk first");

  const source = "Alpha one. Beta two!\n\nGamma three?";
  const sentences = sentenceRanges(source);
  const outputs = [
    fixedChunks(source, 12),
    fixedChunks(source, 12, 1),
    fixedChunks(source, 12, 3),
    structuralChunks(source, 12),
    sentenceWindowChunks(source, 1),
    parentDocumentChunks(source, 7, 14),
    semanticChunks(source, tfidfVectors(sentences.map(([start, end]) => source.slice(start, end)))),
    fixedChunks(source, 24),
  ];
  outputs.forEach((chunks, index) => {
    const coverage = new Uint8Array(source.length);
    chunks.forEach((item) => {
      for (let position = item.start; position < item.end; position += 1) coverage[position] = 1;
    });
    check(coverage.every(Boolean), `chunker ${index + 1} covers every source character`);
  });
  check(fixedChunks("short", 100).length === 1, "short text returns exactly one chunk");
  const summary = `ChunkLab self-test: ${failed ? "FAIL" : "PASS"} (${passed} passed, ${failed} failed)`;
  (failed ? console.error : console.info)(summary);
  return { passed, failed };
}

async function init() {
  $("#api-key").value = storedKey();
  $("#api-key").addEventListener("input", (event) => saveKey(event.target.value.trim()));
  $("#forget-key").addEventListener("click", () => {
    if (!confirm("Forget the API key stored for this tab?")) return;
    saveKey("");
    $("#api-key").value = "";
    setLiveStatus("Key cleared from this tab.", "ok");
  });
  $("#document-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 100000) {
      setLiveStatus("File is too large. Use a plain-text file under 100 KB.", "error");
      return;
    }
    $("#document-text").value = (await file.text()).slice(0, 25000);
  });
  $("#live-form").addEventListener("submit", runLive);
  $("#abort-live").addEventListener("click", () => activeController?.abort());
  $("#inspector-strategy").addEventListener("change", renderInspector);
  $("#inspector-document").addEventListener("change", renderInspector);
  $("#failure-strategy").addEventListener("change", populateFailures);
  $("#failure-example").addEventListener("change", renderFailure);
  document.querySelectorAll("#leaderboard-table button[data-sort]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = button.dataset.sort;
      sortDirection = sortKey === next ? -sortDirection : (next === "name" ? 1 : -1);
      sortKey = next;
      renderLeaderboard();
      syncUrl();
    });
  });
  window.addEventListener("beforeunload", (event) => {
    if (!$("#document-text").value && !$("#qa-pairs").value) return;
    event.preventDefault();
    event.returnValue = "";
  });

  try {
    const responses = await Promise.all(["results", "examples", "meta"].map((name) => fetch(`data/${name}.json`, { cache: "no-store" })));
    if (responses.some((response) => !response.ok)) throw new Error("Committed benchmark artifacts are missing.");
    [benchmark, examples, meta] = await Promise.all(responses.map((response) => response.json()));
    const params = new URLSearchParams(location.search);
    if (["name", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10", "mean_chunk_tokens", "context_tokens_to_90"].includes(params.get("sort"))) {
      sortKey = params.get("sort");
      sortDirection = params.get("dir") === "asc" ? 1 : -1;
    }
    $("#meta-corpus").textContent = meta.source;
    $("#meta-sample").textContent = `${meta.article_count} articles / ${meta.question_count} questions`;
    $("#meta-model").textContent = meta.retriever.toUpperCase();
    $("#meta-seed").textContent = String(meta.seed);
    if (meta.token_counting.estimated) {
      $("#mean-token-heading").textContent = "Est. Mean Chunk Tokens";
      $("#context-token-heading").textContent = "Est. Context Tokens to >90%";
      $("#cost-figure-caption").textContent = "nDCG@10 vs estimated context tokens to >90% recall";
      $("#token-note").textContent = `Every token count in this committed run is estimated using ${meta.token_counting.method}.`;
    }
    renderLeaderboard();
    renderRecallChart();
    renderCostChart();
    populateExplorerControls();
    const formattedDate = new Intl.DateTimeFormat(undefined, { dateStyle: "medium" })
      .format(new Date(`${meta.date}T00:00:00`));
    $("#load-status").textContent = `Committed run loaded: ${formattedDate}.`;
    $("#load-status").dataset.state = "ok";
  } catch (error) {
    ["meta-corpus", "meta-sample", "meta-model", "meta-seed"].forEach((id) => {
      $(id).textContent = "artifact unavailable";
    });
    $("#load-status").textContent = error.message;
    $("#load-status").dataset.state = "error";
  }

  if (new URLSearchParams(location.search).get("selftest") === "1") runSelftest();
}

if (typeof document !== "undefined") init();
if (typeof process !== "undefined" && process.argv.includes("--selftest")) {
  const result = runSelftest();
  if (result.failed) process.exitCode = 1;
}
