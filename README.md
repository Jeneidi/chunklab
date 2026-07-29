# ChunkLab

Chunk boundaries are a retrieval decision, so benchmark them.

**Across 800 SQuAD questions, Fixed 1000 retrieves an answer by rank 10 for 93.125% of queries while semantic (TF-IDF proxy) reaches 87.25%, a measured spread of 5.875 percentage points.**

![Best Recall at 10](https://img.shields.io/badge/best%20R%4010-93.125%25-brightgreen)
![Recall spread](https://img.shields.io/badge/R%4010%20spread-5.875%20points-blue)
![Sample](https://img.shields.io/badge/sample-800%20questions-informational)
![Retriever](https://img.shields.io/badge/BM25-k1%3D1.2%20%7C%20b%3D0.75-8A2BE2)

## The problem

Chunk size, overlap, and returned context are often copied from framework defaults. Those boundaries change which terms occur together, how term frequency is normalized, and how much context retrieval returns. ChunkLab holds the corpus, questions, retriever, and metrics constant so the chunking strategy is the variable.

BM25 is deliberate here, not a fallback presented as embedding retrieval. Chunk boundaries change what each indexed document unit contains, so they change lexical retrieval scores too. The committed benchmark therefore runs locally with no key and no paid service.

## How it works

Seed `20260729` selects 40 source articles and 800 questions from the [SQuAD v1.1 development set](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json). Each strategy indexes its own chunks. A returned context is relevant when it contains any case-insensitive, whitespace-normalized gold answer string.

The 8 strategies are:

1. Fixed 500 characters, no overlap.
2. Fixed 500 characters, 50-character overlap.
3. Fixed 500 characters, 125-character overlap.
4. Recursive 500-character split at paragraph, sentence, whitespace, then hard-cut boundaries.
5. Sentence window: index 1 sentence and return it with 2 neighbors on each side.
6. Parent document: index 250-character children and return their 1,000-character parents.
7. Semantic (TF-IDF proxy): split at the 80th percentile of adjacent-sentence TF-IDF cosine distance, with a 500-character ceiling.
8. Fixed 1,000 characters, no overlap.

## Method

The retriever is standard Okapi BM25 with `k1=1.2` and `b=0.75`. Text is lowercased and split on non-alphanumeric characters with no stemming. Scores use:

```text
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q, d) = sum IDF(t) * tf(t,d) * (k1 + 1)
                   / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

Strategy 7 computes raw term-frequency vectors with smoothed inverse document frequency, then cosine similarity between adjacent sentences. It is labeled `semantic (TF-IDF proxy)` everywhere. The browser-only BYOK mode keeps the embedding-based semantic variant and labels it `Semantic (embeddings)`.

Recall@k is the fraction of questions with at least 1 relevant context by rank k. MRR@10 uses the reciprocal rank of the first hit. nDCG@10 is rank-aware and credits every relevant returned context. Context cost is the average sum of returned context tokens at the first measured k where aggregate recall is greater than 90%.

Research verification fixed two details in the implementation. [Lucene's BM25 documentation](https://lucene.apache.org/core/8_1_1/core/org/apache/lucene/search/similarities/BM25Similarity.html) specifies the positive IDF form and the `k1=1.2`, `b=0.75` defaults used here. [scikit-learn's TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) documents raw term frequency, smoothed IDF, and L2 cosine behavior. ChunkLab mirrors those choices in standard-library code instead of adding either dependency.

## Results

These are the exact values in [`data/results.json`](data/results.json). `n/a` means Recall@k never exceeded 0.9 at a measured rank. `tiktoken` was not installed for this run, so every token number below is an estimate using `ceil(UTF-8 bytes / 4)`. Retrieval metrics are measured, not estimated.

| Strategy | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Estimated mean chunk tokens | Estimated context tokens to >90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 500 | 0.64875 | 0.79875 | 0.83625 | 0.885 | 0.7325897817460317 | 0.5493573843047335 | 124.21954067730634 | n/a |
| Fixed 500 + 10% | 0.6625 | 0.8125 | 0.8675 | 0.90625 | 0.7471731150793651 | 0.5519227706098686 | 124.36328536328536 | 1250.3125 |
| Fixed 500 + 25% | 0.695 | 0.83 | 0.8675 | 0.9175 | 0.7722480158730158 | 0.566983037019141 | 124.49047200234536 | 1249.8325 |
| Recursive | 0.65 | 0.7925 | 0.84625 | 0.89375 | 0.734109126984127 | 0.5494843416484678 | 123.86826811313445 | n/a |
| Sentence window | 0.6875 | 0.8125 | 0.8575 | 0.8925 | 0.7593993055555555 | 0.32271907302533165 | 185.14106657207049 | n/a |
| Parent document | 0.665 | 0.82 | 0.865 | 0.91125 | 0.7493234126984127 | 0.5799763650526358 | 248.73724340175954 | 2496.41625 |
| Semantic (TF-IDF proxy) | 0.645 | 0.78125 | 0.83 | 0.8725 | 0.7216671626984127 | 0.5332221405513783 | 67.130480385987 | n/a |
| Fixed 1000 | 0.715 | 0.85125 | 0.8925 | 0.93125 | 0.7921309523809523 | 0.6154857868407483 | 246.55564142194746 | 2495.0825 |

Fixed 1000 is the recorded winner because it has the highest nDCG@10. The highlighted site row uses context cost only as a tie-breaker.

## Reproduce

No API key is read or required:

```bash
python3 scripts/build_benchmark.py
python3 scripts/build_benchmark.py --selftest
node app.js --selftest
python3 -m http.server 8000
```

The builder downloads SQuAD to the gitignored `data/squad-dev-v1.1.json` cache when it is absent. Install the optional tokenizer before rebuilding if exact OpenAI tokenizer counts are needed:

```bash
python3 -m pip install -r requirements.txt
```

The committed outputs are:

- `data/results.json`: metrics for every strategy.
- `data/examples.json`: real misses and real article excerpts with chunk boundaries.
- `data/meta.json`: source, retriever, BM25 parameters, token-count method, seed, counts, and run date.

## Optional BYOK live mode

The static site has no backend. Live mode sends embedding inputs only to the hardcoded OpenAI HTTPS endpoint and runs the same chunkers and metric formulas in the browser. The key stays in `sessionStorage` for the current tab and the visible Forget Key control removes it.

The live embedding results answer a different retrieval question from the committed BM25 results. They are labeled separately and should not be compared as if only chunking changed.

## Limitations

- Gold-answer string containment is reproducible but misses semantic equivalents and can over-credit a common string.
- SQuAD is English Wikipedia factoid QA. The winner may not transfer to legal text, code, tables, conversations, or multi-hop questions.
- BM25 measures lexical retrieval. Embedding retrievers may react differently to the same boundaries.
- Semantic (TF-IDF proxy) is an honest lexical breakpoint signal, not the embedding-based semantic chunker.
- The committed token counts are estimates, not tokenizer measurements.
- No reranker, query rewriting, hybrid retrieval, generation step, or downstream answer-quality evaluation is included.

## License

Code is MIT. SQuAD data is CC BY-SA 4.0.
