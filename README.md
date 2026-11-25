# RAG Retrieval Eval

[Português](README.pt-BR.md) · [Contributing](CONTRIBUTING.md)

Measure whether a retrieval change brings the right documents into a language model's context. Compare saved rankings against the same human judgments, or generate a local BM25 baseline. No models, credentials, or network calls are needed at runtime.

## Run the example

Use Python 3.11 or newer from the repository root.

```sh
python3 -m rag_retrieval_eval rank examples/corpus.json examples/queries.json --k 3 > /tmp/rag-rankings.json
python3 -m rag_retrieval_eval score examples/judgments.json /tmp/rag-rankings.json --k 2 \
  --baseline examples/reference-ranking.json
```

The first command uses BM25 with `k1=1.2`, `b=0.75`, accent-preserving Unicode tokenization, unique query terms, and deterministic document ID ordering for ties. Unmatched documents are omitted. It is a baseline for small text corpora. [BM25 reference](https://doi.org/10.1561/1500000019)

## Input and output

The corpus and query files are UTF-8 JSON objects mapping IDs to text. Judgments map each query ID to a nonempty list of relevant document IDs. A ranking maps the same query IDs to ordered document ID lists. An empty ranking records a query with zero results. Each list uses unique document IDs.

The score report contains macro-averaged recall, mean reciprocal rank, hit rate, and per-query values within the top `k`. With `--baseline`, deltas are candidate minus baseline at the same cutoff. The tool evaluates document retrieval. Generated answers can be evaluated separately. [NIST TREC RAG Track](https://trec.nist.gov/pubs/trec34/papers/Overview_rag.pdf)

Exit status is `0` for a completed run, `1` when an optional `--min-recall` gate fails, or `2` for invalid input. Use a fraction from `0` to `1` for the gate. `python -m pip install .` also installs the `rag-retrieval-eval` command.
