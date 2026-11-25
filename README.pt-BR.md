# RAG Retrieval Eval

[English](README.md) · [Como contribuir](CONTRIBUTING.md)

Compare recuperadores de documentos para RAG usando os mesmos julgamentos de relevância. Use Python 3.11 ou posterior. O exemplo em português usa documentos sintéticos e não precisa de dependências externas nem de acesso à rede.

```sh
python3 -m rag_retrieval_eval rank examples/corpus.json examples/queries.json --k 3 > /tmp/rag-rankings.json
python3 -m rag_retrieval_eval score examples/judgments.json /tmp/rag-rankings.json --k 2 \
  --baseline examples/reference-ranking.json
```

Os arquivos são objetos JSON em UTF-8. O corpus associa ID de documento ao texto. As consultas associam ID de consulta ao texto. Os julgamentos e rankings associam os mesmos IDs de consulta a listas de IDs de documentos sem duplicatas. Cada julgamento precisa de pelo menos um documento relevante. Uma lista vazia de resultados indica que o recuperador não retornou nada.

O relatório traz recall, posição recíproca média (MRR) e proporção de consultas com ao menos um acerto entre os `k` primeiros resultados. Os deltas usam candidato menos baseline. O `--min-recall` aceita frações de `0` a `1` e retorna código `1` se o limite falhar. Entradas inválidas retornam `2`.
