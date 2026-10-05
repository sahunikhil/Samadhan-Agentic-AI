# Third-party notices

Samadhan Agentic AI is MIT-licensed (see `LICENSE`). It depends on, and the Docker image
redistributes, the third-party components below under their own licenses.

## Python packages

All production dependencies are permissively licensed (MIT, BSD, Apache-2.0, ISC, PSF) except:

| Package | License | Notes |
| --- | --- | --- |
| `psycopg`, `psycopg-binary`, `psycopg-pool` | LGPL-3.0-only | used unmodified as separately installed libraries; replaceable by the user |
| `certifi`, `orjson` (parts), `tqdm` (parts) | MPL-2.0 | used unmodified; source available from PyPI |

No GPL or AGPL code is included. Every installed package ships its license text in its
`*.dist-info/` directory inside the image. Evaluation-only extras (`ragas`, `langchain-community`)
are not part of the production image.

## Machine-learning models bundled in the Docker image

| Model | License |
| --- | --- |
| `BAAI/bge-small-en-v1.5` (dense embeddings) | MIT |
| `Qdrant/bm25` (sparse embeddings) | Apache-2.0 |
| `Xenova/ms-marco-MiniLM-L-6-v2` (reranker, ONNX export of `cross-encoder/ms-marco-MiniLM-L-6-v2`) | Apache-2.0 |

## Models used through hosted APIs (not redistributed)

Use is governed by each provider's terms of service and the model license: `openai/gpt-oss-120b`
and `openai/gpt-oss-20b` (Apache-2.0), `qwen/qwen3.8-27b` (Apache-2.0), Meta Llama Prompt Guard 2
(Meta Llama license; see its model card), Google Gemini models (Google API terms). Free tiers may
use prompts to improve the provider's services; do not send real customer data on free tiers.
