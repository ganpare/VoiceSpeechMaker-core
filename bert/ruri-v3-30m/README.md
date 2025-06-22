---
language:
- ja
tags:
- sentence-similarity
- feature-extraction
base_model: cl-nagoya/ruri-v3-pt-30m
widget: []
pipeline_tag: sentence-similarity
license: apache-2.0
datasets:
- cl-nagoya/ruri-v3-dataset-ft
---

# Ruri: Japanese General Text Embeddings

**Ruri v3** is a general-purpose Japanese text embedding model built on top of [**ModernBERT-Ja**](https://huggingface.co/collections/sbintuitions/modernbert-ja-67b68fe891132877cf67aa0a).
Ruri v3 offers several key technical advantages:
- **State-of-the-art performance** for Japanese text embedding tasks.
- **Supports sequence lengths up to 8192 tokens**  
  - Previous versions of Ruri (v1, v2) were limited to 512.
- **Expanded vocabulary of 100K tokens**, compared to 32K in v1 and v2  
  - The larger vocabulary make input sequences shorter, improving efficiency.
- **Integrated FlashAttention**, following ModernBERT's architecture  
  - Enables faster inference and fine-tuning.
- **Tokenizer based solely on SentencePiece**  
  - Unlike previous versions, which relied on Japanese-specific BERT tokenizers and required pre-tokenized input, Ruri v3 performs tokenization with SentencePiece only—no external word segmentation tool is required.


## Model Series

We provide Ruri-v3 in several model sizes. Below is a summary of each model.

|ID| #Param. | #Param.<br>w/o Emb.|Dim.|#Layers|Avg. JMTEB|
|-|-|-|-|-|-|
|[**cl-nagoya/ruri-v3-30m**](https://huggingface.co/cl-nagoya/ruri-v3-30m)|37M|10M|256|10|**74.51**|
|[cl-nagoya/ruri-v3-70m](https://huggingface.co/cl-nagoya/ruri-v3-70m)|70M|31M|384|13|75.48|
|[cl-nagoya/ruri-v3-130m](https://huggingface.co/cl-nagoya/ruri-v3-130m)|132M|80M|512|19|76.55|
|[cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)|315M|236M|768|25|77.24|


## Usage

You can use our models directly with the transformers library v4.48.0 or higher:

```bash
pip install -U "transformers>=4.48.0" sentence-transformers
```

Additionally, if your GPUs support Flash Attention 2, we recommend using our models with Flash Attention 2.

```
pip install flash-attn --no-build-isolation
```

Then you can load this model and run inference.
```python
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("cl-nagoya/ruri-v3-30m", device=device)

# Ruri v3 employs a 1+3 prefix scheme to distinguish between different types of text inputs:
# "" (empty string) is used for encoding semantic meaning.
# "トピック: " is used for classification, clustering, and encoding topical information.
# "検索クエリ: " is used for queries in retrieval tasks.
# "検索文書: " is used for documents to be retrieved.
sentences = [
    "川べりでサーフボードを持った人たちがいます",
    "サーファーたちが川べりに立っています",
    "トピック: 瑠璃色のサーファー",
    "検索クエリ: 瑠璃色はどんな色？",
    "検索文書: 瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。",
]

embeddings = model.encode(sentences, convert_to_tensor=True)
print(embeddings.size())
# [5, 256]

similarities = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2)
print(similarities)
# [[1.0000, 0.9540, 0.8512, 0.7322, 0.7274],
#  [0.9540, 1.0000, 0.8531, 0.7437, 0.7305],
#  [0.8512, 0.8531, 1.0000, 0.8910, 0.8649],
#  [0.7322, 0.7437, 0.8910, 1.0000, 0.9479],
#  [0.7274, 0.7305, 0.8649, 0.9479, 1.0000]]
```

## Benchmarks

### JMTEB
Evaluated with [JMTEB](https://github.com/sbintuitions/JMTEB).

|Model|#Param.|Avg.|Retrieval|STS|Classfification|Reranking|Clustering|PairClassification|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
||||||||||
|[**Ruri-v3-30m**](https://huggingface.co/cl-nagoya/ruri-v3-30m)<br/>(this model)|**37M**|**74.51**|78.08|82.48|74.80|93.00|52.12|62.40|
|[Ruri-v3-70m](https://huggingface.co/cl-nagoya/ruri-v3-70m)|70M|75.48|79.96|79.82|76.97|93.27|52.70|61.75|
|[Ruri-v3-130m](https://huggingface.co/cl-nagoya/ruri-v3-130m)|132M|76.55|81.89|79.25|77.16|93.31|55.36|62.26|
|[Ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)|315M|77.24|81.89|81.22|78.66|93.43|55.69|62.60|
||||||||||
|[sbintuitions/sarashina-embedding-v1-1b](https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b)|1.22B|75.50|77.61|82.71|78.37|93.74|53.86|62.00|
|[PLaMo-Embedding-1B](https://huggingface.co/pfnet/plamo-embedding-1b)|1.05B|76.10|79.94|83.14|77.20|93.57|53.47|62.37|
||||||||||
|OpenAI/text-embedding-ada-002|-|69.48|64.38|79.02|69.75|93.04|48.30|62.40|
|OpenAI/text-embedding-3-small|-|70.86|66.39|79.46|73.06|92.92|51.06|62.27|
|OpenAI/text-embedding-3-large|-|73.97|74.48|82.52|77.58|93.58|53.32|62.35|
||||||||||
|[pkshatech/GLuCoSE-base-ja](https://huggingface.co/pkshatech/GLuCoSE-base-ja)|133M|70.44|59.02|78.71|76.82|91.90|49.78|66.39|
|[pkshatech/GLuCoSE-base-ja-v2](https://huggingface.co/pkshatech/GLuCoSE-base-ja-v2)|133M|72.23|73.36|82.96|74.21|93.01|48.65|62.37|
|[retrieva-jp/amber-base](https://huggingface.co/retrieva-jp/amber-base)|130M|72.12|73.40|77.81|76.14|93.27|48.05|64.03|
|[retrieva-jp/amber-large](https://huggingface.co/retrieva-jp/amber-large)|315M|73.22|75.40|79.32|77.14|93.54|48.73|60.97|
||||||||||
|[sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)|472M|64.70|40.12|76.56|72.66|91.63|44.88|62.33|
|[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)|118M|69.52|67.27|80.07|67.62|93.03|46.91|62.19|
|[intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)|278M|70.12|68.21|79.84|69.30|92.85|48.26|62.26|
|[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)|560M|71.65|70.98|79.70|72.89|92.96|51.24|62.15|
||||||||||
|[Ruri-Small](https://huggingface.co/cl-nagoya/ruri-small)|68M|71.53|69.41|82.79|76.22|93.00|51.19|62.11|
|[Ruri-Small v2](https://huggingface.co/cl-nagoya/ruri-small-v2)|68M|73.30|73.94|82.91|76.17|93.20|51.58|62.32|
|[Ruri-Base](https://huggingface.co/cl-nagoya/ruri-base)|111M|71.91|69.82|82.87|75.58|92.91|54.16|62.38|
|[Ruri-Base v2](https://huggingface.co/cl-nagoya/ruri-base-v2)|111M|72.48|72.33|83.03|75.34|93.17|51.38|62.35|
|[Ruri-Large](https://huggingface.co/cl-nagoya/ruri-large)|337M|73.31|73.02|83.13|77.43|92.99|51.82|62.29|
|[Ruri-Large v2](https://huggingface.co/cl-nagoya/ruri-large-v2)|337M|74.55|76.34|83.17|77.18|93.21|52.14|62.27|


## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [cl-nagoya/ruri-v3-pt-30m](https://huggingface.co/cl-nagoya/ruri-v3-pt-30m) 
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 256
- **Similarity Function:** Cosine Similarity
- **Language:** Japanese
- **License:** Apache 2.0
- **Paper:** https://arxiv.org/abs/2409.07737


### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: ModernBertModel 
  (1): Pooling({'word_embedding_dimension': 256, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Citation

```bibtex
@misc{
  Ruri,
  title={{Ruri: Japanese General Text Embeddings}}, 
  author={Hayato Tsukagoshi and Ryohei Sasano},
  year={2024},
  eprint={2409.07737},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2409.07737}, 
}
```


## License
This model is published under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).