---
permalink: /
layout: splash
header:
    overlay_color: rgb(171, 74, 73)
    actions:
        - label: "Paper"
          url: https://arxiv.org
          icon: "fas fa-book"
        - label: "Blog"
          url: "https://mcgill-nlp.github.io/llm2vec-gen/blog/"
          icon: "fas fa-newspaper"
        - label: "Code"
          url: "https://github.com/McGill-NLP/llm2vec-gen/"
          icon: "fab fa-github"
        - label: "Models & Data"
          url: "https://huggingface.co/collections/McGill-NLP/llm2vec-gen"
          icon: "fas fa-robot"
        - label: "WandB Logs"
          url: "https://wandb.ai/siva-reddy-mila-org/llm2vec-gen"
          icon: "fas fa-chart-bar"
        

title: "LLM2Vec-Gen: Generative Embeddings from Large Language Models"
excerpt: Parishad BehnamGhader, Vaibhav Adlakha, Fabian David Schmidt, Nicolas Chapados, Marius Mosbach, and Siva Reddy
---

LLM2Vec-Gen is a recipe to train interpretable, generative embeddings that encode the potential answer of an LLM to a query rather than the query itself.

<p align="center">
  <img src="../../assets/llm2vecgen.gif" width="95%" alt="llm2vecgen_main_figure"/>
</p>



You can easily load a HF pretrained model and use it as an encoder or for generation:

```python
from llm2vec_gen import LLM2VecGenModel

model = LLM2VecGenModel.from_pretrained("McGill-NLP/LLM2Vec-Gen-Qwen3-8B")

# Encode text
emb = model.encode("Is Montreal located in Canada?")

# Generate text from Embeddings (encoder + decoder)
answer, enc_before_answer = model.generate("Is Montreal located in Canada?", max_new_tokens=100)
```
This code snippet will return the answer of the LLM2Vec-Gen model generated from the generative embeddings of the input. You can access the embeddings either from the `.encode()` function or from the `.generate()` function. 

> Yes, Montreal is a city in Canada. It is the second-largest city in the country, located in the province of Quebec. Montreal is known for its rich cultural heritage, historic architecture, and vibrant arts scene.
>
> True tensor([[-0.2393,  0.0280, -0.5078,  ...,  0.1270,  0.6484,  0.3574]], device='cuda:0', dtype=torch.bfloat16)
