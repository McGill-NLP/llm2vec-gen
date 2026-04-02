---
permalink: /
layout: splash
header:
    overlay_color: rgb(171, 74, 73)
    actions:
        - label: "Paper"
          url: https://arxiv.org/abs/2603.10913
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
  <img src="./assets/llm2vecgen_main_figure.png" width="95%" alt="llm2vecgen_main_figure"/>
</p>



Load a pretrained model:
```python
import torch
from llm2vec_gen import LLM2VecGenModel

model = LLM2VecGenModel.from_pretrained("McGill-NLP/LLM2Vec-Gen-Qwen3-8B")
```

As an example, you can use the model for **retrieval** using the following code snippet:
```python
q_instruction = "Generate a passage that best answers this question: "
d_instruction = "Summarize the following passage: "

queries = [
  "where do polar bears live and what's their habitat",
  "what does disk cleanup mean on a computer"
]
q_reps = model.encode([q_instruction + q for q in queries])

documents = [
  "Polar bears live throughout the circumpolar North in the Arctic, spanning across Canada, Alaska (USA), Russia, Greenland, and Norway. Their primary habitat is sea ice over the continental shelf, which they use for hunting, mating, and traveling. They are marine mammals that rely on this environment to hunt seals.",
  "Disk Cleanup is a built-in Windows tool that frees up hard drive space by scanning for and deleting unnecessary files like temporary files, cached data, Windows updates, and items in the Recycle Bin. It improves computer performance by removing \"junk\" files, which can prevent the system from running slowly due to low storage.",
]
d_reps = model.encode([d_instruction + d for d in documents])

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
"""
tensor([[0.8789, 0.0938],
        [0.1143, 0.9297]])
"""
```

Note that in all examples, the instructions should be as if you are generating the answer to the input. 
<br>
Other examples to try LLM2Vec-Gen in other tasks (e.g., classification and clustering) are presented in the paper's [GitHub repository](https://github.com/McGill-NLP/llm2vec-gen/).


<br>
LLM2Vec-Gen provides interpretable embeddings. You can use the following code to **decode the content** embedded in the embeddings:

```python
_, recon_hidden_states = model.encode("what does disk cleanup mean on a computer", get_recon_hidden_states=True)
# recon_hidden_states: torch.Tensor with shape (1, compression token size, hidden_dim)

answer = model.generate(recon_hidden_states=recon_hidden_states, max_new_tokens=55)

print(answer)
"""
**Disk Cleanup**" is a built-in utility in Windows that helps you **free up disk space** by **removing unnecessary files** and **temporary data** that are no longer needed. [...]"""
"""
```
This code snippet will return the answer of the LLM2Vec-Gen model generated from the generative embeddings of the input (`recon_hidden_states`).



Check out our paper's thread on X.
<blockquote class="twitter-tweet" data-align="center"><p lang="en" dir="ltr">Your LLM already knows the answer. Why is your embedding model still encoding the question?<br><br>🚨Introducing LLM2Vec-Gen: your frozen LLM generates the answer&#39;s embedding in a single forward pass — without ever generating the answer. Not only that, the frozen LLM can decode the… <a href="https://t.co/XlW6SVTp5t">pic.twitter.com/XlW6SVTp5t</a></p>&mdash; Vaibhav Adlakha (@vaibhav_adlakha) <a href="https://twitter.com/vaibhav_adlakha/status/2032065008603951187?ref_src=twsrc%5Etfw">March 12, 2026</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
