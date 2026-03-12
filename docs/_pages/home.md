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
  <img src="./assets/llm2vecgen.gif" width="95%" alt="llm2vecgen_main_figure"/>
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
tensor([[0.8750, 0.1182],
        [0.0811, 0.9336]])
"""
```

Note that in all examples, the instructions should be as if you are generating the answer to the input. 
<br>
Other examples to try LLM2Vec-Gen in other tasks:
<details markdown="1">
<summary><strong>Sentence Similarity</strong></summary>

```python
instruction = "Generate text that is semantically similar to this text: "

queries = [
  "The traveler was frustrated because the flight had been delayed for several hours.",
  "Stars rotate due to the angular momentum of the gas they formed from."
]
q_reps = model.encode([instruction + q for q in queries])

pairs = [
  "A multi-hour wait at the airport left the passenger feeling quite annoyed.",
  "The rotational motion of a stellar body is a direct consequence of the conservation of angular momentum from its original protostellar cloud.",
]
p_reps = model.encode([instruction + p for p in pairs])

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
p_reps_norm = torch.nn.functional.normalize(p_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, p_reps_norm.transpose(0, 1))

print(cos_sim)
"""
tensor([[0.8008, 0.2539],
        [0.2061, 0.8906]])
"""
```
</details>

<details markdown="1">
<summary><strong>Classification</strong></summary>

```python
q_instruction = "Classify the emotion expressed in the given text into anger and joy: "
p_instruction = "Summarize this text: "

queries = [
  "I just feel irritated right now",
  "I'm feeling really thrilled and excited",
]
q_reps = model.encode([q_instruction + q for q in queries])

pairs = [
  "This text is classified as \"angry\".",
  "This text is classified as \"joy\".",
]
p_reps = model.encode([p_instruction + p for p in pairs])

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
p_reps_norm = torch.nn.functional.normalize(p_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, p_reps_norm.transpose(0, 1))

print(cos_sim)
"""
tensor([[0.8008, 0.6953],
        [0.7070, 0.8008]])
"""
```
</details>

<details markdown="1">
<summary><strong>Clustering</strong></summary>

```python
from sklearn.cluster import KMeans

instruction = "Cluster the following Amazon review: "

sentences = [
  "This product is amazing and I love it",
  "Fantastic experience, highly recommend",
  "Terrible quality, total waste of money",
  "Awful product, very disappointed",
]

reps = model.encode([instruction + s for s in sentences])
reps = reps.float().cpu().numpy()

labels = KMeans(n_clusters=2, random_state=0).fit_predict(reps)

for s, l in zip(sentences, labels):
  print(f"[{l}] {s}")

"""
[1] This product is amazing and I love it
[1] Fantastic experience, highly recommend
[0] Terrible quality, total waste of money
[0] Awful product, very disappointed
"""
```
</details>


<br>

LLM2Vec-Gen provides interpretable embeddings. You can use the following code to **decode the content** embedded in the embeddings:

```python
_, recon_hidden_states = model.encode("what does disk cleanup mean on a computer", get_recon_hidden_states=True)
# recon_hidden_states: torch.Tensor with shape (1, compression token size, hidden_dim)

answer = model.generate(recon_hidden_states=recon_hidden_states, max_new_tokens=55)

print(answer)
"""
* **\n\n**Disk Cleanup** is a built-in utility in Windows that helps you **free up disk space** by **removing unnecessary files** from your computer. It is designed to clean up temporary files, system cache, and other files that are no longer needed.\n\n
"""
```
This code snippet will return the answer of the LLM2Vec-Gen model generated from the generative embeddings of the input (`recon_hidden_states`).



Check out our paper's thread on X.
\[Placeholder\]
<!-- <blockquote class="twitter-tweet" data-align="center"><p lang="en" dir="ltr">We introduce LLM2Vec, a simple approach to transform any decoder-only LLM into a text encoder. We achieve SOTA performance on MTEB in the unsupervised and supervised category (among the models trained only on publicly available data). 🧵1/N<br><br>Paper: <a href="https://t.co/1ARXK1SWwR">https://t.co/1ARXK1SWwR</a> <a href="https://t.co/L4jotnufn2">pic.twitter.com/L4jotnufn2</a></p>&mdash; Vaibhav Adlakha (@vaibhav_adlakha) <a href="https://twitter.com/vaibhav_adlakha/status/1777854148584591441?ref_src=twsrc%5Etfw">April 10, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> -->
