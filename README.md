# *LLM2VEC-GEN: Generative Embeddings from Large Language Models* 

[![arxiv](https://img.shields.io/badge/arXiv-2603.10913-b31b1b.svg)](https://arxiv.org/abs/2603.10913)
[![PyPI](https://img.shields.io/pypi/v/llm2vec-gen?label=PyPI)](https://pypi.org/project/llm2vec-gen/)
[![HF Link](https://img.shields.io/badge/HF%20Models-LLM2Vec--Gen-FFD21E.svg)](https://huggingface.co/collections/McGill-NLP/llm2vec-gen)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/McGill-NLP/llm2vec-gen/blob/main/LICENSE)
[![WandB](https://img.shields.io/badge/WandB-Logs-yellow.svg)](https://wandb.ai/siva-reddy-mila-org/llm2vec-gen)

LLM2Vec-Gen is a recipe to train interpretable, generative embeddings that encode the potential answer of an LLM to a query rather than the query itself. This repository supports codes for training and evaluation on MTEB, AdvBench-IR, BRIGHT, plus analysis scripts (logit lens, latent lens, and generations).


<p align="center">
  <img src="./docs/assets/llm2vecgen.gif" width="95%" alt="llm2vecgen_main_figure"/>
</p>

## Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Response Generation](#-response-generation)
- [Evaluation](#-evaluation)
- [Analysis Scripts](#-analysis-scripts)
- [Project Structure](#-project-structure)


## 📦 Installation
Either use Pypi or clone the repository and install in editable mode:

```bash
pip install llm2vec-gen  # or pip install -e .
```


## 🚀 Quick Start

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
Other examples to try LLM2Vec-Gen in other tasks:
<details>
<summary> <strong>Sentence Similarity </strong> </summary>

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
tensor([[0.8438, 0.2461],
        [0.2334, 0.9023]])
"""
```
</details>
<details>
<summary> <strong>Classification </strong> </summary>

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
tensor([[0.8242, 0.7422],
        [0.7227, 0.8242]])
"""
```
</details>
<details>
<summary> <strong>Clustering </strong> </summary>

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
**Disk Cleanup**" is a built-in utility in Windows that helps you **free up disk space** by **removing unnecessary files** and **temporary data** that are no longer needed. [...]"""
"""
```
This code snippet will return the answer of the LLM2Vec-Gen model generated from the generative embeddings of the input (`recon_hidden_states`).


## 🏋️ Training

Training is driven by **Hydra** configs under `conf/`.

```bash
# LLM2Vec-Gen training for Qwen3-4B
python scripts/train.py \
  training=llm2vec-gen \
  training.per_device_train_batch_size=32 \
  training.learning_rate=3e-4 \
  model=qwen3-4 \
  data=qwen3-4 \
  special_tokens=total_10 \
  run.wandb_run_id=qwen3-4
```

**Useful overrides:**

| Override | Description |
|----------|-------------|
| `model=qwen3-4`, `qwen3-8`, `qwen3-0.6`, … | Different model sizes or variants (see `conf/model/`) |
| `data=qwen3-4`, `tulu`, … | Training data generated by Qwen3-4B or original Tulu QA pairs (see `conf/data/`) |
| `special_tokens=total_10`, … | Special token set (see `conf/special_tokens/`) |
| `run.wandb_run_id=...` | Run name used for logging and saving |

Outputs are written under `outputs/`. You may find all scripts used for training the models in the paper in `train_scripts/` directory. Make sure to identify the correct cuda visible devices in each run. WandB logs are accessible from [this WandB project](https://wandb.ai/siva-reddy-mila-org/llm2vec-gen).


## 💬 Response Generation

Generate model responses for a dataset (e.g. for creating training data). Supports sharding for parallel runs.

```bash
# Tulu dataset, Qwen3-4B, 4 shards
python scripts/response_generation.py \
  --dataset_name tulu \
  --dataset_path mcgill-nlp/llm2vec-gen-tulu \
  --model_name Qwen/Qwen3-4B \
  --max_new_tokens 500 \
  --max_length 1024 \
  --batch_size 8 \
  --num_shards 4 \
  --shard_id 0 \
  --output_dir data
```
Run with `shard_id=0,1,2,3` to process the full dataset in parallel. This repository supports training with data from HF. Use `scripts/upload_responses_to_hf.py` to upload the new generations to a HF dataset with a format like in [our data](https://huggingface.co/datasets/McGill-NLP/llm2vec-gen-tulu).


## 📊 Evaluation

### MTEBv2

Run the **lite** task set (faster) or full MTEB:

```bash
python scripts/mteb_eval.py \
  --model_path McGill-NLP/LLM2Vec-Gen-Qwen3-4B \
  --output_dir outputs/LLM2Vec-Gen-Qwen3-4B \
  --task_set lite  # or `all`
```

### AdvBench-IR

```bash
python scripts/advbenchir_eval.py \
  --model_path McGill-NLP/LLM2Vec-Gen-Qwen3-06B \
  --output_dir outputs/LLM2Vec-Gen-Qwen3-06B
```

### BRIGHT

```bash
python scripts/bright_eval/run.py \
  --task biology \
  --model McGill-NLP/LLM2Vec-Gen-Qwen3-06B \
  --output_dir outputs/LLM2Vec-Gen-Qwen3-06B/bright_eval
```
Other tasks: `earth_science`, `economics`, `pony`, `psychology`, `robotics`, `stackoverflow`, `sustainable_living`, `aops`, `leetcode`, `theoremqa_theorems`, `theoremqa_questions`.

---

## 🔬 Analysis Scripts

### Logit lens

```bash
python scripts/logit_lens_analysis.py \
  --encoder_model_path McGill-NLP/LLM2Vec-Gen-Qwen3-8B \
  --output_dir outputs/hf_model/LLM2Vec-Gen-Qwen3-8B \
  --dataset generation  # or `nq` / `advbenchir`
```

### Latent lens

Requires a prebuilt index. Example:

```bash
# to build the index with Qwen3-8B
python scripts/latent_lens_build_index.py

python scripts/latent_lens_analyze_special_tokens.py \
  --checkpoint McGill-NLP/LLM2Vec-Gen-Qwen3-8B \
  --index /path/to/qwen3-8b_index \
  --top_k 1 \
  --layers 36
```

### Generations analysis

```bash
python scripts/generations_analysis.py \
  --encoder_model_path McGill-NLP/LLM2Vec-Gen-Qwen3-8B \
  --output_dir outputs/LLM2Vec-Gen-Qwen3-8B \
  --dataset generation  # or `nq` / `advbenchir`
```

---

## 📁 Project Structure

```
llm2vec-gen/
├── conf/                    # Hydra configs
│   ├── config.yaml          # Main config
│   ├── model/               # Model presets (qwen3-4, qwen3-8, …)
│   ├── data/                # Dataset presets
│   ├── training/            # Training presets (llm2vec-gen, …)
│   ├── special_tokens/      # Special token configs
│   └── run/                 # Run / wandb settings
├── llm2vec_gen/             # Package
│   ├── models/              # LLM2VecGenModel, encoder-decoder, etc.
│   ├── dataset/             # Data loading and collation
│   └─ trainer/              # Custom trainer
├── train_scripts/           # Different training bash scripts used to train main models and ablations of the paper
└── scripts/
    ├── train.py            
    ├── response_generation.py
    ├── mteb_eval.py
    ├── advbenchir_eval.py
    ├── bright_eval/         
    ├── logit_lens_analysis.py
    ├── latent_lens_analyze_special_tokens.py
    └── generations_analysis.py
```


## Citation

If you use this code, models, or data, please cite the LLM2Vec-Gen paper.

```
@article{behnamghader2026llm2vecgen,
  title={LLM2Vec-Gen: Generative Embeddings from Large Language Models},
  author={BehnamGhader, Parishad and Adlakha, Vaibhav and Schmidt, Fabian David and Chapados, Nicolas and Mosbach, Marius and Reddy, Siva},
  journal={arXiv preprint: arXiv:2603.10913},
  year={2026},
  url={https://arxiv.org/abs/2603.10913}
}
```
