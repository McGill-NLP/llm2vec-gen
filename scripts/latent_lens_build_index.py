import latentlens
from datasets import load_dataset
import json

tulu = load_dataset("McGill-NLP/llm2vec-gen-tulu", split="Qwen3_8B")
tulu = tulu.shuffle()
tulu = tulu.select(range(150000))
test_samples = json.load(open("scripts/latentlens_qwen3_8b_generations.json"))
test_samples = [sample["response"] for sample in test_samples]
corpus = list(tulu["answer"] + test_samples)
index = latentlens.build_index("Qwen/Qwen3-8B", corpus=corpus)
index.save("qwen3-8b_index/")
