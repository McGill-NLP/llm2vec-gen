import argparse
from llm2vec_gen import LLM2VecGenModel


args = argparse.ArgumentParser()
args.add_argument("--model_id", type=str, default="McGill-NLP/LLM2Vec-Gen-Qwen3-8B")
args = args.parse_args()

model_id = args.model_id

model = LLM2VecGenModel.from_pretrained(model_id)

input_text = "Is Montreal located in Canada?"
enc = model.encode(input_text)
answer, enc_before_answer = model.generate(input_text, max_new_tokens=100, get_align_hidden_states=True)
print(answer)
print(enc_before_answer.equal(enc), enc)