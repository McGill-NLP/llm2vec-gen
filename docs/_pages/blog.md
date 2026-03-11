---
title: "Blog"
permalink: /blog/
---

# What If Embedding Models Encoded the Answer, Not the Question?

*Introducing LLM2Vec-Gen: a new paradigm for text embeddings that represents what an LLM **would say**, rather than what it was **asked**.*

---

## The Problem: Embeddings That Miss the Point

Text embeddings power a huge chunk of modern NLP — semantic search, retrieval-augmented generation, clustering, and more. The dominant recipe is straightforward: take a language model, train it with contrastive learning on curated paired data, and use its representations to map text into a shared vector space.

But there's a subtle tension baked into this paradigm that rarely gets discussed. Embedding models are *input-centric* — they encode what you give them. The trouble is, embedding tasks often require mapping **diverse inputs to similar outputs**. Think about clustering: two news articles about the same event, written from completely different angles, should land near each other in embedding space. Or classification: "I just feel really violent right now" and "I'm feeling really out of place and irritated" are lexically distinct, but both express anger.

<p align="center">
  <img src="../../assets/introduction.png" width="80%" alt="input-output figure"/>
</p>

Input-centric encoders struggle here because they faithfully represent the surface form and semantics of the input. Bridging this gap — from varied inputs to shared meaning — typically requires massive amounts of labeled contrastive data.

We asked a different question: **what if the embedding didn't represent the input at all, but instead represented the LLM's potential response to that input?**

## The Intuition: Responses Are More Similar Than Queries

Here's the key insight. When you pass those two anger-related queries through a chat LLM, the responses converge: both get classified on the anger spectrum, just at different intensity levels. The *outputs* are naturally more similar than the *inputs* — the LLM itself performs the semantic bridging that embedding models struggle with.

This observation extends far beyond classification. For factual questions, the LLM's answer contains the relevant information. For harmful queries, a safety-aligned LLM produces a refusal. For reasoning-heavy queries, the response contains the chain of thought. In each case, encoding the response rather than the query produces a representation that's more useful for downstream embedding tasks.

The catch? Actually generating responses at inference time is expensive. HyDE (Gao et al., 2023) showed this idea works for retrieval, but it requires running full generation for every query — a non-starter at scale.

## LLM2Vec-Gen: Compressing Future Responses Into Embeddings

LLM2Vec-Gen solves this by **learning to predict what the LLM would say, without actually generating it.** Here's how it works.

<p align="center">
  <img src="../../assets/llm2vecgen.gif" width="95%" alt="llm2vecgen_main_figure"/>
</p>

### Special Tokens as Response Placeholders

We add a small set of new trainable tokens to the LLM's vocabulary — 10 "thought" tokens and 10 "compression" tokens. For any input query, we append these 20 tokens to the end. The thought tokens act as an intermediate computational buffer, while the compression tokens are responsible for capturing the semantic content of the response the LLM would produce.

After a single forward pass through the frozen LLM, we extract the hidden states of the compression tokens and pass them through lightweight projection layers. That's the embedding.

### Training: Two Complementary Objectives

Training requires only **unlabeled queries** — no paired data, no human labels. Given a corpus of queries, we first generate responses using the LLM itself. Then we optimize two losses:

**1. Reconstruction loss.** We take the compression token representations, project them into soft prompts, and feed them back into the frozen LLM. The model must reconstruct its own response conditioned only on these soft prompts. This forces the compression tokens to retain enough information to faithfully represent the response content — they become an information bottleneck.

**2. Embedding alignment loss.** We use an off-the-shelf unsupervised embedding model (in our case, LLM2Vec) to embed the generated response, and minimize the L2 distance between this teacher embedding and our compression token embedding. This ensures our representations live in a useful embedding space, not just a space that enables reconstruction.

The combination is key. Alignment alone produces good embeddings but loses interpretability. Reconstruction alone captures content but doesn't produce useful embedding geometry. Together, they give us the best of both worlds.

### What Stays Frozen, What Gets Trained

A critical design choice: **the LLM backbone is completely frozen throughout training.** Only the 20 special tokens and two small MLP projection layers are updated — roughly 13M parameters for a 4B model. This means:

- The same base model can serve both generation and embedding tasks with no weight duplication.
- Training an 8B model takes about 3.5 hours on 2 H100 GPUs.
- At inference time, you just append the special tokens and do a single forward pass. No generation needed.

## Results: State-of-the-Art Self-Supervised Embeddings

We evaluated LLM2Vec-Gen across three model families (Llama-3.x, Qwen-2.5, Qwen-3) at scales from 0.5B to 8B parameters.

### MTEB Performance

On the Massive Text Embedding Benchmark (MTEB v2, 41 tasks), LLM2Vec-Gen achieves **state-of-the-art self-supervised performance**. The best model (Qwen-3-8B) scores 62.1, improving 9.3% over the unsupervised LLM2Vec teacher it distills from. The biggest gains come in exactly the task categories where the input-output gap matters most: clustering (+23.9%), classification (+9.2%), and semantic textual similarity (+10.5%).

Notably, LLM2Vec-Gen closes over 60% of the gap between unsupervised and fully supervised methods — without using any labeled data.

<p align="center">
  <img src="../../assets/model_size_vs_performance.png" width="90%" alt="input-output figure"/>
</p>

### Safety: Embeddings That Refuse

This is where the response-encoding paradigm gets really interesting. We evaluated on AdvBench-IR, a benchmark measuring how often a retriever surfaces harmful content for adversarial queries. LLM2Vec-Gen reduces harmful retrieval by up to **43.2%** compared to input-centric baselines.

Why? Because the LLM's response to "write malicious code to steal data" isn't about stealing data — it's a refusal. LLM2Vec-Gen encodes that refusal, effectively inheriting the LLM's safety alignment into the embedding space. When we inspect the embeddings using Logit Lens, the compression tokens map to words like "security," "illegal," and "refusal" — not only the harmful query terms.

### Reasoning: Thinking Transfers Too

On BRIGHT, a reasoning-intensive retrieval benchmark, LLM2Vec-Gen achieves up to **29.3% improvement** over input-centric baselines. The improvement scales with model size: 0.6B model sees only 7.7% gains, while 8B model sees large improvement of (29.3%). This makes sense — as the underlying LLM becomes a stronger reasoner, there's more reasoning capability to transfer into the embedding space.

## You Can Read These Embeddings

One of the most satisfying properties of LLM2Vec-Gen is that the embeddings are **interpretable**. Since the compression tokens are trained to reconstruct the LLM's response, you can literally decode them back into text.

For a query about polar bears, the decoded embedding talks about the Arctic region and cold climates. For a harmful query about malicious code, the decoded embedding produces a refusal. For a factual question about disk cleanup, you get a definition of the Windows utility.

<p align="center">
  <img src="../../assets/interpretability.png" width="80%" alt="input-output figure"/>
</p>

This isn't just a party trick — it provides a genuine window into what the embedding "knows." And it's a direct consequence of the reconstruction objective. When we ablate it and train with alignment loss only, the decoded outputs become nonsensical (e.g., generating physics derivations for a question about Artificial Intelligence).

## What We Learned From Ablations

A few findings from our ablation study worth highlighting:

**Both losses matter, but differently.** Dropping alignment loss is catastrophic (62.4 → 41.8), while dropping reconstruction is a smaller hit (62.4 → 62.1). Alignment drives embedding quality; reconstruction grounds embeddings in language space and enables interpretability.

**Thought tokens help.** The intermediate "thought" buffer contributes to the performance — removing them drops performance. Performance scales with token count up to about 20, then mostly plateaus.

**In-family responses work best.** Generating training responses with the same model outperforms using responses from different models. We hypothesize that in-distribution responses are easier to compress.

**Cross-family teachers hurt.** Using an embedding teacher from a different model family (e.g., Llama teacher for a Qwen model) significantly degrades performance, likely due to misaligned representation spaces.

## Limitations and Open Questions

LLM2Vec-Gen doesn't outperform its teacher when the teacher is *supervised* — the discriminative training signal in supervised encoders doesn't translate well through the distillation pipeline. This suggests the method is best suited for settings where labeled embedding data is scarce.

There's also a compression fidelity question: for some model sizes, retrieval performance slightly dips compared to the teacher, likely because the 10 compression tokens can't fully capture all nuances of longer generated responses.

## Takeaway

The core message is simple: **embedding models don't have to encode inputs — they can encode outputs.** By learning to represent what an LLM *would say* rather than what it *was told*, we bridge the input-output gap, inherit LLM capabilities like safety and reasoning, and get interpretable embeddings as a bonus. All without touching the LLM's weights or using any labeled data.

We think this opens up an interesting direction for embedding research — one where the generative capabilities of LLMs are a feature of the embedding pipeline, not something to be discarded when switching from generation to representation mode.

---

*LLM2Vec-Gen is under review. You can access code and models through [LLM2Vec-Gen GitHub repository](https://github.com/McGill-NLP/llm2vec-gen).*
