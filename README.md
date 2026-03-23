<a name="readme-top"></a>

# When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration

## 📚 Citation



```bibtex
@article{li2026when,
  title={When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration},
  author={Li, Yiping and An, Zhiyu and Du, Wan},
  year={2026}
}
```
**Links:** `Paper (arXiv): coming soon`


This repository studies **KV-cache compression for latent-state communication in multi-agent LLM systems**. In **LatentMAS**, agents relay **transformer KV caches** instead of natural-language messages, which preserves richer internal state but makes inter-agent communication memory- and bandwidth-heavy. We adapt eviction-style KV compression to this relay setting and introduce **OrthoBackfill (OBF)**, a lightweight value-only correction that reinjects the orthogonal residual lost under hard eviction.

The central result is that **better relay does not necessarily require more KV**. Under a fixed prompt-state relay budget, compressed variants can match or outperform full KV relay, and OBF-enhanced methods deliver the strongest overall results across most benchmarks.

**Current support:** the maintained execution path is `--method latent_mas` with `--prompt sequential`. `baseline` and `text_mas` remain in the [methods](./methods) folder for reference, but are not currently supported.

## 🌐 Overview

Recent MAS work is moving from **text messages** to **continuous internal-state communication**. That shift improves fidelity, but it also makes communication substantially more expensive once agents begin passing full KV caches instead of decoded tokens.

LatentMAS sits at the high-fidelity end of this spectrum. Our goal is to make that communication practical: compress the relayed KV cache aggressively enough to reduce cost, while still preserving the latent information that downstream agents need in order to continue reasoning.

![Communication mediums in MAS](assets/mas_communication.png)

This repository focuses on three questions:

- Which prompt KV states should be relayed to the next agent?
- How should single-agent eviction rules be adapted to multi-agent latent relay?
- What information is lost under hard eviction, and how can that loss be compensated?

## 📊 Main Results

Under a fixed relay budget of **32 prompt states per agent round**, compressed relay is not merely competitive with full KV relay. In our experiments, **at least one compressed method outperforms full relay on all 9 benchmarks**, and **OBF variants achieve the best result on 7 of 9 benchmarks**.

![Main results table](assets/table_1.png)

Key takeaways:

- Relay quality depends on **which KV states are kept**, not only on how much KV is transmitted.
- Eviction can act as a useful form of **denoising**, rather than only as an approximation.
- OBF often makes sparsified relay **more usable for downstream continuation**, not just more complete.

## 🧠 Method Overview

### 1. From full relay to selective relay

In LatentMAS, each agent inherits the previous agent's KV message and appends its own local states. That means relay cost grows with interaction depth. We reformulate standard KV eviction for this setting by separating the cache into:

- a shared **attention sink**
- **inherited message history**
- the current agent's **prompt context**
- the current agent's **latent reasoning states**

The compression step acts on the current agent's local prompt states, while inherited relay memory is passed forward.

![KV decomposition](assets/kv_decomposition.png)

### 2. Eviction baselines for LatentMAS

We evaluate two adapted relay baselines:

- **MAS-StreamingLLM / Gen**: keep the global sink and newly generated reasoning states.
- **MAS-H2O**: keep prompt states with the highest attention-based importance under layer-wise or head-wise scoring.

These baselines test whether standard long-context eviction ideas remain effective when the KV cache becomes the communication medium between agents.

### 3. OrthoBackfill (OBF)

Hard eviction can remove prompt-side information that is still useful for the next agent, especially when that information is distributed across discarded states instead of tied to a single retained token. OBF addresses this by:

1. splitting prompt values into **kept** and **discarded** sets
2. projecting discarded values onto the span of retained values
3. isolating the **orthogonal residual** that the retained span cannot represent
4. compressing that residual in a low-rank principal subspace
5. injecting the resulting value-only correction back into retained prompt values

OBF is deliberately conservative: it does **not** try to reconstruct everything that was dropped. It only backfills the part of the discarded signal that is not already expressible by the retained-value span.

![OBF geometry](assets/OBF.png)

## 🧪 Supported Tasks

Current `run.py` supports the following benchmarks:

- `gsm8k`
- `aime2024`
- `aime2025`
- `gpqa`
- `arc_easy`
- `arc_challenge`
- `mbppplus`
- `humanevalplus`
- `medqa`

## ⚙️ Setup

Optional cache location:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

Install dependencies:

```bash
conda create -n compressed_latentams python=3.10 -y
conda activate compressed_latentams
pip install -r requirements.txt
```

Optional logging:

```bash
pip install wandb
```

## 🚀 Quick Start

Clone the repository:

```bash
git clone <your-repo-url>
cd CompressedLatentMAS
```

Run full KV relay:

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --compressor full \
  --latent_steps 20 \
  --generate_bs 1 \
  --max_samples 20
```

Run an eviction baseline:

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --compressor layerwise \
  --latent_steps 20 \
  --generate_bs 1 \
  --max_samples 20
```

Run an OBF-enhanced variant:

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --compressor lobf \
  --latent_steps 20 \
  --generate_bs 1 \
  --max_samples 20
```

Important flags:

- `--latent_steps`: latent rollout steps for each non-judger agent
- `--compressor`: KV relay operator used between agent rounds
- `--prompt`: use `sequential` in the currently supported path
- `--generate_bs`: batch size for evaluation

## 🧩 Available Compressors

The current `run.py` path supports the following compressor names:

- `full`
- `gonly`
- `headwise`
- `layerwise`
- `lobf`
- `lobf_metric`
- `lobf_navie`
- `lobf_no_proj`
- `lobf_no_scale`
- `lobf_max_p`
- `hobf`

For practical evaluation, the most relevant path is:

1. `full`
2. `gonly`, `headwise`, `layerwise`
3. `lobf`, `hobf`

## 📝 Output and Logging

Each run writes per-sample JSONL records to `--output_folder`, including:

- prediction and gold label
- correctness
- agent traces
- communication overhead
- runtime metrics
- OBF diagnostic metrics when available

Enable Weights & Biases logging with:

```bash
--use_wandb
```

## 📌 Notes

- This repository is currently focused on `--method latent_mas`.
- The supported experiment path in this fork is the **sequential LatentMAS** setup.
- `baseline` and `text_mas` are present in [methods](/Users/xxivmk/GitHub/LatentMAS/methods) but are currently not supported.
- Some parser options or legacy modules still exist in the tree, but this README documents only the actively supported path.
- Exact reproduction can depend on model version, seed, and compression hyperparameters.

### This Work

BibTeX for this work can be added once the paper metadata is finalized.

## 🙏 Acknowledgement

This codebase builds on LatentMAS and related work on long-context KV eviction, cache compression, and efficient LLM inference.
