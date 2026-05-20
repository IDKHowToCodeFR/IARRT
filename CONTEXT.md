# IARRT: Technical Context & Implementation Details

This document outlines the "under-the-hood" technical decisions and configurations that drive the IARRT Research Suite.

## 🧠 Model Specifications
- **Base Translator**: `mBART-50` (Large). Multilingual seq2seq model chosen for its robust German-to-English foundation.
- **Detector**: `BERT-Base-German-Cased`. Fine-tuned on synthetic IOB-tagged data for specific idiom span detection.
- **Disambiguator**: `mDeBERTa-v3`. Utilized for its superior zero-shot performance in NLI (Natural Language Inference) tasks to distinguish literal vs. figurative intent.
- **Embedder**: `all-MiniLM-L6-v2`. A lightweight transformer that provides high-quality semantic vectors for the FAISS retrieval index.

## 🛠️ Optimization & Memory Management
- **4-bit Quantization**: Implemented via `bitsandbytes` (NF4) to allow `mBART-50` to run on consumer-grade GPUs with <8GB VRAM.
- **LoRA (Low-Rank Adaptation)**: Rank `r=16`, Alpha `32`. Only `q_proj` and `v_proj` layers are trained to keep the memory footprint low and prevent catastrophic forgetting.
- **FAISS Vector Search**: Uses `IndexFlatIP` (Inner Product) for cosine similarity retrieval over the 60+ idiom knowledge base.

## ⚙️ Decision Logic (The "Invisible" Gate)
- **Confidence Threshold**: $T = 0.5$. The system only performs a "Post-MT Injection" if the combined score (Retrieval Similarity × Detection Density) exceeds this threshold.
- **Smart-Swap Heuristic**: If a literal keyword (e.g., "water", "cat") is detected in the baseline and an idiomatic meaning is retrieved, the system performs a surgical regex-based substitution rather than a full re-generation. This preserves the grammatical integrity of the base model.

## 📈 Metric Rationale
- **chrF++**: Chosen because BLEU often penalizes valid idiomatic variations; character n-grams are more sensitive to semantic shifts in morphologically rich languages.
- **LitTER (Literal Translation Error Rate)**: A custom heuristic that counts occurrences of German-literal words in English output. **Goal: Minimize this value.**
