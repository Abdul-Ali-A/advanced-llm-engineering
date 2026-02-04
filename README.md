# 🧠 Advanced LLM Engineering

[![Status](https://img.shields.io/badge/Status-Research%20Sabbatical-blueviolet)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Framework](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)]()
[![Library](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)]()

> **"I didn't just want to drive the car; I wanted to build the engine."**

---

## 📌 Project Context
**Timeline:** December 2025 – February 2026 (8 Weeks)

After building application-layer projects for 7 weeks, I dedicated **8 weeks** to a rigorous study of the underlying mechanics of Large Language Models.

My goal was to move beyond "API wrapping" and master the low-level engineering required to train, optimize, and deploy Transformers. This repository contains the implementations of core algorithms (BPE, WordPiece, Self-Attention) built from scratch, along with custom PyTorch training loops and production-grade inference pipelines.

---

## 📂 Repository Structure

```text
advanced-llm-engineering/
├── 01-inference-and-production/   # High-performance serving (vLLM, TGI)
├── 02-transformer-basics/         # Core architecture labs
├── 03-fine-tuning-workflows/      # Custom PyTorch training loops
├── 04-advanced-data-engineering/  # Streaming TB-scale datasets
├── 05-nlp-tasks-pipelines/        # NER & QA Pipeline implementations
├── 06-tokenizer-mechanics/        # Algorithms (BPE, WordPiece, Unigram) from scratch
└── 07-experimental/               # Just some Testing
