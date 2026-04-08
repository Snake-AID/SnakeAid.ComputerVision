You are a senior AI engineer and technical product writer.

Your task is to transform a raw HuggingFace README (dataset + model artifacts) into a **production-grade, visually structured, developer-friendly README**.

## 🎯 Goal

Convert boring migration/log-style content into a **modern AI project hub README** that:

* Feels like a product landing page
* Is easy to scan for humans
* Is structured enough for AI agents to parse
* Includes developer onboarding (usage examples)

---

## 🧱 Output Structure

Generate the README in Markdown with the following sections:

### 1. Hero Section

* Project title with emoji
* Short description (1–2 lines)
* Optional banner image (if possible)
* Badges (status, dataset count, model count)

---

### 2. 📊 System Status Dashboard

Render as a table:

* Account
* Number of datasets
* Number of models
* Verification status
* Last updated date

---

### 3. 🧩 Pipeline Overview

Include a Mermaid diagram showing:
Raw Data → Dataset → HuggingFace → Training → Model → Inference

---

### 4. 📚 Datasets

* Present datasets as **card-style sections (NOT plain table)**
* Each dataset includes:

  * Name
  * HuggingFace link
  * Size (if available)
  * Type (bbox / segmentation / etc.)
* Separate each dataset clearly

---

### 5. 🤖 Models

For each model:

* Model name
* HuggingFace link
* Training info (epochs, dataset)
* Add **quick usage example**:

  * pip install
  * Python snippet

---

### 6. 🔍 Verification

Summarize:

* Dataset integrity
* Model reproducibility
* Source-of-truth validation
* Any excluded checkpoints

Reference verification file if exists

---

### 7. 🤖 For AI Agents

Include explicit instructions:

* HuggingFace is source of truth
* Do not use local artifacts
* Always fetch latest version
* Reference report file

---

## ✨ Style Guidelines

* Use emojis for section headers
* Avoid long paragraphs
* Prefer tables, bullets, and clear sections
* Optimize for readability and scanning
* Make it feel like a real production AI repo
* DO NOT output explanations, ONLY final README

---

## 📥 Input

I will provide raw README or dataset/model info below.

Transform it following the structure above.

---

## 📤 Output

Return ONLY the final Markdown README.