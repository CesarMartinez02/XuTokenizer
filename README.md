# XuTokenizer: Llama-2 Style Bilingual BPE Tokenizer (ES / EN)

A production-grade Byte-Pair Encoding (BPE) tokenizer engineered with the **Hugging Face `tokenizers` Rust backend**, specifically optimized for high compression and low fertility across both **Spanish and English**.

---

## 📌 Architecture & Design Specifications

- **Engine:** Rust-backed BPE (`tokenizers`) for high throughput and zero Python-loop bottlenecks.
- **Vocabulary Size:** `32,000` tokens.
- **Byte-Fallback:** 256 byte tokens (`<0x00>` to `<0xFF>`) ensuring 100% character coverage and zero out-of-vocabulary (`<unk>`) issues.
- **Prefix / Metaspace:** SentencePiece-compatible metaspace delimiter ` ` (`\u2581`).
- **Arithemtic Preservation:** `split_digits = True` (isolates individual digits `0-9`).
- **Special Tokens Reservation:**
  - `0`: `<unk>`
  - `1`: `<s>` (BOS)
  - `2`: `</s>` (EOS)
  - `3`: `<pad>`
  - `4..9`: `<|system|>`, `<|user|>`, `<|assistant|>`, `[INST]`, `[/INST]`, `<|end|>`
  - `10..265`: `<0x00>` .. `<0xFF>` (256 Byte-Fallback tokens)
  - `266..31999`: Learned BPE merges (31,734 merges)

---

## 📁 Repository Structure

```text
├── src/
│   ├── __init__.py
│   └── config.py            # Master tokenizer, streaming & special token configurations
├── tests/
│   └── test_config.py       # Comprehensive unit tests for ID mappings and constraints
├── requirements.txt         # Core dependencies
└── README.md
```

---

## 🚀 Quickstart & Testing

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

---

## 🗺️ Roadmap
- [x] **Phase 1: Environment & Architecture Configuration** (Special tokens, byte-fallback layout, Llama-2 parameters)
- [ ] **Phase 2: Bilingual Streaming Data Pipeline** (Balanced 50/50 ES/EN dataset stream)
- [ ] **Phase 3: Rust-Backed Tokenizer Builder & Trainer** (Metaspace normalizer, digit pre-tokenizer, BPE trainer)
- [ ] **Phase 4: LLM Wrapper** (FastTokenizer, batch encoding, padding, attention masks, chat templates)
- [ ] **Phase 5: Evaluation & Benchmark Suite** (Fertility ratio, bytes/token compression, speed)
