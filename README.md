# XuTokenizer: Llama-2 Style Bilingual BPE Tokenizer (ES / EN)

A production-grade Byte-Pair Encoding (BPE) tokenizer engineered with the **Hugging Face `tokenizers` Rust backend**, specifically optimized for high compression and low fertility across both **Spanish and English**.

---

## Architecture & Design Specifications

- **Engine:** Rust-backed BPE (`tokenizers`) for high throughput and zero Python-loop bottlenecks.
- **Vocabulary Size:** `32,000` tokens.
- **Byte-Fallback:** 256 byte tokens (`<0x00>` to `<0xFF>`) ensuring 100% character coverage and zero out-of-vocabulary (`<unk>`) issues.
- **Prefix / Metaspace:** SentencePiece-compatible metaspace delimiter ` ` (`\u2581`).
- **Arithmetic Preservation:** `split_digits = True` (isolates individual digits `0-9`).
- **Special Tokens Reservation:**
  - `0`: `<unk>`
  - `1`: `<s>` (BOS)
  - `2`: `</s>` (EOS)
  - `3`: `<pad>`
  - `4..9`: `<|system|>`, `<|user|>`, `<|assistant|>`, `[INST]`, `[/INST]`, `<|end|>`
  - `10..265`: `<0x00>` .. `<0xFF>` (256 Byte-Fallback tokens)
  - `266..31999`: Learned BPE merges (31,734 merges)

---

## Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── config.py            # Master tokenizer, streaming & special token configurations
│   ├── data_stream.py       # Bilingual 50/50 streaming pipeline & batch generator
│   ├── evaluator.py         # Evaluation metrics (fertility, bytes/token, throughput)
│   ├── llama_tokenizer.py   # High-level PyTorch-ready LLM wrapper (batching, pad, mask, chat)
│   └── tokenizer_builder.py # Rust BPE assembler, trainer, byte-fallback normalizer & exporter
├── scripts/
│   ├── train_tokenizer.py   # CLI training entrypoint with HF streaming & local corpora support
│   └── benchmark.py         # CLI evaluation and side-by-side benchmark runner
├── tests/
│   ├── test_config.py       # Comprehensive unit tests for ID mappings and constraints
│   ├── test_data_stream.py  # Unit tests for streaming, batching, and text filtering
│   ├── test_builder.py      # Unit tests for BPE pipeline, byte-fallback, and roundtrip decoding
│   ├── test_wrapper.py      # Unit tests for batch encoding, padding, attention masks, and chat
│   └── test_evaluator.py    # Unit tests for fertility, compression, zero-UNK, and speed
├── requirements.txt         # Core dependencies
└── README.md
```

---

## Quickstart & Testing

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Train Tokenizer via CLI
```bash
python scripts/train_tokenizer.py --vocab-size 32000 --samples 50000
```

### 4. Run Benchmark & Evaluation Suite
```bash
python scripts/benchmark.py --tokenizer-path artifacts/
```

Optional side-by-side comparison with a reference tokenizer:
```bash
python scripts/benchmark.py --tokenizer-path artifacts/ --compare-with /path/to/reference/tokenizer
```

### 5. Basic Python Usage
```python
from src.llama_tokenizer import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("artifacts/")

# Single encode
ids = tokenizer.encode("Hola mundo")

# Batch encoding with left-padding for generation
batch = tokenizer.encode_batch(
    ["Texto corto", "Un texto mucho más largo para evaluar"],
    padding=True,
    padding_side="left",
)
print("Input IDs:", batch.input_ids)
print("Attention Mask:", batch.attention_mask)
```

---
