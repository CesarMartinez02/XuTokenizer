"""Unit tests for Llama-2 style Tokenizer Builder and Trainer."""

import json
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TokenizerConfig
from src.tokenizer_builder import (
    LlamaTokenizerTrainer,
    build_llama2_pipeline,
    save_tokenizer_artifacts,
)


@pytest.fixture
def sample_bilingual_corpus():
    """Provides a small balanced bilingual corpus for training tests."""
    return [
        "¡Hola! El rápido zorro marrón salta sobre el perro perezoso en 1998.",
        "Hello! The quick brown fox jumps over the lazy dog in 1998.",
        "¿Cómo estás? La inteligencia artificial bilingüe procesa texto en español e inglés.",
        "How are you? Bilingual artificial intelligence processes text in Spanish and English.",
        "Los números 1, 2, 3, 4 y 5 deben estar divididos en dígitos individuales.",
        "Numbers 1, 2, 3, 4 and 5 must be separated into individual single digits.",
        "Prueba con acentos: á, é, í, ó, ú, ñ, ü y símbolos matemáticos: 2 + 2 = 4.",
        "Testing symbols and currencies: €100, ¥500, copyright © 2026, and newlines \n\n with tabs.",
    ]


def test_build_pipeline_structure():
    """Verify that build_llama2_pipeline configures the expected pre_tokenizer and decoder."""
    config = TokenizerConfig()
    tokenizer = build_llama2_pipeline(config)

    assert tokenizer is not None
    assert tokenizer.pre_tokenizer is not None
    assert tokenizer.decoder is not None
    assert tokenizer.post_processor is not None


def test_add_prefix_space_toggle():
    """Verify config.add_prefix_space correctly switches Metaspace prepend_scheme."""
    config_true = TokenizerConfig(add_prefix_space=True)
    tok_true = build_llama2_pipeline(config_true)
    splits_true = tok_true.pre_tokenizer.pre_tokenize_str("palabra")
    assert splits_true[0][0].startswith("▁")

    config_false = TokenizerConfig(add_prefix_space=False)
    tok_false = build_llama2_pipeline(config_false)
    splits_false = tok_false.pre_tokenizer.pre_tokenize_str("palabra")
    assert not splits_false[0][0].startswith("▁")
    assert splits_false[0][0] == "palabra"


def test_train_from_iterator_and_id_reservations(sample_bilingual_corpus):
    """Verify training sets up reserved tokens at exact pre-allocated IDs."""
    config = TokenizerConfig(vocab_size=350, min_frequency=1)
    trainer = LlamaTokenizerTrainer(config=config)

    tokenizer = trainer.train_from_iterator(sample_bilingual_corpus, save_artifacts=False)

    # Check Base special token IDs
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2
    assert tokenizer.token_to_id("<pad>") == 3

    # Check Chat token IDs (4..9)
    assert tokenizer.token_to_id("<|system|>") == 4
    assert tokenizer.token_to_id("<|user|>") == 5
    assert tokenizer.token_to_id("<|assistant|>") == 6
    assert tokenizer.token_to_id("[INST]") == 7
    assert tokenizer.token_to_id("[/INST]") == 8
    assert tokenizer.token_to_id("<|end|>") == 9

    # Check Byte-fallback tokens start at 10 and end at 265
    assert tokenizer.token_to_id("<0x00>") == 10
    assert tokenizer.token_to_id("<0xFF>") == 265


def test_bos_token_prepend(sample_bilingual_corpus):
    """Verify that post-processing automatically prepends <s> (ID 1) to encodings."""
    config = TokenizerConfig(vocab_size=350, min_frequency=1)
    trainer = LlamaTokenizerTrainer(config=config)
    tokenizer = trainer.train_from_iterator(sample_bilingual_corpus)

    encoding = tokenizer.encode("Hola mundo")
    assert encoding.ids[0] == 1
    assert encoding.tokens[0] == "<s>"


def test_lossless_roundtrip_decoding(sample_bilingual_corpus):
    """Verify decode(encode(text)) restores text without loss across ES/EN, accents, digits, and symbols."""
    config = TokenizerConfig(vocab_size=350, min_frequency=1)
    trainer = LlamaTokenizerTrainer(config=config)
    tokenizer = trainer.train_from_iterator(sample_bilingual_corpus)

    test_cases = [
        "¡Hola! ¿Cómo estás?",
        "Text with ñ and accents: cañón, lingüística, murciélago.",
        "Number arithmetic 2026 and 9876543210.",
        "Multiple   spaces   and\nnewlines\n\nare preserved.",
        "Symbol and currency preservation: €100, ¥500, ©, ®, and Greek: α, β, γ.",
    ]

    for original in test_cases:
        encoding = tokenizer.encode(original)
        # Decoding with skip_special_tokens=True strips the BOS <s> but keeps text and symbols
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        assert decoded == original, f"Mismatch for '{original}': got '{decoded}'"


def test_digits_are_isolated(sample_bilingual_corpus):
    """Verify split_digits=True keeps numbers strictly separated into single digit tokens."""
    config = TokenizerConfig(vocab_size=350, min_frequency=1)
    trainer = LlamaTokenizerTrainer(config=config)
    tokenizer = trainer.train_from_iterator(sample_bilingual_corpus)

    encoding = tokenizer.encode("1998")
    # Discard BOS (first token) and leading whitespace/metaspace token if present
    raw_tokens = encoding.tokens[1:]
    digit_tokens = [t.lstrip("▁") for t in raw_tokens if t != "▁"]

    expected_digits = ["1", "9", "9", "8"]
    assert len(digit_tokens) == len(expected_digits), (
        f"Expected exactly {len(expected_digits)} single-digit tokens, but got {digit_tokens}"
    )

    for actual, expected in zip(digit_tokens, expected_digits):
        assert actual == expected or actual == f"<0x{ord(expected):02X}>", (
            f"Expected digit '{expected}', got '{actual}'"
        )


def test_artifact_saving(sample_bilingual_corpus, tmp_path: Path):
    """Verify that artifacts are serialized properly in standard HF fast tokenizer format."""
    config = TokenizerConfig(vocab_size=350, min_frequency=1, output_dir=tmp_path)
    trainer = LlamaTokenizerTrainer(config=config)

    tokenizer = trainer.train_from_iterator(
        sample_bilingual_corpus,
        save_artifacts=True,
        output_dir=tmp_path,
    )

    tokenizer_json = tmp_path / "tokenizer.json"
    named_json = tmp_path / f"{config.model_name}.json"
    config_json = tmp_path / "tokenizer_config.json"
    special_map_json = tmp_path / "special_tokens_map.json"

    assert tokenizer_json.exists()
    assert named_json.exists()
    assert config_json.exists()
    assert special_map_json.exists()

    # Validate json structure and aligned chat template
    with open(config_json, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
        assert cfg_data["bos_token"] == "<s>"
        assert cfg_data["tokenizer_class"] == "LlamaTokenizerFast"
        # Verify chat template uses aligned reserved tokens (<|system|>, <|user|>, <|assistant|>, <|end|>)
        chat_tmpl = cfg_data["chat_template"]
        assert "<|system|>" in chat_tmpl
        assert "<|user|>" in chat_tmpl
        assert "<|assistant|>" in chat_tmpl
        assert "<|end|>" in chat_tmpl

    with open(special_map_json, "r", encoding="utf-8") as f:
        spec_data = json.load(f)
        assert spec_data["eos_token"] == "</s>"
        assert "[INST]" in spec_data["additional_special_tokens"]
        assert "<|system|>" in spec_data["additional_special_tokens"]
