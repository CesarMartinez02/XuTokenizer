"""Unit tests for tokenizer configuration and reserved token layout."""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TokenizerConfig, SpecialTokensConfig, DEFAULT_CONFIG


def test_special_tokens_base_ids():
    """Verify Llama-2 standard IDs: unk=0, bos=1, eos=2, pad=3."""
    spec = DEFAULT_CONFIG.special_tokens
    assert spec.unk_id == 0
    assert spec.bos_id == 1
    assert spec.eos_id == 2
    assert spec.pad_id == 3

    assert spec.id_to_token[0] == "<unk>"
    assert spec.id_to_token[1] == "<s>"
    assert spec.id_to_token[2] == "</s>"
    assert spec.id_to_token[3] == "<pad>"


def test_chat_tokens_reservation():
    """Verify chat/instruct tokens occupy contiguous IDs right after base tokens."""
    spec = DEFAULT_CONFIG.special_tokens
    expected_chat = ["<|system|>", "<|user|>", "<|assistant|>", "[INST]", "[/INST]", "<|end|>"]
    for idx, token in enumerate(expected_chat, start=4):
        assert spec.id_to_token[idx] == token
        assert spec.token_to_id[token] == idx


def test_byte_fallback_tokens():
    """Verify exactly 256 byte fallback tokens (<0x00> .. <0xFF>) starting at ID 10."""
    spec = DEFAULT_CONFIG.special_tokens
    assert len(spec.byte_tokens) == 256
    assert spec.byte_tokens[0] == "<0x00>"
    assert spec.byte_tokens[255] == "<0xFF>"
    assert spec.id_to_token[10] == "<0x00>"
    assert spec.id_to_token[265] == "<0xFF>"


def test_total_reserved_count():
    """Verify total reserved tokens: 4 base + 6 chat + 256 byte fallback = 266."""
    spec = DEFAULT_CONFIG.special_tokens
    assert spec.num_reserved_tokens == 266
    assert len(spec.token_to_id) == 266


def test_vocab_size_and_merges_calculation():
    """Verify BPE merges are calculated accurately relative to 32k total vocab."""
    config = TokenizerConfig(vocab_size=32_000)
    assert config.vocab_size == 32_000
    assert config.num_bpe_merges == 32_000 - 266
    assert config.num_bpe_merges == 31_734


def test_invalid_vocab_size_raises():
    """Ensure configuring a vocab_size smaller than reserved tokens raises ValueError."""
    with pytest.raises(ValueError, match="vocab_size .* must be greater"):
        TokenizerConfig(vocab_size=200)
