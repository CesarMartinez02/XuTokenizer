"""Unit tests for LlamaTokenizer LLM Wrapper."""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TokenizerConfig
from src.llama_tokenizer import BatchEncoding, LlamaTokenizer
from src.tokenizer_builder import LlamaTokenizerTrainer


@pytest.fixture
def trained_wrapper():
    """Builds and returns a trained LlamaTokenizer instance for testing."""
    config = TokenizerConfig(vocab_size=350, min_frequency=1)
    trainer = LlamaTokenizerTrainer(config=config)
    corpus = [
        "El rápido zorro marrón salta sobre el perro perezoso en 1998.",
        "The quick brown fox jumps over the lazy dog in 1998.",
        "Inteligencia artificial bilingüe en español e inglés.",
        "Bilingual artificial intelligence in Spanish and English.",
    ]
    raw_tok = trainer.train_from_iterator(corpus)
    return LlamaTokenizer(raw_tokenizer=raw_tok, config=config)


def test_wrapper_encode_and_decode(trained_wrapper: LlamaTokenizer):
    """Verify single sequence encode and decode roundtrip."""
    text = "Inteligencia artificial"
    token_ids = trained_wrapper.encode(text, add_special_tokens=True)
    assert isinstance(token_ids, list)
    assert token_ids[0] == trained_wrapper.bos_token_id

    decoded = trained_wrapper.decode(token_ids, skip_special_tokens=True)
    assert decoded == text


def test_wrapper_truncation(trained_wrapper: LlamaTokenizer):
    """Verify truncation truncates sequences exceeding max_length."""
    text = "El rápido zorro marrón salta sobre el perro perezoso en 1998."
    full_ids = trained_wrapper.encode(text, add_special_tokens=True, truncation=False)
    truncated_ids = trained_wrapper.encode(text, add_special_tokens=True, max_length=5, truncation=True)

    assert len(truncated_ids) == 5
    assert len(full_ids) > 5
    assert truncated_ids == full_ids[:5]


def test_encode_batch_padding_right(trained_wrapper: LlamaTokenizer):
    """Verify right-side padding for standard training batches."""
    texts = ["Texto corto", "Un texto significativamente más largo para pruebas."]
    batch = trained_wrapper.encode_batch(texts, padding=True, padding_side="right")

    assert isinstance(batch, BatchEncoding)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    assert len(input_ids) == 2
    assert len(input_ids[0]) == len(input_ids[1])
    assert len(attention_mask[0]) == len(attention_mask[1])

    # First sequence is shorter, so its end should be padded with pad_token_id
    pad_id = trained_wrapper.pad_token_id
    assert input_ids[0][-1] == pad_id
    assert attention_mask[0][-1] == 0
    # First tokens must be active
    assert attention_mask[0][0] == 1
    assert attention_mask[1][0] == 1


def test_encode_batch_padding_left(trained_wrapper: LlamaTokenizer):
    """Verify left-side padding for autoregressive LLM generation."""
    texts = ["Corto", "Texto más extenso con múltiples palabras."]
    batch = trained_wrapper.encode_batch(texts, padding=True, padding_side="left")

    input_ids = batch.input_ids
    attention_mask = batch.attention_mask

    assert len(input_ids[0]) == len(input_ids[1])
    pad_id = trained_wrapper.pad_token_id

    # First sequence is shorter, so its beginning should be padded
    assert input_ids[0][0] == pad_id
    assert attention_mask[0][0] == 0
    # Trailing token must be real token
    assert attention_mask[0][-1] == 1


def test_encode_batch_max_length_padding(trained_wrapper: LlamaTokenizer):
    """Verify padding to exact max_length."""
    texts = ["Hola mundo", "Test sentence"]
    batch = trained_wrapper.encode_batch(texts, padding="max_length", max_length=12)

    assert len(batch.input_ids[0]) == 12
    assert len(batch.input_ids[1]) == 12
    assert len(batch.attention_mask[0]) == 12


def test_batch_decode(trained_wrapper: LlamaTokenizer):
    """Verify batch_decode restores multiple sequences simultaneously."""
    texts = ["Primera frase de prueba.", "Second test sentence."]
    batch = trained_wrapper.encode_batch(texts, padding=False)
    decoded_texts = trained_wrapper.batch_decode(batch.input_ids, skip_special_tokens=True)

    assert decoded_texts == texts


def test_apply_chat_template_rendering(trained_wrapper: LlamaTokenizer):
    """Verify conversation formatting with aligned special tokens."""
    messages = [
        {"role": "system", "content": "Asistente bilingüe."},
        {"role": "user", "content": "Hola, ¿cómo estás?"},
        {"role": "assistant", "content": "¡Muy bien! ¿En qué puedo ayudarte?"},
    ]

    formatted = trained_wrapper.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assert "<|system|>\nAsistente bilingüe.<|end|>" in formatted
    assert "<|user|>\nHola, ¿cómo estás?<|end|>" in formatted
    assert "<|assistant|>\n¡Muy bien! ¿En qué puedo ayudarte?<|end|>" in formatted

    # Test with generation prompt
    prompt_with_gen = trained_wrapper.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    assert prompt_with_gen.endswith("<|assistant|>\n")

    # Test tokenized output
    token_ids = trained_wrapper.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0


def test_save_and_from_pretrained(trained_wrapper: LlamaTokenizer, tmp_path: Path):
    """Verify round-trip serialization and loading via from_pretrained."""
    save_dir = tmp_path / "saved_llama_tokenizer"
    trained_wrapper.save_pretrained(save_dir)

    assert (save_dir / "tokenizer.json").exists()
    assert (save_dir / "tokenizer_config.json").exists()

    reloaded = LlamaTokenizer.from_pretrained(save_dir)
    assert reloaded.vocab_size == trained_wrapper.vocab_size
    assert reloaded.bos_token == trained_wrapper.bos_token
    assert reloaded.eos_token == trained_wrapper.eos_token
    assert reloaded.pad_token_id == trained_wrapper.pad_token_id

    # Ensure encoding is identical
    sample = "Validación de consistencia entre tokenizador original y recargado."
    assert reloaded.encode(sample) == trained_wrapper.encode(sample)


def test_callable_interface(trained_wrapper: LlamaTokenizer):
    """Verify tokenizer instance is callable like Hugging Face tokenizers."""
    text = "Llamada directa como funcion."
    ids = trained_wrapper(text)
    assert ids == trained_wrapper.encode(text)

    batch_output = trained_wrapper([text, "Segunda oracion."], padding=True)
    assert isinstance(batch_output, BatchEncoding)
    assert len(batch_output.input_ids) == 2


def test_invalid_padding_side_raises(trained_wrapper: LlamaTokenizer):
    """Verify ValueError when invalid padding_side is passed."""
    with pytest.raises(ValueError, match="padding_side must be 'left' or 'right'"):
        trained_wrapper.encode_batch(["test"], padding_side="center")
