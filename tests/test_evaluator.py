"""Unit tests for Tokenizer Evaluation and Benchmarking Suite."""

import json
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TokenizerConfig
from src.evaluator import (
    BUILTIN_BENCHMARK_CORPUS,
    EvaluationReport,
    LanguageMetrics,
    TokenizerEvaluator,
)
from src.llama_tokenizer import LlamaTokenizer
from src.tokenizer_builder import LlamaTokenizerTrainer


@pytest.fixture
def trained_eval_tokenizer():
    """Builds a calibrated tokenizer for evaluator testing."""
    config = TokenizerConfig(vocab_size=350, min_frequency=1)
    trainer = LlamaTokenizerTrainer(config=config)
    corpus = [
        "El rápido zorro marrón salta sobre el perro perezoso en 1998.",
        "The quick brown fox jumps over the lazy dog in 1998.",
        "Modelos de lenguaje e inteligencia artificial bilingüe en español e inglés.",
        "Language models and bilingual artificial intelligence in Spanish and English.",
    ]
    raw_tok = trainer.train_from_iterator(corpus)
    return LlamaTokenizer(raw_tokenizer=raw_tok, config=config)


def test_language_metrics_computations():
    """Verify computed properties on LanguageMetrics (fertility, bytes/token, etc.)."""
    m = LanguageMetrics(
        language="es",
        total_samples=2,
        total_words=20,
        total_tokens=25,
        total_characters=100,
        total_bytes=110,
        unk_tokens=0,
        byte_fallback_tokens=2,
        elapsed_seconds=0.01,
    )

    assert m.fertility == 25 / 20  # 1.25 tokens/word
    assert m.bytes_per_token == 110 / 25
    assert m.chars_per_token == 100 / 25
    assert m.unk_rate == 0.0
    assert m.byte_fallback_rate == (2 / 25) * 100.0
    assert m.throughput_mb_s > 0
    assert m.tokens_per_second > 0

    d = m.to_dict()
    assert d["fertility_tokens_per_word"] == 1.25
    assert d["unk_rate_percent"] == 0.0


def test_evaluation_report_aggregation():
    """Verify report aggregates totals across multiple languages properly."""
    m_es = LanguageMetrics(language="es", total_words=10, total_tokens=12, total_bytes=60)
    m_en = LanguageMetrics(language="en", total_words=10, total_tokens=14, total_bytes=56)

    report = EvaluationReport(metrics={"es": m_es, "en": m_en})

    assert report.overall_words == 20
    assert report.overall_tokens == 26
    assert report.overall_bytes == 116
    assert report.overall_fertility == 26 / 20
    assert report.overall_bytes_per_token == 116 / 26

    # Test summary table rendering
    table_str = report.summary_table()
    assert "es" in table_str
    assert "en" in table_str
    assert "OVERALL" in table_str

    # Test JSON serialization
    json_out = report.to_json()
    parsed = json.loads(json_out)
    assert parsed["overall_words"] == 20
    assert "es" in parsed["metrics_by_language"]


def test_evaluation_report_dynamic_overall_unk():
    """Verify that overall_unk_rate in EvaluationReport is calculated dynamically."""
    m_clean = LanguageMetrics(language="es", total_words=10, total_tokens=100, unk_tokens=0)
    m_unseen = LanguageMetrics(language="en", total_words=10, total_tokens=100, unk_tokens=10)

    report = EvaluationReport(metrics={"es": m_clean, "en": m_unseen})
    assert report.overall_tokens == 200
    assert report.overall_unk_tokens == 10
    assert report.overall_unk_rate == 5.0

    table_str = report.summary_table()
    assert "5.00%" in table_str
    report_dict = report.to_dict()
    assert report_dict["overall_unk_tokens"] == 10
    assert report_dict["overall_unk_rate_percent"] == 5.0


def test_evaluator_evaluate_corpus(trained_eval_tokenizer: LlamaTokenizer):
    """Verify evaluating a corpus produces accurate word, token, and byte counts."""
    evaluator = TokenizerEvaluator(trained_eval_tokenizer)
    texts = [
        "El rápido zorro marrón salta.",
        "La inteligencia artificial en español.",
    ]

    metrics = evaluator.evaluate_corpus(texts, language="es")
    assert metrics.total_samples == 2
    assert metrics.total_words > 0
    assert metrics.total_tokens > 0
    assert metrics.total_bytes > 0
    assert metrics.unk_tokens == 0
    assert metrics.fertility > 0
    assert metrics.elapsed_seconds > 0


def test_zero_unk_guarantee(trained_eval_tokenizer: LlamaTokenizer):
    """Verify 0% UNK guarantee even when encountering unseen Unicode characters."""
    evaluator = TokenizerEvaluator(trained_eval_tokenizer)
    rare_texts = [
        "Caracteres poco comunes: ©, ®, €, ¥, §, ¶.",
        "Alfabeto griego: α, β, γ, δ, ε.",
        "Símbolos matemáticos: ∑, ∏, √, ∞, ≈, ≠, ≤, ≥.",
    ]

    metrics = evaluator.evaluate_corpus(rare_texts, language="rare")
    assert metrics.unk_tokens == 0
    assert metrics.unk_rate == 0.0
    # Must have utilized byte-fallback tokens (<0x..>) for the unseen symbols
    assert metrics.byte_fallback_tokens > 0


def test_evaluator_evaluate_bilingual(trained_eval_tokenizer: LlamaTokenizer):
    """Verify bilingual evaluation runs across Spanish, English, and code."""
    evaluator = TokenizerEvaluator(trained_eval_tokenizer)
    report = evaluator.evaluate_bilingual()

    assert "es" in report.metrics
    assert "en" in report.metrics
    assert "code" in report.metrics
    assert report.overall_tokens > 0

    for lang, m in report.metrics.items():
        assert m.total_tokens > 0
        assert m.fertility > 0
        assert m.unk_tokens == 0


def test_benchmark_speed(trained_eval_tokenizer: LlamaTokenizer):
    """Verify speed benchmark outputs valid timing and throughput metrics."""
    evaluator = TokenizerEvaluator(trained_eval_tokenizer)
    texts = BUILTIN_BENCHMARK_CORPUS["es"] + BUILTIN_BENCHMARK_CORPUS["en"]

    speed = evaluator.benchmark_speed(texts, warmup_runs=1, benchmark_runs=2)
    assert "throughput_mb_s" in speed
    assert "tokens_per_second" in speed
    assert "samples_per_second" in speed
    assert speed["throughput_mb_s"] > 0
    assert speed["tokens_per_second"] > 0
    assert speed["total_samples"] == len(texts)
