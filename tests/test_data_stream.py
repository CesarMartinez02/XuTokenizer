"""Unit tests for bilingual data streaming pipeline."""

import json
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DataStreamConfig
from src.data_stream import BilingualDataStreamer, StreamingTelemetry


def test_telemetry_metrics():
    """Verify StreamingTelemetry accumulates sample counts, chars and languages accurately."""
    telemetry = StreamingTelemetry()
    assert telemetry.total_samples == 0
    assert telemetry.total_characters == 0

    telemetry.record_sample("Hola mundo en español", lang="es")
    telemetry.record_sample("Hello world in english", lang="en")
    telemetry.record_skipped()

    assert telemetry.total_samples == 2
    assert telemetry.es_samples == 1
    assert telemetry.en_samples == 1
    assert telemetry.skipped_short_samples == 1
    assert telemetry.total_characters == len("Hola mundo en español") + len("Hello world in english")

    summary = telemetry.summary()
    assert summary["total_samples"] == 2
    assert summary["es_samples"] == 1
    assert summary["en_samples"] == 1
    assert summary["skipped_short_samples"] == 1


def test_bilingual_alternation_50_50():
    """Verify streamer alternates strictly between Spanish and English inputs."""
    es_data = [f"Texto en español número {i} con longitud suficiente para superar el filtro mínimo." for i in range(10)]
    en_data = [f"English text sample number {i} with sufficient length to pass the minimum character filter." for i in range(10)]

    config = DataStreamConfig(
        target_sample_count=8,
        min_text_length_chars=20,
        show_progress=False,
    )

    streamer = BilingualDataStreamer(
        config=config,
        custom_es_iterator=es_data,
        custom_en_iterator=en_data,
    )

    samples = list(streamer.iter_texts())
    assert len(samples) == 8

    # Assert alternation (even index -> ES, odd index -> EN)
    for idx, sample in enumerate(samples):
        if idx % 2 == 0:
            assert "español" in sample
        else:
            assert "English" in sample

    assert streamer.telemetry.es_samples == 4
    assert streamer.telemetry.en_samples == 4


def test_min_length_filtering():
    """Verify texts shorter than min_text_length_chars or empty/whitespace are filtered out."""
    es_data = [
        "ok",                                        # Too short (< 25 chars)
        "   ",                                       # Whitespace
        "El quijote de la mancha es una gran obra literaria clásica.",  # Valid
    ]
    en_data = [
        "short",                                     # Too short (< 25 chars)
        "The quick brown fox jumps over the lazy dog.", # Valid
    ]

    config = DataStreamConfig(
        target_sample_count=10,
        min_text_length_chars=25,
        show_progress=False,
    )

    streamer = BilingualDataStreamer(
        config=config,
        custom_es_iterator=es_data,
        custom_en_iterator=en_data,
    )

    samples = list(streamer.iter_texts())
    assert len(samples) == 2
    assert any("quijote" in s for s in samples)
    assert any("quick brown fox" in s for s in samples)
    assert streamer.telemetry.skipped_short_samples >= 3


def test_batch_yielding():
    """Verify iter_batches groups samples into proper chunk sizes."""
    es_data = [f"Texto en español de prueba número {i:03d} con suficiente contenido para pasar." for i in range(7)]
    en_data = [f"English sample sentence number {i:03d} with enough words to pass validation." for i in range(7)]

    config = DataStreamConfig(
        batch_size=3,
        target_sample_count=14,
        min_text_length_chars=20,
        show_progress=False,
    )

    streamer = BilingualDataStreamer(
        config=config,
        custom_es_iterator=es_data,
        custom_en_iterator=en_data,
    )

    batches = list(streamer.iter_batches(batch_size=3))
    # 14 samples with batch_size 3 -> batches of sizes [3, 3, 3, 3, 2]
    assert len(batches) == 5
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 3
    assert len(batches[4]) == 2
    assert sum(len(b) for b in batches) == 14


def test_local_files_streaming(tmp_path: Path):
    """Verify streaming from local .txt and .jsonl files works seamlessly."""
    # Create local txt file
    txt_file = tmp_path / "corpus.txt"
    txt_file.write_text(
        "Primera línea de texto largo en español para validar lectura local.\n\n"
        "Segunda línea de texto largo en español para validar lectura local.\n",
        encoding="utf-8",
    )

    # Create local jsonl file
    jsonl_file = tmp_path / "corpus.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "First line of long English text to validate local jsonl reading."}) + "\n")
        f.write(json.dumps({"text": "Second line of long English text to validate local jsonl reading."}) + "\n")

    config = DataStreamConfig(
        local_es_files=[str(txt_file)],
        local_en_files=[str(jsonl_file)],
        target_sample_count=10,
        min_text_length_chars=20,
        show_progress=False,
    )

    streamer = BilingualDataStreamer(config=config)
    samples = list(streamer.iter_texts())

    assert len(samples) == 4
    assert streamer.telemetry.es_samples == 2
    assert streamer.telemetry.en_samples == 2
