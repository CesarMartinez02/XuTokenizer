"""Bilingual Data Streaming Pipeline for Tokenizer Training.

Provides high-throughput, memory-efficient data streaming for training BPE tokenizers
without needing to download massive corpora to local storage.

Features:
- Streaming from Hugging Face Hub (streaming=True)
- Balanced 50/50 Spanish / English interleaving
- Support for offline local datasets (.txt, .jsonl, .parquet)
- Generator interfaces for both individual text chunks and batches
- Automatic text sanitization and min-length filtering
- Real-time telemetry and progress reporting (tqdm)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class StreamingTelemetry:
    """Real-time metrics tracked during dataset streaming."""

    total_samples: int = 0
    total_characters: int = 0
    total_bytes: int = 0
    es_samples: int = 0
    en_samples: int = 0
    skipped_short_samples: int = 0
    start_time: float = field(default_factory=time.time)

    def record_sample(self, text: str, lang: str = "mixed") -> None:
        """Records metrics for an emitted sample."""
        self.total_samples += 1
        num_chars = len(text)
        self.total_characters += num_chars
        self.total_bytes += len(text.encode("utf-8"))

        if lang == "es":
            self.es_samples += 1
        elif lang == "en":
            self.en_samples += 1

    def record_skipped(self) -> None:
        """Records a sample discarded due to length/content constraints."""
        self.skipped_short_samples += 1

    @property
    def elapsed_seconds(self) -> float:
        return max(1e-5, time.time() - self.start_time)

    @property
    def samples_per_second(self) -> float:
        return self.total_samples / self.elapsed_seconds

    @property
    def mb_processed(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def summary(self) -> Dict[str, Any]:
        """Returns a serializable dictionary of streaming performance metrics."""
        return {
            "total_samples": self.total_samples,
            "total_characters": self.total_characters,
            "approx_mb": round(self.mb_processed, 2),
            "es_samples": self.es_samples,
            "en_samples": self.en_samples,
            "skipped_short_samples": self.skipped_short_samples,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "samples_per_sec": round(self.samples_per_second, 2),
        }


class BilingualDataStreamer:
    """High-throughput bilingual data streamer for BPE tokenizer training.

    Yields sanitized text samples or batches balanced between Spanish and English.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        custom_es_iterator: Optional[Iterable[str]] = None,
        custom_en_iterator: Optional[Iterable[str]] = None,
    ) -> None:
        """Initializes the streamer.

        Args:
            config: DataStreamConfig instance (or default if None).
            custom_es_iterator: Optional in-memory or mock iterator for Spanish texts.
            custom_en_iterator: Optional in-memory or mock iterator for English texts.
        """
        if config is None:
            from .config import DataStreamConfig

            self.config = DataStreamConfig()
        else:
            self.config = config

        self.custom_es_iterator = custom_es_iterator
        self.custom_en_iterator = custom_en_iterator
        self.telemetry = StreamingTelemetry()

    def _sanitize_text(self, raw_text: Any) -> Optional[str]:
        """Cleans and validates candidate text samples.

        Args:
            raw_text: Raw candidate value from dataset.

        Returns:
            Stripped text if valid and meeting length threshold, else None.
        """
        if not isinstance(raw_text, str):
            return None

        cleaned = raw_text.strip()
        if len(cleaned) < self.config.min_text_length_chars:
            self.telemetry.record_skipped()
            return None

        return cleaned

    def _load_hf_streaming_dataset(
        self, dataset_name: str, dataset_config: Optional[str] = None
    ) -> Any:
        """Loads a streaming dataset from Hugging Face Hub."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for HF streaming. "
                "Install it via 'pip install datasets'."
            ) from exc

        try:
            kwargs: Dict[str, Any] = {"streaming": True, "split": "train"}
            if dataset_config:
                return load_dataset(dataset_name, dataset_config, **kwargs)
            return load_dataset(dataset_name, **kwargs)
        except Exception as exc:
            logger.error(
                "Failed to stream dataset '%s' (%s) from Hugging Face: %s",
                dataset_name,
                dataset_config,
                exc,
            )
            raise ConnectionError(
                f"Could not connect to Hugging Face Hub to stream '{dataset_name}'. "
                "Ensure internet connectivity is available or supply local text files."
            ) from exc

    def _build_local_file_iterator(self, file_paths: List[Union[str, Path]]) -> Generator[str, None, None]:
        """Reads text from local .txt or .jsonl files."""
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning("Local corpus file not found: %s", path)
                continue

            if path.suffix.lower() == ".jsonl":
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            for key in ("text", "content", "body"):
                                if key in record and isinstance(record[key], str):
                                    yield record[key]
                                    break
                        except json.JSONDecodeError:
                            continue
            else:
                # Plain text: yield paragraph chunks or lines
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    paragraph: List[str] = []
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            if paragraph:
                                yield " ".join(paragraph)
                                paragraph = []
                        else:
                            paragraph.append(stripped)
                    if paragraph:
                        yield " ".join(paragraph)

    def _create_stream(self, lang: str) -> Iterator[str]:
        """Creates an infinite/exhaustive iterator for the specified language."""
        if lang == "es":
            if self.custom_es_iterator is not None:
                yield from self.custom_es_iterator
                return
            if self.config.local_es_files:
                yield from self._build_local_file_iterator(self.config.local_es_files)
                return

            ds = self._load_hf_streaming_dataset(
                self.config.es_dataset_name, self.config.es_dataset_config
            )
            text_field = self.config.es_text_field
            for item in ds:
                if text_field in item and item[text_field]:
                    yield item[text_field]

        elif lang == "en":
            if self.custom_en_iterator is not None:
                yield from self.custom_en_iterator
                return
            if self.config.local_en_files:
                yield from self._build_local_file_iterator(self.config.local_en_files)
                return

            ds = self._load_hf_streaming_dataset(
                self.config.en_dataset_name, self.config.en_dataset_config
            )
            text_field = self.config.en_text_field
            for item in ds:
                if text_field in item and item[text_field]:
                    yield item[text_field]
        else:
            raise ValueError(f"Unsupported language identifier: {lang}")

    def iter_texts(self, max_samples: Optional[int] = None) -> Generator[str, None, None]:
        """Yields interleaved Spanish and English sanitized text samples.

        Args:
            max_samples: Maximum number of samples to yield (defaults to config.target_sample_count).

        Yields:
            Sanitized str text samples.
        """
        limit = max_samples if max_samples is not None else self.config.target_sample_count
        self.telemetry = StreamingTelemetry()

        es_stream = self._create_stream("es")
        en_stream = self._create_stream("en")

        progress_bar = None
        if self.config.show_progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(
                    total=limit,
                    desc="Streaming Bilingual Corpus (50% ES / 50% EN)",
                    unit="samples",
                )
            except ImportError:
                progress_bar = None

        es_active = True
        en_active = True
        turn_es = True

        try:
            while self.telemetry.total_samples < limit and (es_active or en_active):
                candidate_text = None
                lang = ""

                # Alternating 50/50 round-robin
                if turn_es and es_active:
                    try:
                        raw = next(es_stream)
                        candidate_text = self._sanitize_text(raw)
                        lang = "es"
                    except StopIteration:
                        es_active = False
                    finally:
                        turn_es = False
                elif not turn_es and en_active:
                    try:
                        raw = next(en_stream)
                        candidate_text = self._sanitize_text(raw)
                        lang = "en"
                    except StopIteration:
                        en_active = False
                    finally:
                        turn_es = True
                else:
                    # One stream exhausted; switch to whichever remains active
                    turn_es = not turn_es
                    continue

                if candidate_text:
                    self.telemetry.record_sample(candidate_text, lang=lang)
                    if progress_bar:
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            es=self.telemetry.es_samples,
                            en=self.telemetry.en_samples,
                            mb=f"{self.telemetry.mb_processed:.1f}",
                        )
                    yield candidate_text

        finally:
            if progress_bar:
                progress_bar.close()

    def iter_batches(
        self,
        batch_size: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> Generator[List[str], None, None]:
        """Yields batches of sanitized text samples, ideal for Tokenizer.train_from_iterator.

        Args:
            batch_size: Number of text samples per batch (defaults to config.batch_size).
            max_samples: Maximum total samples to yield across all batches.

        Yields:
            List[str] batches of texts.
        """
        bsize = batch_size if batch_size is not None else self.config.batch_size
        batch: List[str] = []

        for text in self.iter_texts(max_samples=max_samples):
            batch.append(text)
            if len(batch) >= bsize:
                yield batch
                batch = []

        if batch:
            yield batch

    def __iter__(self) -> Generator[str, None, None]:
        """Default iterator protocol yielding individual texts."""
        return self.iter_texts()
