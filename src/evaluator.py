"""Evaluation and Benchmarking Suite for Bilingual Tokenizers.

Provides standardized quantitative metrics for tokenizer performance:
- Fertility (Tokens per Word): Key metric for multilingual fairness and efficiency.
- Compression Ratio (Bytes per Token, Characters per Token).
- Zero-UNK Verification and Byte-Fallback Tracking.
- Throughput and Processing Speed (MB/s, Tokens/s, Samples/s).
- Multi-domain and Bilingual Evaluation Reports.
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Standard word matching pattern across Unicode text
WORD_REGEX = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass
class LanguageMetrics:
    """Quantitative evaluation metrics for a specific language or domain."""

    language: str
    total_samples: int = 0
    total_words: int = 0
    total_tokens: int = 0
    total_characters: int = 0
    total_bytes: int = 0
    unk_tokens: int = 0
    byte_fallback_tokens: int = 0
    elapsed_seconds: float = 0.0

    @property
    def fertility(self) -> float:
        """Tokens per word. Lower indicates higher semantic efficiency."""
        return self.total_tokens / max(1, self.total_words)

    @property
    def bytes_per_token(self) -> float:
        """UTF-8 bytes per token. Higher indicates better byte compression."""
        return self.total_bytes / max(1, self.total_tokens)

    @property
    def chars_per_token(self) -> float:
        """Characters per token."""
        return self.total_characters / max(1, self.total_tokens)

    @property
    def unk_rate(self) -> float:
        """Percentage of tokens that were mapped to UNK. Target is strictly 0.0%."""
        return (self.unk_tokens / max(1, self.total_tokens)) * 100.0

    @property
    def byte_fallback_rate(self) -> float:
        """Percentage of tokens represented via byte-fallback (<0x..>)."""
        return (self.byte_fallback_tokens / max(1, self.total_tokens)) * 100.0

    @property
    def throughput_mb_s(self) -> float:
        """Processing speed in megabytes per second."""
        mb = self.total_bytes / (1024 * 1024)
        return mb / max(1e-6, self.elapsed_seconds)

    @property
    def tokens_per_second(self) -> float:
        """Token processing throughput in tokens per second."""
        return self.total_tokens / max(1e-6, self.elapsed_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a serializable dictionary of computed metrics."""
        return {
            "language": self.language,
            "total_samples": self.total_samples,
            "total_words": self.total_words,
            "total_tokens": self.total_tokens,
            "total_characters": self.total_characters,
            "total_bytes": self.total_bytes,
            "fertility_tokens_per_word": round(self.fertility, 3),
            "bytes_per_token": round(self.bytes_per_token, 3),
            "chars_per_token": round(self.chars_per_token, 3),
            "unk_tokens": self.unk_tokens,
            "unk_rate_percent": round(self.unk_rate, 4),
            "byte_fallback_tokens": self.byte_fallback_tokens,
            "byte_fallback_rate_percent": round(self.byte_fallback_rate, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "throughput_mb_s": round(self.throughput_mb_s, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report covering all evaluated languages and domains."""

    metrics: Dict[str, LanguageMetrics] = field(default_factory=dict)
    tokenizer_name: str = "Llama2-Bilingual-Tokenizer"
    vocab_size: int = 32000

    @property
    def overall_words(self) -> int:
        return sum(m.total_words for m in self.metrics.values())

    @property
    def overall_tokens(self) -> int:
        return sum(m.total_tokens for m in self.metrics.values())

    @property
    def overall_bytes(self) -> int:
        return sum(m.total_bytes for m in self.metrics.values())

    @property
    def overall_fertility(self) -> float:
        return self.overall_tokens / max(1, self.overall_words)

    @property
    def overall_bytes_per_token(self) -> float:
        return self.overall_bytes / max(1, self.overall_tokens)

    @property
    def overall_unk_tokens(self) -> int:
        return sum(m.unk_tokens for m in self.metrics.values())

    @property
    def overall_unk_rate(self) -> float:
        return (self.overall_unk_tokens / max(1, self.overall_tokens)) * 100.0

    def summary_table(self) -> str:
        """Renders a clean ASCII summary table of the evaluation results."""
        header = (
            f"+{'-'*12}+{'-'*10}+{'-'*10}+{'-'*12}+{'-'*12}+{'-'*10}+{'-'*12}+\n"
            f"| Language   | Words    | Tokens   | Fertility  | Bytes/Tok  | UNK Rate | Speed MB/s |\n"
            f"+{'-'*12}+{'-'*10}+{'-'*10}+{'-'*12}+{'-'*12}+{'-'*10}+{'-'*12}+"
        )
        rows = []
        for lang, m in self.metrics.items():
            unk_str = f"{m.unk_rate:.2f}%"
            row = (
                f"| {lang:<10} "
                f"| {m.total_words:<8} "
                f"| {m.total_tokens:<8} "
                f"| {m.fertility:<10.3f} "
                f"| {m.bytes_per_token:<10.3f} "
                f"| {unk_str:<8} "
                f"| {m.throughput_mb_s:<10.2f} |"
            )
            rows.append(row)

        divider = f"+{'-'*12}+{'-'*10}+{'-'*10}+{'-'*12}+{'-'*12}+{'-'*10}+{'-'*12}+"
        overall_unk_str = f"{self.overall_unk_rate:.2f}%"
        summary_row = (
            f"| OVERALL    "
            f"| {self.overall_words:<8} "
            f"| {self.overall_tokens:<8} "
            f"| {self.overall_fertility:<10.3f} "
            f"| {self.overall_bytes_per_token:<10.3f} "
            f"| {overall_unk_str:<8} "
            f"| {'-':<10} |"
        )
        footer = divider

        return "\n".join([header] + rows + [divider, summary_row, footer])

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the report to a dictionary."""
        return {
            "tokenizer_name": self.tokenizer_name,
            "vocab_size": self.vocab_size,
            "overall_words": self.overall_words,
            "overall_tokens": self.overall_tokens,
            "overall_bytes": self.overall_bytes,
            "overall_fertility": round(self.overall_fertility, 3),
            "overall_bytes_per_token": round(self.overall_bytes_per_token, 3),
            "overall_unk_tokens": self.overall_unk_tokens,
            "overall_unk_rate_percent": round(self.overall_unk_rate, 4),
            "metrics_by_language": {lang: m.to_dict() for lang, m in self.metrics.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes the report to JSON format."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# Curated standardized benchmark corpus for offline bilingual evaluation
BUILTIN_BENCHMARK_CORPUS: Dict[str, List[str]] = {
    "es": [
        "El rápido zorro marrón salta sobre el perro perezoso en el bosque templado.",
        "La inteligencia artificial generativa y los modelos de lenguaje transforman el procesamiento del lenguaje natural.",
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo.",
        "El producto interno bruto experimentó una recuperación del cuatro punto cinco por ciento durante el último trimestre.",
        "Los investigadores desarrollaron un nuevo algoritmo cuántico para optimizar redes neuronales convolucionales profundas.",
        "¿Cuáles son las principales diferencias entre el aprendizaje supervisado, no supervisado y por refuerzo?",
        "Las elecciones presidenciales contaron con una participación ciudadana histórica según las actas oficiales del tribunal.",
        "El cielo nocturno sobre el observatorio astronómico reveló nebulosas y galaxias distantes con gran nitidez.",
    ],
    "en": [
        "The quick brown fox jumps over the lazy dog in the temperate forest.",
        "Generative artificial intelligence and large language models are transforming natural language processing.",
        "In a village of La Mancha, the name of which I have no desire to call to mind, there lived not long since a nobleman.",
        "Gross domestic product experienced a recovery of four point five percent during the last fiscal quarter.",
        "Researchers developed a novel quantum algorithm to optimize deep convolutional neural networks.",
        "What are the primary differences between supervised, unsupervised, and reinforcement learning paradigms?",
        "The presidential elections recorded a historic citizen turnout according to the official tribunal records.",
        "The night sky above the astronomical observatory revealed distant nebulae and galaxies with remarkable clarity.",
    ],
    "code": [
        "def compute_bpe_merges(vocab: dict[str, int], min_freq: int = 2) -> list[tuple[str, str]]:\n    pairs = {}\n    return sorted(pairs.items(), key=lambda x: x[1], reverse=True)",
        "SELECT user_id, COUNT(order_id) AS total_orders, SUM(amount) FROM transactions WHERE status = 'completed' GROUP BY user_id HAVING total_orders > 5;",
        "const calculateThroughput = (totalBytes, elapsedSeconds) => (totalBytes / (1024 * 1024)) / Math.max(elapsedSeconds, 1e-5);",
    ],
}


class TokenizerEvaluator:
    """Evaluates tokenizer compression, fertility, and throughput across languages."""

    def __init__(self, tokenizer: Any, unk_token_id: int = 0) -> None:
        """Initializes the evaluator.

        Args:
            tokenizer: Any tokenizer instance implementing .encode(text, add_special_tokens=...)
                       and optionally .vocab_size or .get_vocab_size().
            unk_token_id: ID designated for unknown tokens (0 for Llama-2 standard).
        """
        self.tokenizer = tokenizer
        self.unk_token_id = unk_token_id

        # Determine vocabulary size
        if hasattr(tokenizer, "vocab_size"):
            self.vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, "get_vocab_size"):
            self.vocab_size = tokenizer.get_vocab_size()
        else:
            self.vocab_size = 32000

    def _encode_text(self, text: str) -> List[int]:
        """Extracts token IDs for pure text content without sequence delimiters."""
        if hasattr(self.tokenizer, "encode"):
            try:
                res = self.tokenizer.encode(text, add_special_tokens=False)
                if hasattr(res, "ids"):
                    return res.ids
                return res
            except TypeError:
                return self.tokenizer.encode(text)
        elif callable(self.tokenizer):
            res = self.tokenizer(text)
            if hasattr(res, "ids"):
                return res.ids
            if isinstance(res, dict) and "input_ids" in res:
                return res["input_ids"]
            return res
        raise AttributeError("Tokenizer must provide an 'encode' method or be callable.")

    def evaluate_corpus(self, texts: List[str], language: str = "es") -> LanguageMetrics:
        """Evaluates a collection of texts in a given language.

        Separates pure tokenization execution time from statistical regex and byte computations
        to ensure throughput (MB/s) reflects genuine tokenizer engine speed.

        Args:
            texts: List of text samples to tokenize and measure.
            language: Language or category identifier (e.g. 'es', 'en', 'code').

        Returns:
            LanguageMetrics object containing detailed statistics.
        """
        metrics = LanguageMetrics(language=language)
        metrics.total_samples = len(texts)

        # 1. Pure tokenization timed in isolation
        t0 = time.perf_counter()
        tokenized_sequences: List[List[int]] = [self._encode_text(t) for t in texts]
        metrics.elapsed_seconds = max(1e-6, time.perf_counter() - t0)

        # 2. Text metrics and token inspection outside the timer
        for text, token_ids in zip(texts, tokenized_sequences):
            words = WORD_REGEX.findall(text)
            raw_bytes = text.encode("utf-8")

            metrics.total_words += len(words)
            metrics.total_characters += len(text)
            metrics.total_bytes += len(raw_bytes)
            metrics.total_tokens += len(token_ids)

            # Check for UNK and byte fallback tokens
            for tid in token_ids:
                if tid == self.unk_token_id:
                    metrics.unk_tokens += 1
                # Byte fallback IDs reside at 10..265 in our architecture
                elif 10 <= tid <= 265:
                    metrics.byte_fallback_tokens += 1

        return metrics

    def evaluate_bilingual(
        self,
        es_texts: Optional[List[str]] = None,
        en_texts: Optional[List[str]] = None,
        code_texts: Optional[List[str]] = None,
        custom_corpora: Optional[Dict[str, List[str]]] = None,
    ) -> EvaluationReport:
        """Runs comprehensive multi-domain evaluation across Spanish, English, and code.

        Args:
            es_texts: Spanish text samples (defaults to built-in benchmark corpus).
            en_texts: English text samples (defaults to built-in benchmark corpus).
            code_texts: Code text samples (defaults to built-in benchmark corpus).
            custom_corpora: Optional custom dictionary mapping domain names to text lists.

        Returns:
            EvaluationReport containing per-language metrics and overall averages.
        """
        report = EvaluationReport(
            tokenizer_name=getattr(self.tokenizer, "model_name", "Llama2-Bilingual-Tokenizer"),
            vocab_size=self.vocab_size,
        )

        corpora: Dict[str, List[str]] = {}

        # Add default or supplied corpora
        corpora["es"] = es_texts if es_texts is not None else BUILTIN_BENCHMARK_CORPUS["es"]
        corpora["en"] = en_texts if en_texts is not None else BUILTIN_BENCHMARK_CORPUS["en"]
        if code_texts is not None or "code" in BUILTIN_BENCHMARK_CORPUS:
            corpora["code"] = code_texts if code_texts is not None else BUILTIN_BENCHMARK_CORPUS["code"]

        if custom_corpora:
            corpora.update(custom_corpora)

        for lang, text_list in corpora.items():
            if text_list:
                report.metrics[lang] = self.evaluate_corpus(text_list, language=lang)

        return report

    def benchmark_speed(
        self,
        texts: List[str],
        warmup_runs: int = 2,
        benchmark_runs: int = 5,
    ) -> Dict[str, float]:
        """Benchmarks tokenization speed over multiple repetitions with warm-up.

        Args:
            texts: List of texts to repeatedly tokenize.
            warmup_runs: Number of unmeasured warm-up iterations.
            benchmark_runs: Number of timed iterations to average.

        Returns:
            Dictionary with throughput_mb_s, tokens_per_second, and samples_per_second.
        """
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)

        # Warm-up phase
        for _ in range(warmup_runs):
            for t in texts:
                _ = self._encode_text(t)

        durations: List[float] = []
        total_tokens_per_run: int = 0

        # Benchmark phase
        for _ in range(benchmark_runs):
            run_tokens = 0
            t0 = time.perf_counter()
            for t in texts:
                toks = self._encode_text(t)
                run_tokens += len(toks)
            elapsed = time.perf_counter() - t0
            durations.append(elapsed)
            total_tokens_per_run = run_tokens

        avg_duration = sum(durations) / len(durations)
        mb = total_bytes / (1024 * 1024)

        return {
            "avg_duration_seconds": round(avg_duration, 4),
            "throughput_mb_s": round(mb / max(1e-6, avg_duration), 2),
            "tokens_per_second": round(total_tokens_per_run / max(1e-6, avg_duration), 1),
            "samples_per_second": round(len(texts) / max(1e-6, avg_duration), 1),
            "total_bytes": total_bytes,
            "total_samples": len(texts),
        }
