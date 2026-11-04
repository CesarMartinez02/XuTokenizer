"""CLI Benchmarking and Evaluation entrypoint for Llama-2 Bilingual Tokenizers.

Usage:
  python scripts/benchmark.py --help
  python scripts/benchmark.py --demo
  python scripts/benchmark.py --tokenizer-path artifacts/
  python scripts/benchmark.py --tokenizer-path artifacts/ --output-json benchmark_results.json
  python scripts/benchmark.py --tokenizer-path artifacts/ --compare-with /path/to/other/tokenizer
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TokenizerConfig
from src.evaluator import BUILTIN_BENCHMARK_CORPUS, TokenizerEvaluator
from src.llama_tokenizer import LlamaTokenizer
from src.tokenizer_builder import LlamaTokenizerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def load_lines_from_file(file_path: Optional[str]) -> Optional[List[str]]:
    """Loads non-empty lines from a text file if provided."""
    if not file_path:
        return None
    path = Path(file_path)
    if not path.exists():
        logger.warning("Corpus file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines if lines else None


def train_demo_tokenizer() -> LlamaTokenizer:
    """Trains an in-memory demo tokenizer on the built-in corpus for immediate benchmarking."""
    logger.info("Initializing in-memory demo tokenizer for cold execution...")
    demo_corpus = (
        BUILTIN_BENCHMARK_CORPUS["es"]
        + BUILTIN_BENCHMARK_CORPUS["en"]
        + BUILTIN_BENCHMARK_CORPUS["code"]
    )
    demo_config = TokenizerConfig(vocab_size=350, min_frequency=1)
    trainer = LlamaTokenizerTrainer(config=demo_config)
    raw_tok = trainer.train_from_iterator(demo_corpus, save_artifacts=False)
    logger.info("Demo tokenizer successfully calibrated.")
    return LlamaTokenizer(raw_tokenizer=raw_tok, config=demo_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks on Llama-2 Bilingual Tokenizers."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts"),
        help="Path to tokenizer directory (artifacts/) or .json file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode by training a quick in-memory tokenizer if artifacts are missing.",
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Optional path to a secondary tokenizer to generate side-by-side comparison.",
    )
    parser.add_argument(
        "--eval-es",
        type=str,
        default=None,
        help="Optional custom Spanish evaluation text file.",
    )
    parser.add_argument(
        "--eval-en",
        type=str,
        default=None,
        help="Optional custom English evaluation text file.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of iterations for speed benchmarking (default: 5).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export evaluation report as JSON.",
    )

    args = parser.parse_args()

    # 1. Resolve or train primary tokenizer
    tok_path = Path(args.tokenizer_path)
    primary_tok: Optional[LlamaTokenizer] = None

    if args.demo:
        primary_tok = train_demo_tokenizer()
    elif tok_path.exists():
        logger.info("Loading tokenizer from: %s", tok_path)
        try:
            primary_tok = LlamaTokenizer.from_pretrained(tok_path)
        except Exception as exc:
            logger.warning("Could not load from %s (%s). Falling back to demo tokenizer.", tok_path, exc)
            primary_tok = train_demo_tokenizer()
    else:
        logger.info(
            "Artifacts not found at '%s'. Running in automatic DEMO mode.",
            tok_path,
        )
        primary_tok = train_demo_tokenizer()

    # 2. Prepare corpora
    es_corpus = load_lines_from_file(args.eval_es)
    en_corpus = load_lines_from_file(args.eval_en)

    evaluator = TokenizerEvaluator(primary_tok)

    # 3. Run evaluation report
    logger.info("Running evaluation across domains (Spanish, English, Code)...")
    report = evaluator.evaluate_bilingual(
        es_texts=es_corpus,
        en_texts=en_corpus,
    )

    print("\n" + "=" * 70)
    print(f" EVALUATION REPORT: {report.tokenizer_name} (Vocab Size: {report.vocab_size})")
    print("=" * 70)
    print(report.summary_table())
    print("=" * 70)

    # 4. Run throughput benchmark
    benchmark_texts = (es_corpus or BUILTIN_BENCHMARK_CORPUS["es"]) + (
        en_corpus or BUILTIN_BENCHMARK_CORPUS["en"]
    )
    speed_results = evaluator.benchmark_speed(benchmark_texts, benchmark_runs=args.runs)

    print("\n--- THROUGHPUT BENCHMARK ---")
    print(f"Total Measured Bytes : {speed_results['total_bytes']:,} bytes")
    print(f"Average Duration     : {speed_results['avg_duration_seconds']:.4f} seconds")
    print(f"Speed Throughput     : {speed_results['throughput_mb_s']:.2f} MB/s")
    print(f"Tokens Per Second    : {speed_results['tokens_per_second']:,.1f} tokens/s")
    print(f"Samples Per Second   : {speed_results['samples_per_second']:,.1f} samples/s")
    print("----------------------------\n")

    # 5. Optional Comparison Mode
    if args.compare_with:
        comp_path = Path(args.compare_with)
        logger.info("Loading comparison tokenizer from: %s", comp_path)
        try:
            second_tok = LlamaTokenizer.from_pretrained(comp_path)
            second_eval = TokenizerEvaluator(second_tok)
            second_report = second_eval.evaluate_bilingual(
                es_texts=es_corpus,
                en_texts=en_corpus,
            )

            print("\n" + "=" * 70)
            print(f" COMPARISON REPORT: {args.compare_with}")
            print("=" * 70)
            print(second_report.summary_table())
            print("=" * 70)

            print("\n--- FERTILITY COMPARISON (Tokens/Word) ---")
            for lang in report.metrics:
                if lang in second_report.metrics:
                    f1 = report.metrics[lang].fertility
                    f2 = second_report.metrics[lang].fertility
                    diff = ((f1 - f2) / f2) * 100
                    status = "more compact" if diff < 0 else "less compact"
                    print(f"[{lang.upper()}] Primary: {f1:.3f} vs Reference: {f2:.3f} ({abs(diff):.1f}% {status})")
            print("-----------------------------------------\n")
        except Exception as exc:
            logger.warning("Could not run comparison tokenizer: %s", exc)

    # 6. Save JSON report if requested
    if args.output_json:
        out_file = Path(args.output_json)
        full_data = report.to_dict()
        full_data["speed_benchmark"] = speed_results
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info("Full evaluation report exported to: %s", out_file)


if __name__ == "__main__":
    main()
