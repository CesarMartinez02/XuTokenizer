"""CLI training entrypoint for the Llama-2 Bilingual Tokenizer.

Usage:
  python scripts/train_tokenizer.py --help
  python scripts/train_tokenizer.py --vocab-size 32000 --samples 50000
  python scripts/train_tokenizer.py --local-es data/es.txt --local-en data/en.txt
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DataStreamConfig, TokenizerConfig
from src.data_stream import BilingualDataStreamer
from src.tokenizer_builder import LlamaTokenizerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Llama-2 style Bilingual BPE Tokenizer (Spanish + English)."
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
        help="Target total vocabulary size including special and byte tokens (default: 32000).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for BPE pair merges (default: 2).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50_000,
        help="Target number of bilingual articles/chunks to stream from HF (default: 50000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help="Batch size for Rust streaming iterator (default: 1000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory to save the trained tokenizer.json and configs.",
    )
    parser.add_argument(
        "--local-es",
        nargs="*",
        default=None,
        help="Optional local Spanish text or jsonl files.",
    )
    parser.add_argument(
        "--local-en",
        nargs="*",
        default=None,
        help="Optional local English text or jsonl files.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_stream_cfg = DataStreamConfig(
        batch_size=args.batch_size,
        target_sample_count=args.samples,
        local_es_files=args.local_es,
        local_en_files=args.local_en,
    )

    tokenizer_cfg = TokenizerConfig(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        data_stream=data_stream_cfg,
        output_dir=output_path,
    )

    logger.info("Initializing bilingual streaming pipeline...")
    streamer = BilingualDataStreamer(config=data_stream_cfg)

    logger.info("Assembling Llama-2 Rust BPE Trainer...")
    trainer = LlamaTokenizerTrainer(config=tokenizer_cfg)

    logger.info("Training tokenizer (vocab_size=%d)...", args.vocab_size)
    tokenizer = trainer.train_from_stream(streamer=streamer, save_artifacts=True)

    logger.info("Training complete!")
    logger.info("Final vocabulary size: %d", tokenizer.get_vocab_size())
    logger.info("Artifacts saved to: %s", output_path)


if __name__ == "__main__":
    main()
