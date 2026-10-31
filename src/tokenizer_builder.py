"""Llama-2 Style Bilingual Tokenizer Builder and Trainer.

Assembles and trains a Byte-Pair Encoding (BPE) tokenizer using the Hugging Face
`tokenizers` Rust backend with the exact Llama-2 specification:
- SentencePiece Metaspace delimiter ('\u2581')
- Split digits (0-9 isolated)
- Byte-fallback (256 hexadecimal byte tokens <0x00>..<0xFF>)
- Template post-processing (<s> BOS prefix)
- Export to standard Hugging Face fast tokenizer formats
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from .config import DEFAULT_CONFIG, TokenizerConfig
from .data_stream import BilingualDataStreamer

logger = logging.getLogger(__name__)


def build_llama2_pipeline(config: TokenizerConfig) -> Tokenizer:
    """Builds an untrained Llama-2 style BPE tokenizer pipeline in Rust.

    Args:
        config: TokenizerConfig instance with model hyper-parameters.

    Returns:
        Configured, untrained Tokenizer instance.
    """
    # 1. Base Model: BPE with byte_fallback enabled
    model = models.BPE(
        unk_token=config.special_tokens.unk_token,
        byte_fallback=config.byte_fallback,
    )
    tokenizer = Tokenizer(model)

    # 2. Pre-Tokenizer: Metaspace + Digits
    # Order: Metaspace with prepend_scheme='first' followed by individual Digits
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(
                replacement=config.metaspace_char,
                prepend_scheme="first",
            ),
            pre_tokenizers.Digits(individual_digits=config.split_digits),
        ]
    )

    # 3. Decoder: ByteFallback + Metaspace
    # Reconstructs raw UTF-8 bytes and restores standard space characters
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.ByteFallback(),
            decoders.Metaspace(
                replacement=config.metaspace_char,
                prepend_scheme="first",
            ),
        ]
    )

    # 4. Post-Processor: TemplateProcessing for BOS/EOS handling
    bos_token = config.special_tokens.bos_token
    bos_id = config.special_tokens.bos_id
    eos_token = config.special_tokens.eos_token
    eos_id = config.special_tokens.eos_id

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos_token} $A",
        pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[
            (bos_token, bos_id),
            (eos_token, eos_id),
        ],
    )

    return tokenizer


def _normalize_added_tokens(tokenizer: Tokenizer) -> Tokenizer:
    """Marks byte-fallback tokens (<0x00>..<0xFF>) with special=False.

    During BPE training, passing all reserved tokens to BpeTrainer sets
    their 'special' flag to True. If byte-fallback tokens remain marked as
    special, tokenizer.decode(ids, skip_special_tokens=True) strips them,
    corrupting unseen characters, numbers, and emojis. This function flips
    the 'special' flag of byte-fallback tokens to False.
    """
    raw_json = json.loads(tokenizer.to_str())
    added_tokens = raw_json.get("added_tokens", [])

    for entry in added_tokens:
        content = entry.get("content", "")
        # Byte-fallback tokens: <0x00> to <0xFF>
        if content.startswith("<0x") and content.endswith(">") and len(content) == 6:
            entry["special"] = False

    return Tokenizer.from_str(json.dumps(raw_json))


def save_tokenizer_artifacts(
    tokenizer: Tokenizer,
    config: TokenizerConfig,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Serializes the tokenizer to standard Hugging Face Fast Tokenizer files.

    Generates:
      - tokenizer.json: Fast tokenizer definition in Rust-compatible JSON
      - tokenizer_config.json: HF metadata (BOS, EOS, UNK, PAD, model_max_length)
      - special_tokens_map.json: Standard Hugging Face special tokens mapping
    """
    save_path = Path(output_dir) if output_dir else config.output_dir
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer_json = save_path / f"{config.model_name}.json"
    tokenizer_standard_json = save_path / "tokenizer.json"
    config_json = save_path / "tokenizer_config.json"
    special_map_json = save_path / "special_tokens_map.json"

    # 1. Save core fast tokenizer JSON
    tokenizer.save(str(tokenizer_json))
    tokenizer.save(str(tokenizer_standard_json))

    # 2. Save tokenizer_config.json
    spec = config.special_tokens
    tokenizer_config_data = {
        "add_bos_token": True,
        "add_eos_token": False,
        "add_prefix_space": config.add_prefix_space,
        "bos_token": spec.bos_token,
        "eos_token": spec.eos_token,
        "unk_token": spec.unk_token,
        "pad_token": spec.pad_token,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 4096,
        "tokenizer_class": "LlamaTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        ),
    }
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config_data, f, indent=2, ensure_ascii=False)

    # 3. Save special_tokens_map.json
    special_tokens_map_data = {
        "bos_token": spec.bos_token,
        "eos_token": spec.eos_token,
        "unk_token": spec.unk_token,
        "pad_token": spec.pad_token,
        "additional_special_tokens": spec.chat_tokens,
    }
    with open(special_map_json, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map_data, f, indent=2, ensure_ascii=False)

    logger.info("Tokenizer artifacts successfully saved to %s", save_path)
    return {
        "tokenizer_json": tokenizer_json,
        "tokenizer_config_json": config_json,
        "special_tokens_map_json": special_map_json,
    }


class LlamaTokenizerTrainer:
    """Trainer coordinator for building and training Llama-2 style BPE tokenizers."""

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config if config is not None else DEFAULT_CONFIG

    def train_from_iterator(
        self,
        iterator: Iterable[Union[str, List[str]]],
        save_artifacts: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tokenizer:
        """Trains the BPE tokenizer from any iterable of texts or batches.

        Args:
            iterator: Generator or list yielding strings or list of strings.
            save_artifacts: Whether to serialize tokenizer files to disk after training.
            output_dir: Optional directory override for saving artifacts.

        Returns:
            Trained and calibrated Tokenizer instance.
        """
        # Build initial pipeline
        tokenizer = build_llama2_pipeline(self.config)

        # Configure Rust BPE Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens.all_reserved_tokens,
            show_progress=True,
        )

        logger.info(
            "Starting BPE training (target vocab_size=%d, min_frequency=%d)...",
            self.config.vocab_size,
            self.config.min_frequency,
        )
        tokenizer.train_from_iterator(iterator, trainer=trainer)

        # Normalize byte-fallback tokens so skip_special_tokens=True does not drop them
        tokenizer = _normalize_added_tokens(tokenizer)

        # Re-attach post-processor to ensure BOS/EOS behavior persists
        bos_token = self.config.special_tokens.bos_token
        bos_id = self.config.special_tokens.bos_id
        eos_token = self.config.special_tokens.eos_token
        eos_id = self.config.special_tokens.eos_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{bos_token} $A",
            pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1",
            special_tokens=[
                (bos_token, bos_id),
                (eos_token, eos_id),
            ],
        )

        if save_artifacts:
            save_tokenizer_artifacts(tokenizer, self.config, output_dir=output_dir)

        return tokenizer

    def train_from_stream(
        self,
        streamer: Optional[BilingualDataStreamer] = None,
        save_artifacts: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tokenizer:
        """Trains the tokenizer using the balanced 50/50 bilingual data streamer.

        Args:
            streamer: Optional BilingualDataStreamer instance (or created from config).
            save_artifacts: Whether to save artifacts to disk upon completion.
            output_dir: Output directory path.

        Returns:
            Trained Tokenizer.
        """
        if streamer is None:
            streamer = BilingualDataStreamer(config=self.config.data_stream)

        # Stream in batches for peak multi-threaded throughput in Rust
        batch_generator = streamer.iter_batches()
        return self.train_from_iterator(
            iterator=batch_generator,
            save_artifacts=save_artifacts,
            output_dir=output_dir,
        )
