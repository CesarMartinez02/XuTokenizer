"""Configuration module for Llama-2 style Bilingual (ES/EN) BPE Tokenizer.

Defines all hyper-parameters, token layouts, special token registries,
and dataset streaming settings according to the agreed Llama-2 specification:
- Vocabulary Size: 32,000 tokens
- Byte-Fallback: 256 byte tokens (<0x00> .. <0xFF>)
- Metaspace prefix ('\u2581')
- Isolated single digits (split_digits = True)
- 50/50 Spanish & English dataset streaming
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpecialTokensConfig:
    """Special tokens mapping and ID reservations.

    Llama-2 standard IDs:
      0: <unk> (Unknown token)
      1: <s>   (Beginning of Sequence / BOS)
      2: </s>  (End of Sequence / EOS)
      3: <pad> (Padding token)

    Reserved chat/instruct IDs:
      4: <|system|>
      5: <|user|>
      6: <|assistant|>
      7: [INST]
      8: [/INST]
      9: <|end|>

    Byte-fallback IDs:
      10..265: <0x00> through <0xFF>
    """

    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"

    chat_tokens: List[str] = field(
        default_factory=lambda: [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "[INST]",
            "[/INST]",
            "<|end|>",
        ]
    )

    unk_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 3

    @property
    def base_tokens(self) -> List[str]:
        """Returns foundational control tokens [unk, bos, eos, pad]."""
        return [self.unk_token, self.bos_token, self.eos_token, self.pad_token]

    @property
    def byte_tokens(self) -> List[str]:
        """Returns the 256 byte fallback tokens formatted as <0x00> to <0xFF>."""
        return [f"<0x{i:02X}>" for i in range(256)]

    @property
    def all_reserved_tokens(self) -> List[str]:
        """Returns all pre-allocated special and byte tokens in strict ID order.

        Order:
          0..3: Base tokens (unk, bos, eos, pad)
          4..9: Chat / instruct tokens
          10..265: Byte tokens (<0x00>..<0xFF>)
        Total reserved before BPE merges = 4 + 6 + 256 = 266 tokens.
        """
        return self.base_tokens + self.chat_tokens + self.byte_tokens

    @property
    def token_to_id(self) -> Dict[str, int]:
        """Builds lookup mapping from token string to reserved integer ID."""
        return {token: idx for idx, token in enumerate(self.all_reserved_tokens)}

    @property
    def id_to_token(self) -> Dict[int, str]:
        """Builds lookup mapping from integer ID to token string."""
        return {idx: token for idx, token in enumerate(self.all_reserved_tokens)}

    @property
    def num_reserved_tokens(self) -> int:
        """Total count of reserved tokens prior to BPE merges."""
        return len(self.all_reserved_tokens)


@dataclass
class DataStreamConfig:
    """Configuration for streaming bilingual training data without local corpus."""

    # Spanish streaming source
    es_dataset_name: str = "wikimedia/wikipedia"
    es_dataset_config: str = "20231101.es"
    es_text_field: str = "text"

    # English streaming source
    en_dataset_name: str = "wikimedia/wikipedia"
    en_dataset_config: str = "20231101.en"
    en_text_field: str = "text"

    # Interleave balancing
    es_ratio: float = 0.5
    en_ratio: float = 0.5

    # Streaming parameters
    batch_size: int = 1_000
    target_sample_count: int = 50_000  # Number of articles/chunks to stream
    min_text_length_chars: int = 50   # Discard snippets shorter than 50 chars
    seed: int = 42
    show_progress: bool = True

    # Local filesystem options (optional overrides for offline / private data)
    local_es_files: Optional[List[str]] = None
    local_en_files: Optional[List[str]] = None
    local_dir: Optional[Path] = None


@dataclass
class TokenizerConfig:
    """Master configuration for the Llama-2 style Bilingual Tokenizer."""

    vocab_size: int = 32_000
    min_frequency: int = 2
    metaspace_char: str = "\u2581"  # ' '
    add_prefix_space: bool = True
    split_digits: bool = True
    byte_fallback: bool = True

    special_tokens: SpecialTokensConfig = field(default_factory=SpecialTokensConfig)
    data_stream: DataStreamConfig = field(default_factory=DataStreamConfig)

    # Output artifact directory
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts"
    )
    model_name: str = "llama2_bilingual_32k"

    def __post_init__(self):
        if self.vocab_size <= self.special_tokens.num_reserved_tokens:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be greater than reserved tokens "
                f"({self.special_tokens.num_reserved_tokens})."
            )
        # Ensure output directory path is a Path instance
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    @property
    def num_bpe_merges(self) -> int:
        """Computes the exact number of BPE merge operations to train."""
        return self.vocab_size - self.special_tokens.num_reserved_tokens

    @property
    def tokenizer_json_path(self) -> Path:
        """Standard HuggingFace fast tokenizer file."""
        return self.output_dir / f"{self.model_name}.json"

    @property
    def tokenizer_config_path(self) -> Path:
        """Tokenizer metadata and special tokens configuration."""
        return self.output_dir / "tokenizer_config.json"


# Default singleton instance for easy import
DEFAULT_CONFIG = TokenizerConfig()
