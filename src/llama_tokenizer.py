"""High-level LLM Tokenizer Wrapper for Llama-2 style models.

Provides a production-ready, PyTorch-compatible tokenizer interface
supporting:
- Fast encoding and decoding (powered by Hugging Face Rust tokenizers)
- Multi-threaded batched tokenization via raw_tokenizer.encode_batch
- Dynamic batched tokenization with attention masks
- Left and right padding strategies (crucial for batched generation vs. training)
- Sequence truncation with configurable max_length
- Aligned chat formatting via apply_chat_template
- Serialization and loading via from_pretrained and save_pretrained
- Optional export to Hugging Face PreTrainedTokenizerFast
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tokenizers import Tokenizer

from .config import DEFAULT_CONFIG, TokenizerConfig
from .tokenizer_builder import save_tokenizer_artifacts

logger = logging.getLogger(__name__)


@dataclass
class BatchEncoding:
    """Container for batched tokenization outputs (input_ids and attention_mask).

    Provides dictionary-like access and PyTorch tensor conversion compatibility.
    """

    input_ids: Any
    attention_mask: Any

    def __getitem__(self, item: str) -> Any:
        if item == "input_ids":
            return self.input_ids
        elif item == "attention_mask":
            return self.attention_mask
        raise KeyError(f"BatchEncoding key not found: {item!r}. Valid keys: ['input_ids', 'attention_mask']")

    def keys(self) -> List[str]:
        return ["input_ids", "attention_mask"]

    def items(self) -> List[Tuple[str, Any]]:
        return [("input_ids", self.input_ids), ("attention_mask", self.attention_mask)]

    def to_dict(self) -> Dict[str, Any]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}

    def to(self, device: Any) -> "BatchEncoding":
        """Transfers underlying tensors to target device if PyTorch tensors are present."""
        new_ids = self.input_ids.to(device) if hasattr(self.input_ids, "to") else self.input_ids
        new_mask = self.attention_mask.to(device) if hasattr(self.attention_mask, "to") else self.attention_mask
        return BatchEncoding(input_ids=new_ids, attention_mask=new_mask)


class LlamaTokenizer:
    """Production LLM wrapper for Llama-2 bilingual BPE tokenizers."""

    def __init__(
        self,
        raw_tokenizer: Tokenizer,
        config: Optional[TokenizerConfig] = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        model_max_length: int = 4096,
        padding_side: str = "right",
        chat_template: Optional[str] = None,
    ) -> None:
        """Initializes the tokenizer wrapper.

        Args:
            raw_tokenizer: Underling Hugging Face Tokenizer instance.
            config: Optional TokenizerConfig.
            bos_token: Beginning of sequence string token.
            eos_token: End of sequence string token.
            unk_token: Out-of-vocabulary unknown token.
            pad_token: Padding string token.
            model_max_length: Maximum context length supported.
            padding_side: Direction for padding ("right" for training, "left" for autoregressive generation).
            chat_template: Jinja2 chat template string for apply_chat_template.
        """
        self.raw_tokenizer = raw_tokenizer
        self.config = config if config is not None else DEFAULT_CONFIG
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.model_max_length = model_max_length
        self.padding_side = padding_side.lower()

        if self.padding_side not in ("left", "right"):
            raise ValueError(f"padding_side must be 'left' or 'right', got {self.padding_side!r}")

        # Resolve token IDs
        self.bos_token_id = self.raw_tokenizer.token_to_id(self.bos_token)
        self.eos_token_id = self.raw_tokenizer.token_to_id(self.eos_token)
        self.unk_token_id = self.raw_tokenizer.token_to_id(self.unk_token)
        self.pad_token_id = self.raw_tokenizer.token_to_id(self.pad_token)

        # Fallback to defaults from config if not found in vocab
        if self.pad_token_id is None:
            self.pad_token_id = self.config.special_tokens.pad_id

        self.chat_template = chat_template or (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>\\n' }}"
            "{% endif %}"
        )

    @property
    def vocab_size(self) -> int:
        """Returns total vocabulary size."""
        return self.raw_tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        """Returns the full vocabulary mapping string tokens to integer IDs."""
        return self.raw_tokenizer.get_vocab()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """Encodes a single text string into a list of token IDs.

        Args:
            text: Input text string.
            add_special_tokens: Whether to prepend BOS <s>.
            max_length: Maximum number of tokens to keep.
            truncation: Whether to truncate sequence if exceeding max_length.

        Returns:
            List of integer token IDs.
        """
        encoding = self.raw_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        token_ids = encoding.ids

        if truncation and max_length is not None and len(token_ids) > max_length:
            if self.padding_side == "left":
                token_ids = token_ids[-max_length:]
            else:
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes token ID(s) back into a text string.

        Args:
            token_ids: Integer token ID, list of IDs, or PyTorch/NumPy tensor with .tolist().
            skip_special_tokens: Whether to strip BOS/EOS control tokens.

        Returns:
            Decoded Unicode text string.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.raw_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: List[str],
        padding: Union[bool, str] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding_side: Optional[str] = None,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """Tokenizes a batch of texts with padding, truncation, and attention masks.

        Args:
            texts: List of input string texts.
            padding: Padding strategy: False/None for no padding, True or "longest"
                     to pad to longest sequence in batch, or "max_length" to pad to max_length.
            max_length: Maximum target sequence length for truncation and max_length padding.
            truncation: Whether to truncate sequences exceeding max_length.
            padding_side: Direction for padding ("left" or "right"). Defaults to self.padding_side.
            add_special_tokens: Whether to prepend BOS <s>.
            return_tensors: "pt" for PyTorch tensors, or None for native Python lists.

        Returns:
            BatchEncoding containing input_ids and attention_mask.
        """
        side = (padding_side or self.padding_side).lower()
        if side not in ("left", "right"):
            raise ValueError(f"padding_side must be 'left' or 'right', got {side!r}")

        # 1. Multi-threaded encoding in Rust via raw_tokenizer.encode_batch
        encodings = self.raw_tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        raw_sequences: List[List[int]] = [enc.ids for enc in encodings]

        # 2. Sequence truncation if requested
        if truncation and max_length is not None:
            if side == "left":
                raw_sequences = [s[-max_length:] if len(s) > max_length else s for s in raw_sequences]
            else:
                raw_sequences = [s[:max_length] if len(s) > max_length else s for s in raw_sequences]

        # 3. Determine target padding length
        target_len: Optional[int] = None
        if padding is True or padding == "longest":
            target_len = max((len(seq) for seq in raw_sequences), default=0)
        elif padding == "max_length":
            if max_length is None:
                raise ValueError("max_length must be specified when padding='max_length'")
            target_len = max_length

        # 4. Apply padding and build attention masks
        padded_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        pad_id = self.pad_token_id if self.pad_token_id is not None else 3

        for seq in raw_sequences:
            if target_len is None:
                # No padding requested
                padded_ids.append(seq)
                attention_masks.append([1] * len(seq))
                continue

            pad_count = max(0, target_len - len(seq))
            if side == "left":
                p_seq = [pad_id] * pad_count + seq[:target_len]
                p_mask = [0] * pad_count + [1] * min(len(seq), target_len)
            else:  # right padding
                p_seq = seq[:target_len] + [pad_id] * pad_count
                p_mask = [1] * min(len(seq), target_len) + [0] * pad_count

            padded_ids.append(p_seq)
            attention_masks.append(p_mask)

        # 5. Optional PyTorch Tensor conversion
        if return_tensors == "pt":
            try:
                import torch

                tensor_ids = torch.tensor(padded_ids, dtype=torch.long)
                tensor_mask = torch.tensor(attention_masks, dtype=torch.long)
                return BatchEncoding(input_ids=tensor_ids, attention_mask=tensor_mask)
            except ImportError as exc:
                raise ImportError(
                    "PyTorch ('torch') is required when return_tensors='pt'. "
                    "Please install PyTorch or use return_tensors=None."
                ) from exc

        return BatchEncoding(input_ids=padded_ids, attention_mask=attention_masks)

    def batch_decode(
        self,
        sequences: List[Any],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decodes a batch of token sequences back into strings.

        Args:
            sequences: List of lists of integer token IDs (or 2D tensors).
            skip_special_tokens: Whether to strip BOS/EOS control tokens.

        Returns:
            List of decoded string texts.
        """
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> Union[str, List[int]]:
        """Formats a multi-turn conversation into a prompt string or token ID sequence.

        Args:
            conversation: List of message dictionaries with "role" and "content" keys.
                          Roles typically include "system", "user", and "assistant".
            add_generation_prompt: Whether to append the assistant start delimiter at the end.
            tokenize: If True, returns encoded token IDs; otherwise returns formatted str.

        Returns:
            Formatted prompt string or list of token IDs.
        """
        try:
            import jinja2

            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
            template = env.from_string(self.chat_template)
            rendered = template.render(
                messages=conversation,
                add_generation_prompt=add_generation_prompt,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
            )
        except ImportError:
            # Clean fallback if jinja2 is not available
            rendered_chunks: List[str] = []
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    rendered_chunks.append(f"<|system|>\n{content}<|end|>\n")
                elif role == "user":
                    rendered_chunks.append(f"<|user|>\n{content}<|end|>\n")
                elif role == "assistant":
                    rendered_chunks.append(f"<|assistant|>\n{content}<|end|>\n")

            if add_generation_prompt:
                rendered_chunks.append("<|assistant|>\n")

            rendered = "".join(rendered_chunks)

        if tokenize:
            # Special tokens are already embedded verbatim in the rendered string
            return self.encode(rendered, add_special_tokens=False)

        return rendered

    def save_pretrained(self, save_directory: Union[str, Path]) -> Dict[str, Path]:
        """Saves tokenizer artifacts (tokenizer.json, configs) to the specified directory."""
        return save_tokenizer_artifacts(
            tokenizer=self.raw_tokenizer,
            config=self.config,
            output_dir=save_directory,
        )

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, Path]) -> "LlamaTokenizer":
        """Loads a LlamaTokenizer instance from a serialized directory or file.

        Args:
            pretrained_path: Directory containing tokenizer.json, or direct path to .json file.

        Returns:
            Configured LlamaTokenizer instance.
        """
        path = Path(pretrained_path)
        if path.is_file():
            raw_tok = Tokenizer.from_file(str(path))
            return cls(raw_tokenizer=raw_tok)

        # Directory mode: locate tokenizer.json or named model json, filtering config files
        json_file = path / "tokenizer.json"
        if not json_file.exists():
            candidates = [
                p
                for p in path.glob("*.json")
                if p.name not in ("tokenizer_config.json", "special_tokens_map.json")
            ]
            if not candidates:
                raise FileNotFoundError(f"No tokenizer definition .json file found in {path}")
            json_file = candidates[0]

        raw_tok = Tokenizer.from_file(str(json_file))

        # Check for tokenizer_config.json metadata
        config_file = path / "tokenizer_config.json"
        kwargs: Dict[str, Any] = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                for key in (
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "pad_token",
                    "model_max_length",
                    "padding_side",
                    "chat_template",
                ):
                    if key in cfg_data and cfg_data[key] is not None:
                        kwargs[key] = cfg_data[key]

        return cls(raw_tokenizer=raw_tok, **kwargs)

    def to_pretrained_fast(self) -> Any:
        """Exports to a Hugging Face PreTrainedTokenizerFast instance if transformers is installed."""
        try:
            from transformers import PreTrainedTokenizerFast

            return PreTrainedTokenizerFast(
                tokenizer_object=self.raw_tokenizer,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                unk_token=self.unk_token,
                pad_token=self.pad_token,
                chat_template=self.chat_template,
                model_max_length=self.model_max_length,
                padding_side=self.padding_side,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required to export to PreTrainedTokenizerFast. "
                "Install it via 'pip install transformers'."
            ) from exc

    def __call__(
        self,
        text_or_texts: Union[str, List[str]],
        padding: Union[bool, str] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding_side: Optional[str] = None,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], BatchEncoding]:
        """Convenience callable interface mirroring Hugging Face tokenizer calling convention."""
        if isinstance(text_or_texts, str):
            return self.encode(
                text=text_or_texts,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
            )
        return self.encode_batch(
            texts=text_or_texts,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
