from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor

from model import ModelArgs, Transformer


class LLaMA:
    def __init__(
        self,
        model: Transformer,
        tokenzier: SentencePieceProcessor,
        model_args: ModelArgs,
    ) -> None:
        self.model = model
        self.tokenzier = tokenzier
        self.args = ModelArgs

    @staticmethod 
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, device: str):
        
        