import logging
import os
import pathlib

import torch

logger = logging.getLogger(__name__)

DIRECTORY_PATH = pathlib.Path(os.path.dirname(__file__)).parent

HF_MODEL_PATH = DIRECTORY_PATH / "hf_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSIST_DIRECTORY = DIRECTORY_PATH / "db"

CONFIG_FILE = DIRECTORY_PATH / "config" / "config.yaml"


# Defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_LLM = "gemma:2b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-miniLM-L6-v2"
DEFAULT_PERSONA_NAME = "default"
DEFAULT_PERSONA_TEMPLATE = """You are a helpful assistant. Answer the user's question based on the
context below. If you do not know the answer, say "I don't know".

Context:

{context}

Question: {question}

Answer:
"""
