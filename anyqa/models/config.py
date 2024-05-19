import yaml

from anyqa.constants import CONFIG_FILE
from anyqa.models.query import Persona

import logging


logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        """Initialize."""
        self.chunk_size: int = None
        self.chunk_overlap: int = None
        self.personas: list[Persona] = []
        self.default_llm: str = None

    def load(self):
        """Load configuration."""
        config = yaml.safe_load(CONFIG_FILE.read_text())

        self.chunk_size = config["chunking"]["chunk-size"]
        self.chunk_overlap = config["chunking"]["chunk-overlap"]
        self.personas = [Persona(**persona) for persona in config["personas"]]
        self.default_llm = config["defaults"]["llm"]
        self.default_embedding_model = config["defaults"]["embedding-model"]

        logger.info(f"Successfully loaded config from {CONFIG_FILE}")

        return config

    def to_dict(self):
        return {
            "chunking": {
                "chunk-size": self.chunk_size,
                "chunk-overlap": self.chunk_overlap,
            },
            "personas": [persona.to_dict() for persona in self.personas],
            "defaults": {"llm": self.default_llm, "embedding-model": self.default_embedding_model},
        }

    def save(self):
        """Save configuration items."""
        config_dict = self.to_dict()
        with open(CONFIG_FILE, "w") as f:
            yaml.safe_dump(config_dict, f)
        logger.info(f"Successfully saved config to {CONFIG_FILE}")
