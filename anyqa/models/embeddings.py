import logging

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from anyqa.constants import HF_MODEL_PATH, DEVICE

logger = logging.getLogger(__name__)


class Embeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_kwargs = {"device": DEVICE}
        self.cache_folder = str(HF_MODEL_PATH)

    def get_embedding_function(self):
        logger.info(f"Using embedding model {self.model_name}")
        embedding_function = HuggingFaceEmbeddings(
            model_name=self.model_name, model_kwargs=self.model_kwargs, cache_folder=self.cache_folder, show_progress=True
        )
        return embedding_function
