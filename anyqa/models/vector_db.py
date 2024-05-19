import logging

import chromadb
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
import hashlib

from anyqa.models.embeddings import Embeddings
from anyqa.constants import PERSIST_DIRECTORY


logger = logging.getLogger(__name__)


class ChromaDB:
    """Instance of a ChromaDB database."""

    def __init__(self, collection_name: str | None = None, embedding_model: str | None = None):
        """Initialize ChromaDB object."""
        self.persist_directory = str(PERSIST_DIRECTORY)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection_name = collection_name
        self.collection = None
        self.embeddings = None
        self.embedding_function = None
        self.db = None

        if self.collection_name is not None:
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.embedding_model = self.collection.metadata["embedding_model"]
                logger.info(f"Using existing collection '{self.collection_name}'")
            except ValueError:
                if embedding_model is None:
                    raise ValueError(f"Collection: {self.collection_name} does not exist. You must create a collection with 'load' first.")
                self.embedding_model = embedding_model
                metadata = {"embedding_model": self.embedding_model}
                self.collection = self.client.create_collection(name=self.collection_name, metadata=metadata)
                logger.info(f"Created collection '{self.collection_name}'")

            self.embeddings = Embeddings(model_name=self.embedding_model)
            self.embedding_function = self.embeddings.get_embedding_function()
            self.db = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
            )

    def as_retriever(self, search_kwargs: dict | None = None):
        """Create a Langchain retriever object for the collection."""
        return self.db.as_retriever(search_kwargs=search_kwargs)

    def get_ids(self, where: dict | None = None):
        result = self.collection.get(where=where)
        ids = result["ids"]
        return ids

    def delete_where(self, where: dict):
        """Delete records from the collection meeting some conditions."""
        ids = self.get_ids(where=where)
        self.collection.delete(where=where)
        return ids

    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except ValueError:
            return False

    def delete_all_records(self):
        """Delete all records from the collection, but keep the collection."""
        ids = self.get_ids()
        self.collection.delete(ids=ids)
        return ids

    def load_documents(self, documents: list[Document]):
        """Load documents into the collection."""
        # Hash each document into an id to prevent duplication
        ids = []
        docs = []
        for doc in documents:
            h = hashlib.sha256()
            content = doc.page_content
            h.update(content.encode("UTF-8"))
            source = doc.metadata["source"]
            h.update(source.encode("UTF-8"))
            id = h.hexdigest()
            if id not in ids:
                ids.append(id)
                docs.append(doc)
        # Save documents
        saved_ids = self.db.add_documents(documents=docs, ids=ids)
        return saved_ids
