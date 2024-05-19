import os
import logging
import re

from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders.sitemap import SitemapLoader


logger = logging.getLogger(__name__)

DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


class DirectoryDocumentLoader:
    """Instance of a directory document loader."""

    def __init__(self, path: str, depth: int, pattern: list[str]):
        """Initialize."""
        self.path = path
        self.depth = depth
        self.pattern = pattern

    def load(self):
        docs = self.get_path_documents()
        return docs

    def load_single_document(self, file_path: str) -> Document:
        # Loads a single document from a file path
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
        else:
            raise ValueError("Document type is undefined")
        doc = loader.load()[0]
        doc.metadata["extension"] = file_extension
        return doc

    def get_path_documents(self) -> list[Document]:
        # Loads all documents from the source documents directory, including nested folders
        paths = []
        abs_path = os.path.abspath(self.path)
        for root, _, files in os.walk(self.path):
            if root[len(abs_path) :].count(os.sep) < self.depth or self.depth < 0:
                for file_name in files:
                    if any([re.match(pat, file_name) for pat in self.pattern]):
                        logger.info("Importing: " + file_name)
                        file_extension = os.path.splitext(file_name)[1]
                        source_file_path = os.path.join(root, file_name)
                        if file_extension in DOCUMENT_MAP.keys():
                            paths.append(source_file_path)

        docs = []
        for path in paths:
            docs.append(self.load_single_document(path))

        return docs


class WebDocumentLoader:
    """Instance of a web document loader."""

    def __init__(self, url: str, depth: int, pattern: list[str]):
        """Initialize."""
        self.url = url
        self.depth = 1000 if depth == -1 else depth
        self.pattern = pattern

    def load(self) -> list[Document]:
        """Load documents."""
        loader = SitemapLoader(web_path=self.url, restrict_to_same_domain=True, filter_urls=self.pattern)
        loader.requests_per_second = 2
        documents = loader.load()
        for doc in documents:
            doc.metadata["extension"] = ".html"
        return documents
