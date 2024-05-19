from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

SPLITTER_MAP = {
    ".html": lambda **kwargs: RecursiveCharacterTextSplitter.from_language(language=Language.HTML, **kwargs),
    ".md": lambda **kwargs: RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, **kwargs),
    ".py": lambda **kwargs: RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, **kwargs),
    ".txt": RecursiveCharacterTextSplitter,
    ".pdf": RecursiveCharacterTextSplitter,
    ".docx": RecursiveCharacterTextSplitter,
    ".doc": RecursiveCharacterTextSplitter,
}


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, docs: list[Document]):
        """Chunk documents into smaller subsections"""
        chunks = []
        for doc in docs:
            # Select a splitter based on file extension
            extension = doc.metadata["extension"]
            splitter = SPLITTER_MAP.get(extension, None)

            # Split doc into chunks
            doc_chunks = [doc]
            kwargs = {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}
            if splitter:
                doc_chunks = splitter(**kwargs).split_documents(documents=doc_chunks)
            chunks.extend(doc_chunks)

        return chunks
