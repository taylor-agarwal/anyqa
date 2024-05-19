from langchain_community.llms.ollama import Ollama
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from anyqa.models.vector_db import ChromaDB


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


class Persona:
    def __init__(self, name: str, template: str):
        self.name = name
        self.template = template

    def to_dict(self):
        return {"name": self.name, "template": self.template}


class RAG:
    def __init__(self, collection: ChromaDB, persona: Persona, model_name: str, verbose: bool = False, search_kwargs: dict | None = None):
        self.collection = collection
        self.persona = persona
        self.model_name = model_name

        self.retriever = self.collection.as_retriever(search_kwargs=search_kwargs)
        self.template = self.persona.template
        self.prompt = PromptTemplate.from_template(template=self.template)
        self.llm = Ollama(model=model_name, verbose=verbose)

        rag_chain_from_docs = RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | self.prompt | self.llm | StrOutputParser()

        self.rag = RunnableParallel({"context": self.retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

    def query(self, question: str):
        result = self.rag.invoke(question)
        response, sources = result["answer"], result["context"]
        return response, sources
