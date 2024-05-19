import click
import logging
import json

from anyqa.models import ChromaDB, WebDocumentLoader, DirectoryDocumentLoader, Chunker, Config, RAG, Persona
from anyqa.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM,
    DEFAULT_PERSONA_NAME,
    DEFAULT_PERSONA_TEMPLATE,
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
def setup():
    # Create the config file
    config = Config()

    # Add default config items to Config
    config.chunk_size = DEFAULT_CHUNK_SIZE
    config.chunk_overlap = DEFAULT_CHUNK_OVERLAP
    config.default_embedding_model = DEFAULT_EMBEDDING_MODEL
    config.default_llm = DEFAULT_LLM
    config.personas = [Persona(name=DEFAULT_PERSONA_NAME, template=DEFAULT_PERSONA_TEMPLATE)]

    # Save config
    config.save()


@cli.command("config")
@click.option("--embedding-model", default=None, help="Default HuggingFace model to use for embedding.")
@click.option("--chunk-size", default=None, help="Size of chunks to store in embeddings")
@click.option("--chunk-overlap", default=None, help="Overlap between chunks to store in embeddings")
@click.option("--llm", default=None, help="Default LLM to use. Ensure that the model has been pulled with `ollama pull`")
def update_config(embedding_model: str, chunk_size: int, chunk_overlap: int, llm: str):
    # Load config from config file
    config = Config()
    config.load()

    # Make config changes
    if embedding_model:
        config.default_embedding_model = embedding_model
    if chunk_size:
        config.chunk_size = chunk_size
    if chunk_overlap:
        config.chunk_overlap = chunk_overlap
    if llm:
        config.default_llm = llm

    # Save new config
    config.save()


@cli.command("load")
@click.option("--dir", default=None, help="Local directory to load from. Must be defined if --web option is not used.")
@click.option("--web", default=None, help="Web sitemap.xml to load from. Must be defined if --dir option is not used.")
@click.option("--embedding-model", default=None, help="HuggingFace model to use.")
@click.option("--collection", default="default", help="Chroma collection name.", show_default=True)
@click.option("--depth", default=-1, help="Recursive search depth. When -1, search will extend to maximum recursion depth.", show_default=True)
@click.option("-p", "--pattern", multiple=True, default=[".*"], help="URL/Path regex patterns to match on.")
def load(dir: str, web: str, embedding_model: str, collection: str, depth: int, pattern: list[str]):
    # Load config from config file
    config = Config()
    config.load()

    # Load documents from address
    if web is not None:
        loader = WebDocumentLoader(url=web, depth=depth, pattern=pattern)
    elif dir is not None:
        loader = DirectoryDocumentLoader(path=dir, depth=depth, pattern=pattern)
    else:
        raise ValueError("One of --dir or --web must be defined.")
    documents = loader.load()

    # Chunk documents
    chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = chunker.chunk_documents(docs=documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

    # Save chunks to DB
    if embedding_model is None:
        embedding_model = config.default_embedding_model
    db = ChromaDB(embedding_model=embedding_model, collection_name=collection)
    _ = db.load_documents(documents=chunks)
    logger.info(f"Successfully loaded {len(documents)} documents ({len(chunks)} chunks) into collection {collection}")

    config.save()


@cli.command("remove")
@click.argument("collection")
@click.option(
    "--where",
    default=None,
    help="Where condition for the delete operation. Must be JSON formatted. See https://docs.trychroma.com/guides#using-where-filters for more details",
)
@click.option("--keep", is_flag=True, default=False, help="If present, keep collection", show_default=True)
def remove(collection: str, where: str | None, keep: bool):
    # Load config from config file
    config = Config()
    config.load()

    db = ChromaDB(collection_name=collection)
    if where is not None:
        where = json.loads(where)
        db.delete_where(where)
        logger.info(f"Successfully deleted records from '{collection}' where {where}")
    else:
        if collection != "default" and not keep:
            # TODO: Remove collections from config
            deleted = db.delete_collection()
            if deleted:
                logger.info(f"Successfully deleted collection '{collection}'")
            else:
                logger.info(f"Collection '{collection}' does not exist. Exiting...")
        else:
            db.delete_all_records()
            logger.info(f"Successfully deleted all records from collection '{collection}'")


@cli.command("list")
def list_collections():
    # Load config from config file
    config = Config()
    config.load()

    # Connect to db
    db = ChromaDB()
    collections = db.client.list_collections()

    for collection in collections:
        logger.info(f"{collection.name}: {collection.count()} Embeddings")
        logger.info(f"\tMetadata: {collection.metadata}")
        result = collection.get()
        metadatas = result["metadatas"]
        sources = {m["source"] for m in metadatas}
        logger.info(f"\tSources: {sources}")


@cli.command("query")
@click.argument("question")
@click.option("--collection", default="default", help="Chroma collection name.", show_default=True)
@click.option("--persona", default="default", help="Persona name to use.", show_default=True)
@click.option("--llm", default=None, help="LLM to use. Ensure that the model has been pulled with `ollama pull`")
@click.option("-k", default=3, help="Number of documents to retrieve per query.", show_default=True)
@click.option("-v", "--verbose", count=True, help="Increase verboseness")
def query(question: str, collection: str, persona: str, llm: str, k: int, verbose: bool):
    # Load config from config file
    config = Config()
    config.load()

    # Connect to db
    db = ChromaDB(collection_name=collection)

    # Create RAG
    _persona = [p for p in config.personas if p.name == persona][0]
    if llm is None:
        llm = config.default_llm
    search_kwargs = {"k": k}
    rag = RAG(collection=db, persona=_persona, model_name=llm, verbose=verbose, search_kwargs=search_kwargs)
    response, sources = rag.query(question=question)
    source_names = [source.metadata.get("loc") if source.metadata.get("loc") else source.metadata.get("source") for source in sources]
    logger.info(f"Question: {question}")
    logger.info(f"Response: {response}")
    logger.info(f"Sources: {source_names}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cli()
