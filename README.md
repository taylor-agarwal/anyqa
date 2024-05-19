# AnyQA

Questions and answers for any document.


## Getting Started
1. [Install Ollama](https://ollama.com/)
2. Pull your preferred model with `ollama pull`. We recommend `gemma:2b` for most systems.
3. Clone this repo, then install dependencies with poetry.


## Usage
```bash
$ poetry run python anyqa/cli.py --help
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  config
  list
  load
  query
  remove
  setup

```

### Setup
```bash
$ poetry run python anyqa/cli.py setup
```
Setup the directory and create the config.yaml file.

### Config
```bash
$ poetry run python anyqa/cli.py config --help
Usage: cli.py config [OPTIONS]

Options:
  --embedding-model TEXT  Default HuggingFace model to use for embedding.
  --chunk-size TEXT       Size of chunks to store in embeddings
  --chunk-overlap TEXT    Overlap between chunks to store in embeddings
  --llm TEXT              Default LLM to use. Ensure that the model has been
                          pulled with `ollama pull`
  --help                  Show this message and exit.
```
Update configuration items.

### Load
```bash
$ poetry run python anyqa/cli.py load --help
Usage: cli.py load [OPTIONS]

Options:
  --dir TEXT              Local directory to load from. Must be defined if
                          --web option is not used.
  --web TEXT              Web sitemap.xml to load from. Must be defined if
                          --dir option is not used.
  --embedding-model TEXT  HuggingFace model to use.
  --collection TEXT       Chroma collection name.  [default: default]
  --depth INTEGER         Recursive search depth. When -1, search will extend
                          to maximum recursion depth.  [default: -1]
  -p, --pattern TEXT      URL/Path regex patterns to match on.
  --help                  Show this message and exit.

```
Load documents into vectorstore collection

### List
```bash
$ poetry run python anyqa/cli.py list
```
List collections, including size and metadata.

### Remove
```bash
$ poetry run python anyqa/cli.py remove --help
Usage: cli.py remove [OPTIONS] COLLECTION

Options:
  --where TEXT  Where condition for the delete operation. Must be JSON
                formatted. See https://docs.trychroma.com/guides#using-where-
                filters for more details
  --keep        If present, keep collection
  --help        Show this message and exit.
```
Remove records from a collection or remove a collection entirely.

### Query
```bash
$ poetry run python anyqa/cli.py query --help
Usage: cli.py query [OPTIONS] QUESTION

Options:
  --collection TEXT  Chroma collection name.  [default: default]
  --persona TEXT     Persona name to use.  [default: default]
  --llm TEXT         LLM to use. Ensure that the model has been pulled with
                     `ollama pull`
  -k INTEGER         Number of documents to retrieve per query.  [default: 3]
  -v, --verbose      Increase verboseness
  --help             Show this message and exit.
```
Pose a question against a collection and get an answer from the sources.
