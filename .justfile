default:
    @just --list

format:
    poetry run ruff format anyqa tests

lint:
    poetry run ruff check anyqa tests

test:
    poetry run pytest tests

pre-mr:
    just format
    just lint
    just test
    