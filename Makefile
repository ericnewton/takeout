default: install test check

install:
	uv pip install -e .

test:
	uv run pytest

COMMENT=true

check:  sync
	uv tool run ty check  --color=never $(wildcard src/*/*.py)
	$(COMMENT) uvx mypy $(wildcard src/*/*.py)
	$(COMMENT) uvx pyrefly check $(wildcard src/*/*.py)
	uv tool run ruff check --output-format pylint  $(wildcard src/*/*.py)

format:
	uv format

sync:
	uv sync

clean:
	uv cache clean
	rm -rf ./.venv *.log *~ ./geodata
