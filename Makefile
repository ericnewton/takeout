default: install test check

install:
	uv pip install -e .

test:
	uv run pytest

check:  sync
	uv tool run ty check  --color=never $(wildcard src/*/*.py)
	/bin/true uv tool run pyrefly check $(wildcard src/*/*.py)
	uv tool run ruff check --output-format pylint  $(wildcard src/*/*.py)

format:
	uv format

sync:
	uv sync

clean:
	uv cache clean
	rm -rf ./.venv *.log *~ ./geodata
