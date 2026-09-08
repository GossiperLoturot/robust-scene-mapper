.PHONY: build check fix download run

build:
	@uv sync
	@cd deps/depth-anything-3 && uv sync

check:
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run ty check .

fix:
	@uv run ruff check --fix .
	@uv run ruff format .

download:
	@uv run src/download.py

run:
	@uv run src/main.py
