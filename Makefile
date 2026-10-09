.PHONY: build check fix run_download prepare_tagging prepare_coord prepare_annotation run_pack

build:
	@uv sync

check:
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run ty check .

fix:
	@uv run ruff check --fix .
	@uv run ruff format .

run_download:
	@uv run src/run_download.py

prepare_tagging:
	@uv run src/prepare_tagging.py

prepare_coord:
	@uv run src/prepare_coord.py

prepare_annotation:
	@uv run src/prepare_annotation.py

run_pack:
	@uv run src/run_pack.py
