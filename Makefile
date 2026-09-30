.PHONY: build check fix run_download run_tagging run_recoord run_post

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

run_tagging:
	@uv run src/run_tagging.py

run_recoord:
	@uv run src/run_recoord.py

run_post:
	@uv run src/run_post.py
