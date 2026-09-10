.PHONY: build check fix run_download run_reconstruct run_surface run_lifting

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

run_reconstruct:
	@uv run src/run_reconstruct.py

run_surface:
	@uv run src/run_surface.py

run_lifting:
	@uv run src/run_lifting.py
