.PHONY: build check fix download run

build:
	@echo "installing dependencies"
	@uv sync
	@echo "installing dependencies"
	@cd deps/depth-anything-3 && uv sync

check:
	@echo "running code check"
	@ruff check .
	@ruff format --check .
	@ty check

fix:
	@echo "running code fix"
	@ruff check --fix .
	@ruff format .

download:
	@echo "download model weights"
	@uv run src/download.py

run:
	@echo "running application"
	@uv run src/main.py
