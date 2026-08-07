.PHONY: build check download run

build:
	@echo "installing dependencies"
	@uv sync
	@echo "installing dependencies"
	@cd deps/depth-anything-3 && uv sync

check:
	@echo "running ruff check"
	@uv run ruff check

download:
	@echo "download model weights"
	@uv run src/download.py

run:
	@echo "running application"
	@uv run src/main.py
