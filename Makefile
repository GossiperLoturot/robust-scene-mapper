.PHONY: build check download run

build:
	@echo "installing dependencies"
	@uv sync
	@echo "installing dependencies"
	@cd deps/depth-anything-3 && uv sync
	@echo "build cubic-segmentation"
	@cd deps/cubic-segmentation && cargo build --release

check:
	@echo "running ruff check"
	@uv run ruff check

download:
	@echo "download model weights"
	@uv run src/download.py

run:
	@echo "running application"
	@uv run src/main.py
