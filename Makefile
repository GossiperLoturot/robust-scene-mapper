.PHONY: dev-build dev-check download run view

dev-build:
	@echo "installing dependencies"
	@uv sync
	@echo "build cubic-segmentation"
	@cd deps/cubic-segmentation && cargo build --release
	@echo "build viewer"
	@cd deps/viewer && cargo build --release

dev-check:
	@echo "running ruff check"
	@uv run ruff check

download:
	@echo "download model weights"
	@uv run src/download.py

run:
	@echo "running application"
	@uv run src/main.py

view:
	@echo "running viewer"
	@./deps/viewer/viewer $(ARGS)
