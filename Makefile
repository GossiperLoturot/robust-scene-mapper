.PHONY: dev-build dev-check run view

dev-build:
	@echo "installing dependencies"
	@uv sync
	@echo "build cubic-segmentation"
	@cd deps/cubic-segmentation && cargo build --release
	@echo "build viewer"
	@cd deps/viewer && cargo build --release
	@echo "check docker engine availability"
	@docker version

dev-check:
	@echo "running ruff check"
	@uv run ruff check

run:
	@echo "running application"
	@uv run src/main.py

view:
	@echo "running viewer"
	@./deps/viewer/viewer $(ARGS)
