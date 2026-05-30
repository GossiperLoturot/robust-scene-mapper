.PHONY: dev-build dev-check run view

CONTAINER_RUNTIME = docker compose
CONTAINER_EXEC = $(CONTAINER_RUNTIME) run --rm app

dev-build:
	@echo "building unified runtime container"
	@$(CONTAINER_RUNTIME) build app
	@echo "build cubic-segmentation"
	@$(CONTAINER_EXEC) bash -lc "cd deps/cubic-segmentation && cargo build --release"
	@echo "build viewer"
	@$(CONTAINER_EXEC) bash -lc "cd deps/viewer && cargo build --release"

dev-check:
	@echo "running ruff check"
	@$(CONTAINER_EXEC) ruff check

run:
	@echo "running application"
	@$(CONTAINER_EXEC) python src/main.py

view:
	@echo "running viewer"
	@$(CONTAINER_EXEC) ./deps/viewer/viewer $(ARGS)
