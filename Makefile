.PHONY: dev-build dev-check run view

CONTAINER_RUNTIME = docker compose
CONTAINER_EXEC = $(CONTAINER_RUNTIME) run --rm app

dev-build:
	@echo "building unified runtime container"
	@$(CONTAINER_RUNTIME) build app

dev-check:
	@echo "running ruff check"
	@$(CONTAINER_EXEC) ruff check

run:
	@echo "running application"
	@$(CONTAINER_EXEC) python src/main.py

view:
	@echo "running viewer"
	@$(CONTAINER_EXEC) ./deps/viewer/viewer $(ARGS)
