COMPOSE_FILE ?= docker-compose.yml
DOCKER_COMPOSE = docker compose -f $(COMPOSE_FILE)

.PHONY: up down logs migrate seed
up:
$(DOCKER_COMPOSE) up -d --build

down:
$(DOCKER_COMPOSE) down

logs:
$(DOCKER_COMPOSE) logs -f

migrate:
$(DOCKER_COMPOSE) run --rm migrate

seed:
$(DOCKER_COMPOSE) run --rm api python -m app.seed
