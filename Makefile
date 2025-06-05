# LoRA Easy Training Scripts Backend - Docker Management
.PHONY: help build up down logs shell clean restart dev prod

# Default target
help:
	@echo "Available commands:"
	@echo "  build      - Build Docker images"
	@echo "  up         - Start services"
	@echo "  down       - Stop services"
	@echo "  logs       - View logs"
	@echo "  shell      - Get shell access to the container"
	@echo "  clean      - Clean up containers and images"
	@echo "  restart    - Restart services"
	@echo "  dev        - Start development environment"
	@echo "  prod       - Start production environment"
	@echo "  health     - Check container health"
	@echo "  sync       - Sync dependencies with uv"

# Environment setup
include env.example
export

# Get current user ID
UID := $(shell id -u)
export UID

# Build Docker images
build:
	@echo "Building Docker images..."
	docker-compose build --parallel

# Start services
up:
	@echo "Starting services..."
	docker-compose up -d

# Stop services
down:
	@echo "Stopping services..."
	docker-compose down

# View logs
logs:
	docker-compose logs -f lora-backend

# Get shell access
shell:
	docker-compose exec lora-backend /bin/zsh

# Clean up containers and images
clean:
	@echo "Cleaning up containers and images..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

# Restart services
restart: down up

# Start development environment (with Jupyter and TensorBoard)
dev:
	@echo "Starting development environment..."
	docker-compose --profile development --profile monitoring up -d

# Start production environment
prod:
	@echo "Starting production environment..."
	docker-compose up -d --scale jupyter=0 --scale tensorboard=0

# Check container health
health:
	@echo "Checking container health..."
	docker-compose ps
	@echo "\nContainer health status:"
	docker inspect --format='{{.State.Health.Status}}' lora-backend 2>/dev/null || echo "Health check not available"

# Sync dependencies with uv
sync:
	@echo "Syncing dependencies with uv..."
	docker-compose exec lora-backend uv sync

# Update dependencies
update:
	@echo "Updating dependencies..."
	docker-compose exec lora-backend uv sync --upgrade

# Run tests
test:
	@echo "Running tests..."
	docker-compose exec lora-backend uv run pytest

# Format code
format:
	@echo "Formatting code..."
	docker-compose exec lora-backend uv run black .
	docker-compose exec lora-backend uv run isort .

# Lint code
lint:
	@echo "Linting code..."
	docker-compose exec lora-backend uv run flake8 .

# Show resource usage
stats:
	docker stats lora-backend --no-stream

# Backup volumes
backup:
	@echo "Creating backup of volumes..."
	mkdir -p backups
	docker run --rm -v lora-easy-training-scripts-backend_datasets:/data -v $(PWD)/backups:/backup ubuntu tar czf /backup/datasets-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	docker run --rm -v lora-easy-training-scripts-backend_outputs:/data -v $(PWD)/backups:/backup ubuntu tar czf /backup/outputs-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .

# Show GPU usage
gpu:
	docker-compose exec lora-backend nvidia-smi

# Interactive Python shell
python:
	docker-compose exec lora-backend uv run python

# Generate uv.lock file
lock:
	@echo "Generating uv.lock file..."
	docker-compose exec lora-backend uv lock

# Install new package
install:
	@if [ -z "$(PACKAGE)" ]; then \
		echo "Usage: make install PACKAGE=package_name"; \
	else \
		docker-compose exec lora-backend uv add $(PACKAGE); \
	fi