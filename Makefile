.PHONY: help dev-back dev-front prod down install clean

help: 
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  make dev-back  - Start postgres + backend (recommended: use separate terminal)"
	@echo "  make dev-front - Start frontend only (recommended: use separate terminal)"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start everything in Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  make down         - Stop all services (keeps data/volumes)"
	@echo "  make clean        - Stop all services and remove volumes (deletes all data)"
	@echo "  make install        - Install all dependencies"


# Setup
install: ## Install all dependencies
	@echo "Installing backend dependencies..."
	@cd backend && uv sync
	@echo "Installing frontend dependencies..."
	@cd frontend && npm install
	@echo "✓ Setup complete!"


# Development - Backend (starts postgres if needed, then backend)
dev-back: ## Start postgres + backend
	@echo "Starting postgres..."
	@docker compose up -d postgres
	@echo "Waiting for postgres to be ready..."
	@sleep 3
	@docker compose ps postgres
	@echo "✓ Postgres is ready!"
	@echo "Starting backend..."
	@cd backend && uv run python main.py

# Development - Frontend
dev-front: ## Start frontend only
	@echo "Starting frontend..."
	@cd frontend && npm run dev


# Shutdown
down: ## Stop all services (keeps data/volumes)
	@echo "Stopping all services..."
	@docker compose down
	@pkill -f "python main.py" 2>/dev/null || true
	@pkill -f "vite" 2>/dev/null || true
	@echo "✓ All services stopped"


# Cleanup
clean: ## Stop services and remove volumes (deletes all data)
	@echo "Stopping all services and removing volumes..."
	@docker compose down -v
	@pkill -f "python main.py" 2>/dev/null || true
	@pkill -f "vite" 2>/dev/null || true
	@echo "✓ Cleanup complete (all data removed)"


# Production
prod: ## Start everything in production mode (Docker)
	@echo "Starting production environment..."
	docker compose -f docker compose.yml up --build
