#!/bin/bash

# Zibtek Chatbot Docker Setup Script
echo "🚀 Setting up Zibtek Chatbot with Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "⚠️  Please edit .env file with your API keys before running:"
        echo "   - SUPABASE_URL"
        echo "   - SUPABASE_ANON_KEY" 
        echo "   - PINECONE_API_KEY"
        echo "   - OPENAI_API_KEY"
        echo ""
    else
        echo "❌ .env.example not found. Please create .env manually."
        exit 1
    fi
fi

# Build and start services
echo "🔨 Building and starting Docker services..."
echo "This may take a few minutes on first run..."

# Use docker compose if available, fallback to docker-compose
if command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

$DOCKER_COMPOSE_CMD -f infra/docker/docker-compose.yml up --build -d

# Check if services started successfully
echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)

if [ "$HEALTH_CHECK" = "200" ]; then
    echo "✅ Services started successfully!"
    echo ""
    echo "🌐 Access URLs:"
    echo "   Chat Interface: http://localhost:8501"
    echo "   API Docs:       http://localhost:8000/docs"
    echo "   Health Check:   http://localhost:8000/health"
    echo ""
    echo "🔐 Default access code: zibtek-demo-2024"
    echo ""
    echo "📊 View logs: $DOCKER_COMPOSE_CMD -f infra/docker/docker-compose.yml logs -f"
    echo "🛑 Stop services: $DOCKER_COMPOSE_CMD -f infra/docker/docker-compose.yml down"
else
    echo "⚠️  Services may still be starting. Check logs if needed:"
    echo "$DOCKER_COMPOSE_CMD -f infra/docker/docker-compose.yml logs"
fi