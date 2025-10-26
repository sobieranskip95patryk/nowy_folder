#!/bin/bash
# PinkPlayEvo SWR Deployment Script
# Moduł Świadomego Wnioskowania Resztkowego v2.0

set -e

echo "🧠 PinkPlayEvo SWR Deployment Started..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/sobieranskip95patryk/nowy_folder.git"
DEPLOY_DIR="/opt/pinkplay-swr"
BACKUP_DIR="/opt/pinkplay-swr-backup"
SERVICE_NAME="pinkplay-swr"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "Don't run this script as root for security reasons"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        error "Git is not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed"
        exit 1
    fi
    
    success "All prerequisites are installed"
}

# Backup existing deployment
backup_existing() {
    if [ -d "$DEPLOY_DIR" ]; then
        log "Creating backup of existing deployment..."
        sudo rm -rf "$BACKUP_DIR"
        sudo cp -r "$DEPLOY_DIR" "$BACKUP_DIR"
        success "Backup created at $BACKUP_DIR"
    fi
}

# Clone or update repository
update_repository() {
    log "Updating repository..."
    
    if [ -d "$DEPLOY_DIR" ]; then
        cd "$DEPLOY_DIR"
        git fetch origin
        git reset --hard origin/main
        git clean -fd
    else
        sudo mkdir -p "$DEPLOY_DIR"
        sudo chown $USER:$USER "$DEPLOY_DIR"
        git clone "$REPO_URL" "$DEPLOY_DIR"
        cd "$DEPLOY_DIR"
    fi
    
    success "Repository updated"
}

# Test SWR Core
test_swr_core() {
    log "Testing MŚWR Core system..."
    
    cd "$DEPLOY_DIR"
    
    # Test Python components
    if python3 core/mswr_v2_clean.py; then
        success "MŚWR Core tests passed"
    else
        error "MŚWR Core tests failed"
        exit 1
    fi
    
    # Test PinkPlay integration
    if python3 core/pinkplay_swr_integration.py; then
        success "PinkPlay SWR integration tests passed"
    else
        error "PinkPlay SWR integration tests failed"
        exit 1
    fi
    
    # Test Node.js wrapper
    if node swrModule.js; then
        success "Node.js SWR wrapper tests passed"
    else
        error "Node.js SWR wrapper tests failed"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    cd "$DEPLOY_DIR"
    
    # Install Node.js dependencies
    if [ -f "package.json" ]; then
        npm ci --production
        success "Node.js dependencies installed"
    fi
    
    # Check Python dependencies (all built-in)
    python3 -c "import json, time, hashlib, datetime, typing, dataclasses, enum" || {
        error "Required Python modules not available"
        exit 1
    }
    success "Python dependencies verified"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$DEPLOY_DIR"
    
    # Build SWR image
    docker build -f Dockerfile.swr -t pinkplay-swr:latest .
    
    success "Docker images built"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    cd "$DEPLOY_DIR"
    
    # Stop existing services
    docker-compose -f docker-compose.swr.yml down --remove-orphans || true
    
    # Start new services
    docker-compose -f docker-compose.swr.yml up -d
    
    success "Services deployed"
}

# Health check
health_check() {
    log "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:3000/health > /dev/null; then
            success "Health check passed"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, waiting 10s..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

# Display deployment info
display_info() {
    log "Deployment completed successfully!"
    echo ""
    echo "🧠 PinkPlayEvo SWR v2.0 is now running"
    echo ""
    echo "📡 Available endpoints:"
    echo "   Main API: http://localhost:3000"
    echo "   Health: http://localhost:3000/health"
    echo "   Analytics: http://localhost:3000/api/swr/analytics"
    echo "   Grafana: http://localhost:3001 (admin/swr_admin_2024)"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "🐳 Docker services:"
    docker-compose -f docker-compose.swr.yml ps
    echo ""
    echo "📊 SWR Analytics:"
    curl -s http://localhost:3000/api/swr/analytics | jq . || echo "Analytics endpoint not ready yet"
}

# Cleanup on failure
cleanup() {
    if [ $? -ne 0 ]; then
        error "Deployment failed, performing cleanup..."
        
        if [ -d "$BACKUP_DIR" ]; then
            log "Restoring from backup..."
            sudo rm -rf "$DEPLOY_DIR"
            sudo cp -r "$BACKUP_DIR" "$DEPLOY_DIR"
            warning "Restored from backup"
        fi
    fi
}

# Main deployment function
main() {
    trap cleanup EXIT
    
    log "Starting PinkPlayEvo SWR deployment..."
    
    check_root
    check_prerequisites
    backup_existing
    update_repository
    test_swr_core
    install_dependencies
    build_images
    deploy_services
    
    if health_check; then
        display_info
        success "🎯 PinkPlayEvo SWR deployment completed successfully!"
    else
        error "Deployment completed but health check failed"
        exit 1
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    test)
        check_prerequisites
        update_repository
        test_swr_core
        success "All tests passed!"
        ;;
    build)
        check_prerequisites
        update_repository
        install_dependencies
        build_images
        success "Build completed!"
        ;;
    health)
        health_check && success "System is healthy" || error "System is unhealthy"
        ;;
    *)
        echo "Usage: $0 {deploy|test|build|health}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Full deployment (default)"
        echo "  test    - Run tests only"
        echo "  build   - Build images only"
        echo "  health  - Check system health"
        exit 1
        ;;
esac