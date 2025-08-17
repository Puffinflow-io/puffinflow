export const deploymentMarkdown = `# Deployment Guide

## Overview

This comprehensive guide covers deploying your Puffinflow-based applications to different environments including local development, staging, and production. Learn how to containerize your applications, configure cloud platforms, and implement best practices for scalable deployments.

## Prerequisites

- **Python 3.8+** with your Puffinflow application
- **Docker** for containerization
- **Git** for version control
- Access to a cloud platform (AWS, GCP, Azure, or similar)
- Basic knowledge of deployment concepts

## Quick Deployment Checklist

✅ **Application is production-ready**
✅ **Dependencies are properly documented**
✅ **Environment variables are configured**
✅ **Health checks are implemented**
✅ **Logging and monitoring are set up**
✅ **Security best practices are followed**

## Local Development Setup

### 1. Environment Setup

\`\`\`bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
\`\`\`

### 2. Environment Variables

Create a \`.env\` file for local development:

\`\`\`bash
# .env
PUFFINFLOW_ENV=development
PUFFINFLOW_LOG_LEVEL=DEBUG
PUFFINFLOW_METRICS_ENABLED=true

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/puffinflow_dev

# Redis (for coordination)
REDIS_URL=redis://localhost:6379

# External APIs
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
\`\`\`

### 3. Local Testing

\`\`\`bash
# Run tests
pytest tests/

# Run with development server
python -m uvicorn app.main:app --reload --port 8000
\`\`\`

## Containerization with Docker

### 1. Create Dockerfile

\`\`\`dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

### 2. Docker Compose for Development

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PUFFINFLOW_ENV=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/puffinflow
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app/app
      - ./logs:/app/logs

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: puffinflow
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
\`\`\`

### 3. Build and Run

\`\`\`bash
# Build the image
docker build -t puffinflow-app .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f app
\`\`\`

## Production-Ready Application Structure

### 1. Application Structure

\`\`\`
my-puffinflow-app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── health.py            # Health check endpoints
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   └── data_pipeline.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   └── specialized_agents.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── monitoring.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
\`\`\`

### 2. Configuration Management

\`\`\`python
# app/config.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "Puffinflow Application"
    environment: str = "production"
    debug: bool = False

    # Database
    database_url: str

    # Redis
    redis_url: str

    # Puffinflow
    puffinflow_log_level: str = "INFO"
    puffinflow_metrics_enabled: bool = True
    puffinflow_checkpoint_dir: str = "/app/checkpoints"

    # External APIs
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Security
    cors_origins: list = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
\`\`\`

### 3. Health Checks

\`\`\`python
# app/health.py
from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from app.database import get_db_session
from app.config import settings
import redis
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.environment
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependency status"""
    health_status = {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.environment,
        "dependencies": {}
    }

    # Check database
    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        health_status["dependencies"]["database"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["database"] = "unhealthy"
        health_status["status"] = "unhealthy"

    # Check Redis
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        health_status["dependencies"]["redis"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["redis"] = "unhealthy"
        health_status["status"] = "unhealthy"

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status
\`\`\`

### 4. Production FastAPI Application

\`\`\`python
# app/main.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.health import router as health_router
from app.workflows.document_processor import DocumentProcessor
from app.utils.logging import setup_logging
from app.utils.monitoring import setup_metrics
import uvicorn

# Setup logging
setup_logging()

# Setup metrics
setup_metrics()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Production Puffinflow Application",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include health check router
app.include_router(health_router, prefix="/api/v1")

# Initialize document processor
document_processor = DocumentProcessor()

@app.post("/api/v1/process-document")
async def process_document(
    file_path: str,
    background_tasks: BackgroundTasks
):
    """Process document in background"""
    task_id = f"doc_{hash(file_path)}"

    # Add background task
    background_tasks.add_task(
        document_processor.process_document,
        file_path,
        task_id
    )

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Document processing started"
    }

@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    # Implementation depends on your task tracking system
    return {
        "task_id": task_id,
        "status": "completed",
        "result": "Document processed successfully"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
\`\`\`

## Cloud Platform Deployment

### 1. AWS Deployment

#### Using AWS ECS (Elastic Container Service)

\`\`\`yaml
# aws-ecs-task-definition.json
{
  "family": "puffinflow-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "puffinflow-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/puffinflow-app:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PUFFINFLOW_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:database-url"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/puffinflow-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
\`\`\`

#### Deployment Script

\`\`\`bash
#!/bin/bash
# deploy-aws.sh

set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPO="your-account.dkr.ecr.region.amazonaws.com/puffinflow-app"
ECS_CLUSTER="puffinflow-cluster"
ECS_SERVICE="puffinflow-service"

# Build and push Docker image
echo "Building Docker image..."
docker build -t puffinflow-app .

# Tag for ECR
docker tag puffinflow-app:latest \$ECR_REPO:latest

# Login to ECR
aws ecr get-login-password --region \$AWS_REGION | docker login --username AWS --password-stdin \$ECR_REPO

# Push to ECR
echo "Pushing to ECR..."
docker push \$ECR_REPO:latest

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service \\
    --cluster \$ECS_CLUSTER \\
    --service \$ECS_SERVICE \\
    --force-new-deployment

# Wait for deployment
echo "Waiting for deployment to complete..."
aws ecs wait services-stable \\
    --cluster \$ECS_CLUSTER \\
    --services \$ECS_SERVICE

echo "Deployment completed successfully!"
\`\`\`

### 2. Google Cloud Platform (GCP)

#### Using Google Cloud Run

\`\`\`yaml
# cloudbuild.yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/\$PROJECT_ID/puffinflow-app:latest', '.']

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/\$PROJECT_ID/puffinflow-app:latest']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'puffinflow-app'
      - '--image'
      - 'gcr.io/\$PROJECT_ID/puffinflow-app:latest'
      - '--platform'
      - 'managed'
      - '--region'
      - 'us-central1'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'
      - '--concurrency'
      - '100'
      - '--max-instances'
      - '10'
      - '--set-env-vars'
      - 'PUFFINFLOW_ENV=production'
      - '--set-secrets'
      - 'DATABASE_URL=database-url:latest'
      - '--set-secrets'
      - 'OPENAI_API_KEY=openai-key:latest'

options:
  logging: CLOUD_LOGGING_ONLY
\`\`\`

#### Deployment Script

\`\`\`bash
#!/bin/bash
# deploy-gcp.sh

set -e

# Configuration
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="puffinflow-app"

# Build and deploy
echo "Building and deploying to Google Cloud Run..."
gcloud builds submit --config cloudbuild.yaml

# Get service URL
SERVICE_URL=\$(gcloud run services describe \$SERVICE_NAME --region=\$REGION --format="value(status.url)")

echo "Deployment completed successfully!"
echo "Service URL: \$SERVICE_URL"
\`\`\`

### 3. Azure Deployment

#### Using Azure Container Instances

\`\`\`yaml
# azure-container-group.yaml
apiVersion: '2021-03-01'
location: eastus
name: puffinflow-app-group
properties:
  containers:
  - name: puffinflow-app
    properties:
      image: yourregistry.azurecr.io/puffinflow-app:latest
      resources:
        requests:
          cpu: 2
          memoryInGB: 4
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: PUFFINFLOW_ENV
        value: production
      - name: DATABASE_URL
        secureValue: your-database-url
      - name: OPENAI_API_KEY
        secureValue: your-openai-key
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
type: Microsoft.ContainerInstance/containerGroups
\`\`\`

### 4. Kubernetes Deployment

#### Kubernetes Manifests

\`\`\`yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: puffinflow-app
  labels:
    app: puffinflow-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: puffinflow-app
  template:
    metadata:
      labels:
        app: puffinflow-app
    spec:
      containers:
      - name: puffinflow-app
        image: your-registry/puffinflow-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: PUFFINFLOW_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: puffinflow-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: puffinflow-secrets
              key: openai-key
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: puffinflow-app-service
spec:
  selector:
    app: puffinflow-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
\`\`\`

## Vercel Deployment (for Web Applications)

### 1. Vercel Configuration

\`\`\`json
{
  "version": 2,
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "cleanUrls": true,
  "trailingSlash": false,
  "env": {
    "NODE_ENV": "production",
    "VITE_API_URL": "https://your-api.vercel.app/api/v1"
  },
  "build": {
    "env": {
      "NODE_ENV": "production"
    }
  },
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
\`\`\`

### 2. Deployment Script

\`\`\`bash
#!/bin/bash
# deploy-vercel.sh

set -e

# Install dependencies
npm install

# Build the application
npm run build

# Deploy to Vercel
npx vercel --prod

echo "Deployment to Vercel completed successfully!"
\`\`\`

## Environment-Specific Configurations

### 1. Development Environment

\`\`\`python
# config/development.py
DEBUG = True
PUFFINFLOW_LOG_LEVEL = "DEBUG"
PUFFINFLOW_METRICS_ENABLED = True
PUFFINFLOW_CHECKPOINT_INTERVAL = 10  # seconds

# Database
DATABASE_URL = "postgresql://user:password@localhost:5432/puffinflow_dev"

# External services
OPENAI_API_KEY = "sk-dev-key"
REDIS_URL = "redis://localhost:6379"
\`\`\`

### 2. Staging Environment

\`\`\`python
# config/staging.py
DEBUG = False
PUFFINFLOW_LOG_LEVEL = "INFO"
PUFFINFLOW_METRICS_ENABLED = True
PUFFINFLOW_CHECKPOINT_INTERVAL = 30  # seconds

# Database
DATABASE_URL = "postgresql://user:password@staging-db:5432/puffinflow_staging"

# External services
OPENAI_API_KEY = "sk-staging-key"
REDIS_URL = "redis://staging-redis:6379"
\`\`\`

### 3. Production Environment

\`\`\`python
# config/production.py
DEBUG = False
PUFFINFLOW_LOG_LEVEL = "WARNING"
PUFFINFLOW_METRICS_ENABLED = True
PUFFINFLOW_CHECKPOINT_INTERVAL = 60  # seconds

# Database
DATABASE_URL = "postgresql://user:password@prod-db:5432/puffinflow_prod"

# External services
OPENAI_API_KEY = "sk-production-key"
REDIS_URL = "redis://prod-redis:6379"

# Security
CORS_ORIGINS = ["https://yourdomain.com"]
SECURE_COOKIES = True
\`\`\`

## CI/CD Pipeline

### 1. GitHub Actions

\`\`\`yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: |
        pytest tests/ -v --cov=app --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: \${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build, tag, and push image to Amazon ECR
      env:
        ECR_REGISTRY: \${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: puffinflow-app
        IMAGE_TAG: \${{ github.sha }}
      run: |
        docker build -t \$ECR_REGISTRY/\$ECR_REPOSITORY:\$IMAGE_TAG .
        docker push \$ECR_REGISTRY/\$ECR_REPOSITORY:\$IMAGE_TAG
        docker tag \$ECR_REGISTRY/\$ECR_REPOSITORY:\$IMAGE_TAG \$ECR_REGISTRY/\$ECR_REPOSITORY:latest
        docker push \$ECR_REGISTRY/\$ECR_REPOSITORY:latest

    - name: Deploy to ECS
      env:
        ECS_CLUSTER: puffinflow-cluster
        ECS_SERVICE: puffinflow-service
      run: |
        aws ecs update-service --cluster \$ECS_CLUSTER --service \$ECS_SERVICE --force-new-deployment
        aws ecs wait services-stable --cluster \$ECS_CLUSTER --services \$ECS_SERVICE
\`\`\`

### 2. GitLab CI/CD

\`\`\`yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""

services:
  - postgres:15
  - redis:7

test:
  stage: test
  image: python:3.11
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
  script:
    - pytest tests/ -v --cov=app
  variables:
    DATABASE_URL: postgresql://postgres:postgres@postgres:5432/test_db
    REDIS_URL: redis://redis:6379
  coverage: '/TOTAL.*\\s+(\\d+%)/'

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u \$CI_REGISTRY_USER -p \$CI_REGISTRY_PASSWORD \$CI_REGISTRY
  script:
    - docker build -t \$CI_REGISTRY_IMAGE:latest .
    - docker push \$CI_REGISTRY_IMAGE:latest
  only:
    - main

deploy:
  stage: deploy
  image: google/cloud-sdk:alpine
  before_script:
    - echo \$GCP_SERVICE_KEY | base64 -d > gcp-key.json
    - gcloud auth activate-service-account --key-file gcp-key.json
    - gcloud config set project \$GCP_PROJECT_ID
  script:
    - gcloud run deploy puffinflow-app
        --image \$CI_REGISTRY_IMAGE:latest
        --platform managed
        --region us-central1
        --memory 2Gi
        --cpu 2
        --set-env-vars PUFFINFLOW_ENV=production
  only:
    - main
\`\`\`

## Monitoring and Logging

### 1. Application Monitoring

\`\`\`python
# app/utils/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import logging

# Metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('app_request_duration_seconds', 'Request duration')
ACTIVE_WORKFLOWS = Gauge('app_active_workflows', 'Number of active workflows')

logger = logging.getLogger(__name__)

def setup_metrics():
    """Setup application metrics"""
    logger.info("Setting up monitoring metrics")

def track_request(method: str, endpoint: str):
    """Track API request"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()

def track_duration(duration: float):
    """Track request duration"""
    REQUEST_DURATION.observe(duration)

def set_active_workflows(count: int):
    """Set number of active workflows"""
    ACTIVE_WORKFLOWS.set(count)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Track metrics
    track_request(request.method, request.url.path)
    track_duration(process_time)

    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
\`\`\`

### 2. Structured Logging

\`\`\`python
# app/utils/logging.py
import logging
import json
from datetime import datetime
from app.config import settings

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': settings.app_name,
            'environment': settings.environment,
        }

        if hasattr(record, 'workflow_id'):
            log_entry['workflow_id'] = record.workflow_id

        if hasattr(record, 'agent_id'):
            log_entry['agent_id'] = record.agent_id

        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

def setup_logging():
    """Setup structured logging"""
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.puffinflow_log_level)

    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add JSON handler
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    # Set specific loggers
    logging.getLogger('puffinflow').setLevel(settings.puffinflow_log_level)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
\`\`\`

## Security Best Practices

### 1. Environment Variables

\`\`\`bash
# Use secrets management
export DATABASE_URL="postgresql://user:password@host:5432/db"
export OPENAI_API_KEY="sk-secret-key"
export JWT_SECRET="your-jwt-secret"

# Never commit secrets to version control
echo "*.env" >> .gitignore
echo "secrets/" >> .gitignore
\`\`\`

### 2. Network Security

\`\`\`yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    networks:
      - app-network
    environment:
      - PUFFINFLOW_ENV=production
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    networks:
      - app-network
    environment:
      POSTGRES_DB: puffinflow
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    networks:
      - app-network
    command: redis-server --requirepass \$REDIS_PASSWORD

networks:
  app-network:
    driver: bridge

secrets:
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres_data:
\`\`\`

### 3. Container Security

\`\`\`dockerfile
# Use non-root user
FROM python:3.11-slim

# Install security updates
RUN apt-get update && apt-get upgrade -y && \\
    apt-get install -y --no-install-recommends gcc && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and change ownership
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Use explicit command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

## Troubleshooting Common Issues

### 1. Container Won't Start

\`\`\`bash
# Check logs
docker logs container_name

# Check resource usage
docker stats

# Inspect container
docker inspect container_name

# Common fixes
- Ensure all environment variables are set
- Check file permissions
- Verify network connectivity
- Ensure adequate resources (CPU/Memory)
\`\`\`

### 2. Database Connection Issues

\`\`\`python
# Test database connection
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

async def test_db_connection():
    engine = create_async_engine("postgresql://user:password@host:5432/db")
    async with engine.begin() as conn:
        result = await conn.execute("SELECT 1")
        print(f"Database connection successful: {result.scalar()}")

asyncio.run(test_db_connection())
\`\`\`

### 3. Performance Issues

\`\`\`python
# Monitor workflow performance
from puffinflow.observability import MetricsCollector

metrics = MetricsCollector()

@state
async def monitored_state(context):
    with metrics.timer("state_execution_time"):
        # Your state logic
        pass

    metrics.gauge("context_size", len(context.variables))
    metrics.increment("state_executions")
\`\`\`

## Production Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Security vulnerabilities addressed
- [ ] Environment variables configured
- [ ] Database migrations completed
- [ ] Monitoring and logging configured
- [ ] Health checks implemented
- [ ] Resource limits set
- [ ] Backup strategy in place

### Post-Deployment

- [ ] Application accessible
- [ ] Health checks passing
- [ ] Logs flowing correctly
- [ ] Metrics being collected
- [ ] Database connectivity verified
- [ ] External API integrations working
- [ ] Performance within acceptable limits
- [ ] Error rates acceptable

## Next Steps

After deploying your Puffinflow application:

1. **[Monitor Performance →](#docs/observability)** - Set up comprehensive monitoring
2. **[Scale Resources →](#docs/resource-management)** - Optimize for production load
3. **[Handle Errors →](#docs/error-handling)** - Implement robust error handling
4. **[Backup Data →](#docs/checkpointing)** - Ensure data durability
5. **[Security Hardening →](#docs/troubleshooting)** - Additional security measures

**🚀 Ready to deploy?** Choose your platform and follow the specific deployment guide above!
`.trim();
