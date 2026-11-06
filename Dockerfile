# ============================================================================
# STAGE 1: Build & Dependency Installation
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies if native extensions need compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install required numerical and plotting libraries into a local directory
RUN pip install --no-cache-dir --user numpy matplotlib

# ============================================================================
# STAGE 2: Final Runtime Environment
# ============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /root/.local
COPY . /app

# Update PATH to find user-installed packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create an output directory for generated plots
RUN mkdir -p /app/output

# Run the project script (Assumes your file is named final_project.py)
CMD ["python", "final_project.py"]
