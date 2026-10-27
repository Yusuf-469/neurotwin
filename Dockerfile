# NeuroTwin: single-container deployment (Hugging Face Spaces, Docker SDK, free CPU tier).
# Stage 1 builds the dashboard as static files; stage 2 runs the FastAPI app, which serves both
# the API (/api/...) and the dashboard (/) on port 7860.

FROM node:20-slim AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
# NEXT_PUBLIC_API_URL is deliberately unset: the static site calls the API on its own origin.
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Versions pinned to the environment the test suite passed on (CPU-only PyTorch build).
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy==2.4.6 scipy==1.17.1 gymnasium==1.4.0 fastapi==0.162.0 \
    uvicorn==0.54.0 pydantic==2.14.1

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user neurotwin ./neurotwin
COPY --chown=user config ./config
COPY --chown=user docs/data ./docs/data
COPY --chown=user results/ppo/final_A/agent_*.pt ./results/ppo/final_A/
COPY --chown=user results/ppo/bc_ppo/agent_*.pt ./results/ppo/bc_ppo/
COPY --from=dashboard --chown=user /dashboard/out ./dashboard/out
USER user
ENV PYTHONPATH=/app HOME=/home/user
EXPOSE 7860
# Hosts such as Render set PORT; Hugging Face and local runs use the default 7860.
CMD ["sh", "-c", "exec uvicorn neurotwin.api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
