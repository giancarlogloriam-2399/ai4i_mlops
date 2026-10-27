FROM python:3.14.5-slim

# uv con versión fijada (la misma que generó uv.lock)
COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Capa 1: solo dependencias (se reutiliza si el código cambia)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Capa 2: el código del proyecto
COPY . .
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Entrena dentro de la imagen: si falla la puerta de calidad, la imagen no se construye
RUN preparar-datos && entrenar

# Usuario sin privilegios de root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Render asigna el puerto en la variable PORT; en local se usa 8000
CMD ["sh", "-c", "exec fastapi run src/ai4i_mlops/api.py --port ${PORT:-8000}"]
