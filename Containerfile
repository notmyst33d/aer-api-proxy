FROM docker.io/library/debian:13

RUN apt update -y && apt install -y \
    python3 python3-pip git curl \
    libglib2.0-0t64 libnspr4 libnss3 libdbus-1-3 libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxkbcommon0 libasound2t64 libgbm1

WORKDIR /app

COPY ./src /app/src
COPY ./pyproject.toml /app/

RUN --mount=type=cache,target=/root/.cache/pip pip3 install --break-system-packages .
RUN playwright install chromium --only-shell
RUN python3 -c "from src.aer_api_proxy.models import ensure_models; ensure_models()"

CMD ["fastapi", "run", "src/aer_api_proxy"]
