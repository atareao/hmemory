# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM docker.io/library/rust:alpine3.21 AS builder

RUN apk add --no-cache --update \
    build-base \
    autoconf \
    musl-dev \
    pkgconfig \
    openssl \
    openssl-dev \
    openssl-libs-static

WORKDIR /app

# Cache dependencies (avoid recompiling every time)
RUN cargo init --bin --name hmemory .

COPY Cargo.toml Cargo.lock ./
RUN cargo build --release && \
    rm src/*.rs

# Real compilation
COPY ./src/ ./src
COPY openapi.json ./

RUN touch src/main.rs && \
    cargo build --release && \
    strip target/release/hmemory

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM docker.io/library/alpine:3.21

RUN apk add --update --no-cache \
    ca-certificates \
    curl \
    openssl \
    && \
    adduser -S -u 1000 -D hmem

COPY --from=builder /app/target/release/hmemory /usr/local/bin/

# Environment Variables
ENV PORT=8080 \
    DATABASE_URL=postgres://postgres:hmemory@localhost:5432/hmemory \
    EMBEDDING_PROVIDER=ollama \
    OPENROUTER_API_KEY="" \
    OPENROUTER_MODEL=openai/text-embedding-3-small \
    OLLAMA_BASE_URL=http://localhost:11434 \
    OLLAMA_MODEL=nomic-embed-text \
    HYBRID_ALPHA=0.5

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:$PORT/health || exit 1

USER hmem
EXPOSE 8080

CMD ["hmemory"]