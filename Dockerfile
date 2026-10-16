# CPU image for training and batch scoring.
#
# Note: `rbdaim predict` on raw sequences additionally needs ANARCI, which is
# bioconda-only and deliberately not installed here. Dataset-based paths
# (train / score / predict_numbered) work in this image; see README.
# The tag is mutable, so this build is reproducible only to the day. To pin it,
# resolve the digest and append it to the tag below:
#   docker buildx imagetools inspect python:3.12-slim-bookworm --format '{{.Manifest.Digest}}'
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    APP_HOME=/app

WORKDIR ${APP_HOME}

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

# Dependencies first, from the lockfile, so a source-only edit reuses this layer.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --locked --no-install-project --no-dev

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv sync --locked --no-dev

ENV PATH="${APP_HOME}/.venv/bin:${PATH}"

# Non-root, and a writable place for checkpoints that is not the source tree.
RUN useradd --create-home --uid 1000 rbdaim \
    && mkdir -p /models /data \
    && chown -R rbdaim:rbdaim ${APP_HOME} /models /data
USER rbdaim

ENTRYPOINT ["rbdaim"]
CMD ["--help"]
