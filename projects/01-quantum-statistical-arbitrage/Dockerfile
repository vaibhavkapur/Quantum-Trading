# CPU-only reproduction image: Python 3.12 research stack + Rust replay engine.
FROM rust:1-slim-bookworm AS engine
WORKDIR /build
COPY engine/ engine/
RUN cd engine && cargo test --release && cargo build --release

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps -e .
COPY configs/ configs/
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY reproduce.sh README.md ./
COPY engine/ engine/
COPY --from=engine /build/engine/target/release/statarb-engine engine/target/release/statarb-engine
# cargo is absent in this stage; reproduce.sh then skips the engine rebuild, so run the prebuilt binary explicitly.
CMD ["bash", "-c", "./reproduce.sh ci && python -m project.replay_l2 --config configs/ci.yaml --runs runs/ci && python -m project.report --runs runs/ci --output reports/ci/comparison.html"]
