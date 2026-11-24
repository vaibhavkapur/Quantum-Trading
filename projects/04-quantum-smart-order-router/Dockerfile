# CPU-only reproducibility image. Not built or tested in the authoring environment (no Docker there).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "python -m pytest -q && python -m project.prepare_data --config configs/ci.yaml && python -m project.run --config configs/ci.yaml --solver classical && python -m project.run --config configs/ci.yaml --solver quantum-sim && python -m project.report --runs runs/ --output reports/comparison.html"]
