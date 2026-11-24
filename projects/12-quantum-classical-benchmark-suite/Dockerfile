FROM python:3.12-slim
WORKDIR /suite
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmarks benchmarks
COPY experiments experiments
COPY metrics metrics
COPY project project
COPY configs configs
COPY tests tests
CMD ["python", "-m", "project.reproduce", "--config", "configs/ci.yaml", "--output", "reports/comparison.html", "--strict"]
