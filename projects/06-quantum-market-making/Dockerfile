FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY market_making ./market_making
COPY project ./project
COPY configs ./configs
RUN python -c 'from market_making.native import build; build()'
CMD ["python", "-m", "project.run", "--config", "configs/ci.yaml", "--solver", "all"]
