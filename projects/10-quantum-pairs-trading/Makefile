PYTHON ?= .venv/bin/python

.PHONY: test data demo ci report

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) -m project.prepare_data --config configs/demo.yaml

demo: data
	$(PYTHON) -m project.run --config configs/demo.yaml --solver all
	$(PYTHON) -m project.report --runs runs/ --output reports/comparison.html

ci:
	$(PYTHON) -m project.prepare_data --config configs/ci.yaml
	$(PYTHON) -m project.run --config configs/ci.yaml --solver all
	$(PYTHON) -m project.report --runs runs/ci --output reports/ci.html
	$(PYTHON) -m pytest -q

report:
	$(PYTHON) -m project.report --runs runs/ --output reports/comparison.html
