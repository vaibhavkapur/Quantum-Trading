PYTHON ?= .venv/bin/python

.PHONY: test demo serve native
test:
	$(PYTHON) -m pytest -q -W error::RuntimeWarning

demo:
	$(PYTHON) -m project.reproduce --config configs/demo.yaml

serve:
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port 8088

native:
	cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
	cmake --build native/build
