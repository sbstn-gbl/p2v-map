VIRTUALENV?=env
PORT?=8888
PORTTB?=9999

.PHONY: help
help:
	@echo "Make targets:"
	@echo "  build          create virtualenv and install packages"
	@echo "  build-lab      runs build and installs lab extensions"
	@echo "  freeze         persist installed packaged to requirements.txt"
	@echo "  clean          remove *.pyc files and __pycache__ directory"
	@echo "  distclean      remove virtual environment"
	@echo "  run            run JupyterLab (default port $(PORT))"
	@echo "  runtb          run TensorBoard lab (default port $(PORTTB))"
	@echo "Check the Makefile for more details"

.PHONY: build
build:
	virtualenv $(VIRTUALENV); \
	source $(VIRTUALENV)/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt;

.PHONY: build-lab
build-lab: build
	source $(VIRTUALENV)/bin/activate; \
	jupyter serverextension enable --py jupyterlab_code_formatter

.PHONY: freeze
freeze:
	source $(VIRTUALENV)/bin/activate; \
	pip freeze > requirements.txt

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr

.PHONY: distclean
distclean: clean
	rm -rf $(VIRTUALENV)

.PHONY: run
run:
	source $(VIRTUALENV)/bin/activate; \
	jupyter lab --no-browser --port=$(PORT)

.PHONY: runtb
runtb:
	source $(VIRTUALENV)/bin/activate; \
	tensorboard --logdir results --port=$(PORTTB)

