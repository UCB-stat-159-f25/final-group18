ENV_NAME := stat159-env
CONDA := conda

NOTEBOOKS := EDA.ipynb model_rq1.ipynb model_rq2.ipynb model_rq3.ipynb model_rq4.ipynb

.PHONY: env all run clean html

env:
	@echo ">>> Creating/updating conda env: $(ENV_NAME)"
	@$(CONDA) env list | awk '{print $$1}' | grep -qx "$(ENV_NAME)" && \
		($(CONDA) env update -n $(ENV_NAME) -f environment.yml --prune) || \
		($(CONDA) env create -f environment.yml)

all: run

run:
	@echo ">>> Executing notebooks with nbconvert"
	@for nb in $(NOTEBOOKS); do \
		echo "---- Running $$nb ----"; \
		$(CONDA) run -n $(ENV_NAME) python -m nbconvert --to notebook --execute --inplace "$$nb"; \
	done

clean:
	@echo ">>> Cleaning notebook outputs (optional)"
	@for nb in $(NOTEBOOKS); do \
		$(CONDA) run -n $(ENV_NAME) python -m nbconvert --clear-output --inplace "$$nb"; \
	done

html:
	$(CONDA) run -n $(ENV_NAME) myst build --html


