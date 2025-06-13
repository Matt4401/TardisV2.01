##
## Copyright (C) 2023-2024 TardisV2.01
## TardisV2.01
## File description:
## Makefile
##

ENV := envtardis
NOTEBOOK_DIR := notebooks
OUTPUT_DIR := executed
EXECUTED_FILES := $(wildcard $(NOTEBOOK_DIR)/$(OUTPUT_DIR)/*.ipynb)
EXEC_NOTEBOOKS := $(patsubst $(NOTEBOOK_DIR)/%.ipynb,$(OUTPUT_DIR)/%.ipynb,$(NOTEBOOKS))
STREAMLIT_APP := app/tardis_dashboard.py

.PHONY: all notebook notebooks streamlit clean

all: notebooks

notebook:
ifndef name
	$(error ❌ Try with this format : `make notebook name=my_file`)
endif
	@mkdir -p $(OUTPUT_DIR)
	@cd notebooks && papermill $(name).ipynb $(OUTPUT_DIR)/$(name).ipynb

notebooks:
	@cd $(NOTEBOOK_DIR) && \
	echo "📂 Exécution des notebooks dans $(NOTEBOOK_DIR)..."; \
	for nb in *.ipynb; do \
		name=$$(basename $$nb); \
		echo "🚀 Exécution de $$name..."; \
		papermill $$nb $(OUTPUT_DIR)/$$name || exit 1; \
	done

streamlit:
	@bash -c "source $(ENV)/bin/activate && streamlit run $(STREAMLIT_APP)"

clean:
	@echo "🧹 Suppression des fichiers..."
	@rm -rf $(OUTPUT_DIR)
	@for file in cleaned_dataset.csv comments_dataset.csv model.pkl comments_model.pkl; do \
		if [ -e "$$file" ]; then \
			rm "$$file" > /dev/null 2>&1; \
		fi \
	done
	@for file in notebooks/cleaned_dataset.csv notebooks/comments_dataset.csv notebooks/model.pkl notebooks/comments_model.pkl; do \
		if [ -e "$$file" ]; then \
			rm "$$file" > /dev/null 2>&1; \
		fi \
	done
	@echo "✅ Cleaning terminé."

fclean: clean
	@echo "🧹 Suppression des fichiers exécutés..."
	@rm -f $(EXECUTED_FILES)
	@echo "✅ Fichiers exécutés supprimés."
