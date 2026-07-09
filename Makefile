run-docker-compose:
	uv sync
	docker compose up --build
logs:
	docker compose logs -f
clean-notebook-outputs:
	uv run jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb
run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever
run-evals-retriever-extended:
	uv sync
	cd apps/api/src && PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file ../../../.env python -m evals.eval_retriever_extended