run-docker-compose:
	uv sync
	docker compose up --build
logs:
	docker compose logs -f
clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb
