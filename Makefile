.PHONY: setup data run test all pipeline-definition

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_data.py --cardholders 8000 --days 90 --out data/raw/transactions.csv

run:
	PYTHONPATH=src python -m fraud.pipeline --data data/raw/transactions.csv --out reports

test:
	python -m pytest -q

all: data run test

# Requires requirements-aws.txt and AWS credentials. Prints the pipeline JSON without creating anything.
pipeline-definition:
	PYTHONPATH=src python aws/pipeline.py --role-arn $(ROLE_ARN) --bucket $(BUCKET)
