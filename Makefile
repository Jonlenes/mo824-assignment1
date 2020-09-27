install:
	pip install -r src/requirements.txt

generate_ins:
	python src/generate_instance.py

test:
	python -m pytest .