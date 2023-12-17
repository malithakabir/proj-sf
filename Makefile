install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=src src/test_*.py

format:	
	black src/*.py

lint:
	pylint --disable=R,C,import-error,no-member,unused-import --ignore-patterns=test_.*?py src/*.py 

all: install lint test format

# test:
# 	python -m pytest -vv --cov=main --cov=src test_*.py

# container-lint:
# 	docker run --rm -i hadolint/hadolint < Dockerfile

# refactor: format lint

# deploy:
# 	#deploy goes here
		
# all: install lint test format deploy
