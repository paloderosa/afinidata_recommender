.PHONY: init build run db-migrate test tox

init: build run db-migrate
	docker-compose exec web afinidata_recommender init

build:
	docker-compose build

run:
	docker-compose up -d

db-migrate:
	docker-compose exec web afinidata_recommender db migrate

test:
tox:
	docker-compose run -v $(PWD)/tests:/code/tests:ro web tox -e py37
