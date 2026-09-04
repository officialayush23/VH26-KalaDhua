# Shortcuts. Run from the repository root: `make -f deploy/Makefile up`
COMPOSE = docker compose -f deploy/docker-compose.yml

.PHONY: build up down logs load bench train ps clean

build:      ## build every image
	$(COMPOSE) build

up:         ## start engine, applications, invalidator and dashboard
	$(COMPOSE) up -d
	@echo "engine     http://localhost:8080/healthz"
	@echo "dashboard  http://localhost:5173"

load:       ## start the traffic generator as well
	$(COMPOSE) --profile load up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

bench:      ## run the nine-policy comparison inside the running engine
	curl -s -X POST http://localhost:8080/v1/bench/run \
	  -H "Content-Type: application/json" \
	  -d '{"scenario":"expensive_tail","policies":["lru","lfu","gds","gdsf","tinylfu","s3fifo","sieve","lecar","aura"],"capacity_bytes":134217728,"requests":80000}'

train:      ## train on whatever data exists and publish to the running engine
	cd training && python -m portable train

clean:      ## remove containers and the model volume
	$(COMPOSE) down -v
