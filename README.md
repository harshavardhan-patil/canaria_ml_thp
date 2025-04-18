```bash
docker compose build
docker compose up -d
docker compose exec app bash
python initial_setup.py
```


docker exec -t canaria_ml_thp-postgres-1 pg_dump -U canaria -d canaria_rag > canaria_rag.dump


docker exec -i canaria_ml_thp-postgres-1 psql -U canaria -d canaria_rag < canaria_rag.dump