web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 --worker-class uvicorn.workers.UvicornWorker app:app
