FROM python:3.12.11-slim-bookworm

WORKDIR /app
COPY . /app
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
