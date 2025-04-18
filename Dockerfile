FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Simply keeping app alive for now
CMD ["tail", "-f", "/dev/null"]
