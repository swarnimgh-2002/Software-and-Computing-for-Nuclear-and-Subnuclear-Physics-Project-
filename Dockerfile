FROM python:3.9-slim

RUN apt-get update && apt-get install -y wget dos2unix gawk && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix windows line endings just in case
RUN dos2unix run_pipeline.sh
RUN chmod +x run_pipeline.sh

RUN mkdir -p data output

CMD ["./run_pipeline.sh"]
