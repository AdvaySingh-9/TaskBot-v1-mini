FROM python:3.9

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

EXPOSE 7860

# Run Gunicorn with 4 worker processes on port 7860
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]