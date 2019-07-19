FROM python:3.7-slim

COPY . /optimizer
WORKDIR /optimizer

RUN pip install -r requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["--cfg", "./config/config.yaml", "--host", "0.0.0.0"]
