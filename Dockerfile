FROM python:3.7-slim

COPY . /optimizer
WORKDIR /optimizer

RUN pip install -r requirements.txt

CMD python ./optimizer.py --cfg ./config/config.yaml --host 0.0.0.0
