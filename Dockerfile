FROM python:3.7-slim

COPY . /optimizer
WORKDIR /optimizer

RUN pip install -r requirements.txt

CMD ./optimizer.py --cfg ./config/config.yaml
