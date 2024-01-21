FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y bash git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 ./get-pip.py && \
    pip install pytest

RUN git config --global user.email "intercode@pnlp.org"
RUN git config --global user.name "intercode"

WORKDIR /