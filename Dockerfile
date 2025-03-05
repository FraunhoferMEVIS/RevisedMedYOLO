FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

LABEL maintainer="kai.geissler@mevis.fraunhofer.de"

ADD . /MedYOLO
WORKDIR /MedYOLO
RUN pip install -r requirements.txt 

