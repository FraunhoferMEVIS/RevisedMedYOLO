FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

LABEL maintainer="kai.geissler@mevis.fraunhofer.de"

ADD . /MedYOLO
WORKDIR /MedYOLO
RUN pip install -r requirements.txt 

