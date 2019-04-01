FROM pytorch/pytorch

WORKDIR /project

RUN pip install ipdb && \
    pip install matplotlib && \
    pip install torchvision && \
    pip install pytorch-pretrained-bert
