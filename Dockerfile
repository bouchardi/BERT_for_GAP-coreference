FROM pytorch/pytorch

WORKDIR /project

COPY requirements.txt /project/requirements.txt
COPY pytorch-pretrained-BERT /project/pytorch_pretrained_bert

RUN pip install -r requirements.txt

COPY . /static
