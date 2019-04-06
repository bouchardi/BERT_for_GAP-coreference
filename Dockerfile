FROM pytorch/pytorch

WORKDIR /project

COPY requirements.txt /project/requirements.txt
RUN pip install -r requirements.txt
