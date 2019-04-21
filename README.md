# BERT_for_GAP-coreference

This project was realised in the context of the INF8225 AI course. In this project,
we aim to reduce gender bias in pronoun resolution by creating a coreference 
resolver that performs well on a gender-balanced pronoun dataset, The Gendered 
Ambiguous Pronouns (GAP) dataset. We leverage BERT's strong pre-training tasks on 
large unsupervised datasets and transfer these contextual representations.

We have submitted our best performing model to the [Gendered Pronoun Resolution](https://www.kaggle.com/c/gendered-pronoun-resolution/) Kaggle competition. 


## Setting up
```
git clone --recursive git@github.com:isabellebouchard/BERT_for_GAP-coreference.git
```
Make sure the submodules are properly initialized. 


## First steps

To run the code, first install [Docker](https://docs.docker.com/install/) to be able
to build and run a docker container with all the proper dependencies installed
```
docker build -t IMAGE_NAME .
nvidia-docker run --rm -it -v /path/to/your/code/:/project IMAGE_NAME
```

If you don't have access to GPU, change `nvidia-docker` for `docker`. It is 
highly recommended to run the training on (multiple) GPUs.

Once inside the container you should be able run the training script:
```
python run_GAP.py --data_dir gap-coreference \
                  --bert_model bert-base-cased \
                  --output_dir results \
```
This will run the training script and save checkpoints of the best model in the 
output directory.
