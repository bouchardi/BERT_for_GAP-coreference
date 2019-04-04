import argparse
import torch
import os

from pytorch_pretrained_bert.modeling import (BertForMultipleChoice, BertConfig, WEIGHTS_NAME, CONFIG_NAME)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    #parser.add_argument("--data_dir",
    #                    default=None,
    #                    type=str,
    #                    required=True,
    #                    help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints are be written.")

    args = parser.parse_args()

    # Load a trained model and config that you have fine-tuned
    config = BertConfig(os.path.join(args.model_dir, CONFIG_NAME))
    model = BertForMultipleChoice(config, num_choices=3)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, WEIGHTS_NAME)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



if __name__ == "__main__":
    main()
