import argparse
import os
from tqdm import tqdm, trange
import csv

import torch
from torch.nn import functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert.modeling import (BertForMultipleChoice, BertConfig, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_pretrained_bert.tokenization import BertTokenizer

from run_GAP import read_GAP_examples, convert_examples_to_features, select_field

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints are written.")

    args = parser.parse_args()

    # TODO: Fix hardcoded values
    max_seq_length = 128
    bert_model = 'bert-base-cased'
    do_lower_case = False
    batch_size = 8

    # Load a trained model and config that you have fine-tuned
    config = BertConfig(os.path.join(args.model_dir, CONFIG_NAME))
    model = BertForMultipleChoice(config, num_choices=3)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, WEIGHTS_NAME)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    # Prepare test data
    examples = read_GAP_examples(os.path.join(args.data_dir, 'gap-development.tsv'), is_training=False)
    features = convert_examples_to_features(examples, tokenizer, max_seq_length, True)

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    dataloader = DataLoader(data,
                            sampler=SequentialSampler(data),
                            batch_size=batch_size,
                            shuffle=False)

    model.eval()
    with open(os.path.join(args.model_dir, 'kaggle-dev.csv'), mode='w') as kaggle_file:
        writer = csv.writer(kaggle_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'A', 'B', 'NEITHER'])

        for i, (input_ids, input_mask, segment_ids) in tqdm(enumerate(dataloader), desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = F.softmax(model(input_ids, segment_ids, input_mask))

            logits = logits.detach().cpu().numpy()
            #writer.writerow([filename] + logits)
            for j, l in enumerate(logits):
                filename = features[i * batch_size + j].example_id
                writer.writerow([filename] + list(l))

if __name__ == "__main__":
    main()
