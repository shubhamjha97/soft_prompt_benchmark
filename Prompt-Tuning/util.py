from datasets import Dataset
import numpy as np
import torch
from scipy.special import softmax
from copy import copy, deepcopy

def cast_to_int(str):
    if str.strip().lower() in ['yes', 'true', '1', 'entailment']:
        return 1
    return 0

def clean_str(str_, tokenizer):
    to_remove = [tokenizer.cls_token, tokenizer.mask_token, tokenizer.sep_token, tokenizer.pad_token,
                 tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token]
    to_remove = [token for token in to_remove if token]
    for remove_candidate in to_remove:
        str_ = str_.replace(remove_candidate, "")
    str_ = str_.strip()
    return str_


def compute_metric_batched(trainer, metric_, tokenizer, test_data, eval_batch_size=512, config=None):
    metric = deepcopy(metric_)
    for ix in range(0, len(test_data), eval_batch_size):
        test_batch = Dataset.from_dict(test_data[ix:ix+eval_batch_size])

        if config.model_name.startswith("t5"):
             predictions_ids = trainer.model.generate(torch.tensor(test_batch['input_ids']))
             # predictions_ids = trainer.model.generate(torch.tensor(test_batch['input_ids']).to(torch.device('cuda')))
             decoded_predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
        else:
             predictions_ids = trainer.model(torch.tensor(test_batch['input_ids']))
             # predictions_ids = trainer.model(torch.tensor(test_batch['input_ids']).to(torch.device('cuda')))
             decoded_predictions = tokenizer.batch_decode(predictions_ids.logits.argmax(dim=2), skip_special_tokens=False)

        #try:
        #    predictions_ids = trainer.model.generate(torch.tensor(test_batch['input_ids']))
        #except:
        #predictions_ids = trainer.model.generate(torch.tensor(test_batch['input_ids']).to(torch.device('cuda')))
        #decoded_predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(test_batch['raw_labels'], skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(test_batch['input_ids'], skip_special_tokens=True)

        predictions = [cast_to_int(clean_str(x, tokenizer)) for x in decoded_predictions]
        labels = [cast_to_int(x) for x in decoded_labels]
        metric.add_batch(predictions=predictions, references=labels)

        print('inputs:\n')
        print('\n---\n'.join(decoded_inputs[0:8]))
        print('\n\n\npredictions:\n')
        print('\n---\n'.join([str(x) for x in list(zip(decoded_predictions[0:8], predictions[0:8]))]))
        print('\n\n\nlabels:\n')
        print('\n---\n'.join([str(x) for x in list(zip(decoded_labels[0:8], labels[0:8]))]))

    return metric.compute()