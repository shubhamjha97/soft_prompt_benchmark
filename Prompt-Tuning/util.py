from datasets import Dataset
import numpy as np
import torch
from scipy.special import softmax

def cast_to_int(str):
    if str.strip().lower() in ['yes', 'true', '1']:
        return 1
    return 0


def predictions_to_labels(data, tokenizer, n_prompt_tokens):
    data_predictions = data.predictions[0] if isinstance(data.predictions, tuple) else data.predictions
    predictions_ids = np.argmax(data_predictions, axis=2)
    predictions_ids = predictions_ids[:, n_prompt_tokens:]
    decoded_predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True) # TODO
    decoded_labels = tokenizer.batch_decode(data.label_ids, skip_special_tokens=True)
    print('predictions:')
    print('\n---\n'.join([x[0:40] for x in decoded_predictions[0:8]]))
    print('labels:')
    print('\n---\n'.join([x[0:40] for x in decoded_labels[0:8]]))

    predictions = [cast_to_int(x) for x in decoded_predictions]
    labels = [cast_to_int(x) for x in decoded_labels]
    return predictions, labels


def compute_metric_batched(trainer, metric, tokenizer, test_data, eval_batch_size=512, config=None):
    for ix in range(0, len(test_data), eval_batch_size):
        test_batch = Dataset.from_dict(test_data[ix:ix+eval_batch_size])

        predictions_ids = trainer.model.generate(torch.tensor(test_batch['input_ids']))
        decoded_predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True) # TODO:
        decoded_labels = tokenizer.batch_decode(test_batch['raw_labels'], skip_special_tokens=False)
        decoded_inputs = tokenizer.batch_decode(test_batch['input_ids'], skip_special_tokens=False)

        pred_probs = softmax(trainer.predict(test_batch).predictions[0], axis=2)  # TODO:

        print('inputs:')
        print('\n---\n'.join(decoded_inputs))
        print('predictions:')
        print('\n---\n'.join(decoded_predictions))
        print('labels:')
        print('\n---\n'.join(decoded_labels))

        print("True: ", pred_probs[:, 0, 10998])
        print("False: ", pred_probs[:, 0, 10747])

        # predictions, labels = predictions_to_labels(, tokenizer, config.n_prompt_tokens)
        # del pred_data

        predictions = [cast_to_int(x) for x in decoded_predictions]
        labels = [cast_to_int(x) for x in decoded_labels]
        metric.add_batch(predictions=predictions, references=labels)

    return metric.compute()