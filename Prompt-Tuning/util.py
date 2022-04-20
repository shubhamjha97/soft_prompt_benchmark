from datasets import Dataset
import numpy as np


def cast_to_int(str):
    try:
        return int(str)
    except:
        return 0


def predictions_to_labels(data, tokenizer, n_prompt_tokens):
    data_predictions = data.predictions[0] if isinstance(data.predictions, tuple) else data.predictions
    predictions_ids = np.argmax(data_predictions, axis=2)
    predictions_ids = predictions_ids[:, n_prompt_tokens:]
    decoded_predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
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
        pred_data = trainer.predict(test_batch)
        predictions, labels = predictions_to_labels(pred_data, tokenizer, config.n_prompt_tokens)
        del pred_data
        metric.add_batch(predictions=predictions, references=labels)

    return metric.compute()