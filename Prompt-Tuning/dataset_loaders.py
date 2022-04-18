from datasets import load_dataset, load_metric
import uuid

def tokenize_dataset(dataset, tokenizer, max_source_length, max_target_length, padding, eval=False):
    model_inputs = dataset.map(lambda x: tokenizer(x['source'], max_length=max_source_length, truncation=True, padding=padding))
    if not eval:
        with tokenizer.as_target_tokenizer():
            model_inputs = model_inputs.map(lambda x: {'labels': tokenizer(x['target'], max_length=max_target_length, truncation=True, padding=padding)['input_ids']})
    return model_inputs


def boolq_metric_loader():
    return load_metric('super_glue', 'boolq', experiment_id=str(uuid.uuid1()))

def boolq_loader(tokenizer, soft_prompt_length=0, max_seq_length=128):

    def boolq_preprocessor(x, eval=False):
        src_texts = ["question:", x["question"], "passage:", x["passage"]]
        x['source'] = ' '.join(src_texts)
        if not eval:
            tgt_texts = [str(x["label"])]
            x['target'] = ' '.join(tgt_texts)
        return x

    # TODO: uncomment
    # raw_dataset = load_dataset('super_glue', 'boolq', split=['train', 'validation'])
    raw_dataset = load_dataset('super_glue', 'boolq', split=['train', 'validation'])
    # train_dataset, val_dataset = raw_dataset["train"], raw_dataset["validation"]
    train_dataset, val_dataset = raw_dataset[0], raw_dataset[1]

    # Preprocess dataset
    train_dataset = train_dataset.map(lambda x: boolq_preprocessor(x), batched=False)
    val_dataset = val_dataset.map(lambda x: boolq_preprocessor(x), batched=False)

    # Tokenize dataset
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, val_dataset


DATASET_LOADERS = {
    'boolq': boolq_loader
}

METRIC_LOADERS = {
    'boolq': boolq_metric_loader
}
