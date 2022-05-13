from datasets import load_dataset, load_metric
import uuid

def ignore_pad_tokens(tokenizer, input_ids):
    return [token if token!=tokenizer.pad_token_id else -100 for token in input_ids]

def un_ignore_pad_tokens(tokenizer, input_ids):
    return [token if token!=-100 else tokenizer.pad_token_id for token in input_ids]

def tokenize_dataset(dataset, tokenizer, max_source_length, max_target_length, padding, eval=False):
    model_inputs = dataset.map(lambda x: tokenizer(x['source'], max_length=max_source_length, truncation=True, padding=padding))
    if not eval:
        with tokenizer.as_target_tokenizer():
            model_inputs = model_inputs.map(lambda x: {'raw_labels': tokenizer(x['target'], max_length=max_target_length, truncation=True, padding=padding)['input_ids']})

    model_inputs = model_inputs.map(lambda x: {'labels': ignore_pad_tokens(tokenizer, x['raw_labels'])})

    return model_inputs


def boolq_metric_loader():
    return load_metric('super_glue', 'boolq', experiment_id=str(uuid.uuid1()))

def boolq_loader(tokenizer, soft_prompt_length=0, max_seq_length=128):

    def boolq_preprocessor(x, eval=False):
        src_texts = ["question:", x["question"], "passage:", x["passage"]]
        x['source'] = ' '.join(src_texts)
        if not eval:
            x['target'] = 'yes' if x['label']==1 else 'no'
        return x

    #raw_dataset = load_dataset('super_glue', 'boolq', split=['train', 'validation']) # TODO
    raw_dataset = load_dataset('super_glue', 'boolq', split=['train[0:8]', 'validation[0:8]']) # TODO:
    train_dataset, val_dataset = raw_dataset[0], raw_dataset[1]

    # Preprocess dataset
    train_dataset = train_dataset.map(lambda x: boolq_preprocessor(x), batched=False)
    val_dataset = val_dataset.map(lambda x: boolq_preprocessor(x), batched=False)

    # Tokenize dataset
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'raw_labels'])

    val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'raw_labels'])

    columns_to_remove = ['question', 'passage', 'idx', 'label', 'source', 'target']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    return train_dataset, val_dataset


DATASET_LOADERS = {
    'boolq': boolq_loader
}

METRIC_LOADERS = {
    'boolq': boolq_metric_loader
}
