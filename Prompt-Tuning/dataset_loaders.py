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

def rte_metric_loader():
    return load_metric('super_glue', 'rte')

def wsc_metric_loader():
    return load_metric('super_glue', 'wsc')

def wic_metric_loader():
    return load_metric('super_glue', 'wic')

def copa_metric_loader():
    return load_metric('super_glue', 'copa')


def boolq_loader(tokenizer, soft_prompt_length=0, max_seq_length=128):

    def boolq_preprocessor(x, eval=False):
        src_texts = ["question:", x["question"], "passage:", x["passage"]]
        x['source'] = ' '.join(src_texts)
        if not eval:
            x['target'] = 'yes' if x['label']==1 else 'no'
        return x

    raw_dataset = load_dataset('super_glue', 'boolq', split=['train', 'validation'])
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


def rte_loader(tokenizer, soft_prompt_length=0, max_seq_length=128):

    def rte_preprocessor(x, eval=False):
        src_texts = ["premise:", x["premise"], "hypothesis:", x["hypothesis"]]
        x['source'] = ' '.join(src_texts)
        if not eval:
            x['target'] = 'entailment' if x['label']==1 else 'not_entailment'

        return x

    raw_dataset = load_dataset('super_glue', 'rte', split=['train', 'validation'])
    train_dataset, val_dataset = raw_dataset[0], raw_dataset[1]

    # Preprocess dataset
    train_dataset = train_dataset.map(lambda x: rte_preprocessor(x), batched=False)
    val_dataset = val_dataset.map(lambda x: rte_preprocessor(x), batched=False)

    # Tokenize dataset
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'raw_labels'])

    val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'raw_labels'])

    columns_to_remove = ['premise', 'hypothesis', 'idx', 'label', 'source', 'target']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    return train_dataset, val_dataset

def wsc_loader(tokenizer, soft_prompt_length=0, max_seq_length=128):

    def wsc_preprocessor(examples):
        new_column = []
        for text, span2_index, span2_word in zip(examples["text"], examples["span2_index"], examples["span2_text"]):
            words_a = text.split()
            words_a[span2_index] = "*" + words_a[span2_index] + "*"
            new_column.append(' '.join(words_a))
        new_examples = examples.add_column("span2_word_text", new_column)
        return new_examples

    raw_dataset = load_dataset('super_glue', 'wsc', split=['train', 'validation'])
    train_dataset, val_dataset = raw_dataset[0], raw_dataset[1]

    # Preprocess dataset
    train_dataset = wsc_preprocessor(train_dataset)
    val_dataset = wsc_preprocessor(val_dataset)

    # Tokenize dataset
    # train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    model_inputs = train_dataset.map(lambda x: tokenizer(x['span1_text'], max_length=max_seq_length - soft_prompt_length, truncation=True, padding='max_length'))
    model_inputs = model_inputs.map(lambda x: tokenizer(x['span2_word_text'], max_length=max_seq_length - soft_prompt_length, truncation=True, padding='max_length'))
    # if not eval:
    train_dataset = model_inputs
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'span1_text', 'span2_word_text'])

    # val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    model_inputs = val_dataset.map(
        lambda x: tokenizer(x['span1_text'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    model_inputs = model_inputs.map(
        lambda x: tokenizer(x['span2_word_text'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    val_dataset = model_inputs
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'span1_text', 'span2_word_text'])

    columns_to_remove = ['span2_text', 'span1_text', 'idx']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    return train_dataset, val_dataset

def wic_loader(tokenizer, soft_prompt_length=0, max_seq_length=128):

    def wic_preprocessor(examples, eval=False):
        examples["processed_sentence1"] = []
        nc1 = []
        nc2 = []
        for sentence1, sentence2, word, start1, end1, start2, end2 in zip(examples["sentence1"], examples["sentence2"],
                                                                          examples["word"], examples["start1"],
                                                                          examples["end1"], examples["start2"],
                                                                          examples["end2"]):
            nc1.append(word + ": " + sentence1)
            nc2.append(word + ": " + sentence2)
        ne1 = examples.add_column("processed_sentence1", nc1)
        ne2 = ne1.add_column("processed_sentence2", nc2)
        return ne2

    raw_dataset = load_dataset('super_glue', 'wic', split=['train', 'validation'])
    train_dataset, val_dataset = raw_dataset[0], raw_dataset[1]

    # Preprocess dataset
    train_dataset = wic_preprocessor(train_dataset)
    val_dataset = wic_preprocessor(val_dataset)

    # Tokenize dataset
    #train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    model_inputs = train_dataset.map(
        lambda x: tokenizer(x['processed_sentence1'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    model_inputs = model_inputs.map(
        lambda x: tokenizer(x['processed_sentence2'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    # if not eval:
    train_dataset = model_inputs
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'processed_sentence1', 'processed_sentence2'])

    # val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    model_inputs = train_dataset.map(
        lambda x: tokenizer(x['processed_sentence1'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    model_inputs = model_inputs.map(
        lambda x: tokenizer(x['processed_sentence2'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    # if not eval:
    val_dataset = model_inputs
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'processed_sentence1', 'processed_sentence2'])

    columns_to_remove = ['processed_sentence1', 'processed_sentence2', 'idx', 'input_ids', 'attention_mask']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    return train_dataset, val_dataset

def copa_loader(tokenizer, soft_prompt_length=0, max_seq_length=128):

    def copa_preprocessor(examples, eval=False):
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"
                examples["text_a"].append(text_a)

            result1 = tokenizer(examples["text_a"], examples["choice1"], max_length=max_seq_length, truncation=True)
            result2 = tokenizer(examples["text_a"], examples["choice2"], max_length=max_seq_length, truncation=True)
            result = {}
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1 and key in result2:
                    result[key] = []
                    for value1, value2 in zip(result1[key], result2[key]):
                        result[key].append([value1, value2])
            return result

    raw_dataset = load_dataset('super_glue', 'copa', split=['train', 'validation'])
    train_dataset, val_dataset = raw_dataset[0], raw_dataset[1]

    # Preprocess dataset
    train_dataset = copa_preprocessor(train_dataset)
    val_dataset = copa_preprocessor(val_dataset)

    # Tokenize dataset
    # train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    model_inputs = train_dataset.map(
        lambda x: tokenizer(x['text_a'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    model_inputs = model_inputs.map(
        lambda x: tokenizer(x['choice 1'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    model_inputs = model_inputs.map(
        lambda x: tokenizer(x['choice 2'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    # if not eval:
    train_dataset = model_inputs
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'raw_labels'])

    # val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length - soft_prompt_length, max_seq_length - soft_prompt_length, 'max_length', eval=False)
    model_inputs = val_dataset.map(
        lambda x: tokenizer(x['text_a'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    model_inputs = model_inputs.map(
        lambda x: tokenizer(x['choice 1'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    model_inputs = model_inputs.map(
        lambda x: tokenizer(x['choice 2'], max_length=max_seq_length - soft_prompt_length, truncation=True,
                            padding='max_length'))
    # if not eval:
    val_dataset = model_inputs
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'raw_labels'])

    columns_to_remove = ['text_a', 'choice1', 'choice2', 'idx', 'label', 'source', 'target']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    return train_dataset, val_dataset


DATASET_LOADERS = {
    'boolq': boolq_loader,
    'copa': copa_loader,
    'wsc': wsc_loader,
    'wic': wic_loader,
    'rte': rte_loader
}

METRIC_LOADERS = {
    'boolq': boolq_metric_loader,
    'rte': rte_metric_loader,
    'wsc': wsc_metric_loader,
    'wic': wic_metric_loader,
    'copa': copa_metric_loader
}
