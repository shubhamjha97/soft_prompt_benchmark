import hydra
from transformers import (
    GPT2TokenizerFast,
    T5Tokenizer,
    RobertaTokenizer,
    AdamW,
    get_scheduler,
    Trainer,
    TrainingArguments,
    default_data_collator
)

from dataset_loaders import DATASET_LOADERS, METRIC_LOADERS
from model import GPT2PromptTuningLM, T5PromptTuningLM, RobertaPromptTuningLM

from util import compute_metric_batched

class Config:
    # Same default parameters as run_clm_no_trainer.py in transformers
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py
    num_train_epochs = 3
    weight_decay = 0.01
    learning_rate = 0.01
    lr_scheduler_type = "linear"
    num_warmup_steps = 0
    max_train_steps = num_train_epochs

    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 20
    # If True, soft prompt will be initialized from vocab
    # Otherwise, you can set `random_range` to initialize by randomization.
    init_from_vocab = True
    # random_range = 0.5

def train(tokenizer, model, train_dataset, val_dataset, config, metrics):
    model.train()
    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="no",
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        eval_accumulation_steps=5,
        num_train_epochs=1,
        prediction_loss_only=False
    )

    # Only update soft prompt weights for prompt-tuning. ie, all weights in LM are set as `require_grad=False`.
    optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": config.weight_decay,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.max_train_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=None, #partial(compute_metrics, metric=metrics, tokenizer=tokenizer, config=config),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        optimizers=(optimizer, lr_scheduler)
    )

    for epoch in range(config.num_train_epochs):
        trainer.train(resume_from_checkpoint=None) # TODO: sjha add ability to resume from checkpoint
        computed_metrics = compute_metric_batched(trainer, metrics, tokenizer, val_dataset, eval_batch_size=config.eval_batch_size)
        print(f'epoch: {epoch}, eval_metrics: {computed_metrics}')

    # TODO: save model every n iterations
    save_dir_path = "."
    model.save_soft_prompt(save_dir_path)


def get_tokenizer(tokenizer_name="gpt2"):
    if tokenizer_name.startswith("gpt2"):
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer_name.startswith("t5"):
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError("Tokenizer not supported")

    return tokenizer


def get_model(model_name="gpt2", n_prompt_tokens=20, init_from_vocab=True):
    if model_name.startswith("gpt2"):
        model = GPT2PromptTuningLM.from_pretrained(
            model_name,
            n_tokens=n_prompt_tokens,
            initialize_from_vocab=init_from_vocab)
    elif model_name.startswith("t5"):
        model = T5PromptTuningLM.from_pretrained(
            model_name,
            n_tokens=n_prompt_tokens,
            initialize_from_vocab=init_from_vocab
        )
    elif model_name.startswith("roberta"):
        model = RobertaPromptTuningLM.from_pretrained(
            model_name,
            n_tokens=n_prompt_tokens,
            initialize_from_vocab=init_from_vocab
        )
    else:
        raise ValueError("Tokenizer not supported")
    return model


def get_config(cfg):
    config = Config()

    task_name, method, plm = cfg.task_name, cfg.method, cfg.plm
    task_cfg = cfg[task_name][method][plm]

    config.model_name = plm
    config.dataset = task_name
    config.task_name = task_name
    config.tokenizer_name = task_cfg.tokenizer_name
    config.logging_steps = cfg.logging_steps
    config.eval_steps = cfg.eval_steps
    config.learning_rate = task_cfg.learning_rate
    config.num_train_epochs = task_cfg.num_train_epochs
    config.n_prompt_tokens = task_cfg.n_prompt_tokens
    config.max_seq_length = task_cfg.max_seq_length
    config.init_from_vocab = task_cfg.init_from_vocab
    config.random_range = task_cfg.random_range
    config.eval_batch_size = cfg.eval_batch_size

    return config


@hydra.main(config_name='config')
def main(cfg):
    config = get_config(cfg)
    tokenizer = get_tokenizer(config.tokenizer_name)
    model = get_model(config.model_name, config.n_prompt_tokens, config.init_from_vocab)
    metrics = METRIC_LOADERS[config.task_name]()

    train_dataset, val_dataset = DATASET_LOADERS[config.task_name](
        tokenizer,
        soft_prompt_length=config.n_prompt_tokens,
        max_seq_length=config.max_seq_length
    )
    train(tokenizer, model, train_dataset, val_dataset, config, metrics)


if __name__ == '__main__':
    main()
