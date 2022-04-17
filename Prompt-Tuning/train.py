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


class Config:
    # Same default parameters as run_clm_no_trainer.py in tranformers
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
        evaluation_strategy="steps",
        logging_steps=50,
        eval_steps=300,
        eval_accumulation_steps=5,
        prediction_loss_only=False # TODO: Debug
    )

    # Only update soft prompt'weights for prompt-tuning. ie, all weights in LM are set as `require_grad=False`.
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
        compute_metrics=metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train(resume_from_checkpoint=None) # TODO: sjha add ability to resume from checkpoint

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


@hydra.main(config_name='config')
def main(cfg):
    task_name, method, plm = cfg.task_name, cfg.method, cfg.plm
    task_cfg = cfg[task_name][method][plm]

    config = Config()
    config.tokenizer_name = task_cfg.tokenizer_name
    config.model_name = plm
    config.dataset = task_name
    # TODO: Add other params like learning rate

    tokenizer = get_tokenizer(config.tokenizer_name)
    model = get_model(config.model_name, config.n_prompt_tokens, config.init_from_vocab)
    metrics = METRIC_LOADERS[task_name]()

    train_dataset, val_dataset = DATASET_LOADERS[task_name](tokenizer)
    train(tokenizer, model, train_dataset, val_dataset, config, metrics)


if __name__ == '__main__':
    main()
