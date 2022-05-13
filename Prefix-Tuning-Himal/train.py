import hydra
from opendelta import SoftPromptModel, PrefixModel
from transformers import (
    GPT2TokenizerFast,
    T5Tokenizer,
    RobertaTokenizer,
    AdamW,
    get_scheduler,
    get_constant_schedule,
    Trainer,
    TrainingArguments,
    default_data_collator, T5ForConditionalGeneration, GPT2LMHeadModel
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

    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if p.requires_grad],
        "weight_decay": config.weight_decay,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # TODO:
    # lr_scheduler = get_scheduler(
    #     name="linear", #config.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=config.num_warmup_steps,
    #     num_training_steps= config.num_train_epochs,
    # )
    lr_scheduler = get_constant_schedule(optimizer=optimizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        optimizers=(optimizer, lr_scheduler)
    )

    max_accuracy = 0.0
    for epoch in range(config.num_train_epochs):
        trainer.train(resume_from_checkpoint=None) # TODO: sjha add ability to resume from checkpoint

        if epoch % 30 == 0: # TODO: remove
            computed_metrics = compute_metric_batched(trainer, metrics, tokenizer, val_dataset, # TODO
                                                  eval_batch_size=config.eval_batch_size, config=config)
            max_accuracy = max(max_accuracy, computed_metrics['accuracy'])
            print(f'epoch: {epoch}, eval_metrics: {computed_metrics}, learning_rate: {lr_scheduler.get_lr()}, max_accuracy: {max_accuracy}')
        else:
            print(f'epoch: {epoch}, learning_rate: {lr_scheduler.get_lr()}')

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

    config.num_warmup_steps = task_cfg.num_warmup_steps
    config.num_training_steps = task_cfg.num_training_steps

    return config


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    config = get_config(cfg)
    tokenizer = get_tokenizer(config.tokenizer_name)
    # model = get_model(config.model_name, config.n_prompt_tokens, config.init_from_vocab) # TODO:

    model = T5ForConditionalGeneration.from_pretrained(config.model_name) # TODO:
    # model = GPT2LMHeadModel.from_pretrained(config.model_name)
    #delta_model = SoftPromptModel(backbone_model=model, soft_token_num=config.n_prompt_tokens, token_init=config.init_from_vocab)
    delta_model = PrefixModel(backbone_model=model, prefix_token_num=config.n_prompt_tokens, reparameterize=False)
    delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True) # TODO
    delta_model.log() # This will visualize the backbone after modification and other information.

    metrics = METRIC_LOADERS[config.task_name]()

    train_dataset, val_dataset = DATASET_LOADERS[config.task_name](
        tokenizer,
        soft_prompt_length=config.n_prompt_tokens,
        max_seq_length=config.max_seq_length
    )

    # print (train_dataset['input_ids'])
    # return
    # print((train_dataset['raw_labels']))

    #return
    train(tokenizer, delta_model, train_dataset, val_dataset, config, metrics)


if __name__ == '__main__':
    main()
