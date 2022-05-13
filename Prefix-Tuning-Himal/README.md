# Prompt Tuning
A Pytorch implementation of [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691v2).

## Usage

To train soft prompts for a dataset on an specific PLM:

```bash
python3 "plm=<plm-name>" "task_name=<name-of-the-task>"
```

Default hyperparameters and configuration have already been provided in config.yaml. These configurations can be overridden via command line args.
