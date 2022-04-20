# P-tuning v2

Source code for ACL 2022 Paper
"[P-Tuning v2: Prompt Tuning Can Be Comparable to Finetuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)"

### Setup
Create conda environment:
```shell
conda create -n pt2 python=3.8.5
conda activate pt2
```

Install required packages:
```shell
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```


### Data
We obtain the SuperGLUE benchmark datasets from Huggingface datasets. .

### Training
Run training scripts in [run_script](run_script) (e.g., RoBERTa for RTE):

```shell
bash run_script/run_rte_roberta.sh
```
