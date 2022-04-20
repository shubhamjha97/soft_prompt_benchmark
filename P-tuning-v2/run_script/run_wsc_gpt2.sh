export TASK_NAME=superglue
export DATASET_NAME=wsc
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=5e-4
dropout=0.1
psl=8
epoch=50

python3 run.py \
  --model_name_or_path gpt2-medium \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-gpt2/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 13 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix \
  --prefix_projection yes