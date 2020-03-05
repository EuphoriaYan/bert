# -*- coding: utf-8 -*-

export CUDA_VISIBLE_DEVICES=2

config_path=configs/bert.json
input_file='tfrecords/*.*'
output_dir=model_output

python run_pretraining.py \
--bert_config_file ${config_path} \
--input_file "${input_file}" \
--output_dir ${output_dir} \
--do_train True \
--do_eval True \
--num_train_steps 1000000 \
--num_warmup_steps 100000 \
--save_checkpoints_steps 100000
