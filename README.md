# EAdaBERT

```
python ./examples/run_finetune.py --data_dir $SST_DIR --bert_model bert-base-uncased --task_name sst-2 --do_train --do_eval --num_train_epochs 20 --do_lower_case --learning_rate 2e-5 --train_batch_size 32 --eval_batch_size 32 [--svd_weight_dir SVD_WEIGHT_DIR] --output_dir $OUTPUT_DIR --p_encoder 0.231 --p_embd 0.2 --distill_dir /home/yujwang/maoyh/sst_distill_weight
```
