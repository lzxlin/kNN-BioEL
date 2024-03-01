

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
  --data_dir data/benchmarks/cometa/formatted \
  --bert_dir ../models/SapBERT-from-PubMedBERT-fulltext \
  --output_dir save/cometa \
  --save_checkpoint_best \
  --use_eval \
  --batch_size 128 \
  --n_epochs 20 \
  --lr 0.00004 \
  --online_neg_num 4 \
  --temperature 0.01