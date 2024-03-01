

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
  --data_dir data/benchmarks/ncbi/formatted \
  --bert_dir ../models/SapBERT-from-PubMedBERT-fulltext \
  --output_dir save/ncbi \
  --save_checkpoint_best \
  --use_eval \
  --batch_size 64 \
  --n_epochs 3 \
  --lr 0.00005 \
  --online_neg_num 0 \
  --temperature 0.01