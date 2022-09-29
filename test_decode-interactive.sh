data_dir=./nmt/data/zh_fr_train
model_dir=./nmt/models/zh_fr_train

CUDA_LAUNCH_BLOCKING=1
fairseq-interactive ${data_dir}/data-bin \
    --input ${data_dir}/ptest_toclean_test.zh \
    --path ${model_dir}/checkpoints/checkpoint_best.pt \
    --batch-size 1 --beam 8 --remove-bpe > ${data_dir}/result/bestbeam8.txt

