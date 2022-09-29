model_dir=./nmt/models/zh_fr_train
data_dir=./nmt/data/zh_fr_train
fairseq-generate ${data_dir}/data-bin \
    --path ${model_dir}/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 8 > ${data_dir}/result/bestbeam8.txt

