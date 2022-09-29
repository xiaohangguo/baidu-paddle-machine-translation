# 解码然后评价shell

SCRIPTS=/public/home/hang/桌面/test_train/test_train/mosesdecoder/scripts

DETC=${SCRIPTS}/recaser/detruecase.perl
DETOKENIZER=${SCRIPTS}/tokenizer/detokenizer.perl

data_dir=./nmt/data/zh_fr_train
model_dir=./nmt/models/zh_fr_train


#fairseq-generate ${data_dir}/data-bin \
#    --path ${model_dir}/checkpoints/checkpoint_best.pt \
#    --batch-size 256 --beam 16  > ${data_dir}/result/16_bestbeam.txt \
#    --device-id 4
fairseq-interactive ${data_dir}/data-bin \
    --input ${data_dir}/valid.zh \
    --path ${model_dir}/checkpoints/checkpoint_best.pt \
    --batch-size 1 --beam 8 --remove-bpe > ${data_dir}/result/valid_bestbeam8.txt


grep ^H ${data_dir}/result/valid_bestbeam.txt | cut -f3- > ${data_dir}/result/valid_predict.tok.true.bpe.fr
grep ^T ${data_dir}/result/valid_bestbeam.txt | cut -f2- > ${data_dir}/result/valid_answer.tok.true.bpe.fr

sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/result/valid_predict.tok.true.bpe.fr  > ${data_dir}/result/valid_predict.tok.true.fr
sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/result/valid_answer.tok.true.bpe.fr  > ${data_dir}/result/valid_answer.tok.true.fr


perl ${DETC} < ${data_dir}/result/valid_predict.tok.true.fr > ${data_dir}/result/valid_predict.tok.fr
perl ${DETC} < ${data_dir}/result/valid_answer.tok.true.fr > ${data_dir}/result/valid_answer.tok.fr

MULTI_BLEU=${SCRIPTS}/generic/multi-bleu.perl
perl ${MULTI_BLEU} -lc ${data_dir}/result/valid_answer.tok.fr < ${data_dir}/result/valid_predict.tok.fr

#perl ${DETOKENIZER} -l fr < ${data_dir}/result/16_predict.tok.fr > ${data_dir}/result/16_predict.fr

