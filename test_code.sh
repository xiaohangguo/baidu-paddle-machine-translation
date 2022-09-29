#!/bin/sh

src=zh
tgt=fr

SCRIPTS=/home/hang/桌面/test_train/mosesdecoder/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
DETOKENIZER=${SCRIPTS}/tokenizer/detokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
TRAIN_TC=${SCRIPTS}/recaser/train-truecaser.perl
TC=${SCRIPTS}/recaser/truecase.perl
DETC=${SCRIPTS}/recaser/detruecase.perl
NORM_PUNC=${SCRIPTS}/tokenizer/normalize-punctuation.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl
BPEROOT=/home/hang/桌面/test_train/subword-nmt/subword_nmt
MULTI_BLEU=${SCRIPTS}/generic/multi-bleu.perl
MTEVAL_V14=${SCRIPTS}/generic/mteval-v14.pl

data_dir=./nmt/data/zh_fr_train
model_dir=./nmt/models/zh_fr_train
utils=./nmt/utils



perl ${NORM_PUNC} -l zh < ${data_dir}/zh_fr.test.zh > ${data_dir}/norm_ptest.zh

python -m jieba -d " " ${data_dir}/norm_ptest.zh > ${data_dir}/norm_ptest.seg.zh


perl ${TOKENIZER} -l zh < ${data_dir}/norm_ptest.seg.zh > ${data_dir}/norm_ptest.seg.tok.zh



python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm_ptest.seg.tok.zh  -s 32000 -o ${model_dir}/ptest_bpecode.zh --write-vocabulary ${model_dir}/ptest_voc.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ptest_bpecode.zh --vocabulary ${model_dir}/ptest_voc.zh < ${data_dir}/norm_ptest.seg.tok.zh > ${data_dir}/norm_ptest.seg.tok.bpe.zh

mv ${data_dir}/norm_ptest.seg.tok.bpe.zh ${data_dir}/ptest_toclean_test.zh




