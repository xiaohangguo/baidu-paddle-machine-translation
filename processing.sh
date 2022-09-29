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


#标点符号标准化 ,normalize-punctuation (可以不做)
perl ${NORM_PUNC} -l fr < ${data_dir}/raw.fr > ${data_dir}/norm.fr
perl ${NORM_PUNC} -l zh < ${data_dir}/raw.zh > ${data_dir}/norm.zh

#jieba分词
python -m jieba -d " " ${data_dir}/norm.zh > ${data_dir}/norm.seg.zh

 
#tokenize 标记化处理
perl ${TOKENIZER} -l fr < ${data_dir}/norm.fr > ${data_dir}/norm.tok.fr
perl ${TOKENIZER} -l zh < ${data_dir}/norm.seg.zh > ${data_dir}/norm.seg.tok.zh

#这里在英文里面是做大小写转换处理，会学习大小写的形式。我魔改的法语，我还不清楚法语的，语法。这里语义上可能有问题。**这一步可能有问题
perl ${TRAIN_TC} --model ${model_dir}/truecase-model.fr --corpus ${data_dir}/norm.tok.fr
perl ${TC} --model ${model_dir}/truecase-model.fr < ${data_dir}/norm.tok.fr > ${data_dir}/norm.tok.true.fr

#beq——fr子句处理
python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.tok.true.fr  -s 32000 -o ${model_dir}/bpecode.fr --write-vocabulary ${model_dir}/voc.fr
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.fr --vocabulary ${model_dir}/voc.fr < ${data_dir}/norm.tok.true.fr > ${data_dir}/norm.tok.true.bpe.fr

#beq——zh子句处理
python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.seg.tok.zh  -s 32000 -o ${model_dir}/bpecode.zh --write-vocabulary ${model_dir}/voc.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.zh --vocabulary ${model_dir}/voc.zh < ${data_dir}/norm.seg.tok.zh > ${data_dir}/norm.seg.tok.bpe.zh

#过滤最小长度和最大长度(一个范围)之间的句对
mv ${data_dir}/norm.seg.tok.bpe.zh ${data_dir}/toclean.zh
mv ${data_dir}/norm.tok.true.bpe.fr ${data_dir}/toclean.fr 
perl ${CLEAN} ${data_dir}/toclean zh fr ${data_dir}/clean 1 256


#数据集切分
python ${utils}/split.py ${data_dir}/clean.zh ${data_dir}/clean.fr ${data_dir}/

#生成dict 词表和二进制文件
fairseq-preprocess -s ${src} -t ${tgt} --trainpref ${data_dir}/train --validpref ${data_dir}/valid --testpref ${data_dir}/test --destdir ${data_dir}/data-bin

# 训练，设置的保存一个checkpoint，多了很占内存，不过他会保存best的点，和last的点
CUDA_VISIBLE_DEVICES=1 nohup fairseq-train ${data_dir}/data-bin --arch transformer \
	--source-lang ${src} --target-lang ${tgt}  \
    --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 1 --num-workers 8 \
	--save-dir ${model_dir}/checkpoints &

