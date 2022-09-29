src=zh
tgt=fr
data_dir=./nmt/data/zh_fr_train

fairseq-preprocess -s ${src} -t ${tgt} --trainpref ${data_dir}/train --validpref ${data_dir}/valid --testpref ${data_dir}/test --destdir ${data_dir}/data-bin
