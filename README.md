# baidu-paddle-machine-translation
这份代码是参加百度机器翻译比赛https://aistudio.baidu.com/aistudio/competition/detail/477/0/introduction
## 用fairseq训练的机器翻译模型

参考博客https://blog.csdn.net/qq_42734797/article/details/112916511
  魔改了一份代码，pre中是预处理的脚本，还有一些脚本.
  processing是预处理到训练的代码，下载数据后开箱即用。
## 数据集
https://dataset-bj.cdn.bcebos.com/qianyan/datasets.tar.gz 不知道什么时候过期

data-bin:预处理后的一些文件，用于训练模型

result-- 有两种方法.sh，里面有详细生成代码。

解码以后 做的bleu计算。

写于2022国庆，因为训练效果差，所以不多做记录，小语种训练出来纯牛马模型。
## 附件
附上一些文件包括：checkpoint、团队最后提交result、预处理数据data-bin、数据集dataset
https://pan.baidu.com/s/1__Fw5Eye5GOh4GKhD-uA_w

顾威学长分享的baseline代码
https://www.kaggle.com/code/shuhanglv/ikcest-2022-baseline/edit
