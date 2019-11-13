# JointNRE

Codes and datasets for our paper "Neural Knowledge Acquisition via Mutual Attention between Knowledge Graph and Text"


Some Introduction
===

This implementation is a fast and stable version. 

We have made some simplifications for the original model so that to train a joint model just needs around 15min.

We also encapsulate more neural architectures into our framework to encode sentences.

The code and datasets mainly for the task relation extraction.

Data
==========

We provide the datasets used for the task relation extraction.

New York Times Corpus: The data used in relation extraction from text is published by "Modeling relations and their mentions without labeled text". The data should be obtained from [[LDC]](https://catalog.ldc.upenn.edu/LDC2008T19) first.

Datasets are required in the folder data/ in the following format, containing at least 4 files:

+ kg/train.txt: the knowledge graph for training, format (e1, e2, rel).

+ text/relation2id.txt: the relation needed to be predicted for RE, format (rel, id).

+ text/train.txt: the text for training, format (e1, e2, name1, name2, rel, sentence).

+ text/vec.txt: the initial word embeddings.

+ [[Download (Baidu Cloud)]](https://pan.baidu.com/s/1q7rctsoJ_YdlLa55yckwbQ)
+ [[Download (Tsinghua Cloud)]](https://cloud.tsinghua.edu.cn/f/28ba8ac5262349dd9622/?dl=1)



Run the experiments
==========

### To run the experiments, unpack the datasets first:

```
unzip origin_data.zip -d origin_data/
mkdir data/
python initial.py
```

### Run the corresponding python scripts to train models:

```
cd jointE
bash make.sh
python train.py
```

### Change the corresponding python code to set hyperparameters:

```
tf.app.flags.DEFINE_float('nbatch_kg',100,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_float('ent_total',lib.getEntityTotal(),'total of entities')
tf.app.flags.DEFINE_float('rel_total',lib.getRelationTotal(),'total of relations')
tf.app.flags.DEFINE_float('tri_total',lib.getTripleTotal(),'total of triples')
tf.app.flags.DEFINE_float('katt_flag', 1, '1 for katt, 0 for att')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_float('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_float('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_float('num_classes', config['textual_rel_total'],'maximum of relations')

tf.app.flags.DEFINE_float('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_float('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_float('max_epoch',20,'maximum of training epochs')
tf.app.flags.DEFINE_float('batch_size',160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'learning rate for nn')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')
```

### Run the corresponding python scripts to test models:

```
cd jointE
bash make.sh
python test.py
```

Note that the hyperparameters in the train.py and the test.py must be the same.

### Run the corresponding python script to get PR-curve results:

```
cd jointE
python pr_plot.py
```

Citation
===

```
 @inproceedings{han2018neural,
   title={Neural Knowledge Acquisition via Mutual Attention between Knowledge Graph and Text},
   author={Han, Xu and Liu, Zhiyuan and Sun, Maosong},
   booktitle={Proceedings of AAAI},
   year={2018}
 }
```



 




