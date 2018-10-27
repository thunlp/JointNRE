import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
import sys
from sklearn.metrics import average_precision_score
import ctypes

export_path = "../data/"

word_vec = np.load(export_path + 'vec.npy')
f = open(export_path + "config", 'r')
config = json.loads(f.read())
f.close()

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")
lib.setInPath("../data/")
lib.init()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nbatch_kg',100,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_integer('ent_total',lib.getEntityTotal(),'total of entities')
tf.app.flags.DEFINE_integer('rel_total',lib.getRelationTotal(),'total of relations')
tf.app.flags.DEFINE_integer('tri_total',lib.getTripleTotal(),'total of triples')
tf.app.flags.DEFINE_integer('katt_flag', 1, '1 for katt, 0 for att')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', config['textual_rel_total'],'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch',30,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size',131*2,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.1,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',1.0,'dropout rate')

tf.app.flags.DEFINE_integer('test_batch_size',131*2,'entity numbers used each test time')
tf.app.flags.DEFINE_string('checkpoint_path','./model/','path to store model')


def make_shape(array,last_dim):
	output = []
	for i in array:
		for j in i:
			output.append(j)
	output = np.array(output)
	if np.shape(output)[-1]==last_dim:
		return output

	else:
		print 'Make Shape Error!'

def main(_):

	print 'reading word embedding'
	word_vec = np.load(export_path + 'vec.npy')
	print 'reading test data'
	test_instance_triple = np.load(export_path + 'test_instance_triple.npy')
	test_instance_scope = np.load(export_path + 'test_instance_scope.npy')
	test_len = np.load(export_path + 'test_len.npy')
	test_label = np.load(export_path + 'test_label.npy')
	test_word = np.load(export_path + 'test_word.npy')
	test_pos1 = np.load(export_path + 'test_pos1.npy')
	test_pos2 = np.load(export_path + 'test_pos2.npy')
	test_mask = np.load(export_path + 'test_mask.npy')
	test_head = np.load(export_path + 'test_head.npy')
	test_tail = np.load(export_path + 'test_tail.npy')
	print 'reading finished'
	print 'mentions 		: %d' % (len(test_instance_triple))
	print 'sentences		: %d' % (len(test_len))
	print 'relations		: %d' % (FLAGS.num_classes)
	print 'word size		: %d' % (len(word_vec[0]))
	print 'position size 	: %d' % (FLAGS.pos_size)
	print 'hidden size		: %d' % (FLAGS.hidden_size)
	print 'reading finished'

	print 'building network...'
	sess = tf.Session()
	if FLAGS.model.lower() == "cnn":
		model = network.CNN(is_training = False, word_embeddings = word_vec)
	elif FLAGS.model.lower() == "pcnn":
		model = network.PCNN(is_training = False, word_embeddings = word_vec)
	elif FLAGS.model.lower() == "lstm":
		model = network.RNN(is_training = False, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "gru":
		model = network.RNN(is_training = False, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
		model = network.BiRNN(is_training = False, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
		model = network.BiRNN(is_training = False, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	def test_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope):
		feed_dict = {
			model.head_index: head,
			model.tail_index: tail,
			model.word: word,
			model.pos1: pos1,
			model.pos2: pos2,
			model.mask: mask,
			model.len : leng,
			model.label_index: label_index,
			model.label: label,
			model.scope: scope,
			model.keep_prob: FLAGS.keep_prob
		}
		output = sess.run(model.test_output, feed_dict)
		return output

	f = open('results.txt','w')
	f.write('iteration\taverage precision\n')
	for iters in range(1,30):
		print iters
		saver.restore(sess, FLAGS.checkpoint_path + FLAGS.model+str(FLAGS.katt_flag)+"-"+str(3664*iters))

		stack_output = []
		stack_label = []
		
		iteration = len(test_instance_scope)/FLAGS.test_batch_size
		for i in range(iteration):
			temp_str= 'running '+str(i)+'/'+str(iteration)+'...'
			sys.stdout.write(temp_str+'\r')
			sys.stdout.flush()
			input_scope = test_instance_scope[i * FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size]
			index = []
			scope = [0]
			label = []
			for num in input_scope:
				index = index + range(num[0], num[1] + 1)
				label.append(test_label[num[0]])
				scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
			label_ = np.zeros((FLAGS.test_batch_size, FLAGS.num_classes))
			label_[np.arange(FLAGS.test_batch_size), label] = 1
			output = test_step(test_head[index], test_tail[index], test_word[index,:], test_pos1[index,:], test_pos2[index,:], test_mask[index,:], test_len[index], test_label[index], label_, np.array(scope))
			stack_output.append(output)
			stack_label.append(label_)
			
		print 'evaluating...'

		stack_output = np.concatenate(stack_output, axis=0)
		stack_label = np.concatenate(stack_label, axis = 0)

		exclude_na_flatten_output = stack_output[:,1:]
		exclude_na_flatten_label = stack_label[:,1:]
		print exclude_na_flatten_output.shape
		print exclude_na_flatten_label.shape

		average_precision = average_precision_score(exclude_na_flatten_label,exclude_na_flatten_output, average = "micro")

		np.save('./'+FLAGS.model+'+sen_att_all_prob_'+str(iters)+'.npy', exclude_na_flatten_output)
		np.save('./'+FLAGS.model+'+sen_att_all_label_'+str(iters)+'.npy',exclude_na_flatten_label)

		print 'pr: '+str(average_precision)
		f.write(str(average_precision)+'\n')
	f.close()

if __name__ == "__main__":
	tf.app.run()
