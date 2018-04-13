import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys
import ctypes
import threading

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

tf.app.flags.DEFINE_float('nbatch_kg',100,'entity numbers used each training time')
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

tf.app.flags.DEFINE_integer('max_epoch',20,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size',160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'learning rate for nn')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')


def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

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
	print 'reading training data'
	
	instance_triple = np.load(export_path + 'train_instance_triple.npy')
	instance_scope = np.load(export_path + 'train_instance_scope.npy')
	train_len = np.load(export_path + 'train_len.npy')
	train_label = np.load(export_path + 'train_label.npy')
	train_word = np.load(export_path + 'train_word.npy')
	train_pos1 = np.load(export_path + 'train_pos1.npy')
	train_pos2 = np.load(export_path + 'train_pos2.npy')
	train_mask = np.load(export_path + 'train_mask.npy')
	train_head = np.load(export_path + 'train_head.npy')
	train_tail = np.load(export_path + 'train_tail.npy')

	print 'reading finished'
	print 'mentions 		: %d' % (len(instance_triple))
	print 'sentences		: %d' % (len(train_len))
	print 'relations		: %d' % (FLAGS.num_classes)
	print 'word size		: %d' % (len(word_vec[0]))
	print 'position size 	: %d' % (FLAGS.pos_size)
	print 'hidden size		: %d' % (FLAGS.hidden_size)
	reltot = {}
	for index, i in enumerate(train_label):
		if not i in reltot:
			reltot[i] = 1.0
		else:
			reltot[i] += 1.0
	for i in reltot:
		reltot[i] = 1/(reltot[i] ** (0.05)) 
	print 'building network...'
	sess = tf.Session()
	if FLAGS.model.lower() == "cnn":
		model = network.CNN(is_training = True, word_embeddings = word_vec)
	elif FLAGS.model.lower() == "pcnn":
		model = network.PCNN(is_training = True, word_embeddings = word_vec)
	elif FLAGS.model.lower() == "lstm":
		model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "gru":
		model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
		model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
		model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	
	global_step = tf.Variable(0,name='global_step',trainable=False)
	global_step_kg = tf.Variable(0,name='global_step_kg',trainable=False)
	tf.summary.scalar('learning_rate', FLAGS.learning_rate)
	tf.summary.scalar('learning_rate_kg', FLAGS.learning_rate_kg)

	optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
	grads_and_vars = optimizer.compute_gradients(model.loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

	optimizer_kg = tf.train.GradientDescentOptimizer(FLAGS.learning_rate_kg)
	grads_and_vars_kg = optimizer_kg.compute_gradients(model.loss_kg)
	train_op_kg = optimizer_kg.apply_gradients(grads_and_vars_kg, global_step = global_step_kg)

	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=None)
	print 'building finished'

	def train_kg(coord):
		def train_step_kg(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
			feed_dict = {
				model.pos_h: pos_h_batch,
				model.pos_t: pos_t_batch,
				model.pos_r: pos_r_batch,
				model.neg_h: neg_h_batch,
				model.neg_t: neg_t_batch,
				model.neg_r: neg_r_batch
			}
			_, step, loss = sess.run(
				[train_op_kg, global_step_kg, model.loss_kg], feed_dict)
			return loss

		batch_size = (FLAGS.ent_total / FLAGS.nbatch_kg)
		ph = np.zeros(batch_size, dtype = np.int32)
		pt = np.zeros(batch_size, dtype = np.int32)
		pr = np.zeros(batch_size, dtype = np.int32)
		nh = np.zeros(batch_size, dtype = np.int32)
		nt = np.zeros(batch_size, dtype = np.int32)
		nr = np.zeros(batch_size, dtype = np.int32)
		ph_addr = ph.__array_interface__['data'][0]
		pt_addr = pt.__array_interface__['data'][0]
		pr_addr = pr.__array_interface__['data'][0]
		nh_addr = nh.__array_interface__['data'][0]
		nt_addr = nt.__array_interface__['data'][0]
		nr_addr = nr.__array_interface__['data'][0]
		lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
		times_kg = 0
		while not coord.should_stop():
			times_kg += 1
			res = 0.0
			for batch in range(FLAGS.nbatch_kg):
				lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, batch_size)
				res += train_step_kg(ph, pt, pr, nh, nt, nr)
			time_str = datetime.datetime.now().isoformat()
			print "batch %d time %s | loss : %f" % (times_kg, time_str, res)


	def train_nn(coord):
		def train_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope, weights):
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
				model.keep_prob: FLAGS.keep_prob,
				model.weights: weights
			}
			_, step, loss, summary, output, correct_predictions = sess.run([train_op, global_step, model.loss, merged_summary, model.output, model.correct_predictions], feed_dict)
			summary_writer.add_summary(summary, step)
			return output, loss, correct_predictions

		stack_output = []
		stack_label = []
		stack_ce_loss = []

		train_order = range(len(instance_triple))

		save_epoch = 2
		eval_step = 300

		for one_epoch in range(FLAGS.max_epoch):

			print('epoch '+str(one_epoch+1)+' starts!')
			np.random.shuffle(train_order)
			s1 = 0.0
			s2 = 0.0
			tot1 = 0.0
			tot2 = 0.0
			losstot = 0.0
			for i in range(int(len(train_order)/float(FLAGS.batch_size))):
				input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
				index = []
				scope = [0]
				label = []
				weights = []
				for num in input_scope:
					index = index + range(num[0], num[1] + 1)
					label.append(train_label[num[0]])
					if train_label[num[0]] > 53:
						print train_label[num[0]]
					scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
					weights.append(reltot[train_label[num[0]]])
				label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
				label_[np.arange(FLAGS.batch_size), label] = 1
				output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:], train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights)
				num = 0
				s = 0
				losstot += loss
				for num in correct_predictions:
					if label[s] == 0:
						tot1 += 1.0
						if num:
							s1+= 1.0
					else:
						tot2 += 1.0
						if num:
							s2 += 1.0
					s = s + 1

				time_str = datetime.datetime.now().isoformat()
				print "batch %d step %d time %s | loss : %f, NA accuracy: %f, not NA accuracy: %f" % (one_epoch, i, time_str, loss, s1 / tot1, s2 / tot2)
				current_step = tf.train.global_step(sess, global_step)

			if (one_epoch + 1) % save_epoch == 0:
				print 'epoch '+str(one_epoch+1)+' has finished'
				print 'saving model...'
				path = saver.save(sess,FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag), global_step=current_step)
				print 'have savde model to '+path

		coord.request_stop()


	coord = tf.train.Coordinator()
	threads = []
	threads.append(threading.Thread(target=train_kg, args=(coord,)))
	threads.append(threading.Thread(target=train_nn, args=(coord,)))
	for t in threads: t.start()
	coord.join(threads)

if __name__ == "__main__":
	tf.app.run() 
