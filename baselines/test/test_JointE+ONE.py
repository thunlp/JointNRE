#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import threading
import json

ll1 = ctypes.cdll.LoadLibrary   
lib_cnn = ll1("./init_cnn.so")
ll2 = ctypes.cdll.LoadLibrary   
lib_kg = ll2("./init_know.so")
	
class Config(object):
	def __init__(self):
		self.instanceTot = lib_cnn.getInstanceTot()
		self.sequence_size = lib_cnn.getLenLimit()
		self.num_classes = lib_cnn.getRelationTotal()
		self.num_words = lib_cnn.getWordTotal()
		self.num_positions = 2 * lib_cnn.getPositionLimit() + 1
		self.word_size = lib_cnn.getWordDimension()
		self.position_size = 5
		self.embedding_size = self.word_size + self.position_size * 2
		self.filter_size = 3
		self.num_filters = 230
		self.relation_size = self.word_size
		self.dropout_keep_prob = 0.5
		self.l2_lambda = 0.0001
		lib_cnn.setNA(51)
		self.margin = 1.0
		self.nbatches = 100
		self.trainTimes = 15
		self.entityTotal = 0
		self.relationTotal = 0

class Model(object):

	def __init__(self, config):
		sequence_size = config.sequence_size
		num_classes = config.num_classes
		num_words = config.num_words
		num_positions = config.num_positions

		embedding_size = config.embedding_size
		word_size = config.word_size
		position_size = config.position_size
		relation_size = config.relation_size
		filter_size = config.filter_size
		num_filters = config.num_filters
		dropout_keep_prob = config.dropout_keep_prob

		margin = config.margin
		l2_lambda = config.l2_lambda

		self.input_x = tf.placeholder(tf.int32, [None, sequence_size], name = "input_x")
		self.input_p_h = tf.placeholder(tf.int32, [None, sequence_size], name = "input_p_h")
		self.input_p_t = tf.placeholder(tf.int32, [None, sequence_size], name = "input_p_t")
		self.input_r = tf.placeholder(tf.int32, [1, 1], name = "input_r")
		self.input_r_n = tf.placeholder(tf.float32, [1, 1], name = "input_r_n")
		self.input_h = tf.placeholder(tf.int32, [1, 1], name = "input_t")
		self.input_t = tf.placeholder(tf.int32, [1, 1], name = "input_t")
		self.input_y = tf.placeholder(tf.float32, [1, num_classes], name = "input_y")

		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])
		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

		l2_loss = tf.constant(0.0)
		with tf.name_scope("embedding-lookup"):
			self.word_embeddings = tf.Variable(word_embeddings, name="word_embeddings")
			self.relation_embeddings = tf.Variable(relation_embeddings, name="relation_embeddings")
			self.position_embeddings = tf.Variable(position_embeddings, name="position_embeddings")
			self.relation_attention = tf.Variable(relation_attention, name="relation_attention")
			self.NAattention = tf.Variable(NAattention, name="NAattention")
			self.attention = tf.Variable(attention, name="attention")

			#know
			pos_h_e = tf.nn.embedding_lookup(self.word_embeddings, self.pos_h)
			pos_t_e = tf.nn.embedding_lookup(self.word_embeddings, self.pos_t)
			pos_r_e = tf.nn.embedding_lookup(self.relation_embeddings, self.pos_r)
			neg_h_e = tf.nn.embedding_lookup(self.word_embeddings, self.neg_h)
			neg_t_e = tf.nn.embedding_lookup(self.word_embeddings, self.neg_t)
			neg_r_e = tf.nn.embedding_lookup(self.relation_embeddings, self.neg_r)

			#cnn
			self.x_initial = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)
			self.x_p_h = tf.nn.embedding_lookup(self.position_embeddings, self.input_p_h)
			self.x_p_t = tf.nn.embedding_lookup(self.position_embeddings, self.input_p_t)
			self.x = tf.expand_dims(tf.concat(2, [self.x_initial, self.x_p_h, self.x_p_t]), -1)
			self.head = tf.nn.embedding_lookup(self.word_embeddings, self.input_h)
			self.tail = tf.nn.embedding_lookup(self.word_embeddings, self.input_t)
			l2_loss += tf.nn.l2_loss(self.attention)

		with tf.name_scope("conv-maxpool"):
			self.W = tf.Variable(W, name="W")
			self.b = tf.Variable(B, name="b")
			conv = tf.nn.conv2d(self.x, self.W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
			h = tf.nn.tanh(tf.nn.bias_add(conv, self.b), name="tanh")
			self.y = tf.nn.max_pool(h, ksize=[1, sequence_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
			l2_loss += tf.nn.l2_loss(self.W)
			l2_loss += tf.nn.l2_loss(self.b)
			self.y = tf.reshape(self.y, [-1, num_filters])

		with tf.name_scope('attention'):
			self.y_attention = tf.reduce_max(self.y, 0 , keep_dims = True)
			

		with tf.name_scope("dropout"):
			print self.y_attention.get_shape()
			self.transfer_w = tf.Variable(transfer_w, name="transfer_w") 
			self.h_drop = tf.nn.l2_normalize(self.y_attention, 1)
			self.scores = tf.matmul(self.h_drop, self.transfer_w) 
			self.scoress = tf.nn.softmax(tf.reshape(self.scores, [1, -1]))

bags_sum = 0.0
bags_hit_NA = 0.0
sum_NA = 0.0
sum_fNA = 0.0
bags_hit = 0.0
loss_sum = 0.0
ss = []
flag = True

if __name__ == "__main__":
	
	lib_cnn.readWordVec()
	lib_cnn.readFromFile()
	lib_kg.init()

	np.random.seed(0)
	tf.set_random_seed(0)
	config = Config()

	word_embeddings = np.zeros(config.num_words * config.word_size, dtype = np.float32)
	lib_cnn.getWordVec.argtypes = [ctypes.c_void_p]
	lib_cnn.getWordVec(word_embeddings.__array_interface__['data'][0])
	word_embeddings.resize((config.num_words,config.word_size))
	
	config.batch_size = lib_kg.getTripleTotal() / config.nbatches
	config.entityTotal = lib_kg.getEntityTotal()
	config.relationTotal = lib_kg.getRelationTotal()

	config.num_classes = 56
	log = open("log14.txt", "r")
	word_embeddings = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.num_words , config.word_size))
	relation_embeddings = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.relationTotal, config.word_size))
	position_embeddings = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.num_positions, config.position_size))
	relation_attention = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.num_classes, config.relation_size))
	attention = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.num_filters, config.relation_size))
	W = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.filter_size, config.embedding_size, 1, config.num_filters))
	B = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.num_filters))
	transfer_w = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape(config.num_filters, config.num_classes)
	NAattention = np.array(json.loads(log.readline().strip()), dtype = np.float32).reshape((config.relation_size, 1))
	log.close()

	
	with tf.Graph().as_default():
		conf = tf.ConfigProto()
		sess = tf.Session(config=conf)
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				m = Model(config = config)
			
			global_step_cnn = tf.Variable(0, name="global_step_cnn", trainable=False)
			sess.run(tf.initialize_all_variables())
			
			x_batch = np.zeros((config.instanceTot,config.sequence_size), dtype = np.int32)
			p_t_batch = np.zeros((config.instanceTot,config.sequence_size), dtype = np.int32)
			p_h_batch = np.zeros((config.instanceTot,config.sequence_size), dtype = np.int32)
			r_batch = np.zeros((1, 1), dtype = np.int32)
			y_batch = np.zeros((1, config.num_classes), dtype = np.int32)
			r_n_batch = np.zeros((1, 1), dtype = np.int32)
			h_batch = np.zeros((1, 1), dtype = np.int32)
			t_batch = np.zeros((1, 1), dtype = np.int32)

			x_batch_addr = x_batch.__array_interface__['data'][0]
			p_t_batch_addr = p_t_batch.__array_interface__['data'][0]
			p_h_batch_addr = p_h_batch.__array_interface__['data'][0]
			y_batch_addr = y_batch.__array_interface__['data'][0]
			r_batch_addr = r_batch.__array_interface__['data'][0]
			r_n_batch_addr = r_n_batch.__array_interface__['data'][0]
			h_batch_addr = h_batch.__array_interface__['data'][0]
			t_batch_addr = t_batch.__array_interface__['data'][0]
			lib_cnn.batch_iter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
			tipTotal = lib_cnn.getTipTotal()

			def train_step_cnn(x_batch, p_h_batch, p_t_batch, y_batch, r_batch, r_n_batch, h_batch, t_batch,i):
					global ss, bags_sum, bags_hit, loss_sum, bags_hit_NA, bags_hit, sum_fNA, sum_NA, flag
					feed_dict = {
						m.input_x: x_batch,
						m.input_p_h: p_h_batch,
						m.input_p_t: p_t_batch,
						m.input_r: r_batch,
						m.input_r_n: r_n_batch,
						m.input_y: y_batch,
						m.input_h: h_batch,
						m.input_t: t_batch
					}
					scores, step, accuracy, accuracy1,scoresgg,scores1gg = sess.run(
						[m.scoress, global_step_cnn, m.accuracy,m.accuracy1,m.y_attention,m.h_drop], feed_dict)
					time_str = datetime.datetime.now().isoformat()
					for i in range(config.num_classes):
						if (r_batch[0] == i):
							ss.append((i,scores[0][i],1))
						else:
							ss.append((i,scores[0][i],0))
					bags_sum += 1
					if bags_sum % 100 == 0:
						print("{}: step {}".format(time_str, step))

			for i in range(tipTotal):
				length = lib_cnn.batch_iter(x_batch_addr, p_h_batch_addr, p_t_batch_addr, y_batch_addr, r_batch_addr, r_n_batch_addr, h_batch_addr, t_batch_addr)
				train_step_cnn(x_batch[0:length,], p_h_batch[0:length,], p_t_batch[0:length,], y_batch, r_batch, r_n_batch, h_batch, t_batch,i)
				if not flag:
					break

f = open("res.txt", "w")
for i in ss:
	f.write("%d\t%f\t%d\n"%(i[0],i[1],i[2]))
f.close()
