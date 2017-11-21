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
		self.NA = 51
		lib_cnn.setNA(self.NA)
		lib_cnn.setRate(3)
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
		self.input_h = tf.placeholder(tf.int32, [1, 1], name = "input_h")
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
			self.relation_embeddings = tf.get_variable("relation_embeddings", [config.relationTotal, word_size])
			self.position_embeddings = tf.get_variable("position_embeddings", [num_positions, position_size])
			self.relation_attention = tf.get_variable("relation_attention", [num_classes, relation_size])
			self.NAattention = tf.get_variable("NAattention", [relation_size, 1])
			self.attention = tf.get_variable("attention", [num_filters, relation_size])
			self.r = tf.nn.embedding_lookup(self.attention, self.input_r)

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
			self.W = tf.get_variable("W", [filter_size, embedding_size, 1, num_filters])
			self.b = tf.get_variable("b", [num_filters])
			conv = tf.nn.conv2d(self.x, self.W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
			h = tf.nn.tanh(tf.nn.bias_add(conv, self.b), name="tanh")
			self.y = tf.nn.max_pool(h, ksize=[1, sequence_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
			l2_loss += tf.nn.l2_loss(self.W)
			l2_loss += tf.nn.l2_loss(self.b)
			self.y = tf.reshape(self.y, [-1, num_filters])

		with tf.name_scope('attention'):
			self.r = tf.reshape(self.r, [relation_size, -1])
			self.e = tf.matmul(tf.matmul(self.y, self.attention), self.r)
			alpha = tf.reshape(self.e, [1, -1])
			self.alpha_reshape = tf.nn.softmax(alpha)
			self.y_attention = tf.matmul(self.alpha_reshape, self.y)

		with tf.name_scope("dropout"):
			self.y_attention = tf.nn.l2_normalize(self.y_attention, 1)
			self.h_drop = tf.nn.dropout(self.y_attention, dropout_keep_prob)
			self.transfer_w = tf.get_variable("transfer_w", [num_filters, num_classes])
			self.scores = tf.matmul(self.h_drop, self.transfer_w)
			l2_loss += tf.nn.l2_loss(self.transfer_w)

		with tf.name_scope("loss"):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss_cnn = tf.reduce_mean(cross_entropy) + l2_lambda * l2_loss

			pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
			self.loss_kg = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

		with tf.name_scope("accuracy"):
			self.predictions = tf.argmax(self.scores, 1, name="predictions")
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
			

bags_sum = 0.0
bags_hit_NA = 0.0
sum_NA = 0.0
sum_fNA = 0.0
bags_hit = 0.0
loss_sum = 0.0

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

	with tf.Graph().as_default():
		conf = tf.ConfigProto()
		sess = tf.Session(config=conf)
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				m = Model(config = config)
			
			global_step_cnn = tf.Variable(0, name="global_step_cnn", trainable=False)

			optimizer_cnn = tf.train.GradientDescentOptimizer(0.01)
			grads_and_vars_cnn = optimizer_cnn.compute_gradients(m.loss_cnn)
			train_op_cnn = optimizer_cnn.apply_gradients(grads_and_vars_cnn, global_step = global_step_cnn)

			global_step_kg = tf.Variable(0, name="global_step_kg", trainable=False)
			optimizer_kg = tf.train.GradientDescentOptimizer(0.001)
			grads_and_vars_kg = optimizer_kg.compute_gradients(m.loss_kg)
			train_op_kg = optimizer_kg.apply_gradients(grads_and_vars_kg, global_step=global_step_kg)

			sess.run(tf.initialize_all_variables())

			def outEmbedding(str1):
				word_embeddings, relation_embeddings, position_embeddings, relation_attention, attention, W, B, transfer_w, transfer_b, softmax_w, softmax_b = sess.run([m.word_embeddings, m.relation_embeddings, m.position_embeddings, m.relation_attention, m.attention, m.W, m.b, m.transfer_w, m.transfer_b, m.softmax_w, m.softmax_b])
				log = open("log"+str1+".txt", "w")
				log.write(json.dumps(word_embeddings.tolist())+"\n")
				log.write(json.dumps(relation_embeddings.tolist())+"\n")
				log.write(json.dumps(position_embeddings.tolist())+"\n")
				log.write(json.dumps(relation_attention.tolist())+"\n")
				log.write(json.dumps(attention.tolist())+"\n")
				log.write(json.dumps(W.tolist())+"\n")
				log.write(json.dumps(B.tolist())+"\n")
				log.write(json.dumps(transfer_w.tolist())+"\n")
				NAattention = sess.run(m.NAattention)
				log.write(json.dumps(NAattention.tolist()) + "\n")
				log.close()

			x_batch = np.zeros((config.instanceTot,config.sequence_size), dtype = np.int32)
			p_t_batch = np.zeros((config.instanceTot,config.sequence_size), dtype = np.int32)
			p_h_batch = np.zeros((config.instanceTot,config.sequence_size), dtype = np.int32)
			r_batch = np.zeros((1, 1), dtype = np.int32)
			y_batch = np.zeros((1, config.num_classes), dtype = np.int32)
			r_n_batch = np.zeros((1, 1), dtype = np.float32)
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
			loop = 0

			def train_cnn(coord):
				def train_step_cnn(x_batch, p_h_batch, p_t_batch, y_batch, r_batch, r_n_batch, h_batch, t_batch):
					global bags_sum, bags_hit, loss_sum, bags_hit_NA, bags_hit, sum_fNA, sum_NA
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
					_, step, loss, accuracy = sess.run(
					 	[train_op_cnn, global_step_cnn, m.loss_cnn, m.accuracy], feed_dict)
				 	time_str = datetime.datetime.now().isoformat()
				 	loss_sum += loss
				 	bags_sum += 1
				 	if (r_batch[0]!=config.NA):
				 		sum_fNA += 1
				 		if accuracy > 0.5:
				 			bags_hit += 1.0
				 	else:
				 		sum_NA += 1
				 		if accuracy > 0.5:
				 			bags_hit_NA += 1.0
				 	if bags_sum % 1000 == 0:
				 		if (sum_NA == 0):
				 			sum_NA+=1
				 		if (sum_fNA == 0):
				 			sum_fNA+=1
				 		print("{}: step {}, loss {:g}, acc {:g} acc {:g} {} {}".format(time_str, step, loss_sum/bags_sum, bags_hit_NA/sum_NA, bags_hit/sum_fNA, sum_NA, sum_fNA))


				global loop
				while not coord.should_stop():
					print 'Looping ', loop
					outEmbedding(str(loop))
					for i in range(tipTotal):
						length = lib_cnn.batch_iter(x_batch_addr, p_h_batch_addr, p_t_batch_addr, y_batch_addr, r_batch_addr, r_n_batch_addr, h_batch_addr, t_batch_addr)
						train_step_cnn(x_batch[0:length,], p_h_batch[0:length,], p_t_batch[0:length,], y_batch, r_batch, r_n_batch, h_batch, t_batch)
					global bags_sum, bags_hit, loss_sum, bags_hit_NA, bags_hit, sum_fNA, sum_NA
					bags_sum = 0
					bags_hit = 0
					bags_hit_NA = 0
					loss_sum = 0
					sum_fNA = 0
					sum_NA = 0	
					loop += 1
					if loop == config.trainTimes:
						coord.request_stop()

			ph = np.zeros(config.batch_size, dtype = np.int32)
			pt = np.zeros(config.batch_size, dtype = np.int32)
			pr = np.zeros(config.batch_size, dtype = np.int32)
			nh = np.zeros(config.batch_size, dtype = np.int32)
			nt = np.zeros(config.batch_size, dtype = np.int32)
			nr = np.zeros(config.batch_size, dtype = np.int32)
			ph_addr = ph.__array_interface__['data'][0]
			pt_addr = pt.__array_interface__['data'][0]
			pr_addr = pr.__array_interface__['data'][0]
			nh_addr = nh.__array_interface__['data'][0]
			nt_addr = nt.__array_interface__['data'][0]
			nr_addr = nr.__array_interface__['data'][0]
			lib_kg.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]		
			times_kg = 0

			def train_kg(coord):
				def train_step_kg(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
					feed_dict = {
						m.pos_h: pos_h_batch,
						m.pos_t: pos_t_batch,
						m.pos_r: pos_r_batch,
						m.neg_h: neg_h_batch,
						m.neg_t: neg_t_batch,
						m.neg_r: neg_r_batch
					}
					_, step, loss = sess.run(
						[train_op_kg, global_step_kg, m.loss_kg], feed_dict)
					return loss
				global times_kg
				while not coord.should_stop():
					times_kg += 1
					res = 0.0
					for batch in range(config.nbatches):
						lib_kg.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
						res += train_step_kg(ph, pt, pr, nh, nt, nr)

			coord = tf.train.Coordinator()
			threads = []
			threads.append(threading.Thread(target=train_kg, args=(coord,)))
			threads.append(threading.Thread(target=train_cnn, args=(coord,)))
			for t in threads: t.start()
			coord.join(threads)



