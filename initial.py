import numpy as np
import os
import json

# folder of training datasets
data_path = "./origin_data/"
# files to export data
export_path = "./data/"
#length of sentence
fixlen = 120
#max length of position embedding is 100 (-100~+100)
maxlen = 100

word2id = {}
relation2id = {}
word_size = 0
word_vec = None

def pos_embed(x):
	return max(0, min(x + maxlen, maxlen + maxlen + 1))

def find_index(x,y):
	for index, item in enumerate(y):
		if x == item:
			return index
	return -1

def init_word():
	# reading word embedding data...
	global word2id, word_size
	res = []
	ff = open(export_path + "/entity2id.txt", "w")
	f = open(data_path + "kg/train.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h, t, r = content.strip().split("\t")
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	f = open(data_path + "text/train.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h,t = content.strip().split("\t")[:2]
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	f = open(data_path + "text/test.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h,t = content.strip().split("\t")[:2]
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	res.append(len(word2id))
	ff.close()

	print 'reading word embedding data...'
	f = open(data_path + 'text/vec.txt', "r")
	total, size = f.readline().strip().split()[:2]
	total = (int)(total)
	word_size = (int)(size)
	vec = np.ones((total + res[0], word_size), dtype = np.float32)
	for i in range(total):
		content = f.readline().strip().split()
		word2id[content[0]] = len(word2id)
		for j in range(word_size):
			vec[i + res[0]][j] = (float)(content[j+1])
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	global word_vec
	word_vec = vec
	res.append(len(word2id))
	return res

def init_relation():
	# reading relation ids...
	global relation2id
	print 'reading relation ids...'	
	res = []
	ff = open(export_path + "/relation2id.txt", "w")
	f = open(data_path + "text/relation2id.txt","r")
	total = (int)(f.readline().strip())
	for i in range(total):
		content = f.readline().strip().split()
		if not content[0] in relation2id:
			relation2id[content[0]] = len(relation2id)
			ff.write("%s\t%d\n"%(content[0], relation2id[content[0]]))
	f.close()
	res.append(len(relation2id))
	f = open(data_path + "kg/train.txt", "r")
	for i in f.readlines():
		h, t, r = i.strip().split("\t")
		if not r in relation2id:
			relation2id[r] = len(relation2id)
			ff.write("%s\t%d\n"%(r, relation2id[r]))
	f.close()
	ff.close()
	res.append(len(relation2id))
	return res

def sort_files(name, limit):
	hash = {}
	f = open(data_path + "text/" + name + '.txt','r')
	s = 0
	while True:
		content = f.readline()
		if content == '':
			break
		s = s + 1
		origin_data = content
		content = content.strip().split()
		en1_id = content[0]
		en2_id = content[1]
		rel_name = content[4]
		if (rel_name in relation2id) and ((int)(relation2id[rel_name]) < limit[0]):
			relation = relation2id[rel_name]
		else:
			relation = relation2id['NA']
		id1 = str(en1_id)+"#"+str(en2_id)
		id2 = str(relation)
		if not id1 in hash:
			hash[id1] = {}
		if not id2 in hash[id1]:
			hash[id1][id2] = []
		hash[id1][id2].append(origin_data)
	f.close()
	f = open(data_path + name + "_sort.txt", "w")
	f.write("%d\n"%(s))
	for i in hash:
		for j in hash[i]:
			for k in hash[i][j]:
				f.write(k)
	f.close()

def init_train_files(name, limit):
	print 'reading ' + name +' data...'
	f = open(data_path + name + '.txt','r')
	total = (int)(f.readline().strip())
	sen_word = np.zeros((total, fixlen), dtype = np.int32)
	sen_pos1 = np.zeros((total, fixlen), dtype = np.int32)
	sen_pos2 = np.zeros((total, fixlen), dtype = np.int32)
	sen_mask = np.zeros((total, fixlen), dtype = np.int32)
	sen_len = np.zeros((total), dtype = np.int32)
	sen_label = np.zeros((total), dtype = np.int32)
	sen_head = np.zeros((total), dtype = np.int32)
	sen_tail = np.zeros((total), dtype = np.int32)
	instance_scope = []
	instance_triple = []
	for s in range(total):
		content = f.readline().strip().split()
		sentence = content[5:-1]
		en1_id = content[0]
		en2_id = content[1]
		en1_name = content[2]
		en2_name = content[3]
		rel_name = content[4]
		if rel_name in relation2id and ((int)(relation2id[rel_name]) < limit[0]):
			relation = relation2id[rel_name]
		else:
			relation = relation2id['NA']
		en1pos = 0
		en2pos = 0
		for i in range(len(sentence)):
			if sentence[i] == en1_name:
				sentence[i] = en1_id
				en1pos = i
				sen_head[s] = word2id[en1_id]
			if sentence[i] == en2_name:
				sentence[i] = en2_id
				en2pos = i
				sen_tail[s] = word2id[en2_id]
		en_first = min(en1pos,en2pos)
		en_second = en1pos + en2pos - en_first
		for i in range(fixlen):
			sen_word[s][i] = word2id['BLANK']
			sen_pos1[s][i] = pos_embed(i - en1pos)
			sen_pos2[s][i] = pos_embed(i - en2pos)
			if i >= len(sentence):
				sen_mask[s][i] = 0
			elif i - en_first<=0:
				sen_mask[s][i] = 1
			elif i - en_second<=0:
				sen_mask[s][i] = 2
			else:
				sen_mask[s][i] = 3
		for i, word in enumerate(sentence):
			if i >= fixlen:
				break
			elif not word in word2id:
				sen_word[s][i] = word2id['UNK']
			else:
				sen_word[s][i] = word2id[word]
		sen_len[s] = min(fixlen, len(sentence))
		sen_label[s] = relation
		#put the same entity pair sentences into a dict
		tup = (en1_id,en2_id,relation)
		if instance_triple == [] or instance_triple[len(instance_triple) - 1] != tup:
			instance_triple.append(tup)
			instance_scope.append([s,s])
		instance_scope[len(instance_triple) - 1][1] = s
		if (s+1) % 100 == 0:
			print s
	return np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask, sen_head, sen_tail

def init_kg():
	ff = open(export_path + "/triple2id.txt", "w")
	f = open(data_path + "kg/train.txt", "r")
	content = f.readlines()
	ff.write("%d\n"%(len(content)))
	for i in content:
		h,t,r = i.strip().split("\t")
		ff.write("%d\t%d\t%d\n"%(word2id[h], word2id[t], relation2id[r]))
	f.close()
	ff.close()

	f = open(export_path + "/entity2id.txt", "r")
	content = f.readlines()
	f.close()
	f = open(export_path + "/entity2id.txt", "w")
	f.write("%d\n"%(len(content)))
	for i in content:
		f.write(i.strip()+"\n")
	f.close()

	f = open(export_path + "/relation2id.txt", "r")
	content = f.readlines()
	f.close()
	f = open(export_path + "/relation2id.txt", "w")
	f.write("%d\n"%(len(content)))
	for i in content:
		f.write(i.strip()+"\n")
	f.close()

textual_rel_total, rel_total = init_relation()
entity_total, word_total = init_word()

print textual_rel_total
print rel_total
print entity_total
print word_total
print word_vec.shape
f = open(data_path + "word2id.txt", "w")
for i in word2id:
	f.write("%s\t%d\n"%(i, word2id[i]))
f.close()

init_kg()
np.save(export_path+'vec', word_vec)
f = open(export_path+'config', "w")
f.write(json.dumps({"word2id":word2id,"relation2id":relation2id,"word_size":word_size, "fixlen":fixlen, "maxlen":maxlen, "entity_total":entity_total, "word_total":word_total, "rel_total":rel_total, "textual_rel_total":textual_rel_total}))
f.close()
sort_files("train", [textual_rel_total, rel_total])
sort_files("test", [textual_rel_total, rel_total])

# word_vec = np.load(export_path + 'vec.npy')
# f = open(export_path + "config", 'r')
# config = json.loads(f.read())
# f.close()
# relation2id = config["relation2id"]
# word2id = config["word2id"]

instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask, train_head, train_tail = init_train_files("train_sort",  [textual_rel_total, rel_total])
np.save(export_path+'train_instance_triple', instance_triple)
np.save(export_path+'train_instance_scope', instance_scope)
np.save(export_path+'train_len', train_len)
np.save(export_path+'train_label', train_label)
np.save(export_path+'train_word', train_word)
np.save(export_path+'train_pos1', train_pos1)
np.save(export_path+'train_pos2', train_pos2)
np.save(export_path+'train_mask', train_mask)
np.save(export_path+'train_head', train_head)
np.save(export_path+'train_tail', train_tail)

instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask, test_head, test_tail = init_train_files("test_sort",  [textual_rel_total, rel_total])
np.save(export_path+'test_instance_triple', instance_triple)
np.save(export_path+'test_instance_scope', instance_scope)
np.save(export_path+'test_len', test_len)
np.save(export_path+'test_label', test_label)
np.save(export_path+'test_word', test_word)
np.save(export_path+'test_pos1', test_pos1)
np.save(export_path+'test_pos2', test_pos2)
np.save(export_path+'test_mask', test_mask)
np.save(export_path+'test_head', test_head)
np.save(export_path+'test_tail', test_tail)
