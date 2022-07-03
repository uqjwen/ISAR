
import argparse
import numpy as np 
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras
import os
import pickle
import sys


class Tokenizer(object):
	"""docstring for Tokenizer"""
	def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
		super(Tokenizer, self).__init__()
		# self.arg = arg
		self.num_words = num_words
		self.filters   = list(filters)+['<sssss>']

	def fit_on_texts(self, texts, min_count = 20):
		count = {}
		for text in texts:
			tokens = text.split()
			for token in tokens:
				if token not in self.filters:
					count[token] = count.get(token, 0)+1
		# word_count = sorted(count.items(), key=lambda d:d[1], reverse = True)[:self.num_words-1]

		# self.word_index = dict((tp[0], i+1) for i,tp in enumerate(word_count))
		# self.index_word = dict((i+1, tp[0]) for i,tp in enumerate(word_count))
		self.word_index = {}
		self.index_word = {}
		for (k,v) in count.items():
			if v>= min_count:
				index = len(self.word_index)
				self.word_index[k]       = index+1
				self.index_word[index+1] = k

		self.vocab_size = len(self.word_index)+1

		# print(self.num_words, self.vocab_size)

	def texts_to_ids(self, texts):
		# return [self.word_index[token] if token in self.word_index for text in texts for token in text]
		return [[self.word_index[token] for token in text if token in self.word_index] for text in texts]

	def pad_sequence(self, sequences, max_length):
		input_ids      = [sequence[:max_length] if len(sequence)>max_length else sequence+[0]*(max_length-len(sequence)) for sequence in sequences]
		attention_mask = [(np.array(sequence)>0).astype(np.float32) for sequence in input_ids]
		return input_ids, attention_mask

	def get_input_embeddings(self, vocab_file, emb_size = 300):
		embeddings = np.random.uniform(-0.25, 0.25, (self.vocab_size, emb_size))
		fr = open(vocab_file, 'r', encoding='utf-8', errors='ignore')
		for aline in fr:
			line = aline.split(' ')
			if(line[0]) in self.word_index:
				# print(line)
				embeddings[self.word_index[line[0]]] = np.array(list(map(np.float, line[1:])))
		return embeddings


class DataLoader(object):
	"""docstring for ClassName"""
	def __init__(self, args):
		super(DataLoader, self).__init__()
		# self.arg = arg
		self.batch_size = args.batch_size
		self.max_length = args.max_length
		self.num_class  = args.num_class
		self.load_data('./data/', args.dataset)

	def load_data(self, directory, dataset):
		if not os.path.exists(directory+dataset+'.pkl'):

			filenames = [directory+dataset+suffix+'.txt.ss' for suffix in ['.train', '.dev', '.test']]
			texts  = []
			labels = []
			split  = [0,0,0,0]
			for i,filename in enumerate(filenames):
				fr = open(filename, 'r', encoding = 'utf-8', errors = 'ignore')
				size = 0
				for line in fr:
					size+=1
					line = line.strip()
					line = line.split('\t\t')
					assert len(line)==4
					texts.append(line[-1])
					labels.append(int(line[-2]))
				split[i+1] = size
				fr.close()
			# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 30000)
			tokenizer = Tokenizer(num_words=30000)
			print('getting vocab...')
			tokenizer.fit_on_texts(texts)
			print('covert to ids...')
			sequences = tokenizer.texts_to_ids(texts)
			print('pad to length...')
			sequences, attention_mask = tokenizer.pad_sequence(sequences, 500)
			print(np.array(sequences).shape, np.array(attention_mask).shape)
			print('getting embeddings...')
			self.embeddings     = tokenizer.get_input_embeddings('./glove.840B.300d.txt', 300)


			self.labels         = []
			self.input_ids      = []
			self.attention_mask = []
			split               = np.cumsum(split)
			# print(split)
			for i in range(len(split)-1):
				self.input_ids.append(np.array(sequences[split[i]:split[i+1]]))
				self.attention_mask.append(np.array(attention_mask[split[i]:split[i+1]]))
				self.labels.append(np.array(labels[split[i]:split[i+1]]))
			for i in range(3):
				self.labels[i]         = tf.keras.utils.to_categorical(self.labels[i]-1, self.num_class)
				pmtt                   = np.random.permutation(len(self.labels[i]))
				self.input_ids[i]      = self.input_ids[0][pmtt]
				self.attention_mask[i] = self.attention_mask[i][pmtt]
				self.labels[i]         = self.labels[i][pmtt]
			save_to_pkl                   = {}
			save_to_pkl['input_ids']      = self.input_ids
			save_to_pkl['attention_mask'] = self.attention_mask
			save_to_pkl['labels']         = self.labels
			save_to_pkl['embeddings']     = self.embeddings
			pickle.dump(save_to_pkl, open(directory+dataset+'.pkl', 'wb'))


		else:
			print("loading data...")
			load_from_pkl       = pickle.load(open(directory+dataset+'.pkl', 'rb'))
			self.input_ids      = load_from_pkl['input_ids']
			self.attention_mask = load_from_pkl['attention_mask']
			self.labels         = load_from_pkl['labels']
			self.embeddings     = load_from_pkl['embeddings']

		self.train_size, self.dev_size, self.test_size = len(self.labels[0]), len(self.labels[1]), len(self.labels[2])

	def reset_train_pt(self):
		self.train_pt = 0
	def next_batch(self):
		begin = self.train_pt*self.batch_size
		end   = (self.train_pt+1)*self.batch_size
		self.train_pt += 1
		if(self.train_pt*self.batch_size >= self.train_size):
			self.train_pt = 0
		end   = min(end, self.train_size)

		return self.input_ids[0][begin:end],\
		       self.attention_mask[0][begin:end],\
		       self.labels[0][begin:end]

	def reset_test_pt(self):
		self.test_pt = 0

	def test_next_batch(self):
		begin = self.test_pt*self.batch_size
		end   = (self.test_pt+1)*self.batch_size
		self.test_pt +=1
		if(self.test_pt*self.batch_size >= self.test_size):
			self.test_pt = 0
		end   = min(end, self.test_size)

		return self.input_ids[2][begin:end], \
			   self.attention_mask[2][begin:end], \
			   self.labels[2][begin:end]


# class DataLoader(object):
# 	"""docstring for DataLoader"""
# 	def __init__(self, args):
# 		super(DataLoader, self).__init__()
# 		# self.arg = arg
# 		self.batch_size   = args.batch_size
# 		self.max_length   = args.max_length
# 		self.tokenizer    = BertTokenizer.from_pretrained("bert-base-uncased")

# 		self.get_dataset('./data/', args.dataset)
# 		self.concat_label = np.concatenate(self.label)
# 		self.concat_data  = np.concatenate(self.data)


# 	def get_dataset(self, directory, dataset):
# 		data  = []
# 		label = []

# 		if not os.path.exists(directory+dataset+'.pkl'):
# 			filename = [directory+dataset+'.train.txt.ss', directory+dataset+'.dev.txt.ss', directory+dataset+'.test.txt.ss']
# 			# filename = [directory+dataset+'-test.txt.ss']
# 			for file in filename:
# 				split_data, split_label = self.get_datafile(file)
# 				data.append(split_data)
# 				label.append(split_label)
# 			save_to_pkl = {}
# 			save_to_pkl['data'] = data
# 			save_to_pkl['label'] = label
# 			# fr = open(directory+dataset+'.pkl', 'wb')
# 			pickle.dump(save_to_pkl, open(directory+dataset+'.pkl', 'wb'))
# 			# seq_lens = [len(doc) for split_data in data for doc in split_data]
# 			# print(len(seq_lens), np.percentile(seq_lens,90))
# 		print("data loading...")
# 		load_from_pkl = pickle.load(open(directory+dataset+'.pkl', 'rb'))
# 		# assert(s)
# 		self.data  = [np.array(split_data['input_ids']) for split_data in load_from_pkl['data']]
# 		self.label = [np.array(split_label) for split_label in  load_from_pkl['label']]


# 		self.train_size = len(self.label[0])
# 		self.dev_size   = len(self.label[1])
# 		self.test_size  = len(self.label[2])

# 	def get_datafile(self,filename):
# 		print(filename)
# 		fr    = open(filename, 'r')
# 		data  = fr.readlines()
# 		fr.close()
# 		split_data  = {}
# 		split_data['input_ids'] = []
# 		split_data['attention_mask'] = []
# 		split_label = []
# 		lines = len(data)
# 		for i,line in enumerate(data):
# 			line = line.strip()
# 			listfromline = line.split('\t\t')
# 			assert(len(listfromline) == 4)
# 			listfromline[-1].replace("<sssss>",'')
# 			# print(self.tokenizer(listfromline[-1]))
# 			# break
# 			split_label.append(int(listfromline[-2]))
# 			# split_data.append(listfromline[-1])
# 			sys.stdout.write('\rline:{}/{}'.format(i,len(data)))
# 			outputs = self.tokenizer(listfromline[-1], max_length=self.max_length, padding='max_length', truncation=True)
# 			split_data['input_ids'].append(outputs['input_ids'])
# 			split_data['attention_mask'].append(outputs['attention_mask'])
# 			sys.stdout.write('\r{}/{}'.format(i, lines))
# 			sys.stdout.flush()
# 			# split_data.append(self.tokenizer(listfromline[-1], max_length = self.max_length, padding='max_length', truncation=True)['input_ids'])
# 		# print(split_data)

# 		# print("tokenizing...")	
# 		# split_data = self.tokenizer(split_data, max_length = self.max_length, padding='max_length', truncation=True)['input_ids']

# 		return split_data, split_label


# 	def reset_train_pt(self):
# 		self.train_pt = 0

# 	def next_batch(self):
# 		begin = self.train_pt*self.batch_size
# 		end   = (self.train_pt+1)*self.batch_size
# 		self.train_pt+=1
# 		if(self.train_pt*self.batch_size>=self.train_size):
# 			self.train_pt = 0
# 		end = min(end, self.train_size)
# 		return self.label[0][begin:end], self.data[0][begin:end]



# 	def gan_samples(self, batch_label, gen_samples = 20):
# 		# concat_data = np.concatenate(self.data)
# 		# concat_label = np.concatenate(self.label)
# 		real_label = []
# 		real_data  = []
# 		fake_label = []
# 		fake_data  = []
# 		for label in batch_label:
# 			index = np.random.randint(0,len(self.concat_label))
# 			while self.concat_label[index]!=label:
# 				index = np.random.randint(0,len(self.concat_label))
# 			# real_index.append(index)
# 			# real_samples.append()
# 			# index = np.random.choice()
# 			real_label.append(self.concat_label[index])
# 			real_data.append(self.concat_data[index])

# 			index = np.random.randint(0,len(self.concat_label), gen_samples)
# 			# fake_index.append(index)
# 			fake_label.append(self.concat_label[index])
# 			fake_data.append(self.concat_data[index])
# 		return np.array(real_label), \
# 				np.array(real_data), \
# 				np.array(fake_label), \
# 				np.array(fake_data)
# 		# return np.array(real_index), np.array(fake_index)
# 	# def get_doc_embeddings(self):
# 	# 	self.doc_embed = np.random.uniform(-0.25,0.25, (self.doc_size, 768))
# 	# 	bert_model     = TFBertModel.from_pretrained("bert-base-uncased")
# 	# 	print("doc embedding initialized")
# 		# for doc in self.concat_data:
# 	def reset_test_pt(self):
# 		self.test_pt = 0

# 	def test_next_batch(self):
# 		begin        = self.test_pt*self.batch_size
# 		end          = (self.test_pt+1)*self.batch_size
# 		self.test_pt += 1
# 		if(self.test_pt*self.batch_size >= self.test_size*5):
# 			self.test_pt = 0

# 		end   = min(end, self.test_size)
# 		return self.label[0][begin:end], self.data[0][begin:end]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-batch_size', type=int, default = 128)
	parser.add_argument('-max_length', type=int, default = 500)
	parser.add_argument('-dataset', type=str, default='imdb')
	parser.add_argument('-num_class', type=int, default=5)
	args = parser.parse_args()

	data_loader = DataLoader(args)
	# data_loader.get_data()
	# data_loader.get_dataset("./data/", 'imdb')
	data_loader.reset_train_pt()
	input_ids, attention_mask, labels = data_loader.next_batch()
	# print(np.array(input_ids).shape, np.array(attention_mask).shape, np.array(labels).shape)
	print(input_ids)
	print('=============================')
	print(labels)
	print('==============================')
	print(attention_mask)
	# print(batch_data)

	# print(data_loader.concat_data.shape)
