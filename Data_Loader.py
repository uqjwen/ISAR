import pickle 
import argparse
import numpy as np 
import sys
import tensorflow as tf 
import utils
import os
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from process import Trans_X
from transformers import BertTokenizer, TFBertModel


####['x','y','l','u','p','uc','pc']  before split self.x_dat; after split: self.train_x, self.dev_x


class Data_Loader():

	def __init__(self, flags):

		if flags.dataset    == 'imdb':
			self.size_split =  [67426, 8381, 9112]
		elif flags.dataset  == 'yelp-2013':
		# else:
			self.size_split =  [62522, 7773, 8671]
		elif flags.dataset  == 'yelp-2014':
			self.size_split =  [183019, 22745, 25399]



		self.batch_size 	= flags.batch_size



		self.vec_file 		= './glove.840B.300d.txt'

		self.embed_size 	= flags.embed_size

		self.num_class 		= flags.num_class

		self.maxlen     	= flags.max_length

		self.flags 			= flags

		# self.bert           = TFBertModel.from_pretrained('bert-base-uncased')
		# x_dict = utils.my_get_dict(filenames)

		self.get_data(flags)

		# self.get_pt(flags.data_dir)
		self.get_reke_data(flags)

		self.split(flags)


	def get_emb_mat(self, filename, embed_size):

		fr = open(filename,'r', encoding='utf-8', errors = 'ignore')
		# for line in fr.readline()

		embed_mat = np.random.uniform(-0.5,0.5,(len(self.x_dict)+1, embed_size)).astype(np.float32)

		for line in fr:

			line = line.strip()

			listfromline = line.split(' ')

			token,vector = listfromline[0],listfromline[1:]

			if token in self.x_dict:

				index = self.x_dict[token]

				try:

					embed_mat[index] = list(map(float,vector))

				except:

					print(listfromline[:5], len(listfromline))

					continue

		fr.close()

		return embed_mat


	def get_flat_data(self, filenames):
		# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		names     = locals()
		res       = ['x', 'y', 'l', 'u', 'p']

		for r in res:
			names[r+'_dat'] = []

		for filename in filenames:
			fr = open(filename, 'r', encoding='utf-8', errors='ignore')
			for i,line in enumerate(fr):
				line = line.split('\t\t')
				y    = int(line[2])-1
				u_id = line[0]
				p_id = line[1]
				u    = self.u_dict[u_id]
				p    = self.p_dict[p_id]

				# outputs = tokenizer(line[-1], max_length = self.maxlen, padding='max_length', truncation=True)
				# sys.stdout.write('\r{}'.format(i))
				tokens = line[-1].split()
				x = [self.x_dict[token]  for token in tokens if token in self.x_dict]

				names['y_dat'].append(y)
				names['u_dat'].append([u])
				names['p_dat'].append([p])
				names['x_dat'].append(x)
				names['l_dat'].append(len(x))
				# names['x_dat']
			# print('\n')

		for r in res:
			names[r+'_dat'] = np.array(names[r+'_dat'])
		res_dat = [names[r+'_dat'] for r in res]
		return res_dat



	def get_data(self, flags):

		data_dir = './data'

		names = self.__dict__

		dict_data = ['x_dict', 'u_dict', 'p_dict']

		dat_data = ['x','y','l','u','p']

		#x_dat: input sentence of shape [batch_size, maxlen] for training

		#y_dat: sentiment labels of shape [batch_size, num_class] for training

		#l_dat: input sentence length before pad_sequence [batch_size,] for training

		#u_dat: input user id of shape [batch_size, 1] for training

		#p_dat: input item(prod) id of shape [batch_size,1] for training 

		files = ['train', 'dev', 'test']

		# source_files = [data_dir+'/'+file+'.txt' for file in files]
		source_files = ['./data/'+flags.dataset+'.'+file+'.txt.ss' for file in files]

		save_file = './data/'+flags.dataset+'.pkl'

		if not os.path.exists(save_file):
			print('x_dict...')
			self.x_dict = utils.my_get_dict(source_files, 3)
			print('u_dict, p_dict ...')
			self.u_dict, self.p_dict = utils.my_get_up_dict(source_files)
			print('train data...')
			data = self.get_flat_data(source_files)
			print('embedding matrix...')
			self.embeddings = self.get_emb_mat(self.vec_file, self.embed_size)
			self.pmtt       = np.random.permutation(self.size_split[0])
			# self.embeddings = self.bert.get_input_embeddings().word_embeddings.numpy()
			for i,dat in enumerate(dat_data):
				names[dat+'_dat'] = data[i]

			names['y_dat'] = to_categorical(names['y_dat'],self.num_class)

			pickle_data = {}

			pickle_data['data'] = [names[dat+'_dat'] for dat in dat_data]

			for item in dict_data:
				pickle_data[item] = names[item]

			pickle_data['embed_mat'] = self.embeddings
			pickle_data['pmtt']      = self.pmtt
			pickle.dump(pickle_data,open(save_file,'wb'))

		else:
			pickle_data = pickle.load(open(save_file, 'rb'))
			for i,dat in enumerate(dat_data):
				names[dat+'_dat'] = pickle_data['data'][i]
			for item in dict_data:
				names[item] = pickle_data[item]
			self.embeddings = pickle_data['embed_mat']
			self.pmtt       = pickle_data['pmtt']
			# self.embed_mat = np.load(ckpt_dir+'/word.npy')



		# lens = [len(x) for x in self.x_dat]

		# self.maxlen = np.max(lens)

		# self.maxlen = 500



		self.x_dat = pad_sequences(self.x_dat, self.maxlen, padding = 'post')
		self.l_dat = np.array([[1.]*self.maxlen if length>self.maxlen else [1.]*length+[0.]*(self.maxlen-length) for length in self.l_dat], dtype=np.float32)
		# self.l_dat[self.l_dat > self.maxlen] = self.maxlen

		# self.vocab_size = len(self.x_dict)+1
		self.vocab_size = len(self.embeddings)

		self.num_user 	= len(self.u_dict)

		self.num_item 	= len(self.p_dict)

		print('num_user: ', self.num_user, 'num_item', self.num_item)
		# trans_x = Trans_X(self.x_dat, self.y_dat, self.p_dat,self.u_dat, self.num_user, self.num_item, data_dir, flags.dataset)

		# self.s_dat = trans_x.sent
		self.s_dat = np.arange(len(self.x_dat))

	def gen_real_fake_data(self, batch_y):
		real_ids  = []
		fake_label= []
		fake_ids  = []
		length    = len(self.y_dat)
		for i,item in enumerate(batch_y):
			y = np.argmax(item)
			pos_ids = []
			for i in range(self.flags.pos_samples):
				index = np.random.randint(length)
				while y != np.argmax(self.y_dat[index]):
					index = np.random.randint(length)
				pos_ids.append(index)
			# real_data.append(self.x_dat[index])
			real_ids.append(pos_ids)
			index = np.random.randint(0, length, self.flags.neg_samples)
			# fake_data.append(self.x_dat[index])
			fake_ids.append(index)
			fake_label.append(np.argmax(self.y_dat[index], axis=1)) # [batch_size, neg_samples]

		return np.array(real_ids), np.array(fake_label), np.array(fake_ids)

	def get_reke_data(self, flags):
		real_ids   = []
		fake_label = []
		fake_ids   = []
		length     = len(self.y_dat)
		filename = './data/'+self.flags.dataset+'-helper.pkl'
		if not os.path.exists(filename):
			print('getting helper data from random...')
			for tmp_y in self.y_dat:
				tmp_y = np.argmax(tmp_y)
				pos_ids = []
				for i in range(flags.pos_samples):
					idx   = np.random.randint(length)
					while tmp_y != np.argmax(self.y_dat[idx]):
						idx = np.random.randint(length)
					pos_ids.append(idx)
				real_ids.append(pos_ids)
				neg_ids = np.random.randint(0,length, flags.neg_samples)
				fake_ids.append(neg_ids)
				fake_label.append(np.argmax(self.y_dat[neg_ids], axis=-1))
			self.ri_dat   = np.array(real_ids)
			self.fi_dat   = np.array(fake_ids)
			self.fl_dat   = np.array(fake_label)
			reke_data     = {}
			reke_data['ri_dat'] = self.ri_dat
			reke_data['fi_dat'] = self.fi_dat
			reke_data['fl_dat'] = self.fl_dat
			pickle.dump(reke_data,open(filename,'wb'))
		else:
			print('getting helper data from previous saved...')
			reke_data   = pickle.load(open(filename, 'rb'))
			self.ri_dat =  reke_data['ri_dat']
			self.fi_dat =  reke_data['fi_dat']
			self.fl_dat =  reke_data['fl_dat']



	def split(self, flags):

		names = self.__dict__

		dat_data = ['x','y','l','u','p','s', 'ri', 'fi', 'fl']

		files = ['train', 'dev', 'test']

		my_split = np.cumsum(self.size_split)

		for i,(to_split, file) in enumerate(zip(my_split,files)):

			# end = to_split if i!=0 else my_split[-1]

			end = to_split

			begin = 0 if i==0 else my_split[i-1]

			# print(file, begin,end, to_split)

			for dat in dat_data:

				names[file+'_'+dat] = names[dat+'_dat'][begin:end]# self.train_x...||self.dev_x...||self.test_x

		self.train_size = len(self.train_x)
		self.dev_size   = len(self.dev_x)
		self.test_size  = len(self.test_x)
		self.doc_size   = len(self.x_dat)

		# pmtt = np.random.permutation(self.train_size)

		for dat in dat_data:

			names['train_'+dat] = names['train_'+dat][self.pmtt]

	def reset_train_pt(self):

		self.train_pt = 0

	def next_batch(self):

		names = self.__dict__

		dat_data = ['x','y','l','u','p', 's']

		begin = self.train_pt*self.batch_size

		end  = (self.train_pt+1)*self.batch_size

		self.train_pt+=1
		if(self.train_pt*self.batch_size>=self.train_size):
			self.train_pt = 0

		end = min(end,self.train_size)

		res_dat = [names['train_'+dat][begin:end]  for dat in dat_data]

		return self.train_x[begin:end], self.train_l[begin:end], self.train_s[begin:end], self.train_y[begin:end], self.train_ri[begin:end], self.train_fi[begin:end], self.train_fl[begin:end]

	def reset_test_pt(self):
		self.test_pt = 0

	def test_next_batch(self):

		names    = self.__dict__

		dat_data = ['x','y','l','u','p','s']
		begin    = self.test_pt*self.batch_size
		end      = (self.test_pt+1)*self.batch_size
		self.test_pt+=1
		if(self.test_pt*self.batch_size >= self.test_size):
			self.test_pt = 0
		end      = min(end, self.test_size)

		res_dat  = [names['test_'+dat][begin:end] for dat in dat_data]

		# return res_dat
		return self.test_x[begin:end], self.test_l[begin:end], self.test_s[begin:end], self.test_y[begin:end], self.test_ri[begin:end], self.test_fi[begin:end], self.test_fl[begin:end]
		# return self.train_x[begin:end], self.train_l[begin:end], self.train_y[begin:end]


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-l1_kernel_size', type=list, default = [2,3,4,5])
	parser.add_argument('-l1_hidden_size', type=int, default = 512)
	parser.add_argument('-l2_kernel_size', type=list, default=[3,5])
	parser.add_argument('-l2_hidden_size', type=int, default = 256)
	parser.add_argument('-fc1_size', type=int, default = 128)
	parser.add_argument('-fc2_size', type=int, default = 64)
	parser.add_argument('-batch_size', type=int, default = 128)
	parser.add_argument('-max_length', type=int, default = 500)
	parser.add_argument('-embed_size', type=int, default = 300)
	parser.add_argument('-neg_samples', type=int, default = 30)
	parser.add_argument('-pos_samples', type=int, default = 10)
	parser.add_argument("-learning_rate", type=float, default=5e-4)
	parser.add_argument('-epoch', type=int, default = 10)
	parser.add_argument('-task', type=str, default='clf')
	parser.add_argument('-dataset', type=str, default='imdb')
	parser.add_argument('-embeddings', type = np.ndarray, default = None)
	parser.add_argument('-doc_size', type=int, default = 90000)


	task    = parser.parse_args().task
	dataset = parser.parse_args().dataset

	# print(task, dataset)

	num_class = 1 if task == 'reg' else 10 if dataset=='imdb' else 5

	parser.add_argument('-num_class', type=int, default= num_class)

	args = parser.parse_args()


	data_loader = Data_Loader(args)



	data_loader.reset_train_pt()

	data = data_loader.next_batch()
	x, att_mask,s, y, real_ids, fake_ids, fake_label = data 
	print(x.shape, att_mask.shape, y.shape )

	# data_loader.val()
	data_loader.reset_test_pt()
	data = data_loader.test_next_batch()
	x, att_mask,s, y, real_ids, fake_ids, fake_label = data