import numpy as np 
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras
import argparse
import os



h_dim = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert      = TFBertModel.from_pretrained('bert-base-cased')
# input_ids = keras.layers.Input(shape = (max_sen_len), dtype = 'int32')
# outputs   = bert(input_ids)
# bert_model = keras.Model(inputs = input_ids, outputs = outputs)

# class Model(tf.keras.layers):

class Generator(tf.keras.Model):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
		# self.arg = arg
		self.fc1 = tf.keras.layers.Dense(h_dim/2, activation = 'relu')
		self.fc2 = tf.keras.layers.Dense(1)
		# self.doc_embed = tf.Variable(np.random.uniform(-1,1,(350000,768)))
	def get_vector(self, anchor_h, random_h):
		# shape = random_h.shape.as_list()
		# anchor_b = tf.broadcast_to(anchor_h, )

		anchor_t = tf.expand_dims(anchor_h, axis=1)
		anchor_b = tf.broadcast_to(anchor_t, shape = random_h.shape)
		hidden   = tf.concat([anchor_b, random_h, anchor_b*random_h], axis=-1) #[batch_size, num_random, 3*hidden_size]
		hidden_1 = self.fc1(hidden)
		hidden_2 = self.fc2(hidden_1) # [batch_size, num_random, 1]
		# hidden_2 = tf.squeeze(hidden_2)
		hidden_softmax  = tf.math.softmax(50*hidden_2, axis=1) # [batch_size, num_random, 1]

		self.argmax_idx = tf.argmax(hidden_softmax, axis=1)

		return tf.reduce_sum(hidden_softmax*random_h, axis=1)

	# def get_max_idx(self, anchor_h, random_h):



class Discrimitor(tf.keras.Model):
	"""docstring for Discrimitor"""
	def __init__(self, args):
		super(Discrimitor, self).__init__()
		self.args= args
		self.fc1 = tf.keras.layers.Dense(h_dim/2, activation = "relu")
		self.fc2 = tf.keras.layers.Dense(1)
		self.doc_embed = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(args.doc_size, 100)))
	def get_logits(self, anchor_h, candidate_h):
		anchor_h = tf.nn.embedding_lookup(self.doc_embed, anchor_h)
		candidate_h = tf.nn.embedding_lookup(self.doc_embed, candidate_h)

		hidden   = tf.concat([anchor_h, candidate_h, anchor_h*candidate_h], axis=-1)
		hidden_1 = self.fc1(hidden)
		logits   = self.fc2(hidden_1)

		return logits
	def get_full_logits(self, anchor_h, candidate_h):
		anchor_h = tf.nn.embedding_lookup(self.doc_embed, anchor_h)
		candidate_h = tf.nn.embedding_lookup(self.doc_embed, candidate_h)

		anchor_t = tf.expand_dims(anchor_h, axis=1)
		anchor_b = tf.broadcast_to(anchor_t, shape = candidate_h.shape) #[batch_size, neg_samples, hidden_size]

		hidden   = tf.concat([anchor_b, candidate_h, anchor_b*candidate_h], axis=-1) # batch_size, neg_samples, hidden_size*3
		hidden_1 = self.fc1(hidden)
		logits   = self.fc2(hidden_1) # batch_size, neg_samples,1
		# logits   = tf.argmax()
		return logits



# l1_kernel_size = [2,3,4,5]
# l1_hidden_size = 128
# l2_kernel_size = [3,5]

class Model(tf.keras.Model):	
	"""docstring for ClassName"""
	def __init__(self, args):
		super(Model, self).__init__()
		self.args   = args
		# self.fc1    = keras.layers.Dense(h_dim, activation='relu')
		# self.fc2    = keras.layers.Dense(num_class)
		# self.bert = TFBertModel.from_pretrained('bert-base-uncased')
		l1_filters     = int(args.l1_hidden_size/len(args.l1_kernel_size))
		l2_filters     = int(args.l2_hidden_size/len(args.l2_kernel_size))
		print("filters per layer: ", l1_filters, l2_filters)

		self.l1_conv1d = [tf.keras.layers.Conv1D(l1_filters, k_size, padding='same') for k_size in args.l1_kernel_size]
		self.l1_bn     = [tf.keras.layers.BatchNormalization() for k_size in args.l1_kernel_size]
		# self.bn1       = tf.keras.layers.BatchNormalization()
		self.l2_conv1d = [tf.keras.layers.Conv1D(l2_filters, k_size, padding='same') for k_size in args.l2_kernel_size]
		self.l2_bn     = [tf.keras.layers.BatchNormalization() for k_size in args.l2_kernel_size]

		self.att_layer = tf.keras.layers.Dense(1)

		self.fc1       = tf.keras.layers.Dense(args.fc1_size)
		self.fc1_bn    = tf.keras.layers.BatchNormalization()
		self.fc2       = tf.keras.layers.Dense(args.fc2_size)
		self.fc2_bn    = tf.keras.layers.BatchNormalization()
		self.fc3       = tf.keras.layers.Dense(args.num_class)

		self.fc        = [tf.keras.layers.Dense(int(args.l2_hidden_size/(2**(i+1)))) for i in range(args.fc_layers)]

		self.embeddings= tf.Variable(args.embeddings)
		self.doc_embed = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(args.doc_size, args.l2_hidden_size)))

		# def call(self, inputs, training=None, mask=None):


		self.bi_lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(args.l1_hidden_size/2), return_sequences=True))

		self.bi_lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(args.l2_hidden_size/2)))

	def get_hidden_with_ids(self, doc_ids):
		# doc_ids: [batch_size, neg_samples]
		return tf.nn.embedding_lookup(self.doc_embed, doc_ids) # [batch_size, neg_samples, hidden_size]


	def get_hidden_states(self, inputs, mask=None):
		inputs         = tf.nn.embedding_lookup(self.embeddings, inputs)

		l1_conv        = [l1_conv1d(inputs) for l1_conv1d in self.l1_conv1d]
		# l1_conv_bn     = [l1bn(l1conv) for l1conv, l1bn in zip(l1_conv, self.l1_bn)]
		l1_conv_bn     = l1_conv
		l1_conv_lk     = [tf.nn.relu(l1convbn) for l1convbn in l1_conv_bn]

		l1_concat      = tf.concat(l1_conv_lk, axis=-1)


		l2_conv        = [l2_conv1d(l1_concat) for l2_conv1d in self.l2_conv1d]
		# l2_conv_bn     = [l2bn(l2conv) for l2conv, l2bn in zip(l2_conv, self.l2_bn)]
		l2_conv_bn     = l2_conv
		l2_conv_lk     = [tf.nn.relu(l2convbn) for l2convbn in l2_conv_bn]
		l2_concat      = tf.concat(l2_conv_lk, axis=-1)

		atts           = self.att_layer(l2_concat) # [batch_size, maxlen, 1]
		if mask == None:
			self.atts_norm = tf.nn.softmax(atts, axis=1)
		else:
			expand_mask    = tf.expand_dims(mask, axis=-1)
			self.atts_norm = tf.exp(atts)*expand_mask/tf.reduce_sum(tf.exp(atts)*expand_mask, axis=1, keepdims = True)

		self.hidden      = tf.reduce_sum(l2_concat*self.atts_norm, axis=1)

		return self.hidden

		# print(logits.shape)
	def get_helper_pool(self, hidden_states, doc_embeddings):
		if self.args.pool == 'mean':
			return tf.reduce_mean(doc_embeddings, axis=1)
		elif self.args.pool == 'max':
			return tf.reduce_max(doc_embeddings, axis=1)
		else:
			transforms = tf.keras.layers.Dense(doc_embeddings.shape[-1])(hidden_states)
			transforms = tf.expand_dims(transforms, 1)
			sims       = tf.reduce_sum(transforms*doc_embeddings, axis=-1, keepdims = True)
			atts       = tf.nn.softmax(sims, axis=1)
			return tf.reduce_sum(atts*doc_embeddings, axis=1)

	def get_logits(self, hidden_states, h_doc_ids, helper=False):
		doc_embeddings = tf.nn.embedding_lookup(self.doc_embed, h_doc_ids)
		helper_pool    = tf.reduce_max(doc_embeddings, axis=1)

		hidden_vector  = (hidden_states+helper_pool)/2 if helper==True else hidden_states

		# fc1            = tf.nn.leaky_relu(self.fc1(hidden_vector))
		# fc2            = tf.nn.leaky_relu(self.fc2(fc1))
		for hidden_layer in self.fc:
			hidden_vector = tf.nn.relu(hidden_layer(hidden_vector))
		self.logits    = self.fc3(hidden_vector)
		
	def get_my_loss(self, m_doc_ids, labels):
		doc_embeddings = tf.nn.embedding_lookup(self.doc_embed, m_doc_ids)
		reg_loss       = tf.reduce_mean(tf.square(doc_embeddings - self.hidden))


		if self.args.task == 'clf':
			m_loss         = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = labels))
		else:
			# m_loss         = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = final_logits, labels = labels))
			classes        = np.expand_dims(np.arange(self.args.num_class),0)
			logits         = tf.nn.softmax(self.logits*self.args.train_factor, axis=-1)
			soft_class     = tf.reduce_sum(logits*classes, axis=-1)
			# m_loss         = tf.keras.losses.MeanSquaredError()(tf.squeeze(self.logits), tf.cast(tf.argmax(labels, axis=1), tf.float32))
			m_loss         = tf.keras.losses.MeanSquaredError()(soft_class, tf.cast(tf.argmax(labels, axis=1), tf.float32))

		# print('\n', reg_loss)
		return m_loss+reg_loss


	def get_hidden_states_lstm(self, inputs, mask=None):
		embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
		lstm_1     = self.bi_lstm_1(embeddings)
		self.hidden= self.bi_lstm_2(lstm_1)
		return self.hidden

	def get_prediction(self, inputs, mask=None):
		# logits = self.get_hidden_states(inputs, mask)
		logits = self.m_logits+self.h_logits
		return tf.argmax(logits, axis=-1)
		# [for ]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l1_kernel_size', type=list, default = [3,5])
	parser.add_argument('-l1_hidden_size', type=int, default = 256)
	parser.add_argument('-l2_kernel_size', type=list, default=[5])
	parser.add_argument('-l2_hidden_size', type=int, default = 128)
	parser.add_argument('-fc1_size', type=int, default = 128)
	parser.add_argument('-fc2_size', type=int, default = 64)
	parser.add_argument('-num_class', type=int, default= 10)

	args = parser.parse_args()

	model = Model(args)
	optimizer = keras.optimizers.Adam(0.0001)

	inputs = np.random.random((64,128,768))
	with tf.GradientTape() as tape:
		model.get_hidden_states(inputs)



if __name__ == '__main__':
	main()
