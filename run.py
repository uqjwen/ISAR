import numpy as np 
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras
import os
import pickle
from Model import Model, Generator, Discrimitor
from DataLoader import DataLoader
from Data_Loader import Data_Loader
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def celoss_ones(logits, smooth = 0.0):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
																  labels = tf.ones_like(logits)*(1-smooth)))

def celoss_zeros(logits, smooth = 0.0):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
																  labels = tf.zeros_like(logits)*(1-smooth)))


def get_d_loss(generator, discrimitor, anchor_vectors, real_vectors, fake_vectors, fake_label, real_label):
	real_label    = np.expand_dims(np.argmax(real_label, axis=-1), -1)# batch_size, 1    fake_label:[batch_size, neg_samples]
	rf_equal      = real_label == fake_label    # batch_size, neg_samples



	fake_vectors  = generator.get_vector(anchor_vectors, fake_vectors)
	argmax_idx    = generator.argmax_idx
	d_real_logits = discrimitor.get_logits(anchor_vectors, real_vectors)
	d_fake_logits = discrimitor.get_logits(anchor_vectors, fake_vectors)
	# d_full_logits = discrimitor.get_full_logits(anchor_vectors, fake_vectors)
	labels        = tf.cast(tf.gather(rf_equal, argmax_idx, batch_dims = 1), tf.float32)


	d_loss_real   = celoss_ones(d_real_logits, smooth=0.1)
	# d_loss_fake   = celoss_zeros(d_fake_logits, smooth = 0.0)
	d_loss_fake   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_logits, labels = labels))
	loss = d_loss_real + d_loss_fake
	return loss

def get_d_loss_1(generator, discrimitor, anchor_ids, real_ids, fake_ids, fake_label, real_label, args):
	real_label    = np.expand_dims(np.argmax(real_label, axis=-1), -1)
	reke_equal    = (real_label == fake_label).astype(np.float32)



	# d_real_logits = discrimitor.get_logits(anchor_ids, real_ids)
	d_real_logits = discrimitor.get_full_logits(anchor_ids, real_ids)

	d_fake_logits = discrimitor.get_full_logits(anchor_ids, fake_ids)
	d_fake_logits = tf.squeeze(d_fake_logits)
	# print('\n', reke_equal.shape, d_fake_logits.shape)
	d_loss_real   = celoss_ones(d_real_logits, smooth=0.1)
	d_loss_fake   = celoss_zeros(d_fake_logits, smooth=0.0)
	# d_loss_fake   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_logits, labels = reke_equal))
	l2_loss       = 0
	for var in discrimitor.trainable_variables:
		l2_loss += tf.nn.l2_loss(var)
	return d_loss_real*(args.neg_samples/args.pos_samples)+d_loss_fake+l2_loss*1e-5

def get_g_loss(generator, discrimitor, anchor_vectors, fake_vectors, fake_label, real_label):
	real_label    = np.expand_dims(np.argmax(real_label, axis=-1), -1)
	rf_equal      = real_label == fake_label


	fake_vectors  = generator.get_vector(anchor_vectors, fake_vectors)
	argmax_idx    = generator.argmax_idx # batch_size, 1
	d_fake_logits = discrimitor.get_logits(anchor_vectors, fake_vectors)
	labels        = tf.cast(tf.gather(rf_equal, argmax_idx, batch_dims = 1), tf.float32)
	loss          = celoss_ones(d_fake_logits, smooth = 0.1)
	# loss          = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_logits, labels = labels))
	return loss, d_fake_logits


def get_logits(model, data):
	shape = data.shape
	reshape = True if len(shape)>2 else False
	# tmp_data = data
	if reshape:
		data = np.reshape(data, (-1, shape[-1]))
	outputs = model.get_vector(data)
	if reshape:
		outputs = tf.reshape(outputs, (shape[0], shape[1], -1))
	return outputs




def main(args):
	# print("num_class:", args.num_class)
	# data_loader     = DataLoader(args)
	data_loader     = Data_Loader(args)
	args.embeddings = data_loader.embeddings
	args.doc_size   = data_loader.doc_size

	model           = Model(args)
	generator       = Generator()
	discrimitor     = Discrimitor(args)

	train(model, data_loader, generator, discrimitor, args)
	# train_1(model, data_loader, generator, discrimitor, args)


# def val(model, data_loader, embeddings):
def val(model, data_loader, discrimitor, args, helper):
	total_batch = int(data_loader.test_size/data_loader.batch_size)+1
	data_loader.reset_test_pt()
	metric      = tf.keras.metrics.CategoricalAccuracy() if args.task == 'clf' else tf.keras.metrics.RootMeanSquaredError()
	# rmse        = tf.keras.metrics.RootMeanSquaredError()
	acc         = tf.keras.metrics.CategoricalAccuracy()
	rmse        = tf.keras.metrics.RootMeanSquaredError()
	# data = data_loader.val()
	# x,y,l,u,p,s = data
	# logits = model.get_hidden_states(x)
	# if args.task == 'clf':
	# 	metric.update_state(y, logits)
	for b in range(total_batch):
		x, att_mask,s, y, real_ids, fake_ids, fake_label = data_loader.test_next_batch()
		# real_ids, fake_label, fake_ids                   = data_loader.gen_real_fake_data(y)
		helper_ids       = get_reke(s, fake_ids, fake_label, y, discrimitor, args, helper)


		m_hidden     = model.get_hidden_states(x)
		logits       = model.get_logits(m_hidden, helper_ids, helper = helper)
		loss         = model.get_my_loss(s,y)


		logits       = model.logits

		if args.task == 'clf':
			acc.update_state(y, logits)
			classes    = np.expand_dims(np.arange(args.num_class), 0)
			logits     = tf.nn.softmax(5*logits, -1)
			soft_class = np.sum(classes*logits, axis=-1) 
			rmse.update_state(tf.cast(tf.argmax(y,-1), tf.float32), soft_class)
			# metric.update_state(tf.keras.utils.to_categorical(labels-1,10), logits)
		else:
			classes    = np.expand_dims(np.arange(args.num_class), 0)
			max_logits = tf.nn.softmax(logits*args.test_factor, -1)
			soft_class = np.sum(classes*max_logits, axis=-1)
			# rmse.update_state(tf.cast(tf.argmax(y,-1), tf.float32), tf.squeeze(logits))
			rmse.update_state(tf.cast(tf.argmax(y,-1), tf.float32), soft_class)

	if args.task == 'clf':
		print("\nacc={}, rmse={}".format(acc.result().numpy(), rmse.result()))
		result = acc.result().numpy()
	else:
		print('\nrmse={}'.format(rmse.result()))
		result = rmse.result().numpy()
	return result

def statistics(real_label, fake_label, max_logits, argmax_fake_label):
	real_label = np.argmax(real_label, axis=1).reshape(-1,1)
	equal_1    = (real_label == fake_label)
	mean_1     = np.mean(equal_1)

	fake_label = tf.gather(fake_label, argmax_fake_label, batch_dims=1).numpy()
	equal_2    = (fake_label == real_label)
	mean_2     = np.mean(equal_2)

	# max_logits = tf.gather(full_logits, argmax_fake_label, batch_dims=1).numpy()
	# max_logits = tf.reduce_max(full_logits, axis=1)
	max_logits = tf.sigmoid(max_logits).numpy()



	print('\n', equal_1.shape, equal_2.shape, mean_1, mean_2, np.mean(equal_2[max_logits>0.7]), len(equal_2[max_logits>0.7]))

	# for eq, logit in zip(equal_2, max_logits):
	# 	print(eq,'==>',tf.sigmoid(logit))
def get_reke(anchor_ids, fake_ids, fake_label,y, discrimitor, args, helper):
	fake_logits = discrimitor.get_full_logits(anchor_ids, fake_ids).numpy() # [batch_size, neg_samples]
	fake_logits = np.squeeze(fake_logits)
	# print(fake_logits.shape, fake_ids.shape)
	assert fake_logits.shape == fake_ids.shape
	arg_logits  = np.argsort(fake_logits, axis=-1)[:,::-1]

	real_label  = np.argmax(y, axis=-1).reshape(-1,1)
	equal_label = (fake_label == real_label).astype(np.float32)

	# arg_logits  = np.argsort(equal_label, axis=-1)[:,::-1]

	max_ids     = tf.gather(fake_ids, arg_logits, batch_dims = 1)
	max_labels  = tf.gather(fake_label, arg_logits, batch_dims = 1)
	max_equal   = tf.gather(equal_label, arg_logits, batch_dims = 1)


	# truncate, rand = (23,0) if args.dataset == 'imdb' else (29,0.9) if args.dataset == 'yelp-2013' else (27,0.2)
	if args.task == 'clf':
		truncate, rand = (23,0) if args.dataset == 'imdb' else (29,0.9) if args.dataset == 'yelp-2013' else (27,0.2)
	else:
		truncate, rand = (5,0) if args.dataset == 'imdb' else (29,0.9) if args.dataset == 'yelp-2013' else (27,0.2)

	truncate = truncate-1 if np.random.random()<rand else truncate

	# print('\n', np.mean(equal_label[:,:truncate]), np.mean(max_equal.numpy()[:,:truncate]))

	return max_ids[:,:truncate]

def check_data(data_loader, y, real_ids, fake_ids, fake_label):
	for i in range(len(y)):
		assert np.argmax(y[i]) == np.argmax(data_loader.y_dat[real_ids[i]])
		for ids,label in zip(fake_ids[i], fake_label[i]):
			assert np.argmax(data_loader.y_dat[ids]) == label


def train(model, data_loader, generator, discrimitor, args):


	d_optimizer = keras.optimizers.Adam(learning_rate = args.learning_rate*2, beta_1 = 0.5)
	g_optimizer = keras.optimizers.Adam(learning_rate = args.learning_rate*2, beta_1 = 0.5)

	optimizer   = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
	if args.task == 'clf':
		ckpt = tf.train.Checkpoint(genModel = generator, disModel = discrimitor)
	else:
		ckpt = tf.train.Checkpoint(mainModel=model)
	ckpt.restore(tf.train.latest_checkpoint('./ckpt/'+args.dataset+'/'+args.task))
	# bert        = TFBertModel.from_pretrained('bert-base-uncased')
	# embeddings  = bert.get_input_embeddings().word_embeddings
	acc    = 0
	rmse   = 10.0
	helper = False
	gan_i  = 1
	for i in range(args.epoch):
		total_batch = int(data_loader.train_size/args.batch_size)+1
		data_loader.reset_train_pt()
		data_loader.split(args)
		for b in range(total_batch):
			# label, data = data_loader.next_batch()
			x, att_mask, s, y, real_ids, fake_ids, fake_label = data_loader.next_batch()
			# check_data(data_loader, y, real_ids, fake_ids, fake_label)
			# real_ids, fake_label, fake_ids                    = data_loader.gen_real_fake_data(y) 

			# anchor_hidden = model.get_hidden_states(x)
			# anchor_hidden = model.get_hidden_with_ids(s)


			if i > gan_i:
				if helper == False:
					data_loader.split(args)
					helper = True
				# optimizer   = keras.optimizers.Adam(args.learning_rate/5.)
			loss   = 0
			d_loss = 0

			# if helper == False:
			# 	for j in range(2):
			# 		with tf.GradientTape() as d_tape:
			# 			# d_loss = get_d_loss(generator, discrimitor, anchor_hidden, real_hidden, fake_hidden, fake_label, y)
			# 			d_loss = get_d_loss_1(generator, discrimitor, s, real_ids, fake_ids, fake_label, y, args)
			# 		grads = d_tape.gradient(d_loss, discrimitor.trainable_variables)
			# 		d_optimizer.apply_gradients(zip(grads, discrimitor.trainable_variables))



			helper_ids = get_reke(s, fake_ids, fake_label, y, discrimitor, args, helper)

			with tf.GradientTape(persistent = True) as tape:
				m_hidden     = model.get_hidden_states(x)
				logits       = model.get_logits(m_hidden, helper_ids, helper = helper)
				loss         = model.get_my_loss(s,y)
			gradients = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(gradients, model.trainable_variables))

			del tape

			sys.stdout.write('\repoch:{}/{}, batch:{}/{}, loss={}, d_loss={}'.format(i, args.epoch, b, total_batch, loss, d_loss))
			sys.stdout.flush()
			check_freq = 100 if args.dataset == 'yelp-2014' else 30
			if((i*total_batch+b+1)%check_freq == 0):

				result = val(model, data_loader, discrimitor, args, helper)
				# if i<gan_i:
				# 	ckpt = tf.train.Checkpoint(genModel = generator, disModel = discrimitor)
				# 	ckpt.save('./ckpt/'+args.dataset+'/'+args.task+'/model.ckpt')

				if args.task == 'clf':
					if result>acc:
						acc = result
						# ckpt = tf.train.Checkpoint(mainModel = model, genModel = generator, disModel = discrimitor)
						# ckpt.save('./ckpt/'+args.dataset+'/'+args.task+'/model.ckpt')
				else:
					if result<rmse:
						rmse = result
						# ckpt = tf.train.Checkpoint(mainModel = model)
						# ckpt.save('./ckpt/'+args.dataset+'/'+args.task+'/model.ckpt')
				# save_file = str(args.l1_kernel_size[0])+'_'+str(args.l1_hidden_size)+'.txt'
				# save_file = 'base.txt'
				# save_file = 'lens_'+str(args.max_length)+'.txt'
				# save_file = 'lstm.txt'
				# save_file = 'fc_layers_'+str(args.fc_layers)+'.txt'
				# save_file = 'pool_'+args.pool+'.txt'
				save_file = str(args.train_factor)+'_'+str(args.test_factor)+'_reg.txt'
				# fr = open('./ckpt/'+args.dataset+'/'+args.task+'/'+save_file, 'a+')
				# fr.write(str(loss.numpy())+'\t'+str(result)+'\n')
				# fr.close()


		# break


if __name__ == '__main__':
	# python3 run.py dataset task, example: python3 run.py imdb clf/reg
	# dataset = sys.argv[1]
	# task    = sys.argv[2]

	parser = argparse.ArgumentParser()
	parser.add_argument('-l1_kernel_size', type=int,nargs='+', default = [2,3,4,5])
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
	parser.add_argument('-epoch', type=int, default = 6)
	parser.add_argument('-task', type=str, default='clf')
	parser.add_argument('-dataset', type=str, default='imdb')
	parser.add_argument('-embeddings', type = np.ndarray, default = None)
	parser.add_argument('-fc_layers', type=int, default = 1)
	parser.add_argument('-doc_size', type=int, default = 90000)
	parser.add_argument('-pool', type=str, default ='atts')
	parser.add_argument('-test_factor', type=int, default=5)
	parser.add_argument('-train_factor', type=int, default=5)
	task    = parser.parse_args().task
	dataset = parser.parse_args().dataset


	# num_class = 1 if task == 'reg' else 10 if dataset=='imdb' else 5
	num_class = 10 if dataset=='imdb' else 5

	parser.add_argument('-num_class', type=int, default= num_class)

	args = parser.parse_args()

	args.l2_hidden_size = int(args.l1_hidden_size/2)
	args.fc1_size       = int(args.l1_hidden_size/4)
	args.fc2_size       = int(args.l1_hidden_size/8)
	args.l2_kernel_size = args.l1_kernel_size

	main(args)