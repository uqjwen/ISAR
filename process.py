import sys

import numpy as np 

import pickle

from keras.preprocessing.sequence import pad_sequences

import os 

class Trans_X():

	def __init__(self,x_dat, y_dat,p_dat,u_dat, num_user, num_item, data_dir, domain):

		self.x_dat = x_dat

		self.y_dat = y_dat

		self.p_dat = p_dat

		self.u_dat = u_dat

		self.num_user = num_user

		self.num_item = num_item

		self.data_dir = data_dir

		self.domain = domain



		filename = data_dir+'/x_neighbor.npy'

		# ypred = np.load(data_dir+'/ypred.npy')

		self.y_dat = np.argmax(self.y_dat, axis=-1)

		# self.y_dat[-len(ypred):] = ypred

		# self.get_neighbors_from_rating()





		# fr1 = data_dir+'/x_neighbor_item.npy'

		# fr2 = data_dir+'/x_neighbor_user.npy'

		# fr3 = data_dir+'/x_neighbor_ui.npy'



		# if os.path.exists(fr1):

		# 	mat1 = np.load(fr1)

		# else:

		# 	mat1 = self.get_data('item')

		# 	np.save(fr1, mat1)



		# if os.path.exists(fr2):

		# 	mat2 = np.load(fr2)

		# else:

		# 	mat2 = self.get_data('user')

		# 	np.save(fr2, mat2)



		# mat = np.concatenate([mat1[:,:10], mat2[:,:10]], axis=-1)

		# np.save(fr3, mat)



		# self.res = mat1

		self.sent = np.array(list(range(len(self.x_dat))))

	# def get_gather():

	def get_data(self, name = 'item'):

		res = []

		length = len(self.x_dat)

		my_golden = []

		his_golden = []

		my_diff = []

		his_diff = []

		my_dist = []

		his_dist = []

		num = 20

		position = []

		my_counter = {}

		his_counter = {}

		data1 = self.p_dat if name == 'item' else self.u_dat

		data2 = self.u_dat if name == 'item' else self.p_dat

		neighbor_mat = self.u_neighbor_mat if name == 'item' else self.i_neighbor_mat

		for i in range(length):

			label = self.y_dat[i]

			item = data1[i]



			index = np.where(data1 == item)[0]

			# index = np.random.choice(range(length),100)

			user = data2[i]      # user in this instance

			users = data2[index].flatten()   # other users who have also rated the item: item

			sims = neighbor_mat[user].flatten() # similarities between user and all the other users

			similarity = sims[users] # similarities between current user and all the other users who have rated the item

			similarity = np.round(similarity,3)



			sort_index = np.argsort(similarity)[::-1]



			if len(index)-1>=num:

				neighbors_1 = index[sort_index][1:num+1]

			else:

				neighbors_1 = np.concatenate([index[sort_index][1:],np.array([i]*(num-len(index)+1))])





			labels = self.y_dat[index]

			diffs = np.abs(labels - label)

			sort_index_2 = list(index[np.argsort(diffs)])

			golden = list(index[diffs == 0])

			if i in golden:

				golden.remove(i)



			# rate = len(golden)*1.8/len(index)

			# neighbors_1 = []

			# for item in index[sort_index][1:]:

			# 	sample = np.random.uniform(0,1)

			# 	to_add = item if sample<=rate else i

			# 	neighbors_1.append(to_add)

			# 	if len(neighbors_1) == num:

			# 		break

			# if len(neighbors_1)<num:

			# 	neighbors_1 += [i]*(num-len(neighbors_1))









			# sort_index.remove(i)

			neighbors_2 = [golden[0] if len(golden)>0 else i]+list(np.random.choice(sort_index_2,num-1))

			# np.random.shuffle(neighbors_2)



			my_golden_num = len(np.intersect1d(neighbors_1, golden))

			his_golden_num = len(np.intersect1d(neighbors_2, golden))

			my_golden.append(my_golden_num)

			his_golden.append(his_golden_num)



			my_counter[my_golden_num] = my_counter.get(my_golden_num,0)+1

			his_counter[his_golden_num] = his_counter.get(his_golden_num, 0)+1





			my_dist.append([1 if item < 62522 else 0 for item in neighbors_1])

			his_dist.append([1 if item < 62522 else 0 for item in neighbors_2])

			# neighbors_1 = np.intersect1d(neighbors_1, golden)

			# remain_len = num - len(neighbors_1)

			# additional = np.random.choice(index, remain_len)

			# neighbors_1 = np.concatenate([neighbors_1, additional])

			print('all my neighbors: ', self.get_neighbor_sim(index[sort_index], similarity[sort_index]))

			print('my_neighbor:', neighbors_1)

			my_diff_individual = np.abs(self.y_dat[neighbors_1]-label)

			my_diff.append(my_diff_individual)

			print('my_neighbor_diff: ', np.abs(self.y_dat[neighbors_1]-label))



			print('his_neighbor: ', neighbors_2)

			his_diff_individual = np.abs(self.y_dat[neighbors_2]-label)

			his_diff.append(his_diff_individual)

			print('his_neighbor_diff: ', np.abs(self.y_dat[neighbors_2]-label))

			print('golden: ', golden)

			print('-----------------------------------------------------------')

			# neighbors = sort_index[:5]

			# neighbors_1 = [item if my_diff_individual[i]==0 else np.random.choice(neighbors_1[my_diff_individual!=0])     for i,item in enumerate(neighbors_1)]

			np.random.shuffle(neighbors_1)

			res.append(neighbors_1)

			sys.stdout.write('\r {}/{}. {}'.format(i,length,len(neighbors_1)))

			sys.stdout.flush()



		# position = np.array(position)

		# print(np.sum(position, axis=0))

		res = np.array(res)



		my_dist = np.array(my_dist)

		his_dist = np.array(his_dist)

		my_dist_t = np.mean(np.sum(my_dist[:62522],axis=-1))

		my_dist_v = np.mean(np.sum(my_dist[62522:], axis=-1))





		his_dist_t = np.mean(np.sum(his_dist[:62522],axis=-1))

		his_dist_v = np.mean(np.sum(his_dist[62522:], axis=-1))











		print('\n my_golden: ',np.mean(my_golden))

		print(' his_golden: ', np.mean(his_golden))

		print('my_mean_diff:', np.mean(my_diff))

		print('his_mean_diff: ', np.mean(his_diff))



		print('my_counter: ', my_counter)

		print('his_counter: ', his_counter)





		print('my_dist_t: ', my_dist_t, num - my_dist_t)

		print('my_dist_v: ', my_dist_v, num - my_dist_v)





		print('his_dist_t: ', his_dist_t, num - his_dist_t)

		print('his_dist_v: ', his_dist_v, num - his_dist_v)

		print(res.shape)

		# np.save(filename,res)

		return res





	# def get_neighbors(self,mat):

	# 	rows, cols = mat.shape

	# 	mat = mat/np.max(mat)

	# 	res_mat = np.zeros((rows,rows))

	# 	for i in range(rows):

	# 		sys.stdout.write('\r {}/{}'.format(i,rows))

	# 		for j in range(i,rows):

	# 			a1 = mat[i]

	# 			a2 = mat[j]



	# 			res = np.sum(a1*a2)

	# 			norm = ((np.sum(a1**2))**0.5) * ((np.sum(a2**2))**0.5)

	# 			res = res/norm 

	# 			res_mat[i,j] = res 

	# 			res_mat[j,i] = res 

	# 	return res_mat

	def get_neighbor_sim(self,neighbors, similarity):

		res_str = ''

		for n,s in zip(neighbors, similarity):

			res_str += str(n)+':'+str(s)+' '

		return res_str



	def get_neighbors(self, mat):

		rows, cols = mat.shape

		res_mat = np.zeros((rows,rows))

		for i in range(rows):

			sys.stdout.write('\r {}/{}'.format(i,rows))

			for j in range(i,rows):

				a1 = mat[i]

				a2 = mat[j]

				commo_vec = (a1*a2!=0).astype(int)

				equal_vec = (a1-a2==0).astype(int)



				coeq_vote = np.sum(commo_vec*equal_vec)

				comm_vote = np.sum(commo_vec)

				norm = np.sqrt(comm_vote)

				norm = max(norm,1)

				res = coeq_vote/norm



				res_mat[i,j] = res 

				res_mat[j,i] = res

		return res_mat



	def get_neighbors_from_rating(self,n_neighbors=20):

		# user,item,label = self.train_u, self.train_p, self.train_y 

		train_size = 62522 if self.domain == 'yelp' else 67426

		user,item,label = self.u_dat[:train_size], self.p_dat[:train_size], self.y_dat[:train_size]

		up_mat = np.zeros((self.num_user, self.num_item))

		for u,p,r in zip(user,item,label):

			u = u[0]

			p = p[0]

			up_mat[u,p] = r 



		self.u_neighbor_mat = np.zeros((self.num_user, self.num_user))

		# print("getting user similarity matrix")

		filename = self.data_dir+'/u_neighbors.npy'

		if not os.path.exists(filename):

			print('user neighbors file not exists')

			self.u_neighbor_mat = self.get_neighbors(up_mat)

			np.save(filename, self.u_neighbor_mat)

		else:

			self.u_neighbor_mat = np.load(filename)



		filename = self.data_dir+'/i_neighbors.npy'

		if not os.path.exists(filename):

			print('item neighbors file not exists')

			self.i_neighbor_mat = self.get_neighbors(up_mat.T)

			np.save(filename, self.i_neighbor_mat)

		else:

			self.i_neighbor_mat = np.load(filename)





	# def get_neighbors_from_rating(self,n_neighbors=20):

	# 	# user,item,label = self.train_u, self.train_p, self.train_y 

	# 	user,item,label = self.u_dat, self.p_dat, self.y_dat

	# 	up_mat = np.zeros((self.num_user, self.num_item))

	# 	for u,p,r in zip(user,item,label):

	# 		u = u[0]

	# 		p = p[0]

	# 		up_mat[u,p] = r 



	# 	self.u_neighbor_mat = np.zeros((self.num_user, self.num_user))

	# 	print("getting user similarity matrix")

	# 	filename = self.data_dir+'/u_neighbors.npy'

	# 	if not os.path.exists(filename):

	# 		for i in range(self.num_user):

	# 			for j in range(i,self.num_user):

	# 				a1 = up_mat[i]

	# 				a2 = up_mat[j]

	# 				commo_vec = (a1*a2!=0).astype(int)



	# 				equal_vec = (a1-a2==0).astype(int)



	# 				coeq_vote = np.sum(commo_vec*equal_vec)

	# 				comm_vote = np.sum(commo_vec)



	# 				norm = np.sqrt(comm_vote)

	# 				norm = max(norm,1)

	# 				res = coeq_vote/norm



	# 				self.u_neighbor_mat[i,j] = res

	# 				self.u_neighbor_mat[j,i] = res

	# 		np.save(filename, self.u_neighbor_mat)

	# 	else:

	# 		self.u_neighbor_mat = np.load(filename)







if __name__ == '__main__':

	dat_data = ['x','y','l','u','p']

	filename = '/home/wenjh/data/yelp/data.pkl'

	fr = open(filename, 'rb')

	pickle_data = pickle.load(fr)

	fr.close()

	num_user = 1631

	num_item = 1633

	x_dat,y_dat,_,u_dat,p_dat = pickle_data['data']

	x_dat = pad_sequences(x_dat, 300, padding = 'post')

	trans_x = Trans_X(x_dat, y_dat, p_dat, u_dat, num_user, num_item,'/home/wenjh/data/yelp', 'yelp')