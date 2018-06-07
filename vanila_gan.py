#https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf

import tensorflow as tf #version 1.4
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt


tensorboard_path = './tensorboard/'
saver_path = './saver/'
make_image_path = './generate/'

batch_size = 256


class GAN:

	def __init__(self, sess):
		self.noise_dim = 128 # 노이즈 차원
		self.input_dim = 784
		self.hidden_dim = 256
		self.train_rate = 0.0002

		with tf.name_scope("placeholder"):
			#class 밖에서 모델 실행시킬때 학습데이터 넣어주는곳.
			self.X = tf.placeholder(tf.float32, [None, self.input_dim])
			#class 밖에서 모델 실행시킬때 class의 Generate_noise 실행한 결과를 넣어주는 곳.
			self.noise_source = tf.placeholder(tf.float32, [None, self.noise_dim])

		
		with tf.name_scope("generate_image_from_noise"):
			#노이즈로 데이터 생성. 
			self.Gen = self.Generator(self.noise_source) #batch_size, input_dim

		
		with tf.name_scope("result_from_Discriminator"):
			#학습데이터가 진짜일 확률
			self.D_X = self.Discriminator(self.X) #batch_size, 1
			#노이즈로부터 생성된 데이터가 진짜일 확률 
			self.D_Gen = self.Discriminator(self.Gen, True) #batch_size, 1


		with tf.name_scope("for_check_Discriminator_values"):
			#학습데이터 진짜일 확률 batch끼리 합친거. 나중에 총 데이터 셋으로 나눠줘서 평균 볼 용도.
			self.D_X_sum = tf.reduce_sum(self.D_X)
			self.D_Gen_sum = tf.reduce_sum(self.D_Gen)


		with tf.name_scope("loss"):
			#Discriminator 입장에서 최대화 해야 하는 값
			self.D_loss = self.Discriminator_loss_function(self.D_X, self.D_Gen)
			#Generator 입장에서 최대화 해야 하는 값.
			self.G_loss = self.Generator_loss_function(self.D_Gen)


		with tf.name_scope("train"):
			#학습 코드
			self.optimizer = tf.train.AdamOptimizer(self.train_rate)
				
				#Discriminator와 Generator에서 사용된 variable 분리.
			self.D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')
			self.G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
				
				#maximize 해야하므로 minimize에 -1 곱해줌.
			self.D_Maximize = self.optimizer.minimize(-self.D_loss, var_list=self.D_variables) #G 변수는 고정하고 D로만 학습.
			self.G_Maximize = self.optimizer.minimize(-self.G_loss, var_list=self.G_variables) #D 변수는 고정하고 G로만 학습.


		with tf.name_scope("tensorboard"):
			#tensorboard
			self.D_X_tensorboard = tf.placeholder(tf.float32) #학습데이터가 진짜일 확률
			self.D_Gen_tensorboard = tf.placeholder(tf.float32) #노이즈로부터 생성된 데이터가 진짜일 확률 
			self.D_value_tensorboard = tf.placeholder(tf.float32) #Discriminator 입장에서 최대화 해야 하는 값
			self.G_value_tensorboard = tf.placeholder(tf.float32) #Generator 입장에서 최대화 해야 하는 값.

			self.D_X_summary = tf.summary.scalar("D_X", self.D_X_tensorboard) 
			self.D_Gen_summary = tf.summary.scalar("D_Gen", self.D_Gen_tensorboard) 
			self.D_value_summary = tf.summary.scalar("D_value", self.D_value_tensorboard) 
			self.G_value_summary = tf.summary.scalar("G_value", self.G_value_tensorboard) 
			
			self.merged = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter(tensorboard_path, sess.graph)


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)


		sess.run(tf.global_variables_initializer())



	#노이즈 생성
	def Generate_noise(self, batch_size): #batch_size, nose_dim
		return np.random.normal(size=[batch_size, self.noise_dim])


	#데이터의 진짜일 확률
	def Discriminator(self, data, reuse=False): #batch_size, 1
		with tf.variable_scope('Discriminator') as scope:
			if reuse == True: #Descriminator 함수 두번 부르는데 두번째 부르는 때에 같은 weight를 사용하려고 함.
				scope.reuse_variables()
			
			hidden = tf.layers.dense(data, self.hidden_dim, activation=tf.nn.relu) #probability
			P = tf.layers.dense(hidden, 1, activation=tf.nn.sigmoid) #probability #확률이므로 0~1 나오는 시그모이드 써야됨.
			
			return P


	#노이즈로 진짜같은 데이터 생성
	def Generator(self, noise): #batch_size * input_dim
		with tf.variable_scope('Generator'):

			hidden = tf.layers.dense(noise, self.hidden_dim, activation=tf.nn.relu)
			Z = tf.layers.dense(hidden, self.input_dim, activation=tf.nn.sigmoid) #mnist데이터가 0~1 사이니까 sigmoid 씀.
			
			return Z #생성된 이미지


	
	#Discriminator 입장에서 최대화 해야 하는 값
	def Discriminator_loss_function(self, D_X, D_Gen):
		return tf.reduce_mean(tf.log(D_X) + tf.log(1-D_Gen))



	#Generator 입장에서 최대화 해야 하는 값.
	def Generator_loss_function(self, D_Gen):
		return tf.reduce_mean(tf.log(D_Gen))
		# tf.reduce_mean(tf.log(1-D_Gen)) 를 최소화 하도록 해도 되지만 학습이 느림.
			# 학습 초에는 D_Gen가 0에 근접한데 이 경우 log(1-0) 즉 0에 근사하게됨. 따라서 느림
 
		#log(1-D_Gen)가 최소화 되려면 D_Gen가 커져야함. 따라서 새로운 식으로 활용 가능.
		#이 이유로 tf.log(D_Gen)를 최대화하도록 학습.
			# 학습 초기 D_Gen가 0에 근사하더라도 -무한대에 가까움 따라서 학습이 잘 됨.



def train(model, data):
	total_D_value = 0
	total_G_value = 0
	total_D_X = 0
	total_D_Genb = 0


	np.random.shuffle(data)
	iteration = int(np.ceil(len(data)/batch_size))


	for i in range( iteration ):
		#train set. mini-batch
		input_ = data[batch_size * i: batch_size * (i + 1)]

		#노이즈 생성.
		noise = model.Generate_noise(len(input_)) # len(input_) == batch_size, noise = (batch_size, model.noise_dim)
			
		#Discriminator 학습.
		_, D_value = sess.run([model.D_Maximize, model.D_loss], {model.X:input_, model.noise_source:noise})

		#Generator 학습.
		_, G_value = sess.run([model.G_Maximize, model.G_loss], {model.noise_source:noise})

		#학습데이터가 진짜일 확률(D_X)와 노이즈로부터 생성된 데이터가 진짜일 확률(D_Gen).  mini-batch니까 합으로 구하고 나중에 토탈크기로 나누자.
		D_X, D_Gen = sess.run([model.D_X_sum, model.D_Gen_sum], {model.X:input_, model.noise_source:noise})
		

		#parameter sum
		total_D_value += D_value
		total_G_value += G_value
		total_D_X += D_X
		total_D_Genb += D_Gen

	
	return total_D_value/iteration, total_G_value/iteration, total_D_X/len(data), total_D_Genb/len(data)



def write_tensorboard(model, D_X, D_Gen, D_value, G_value, epoch):
	summary = sess.run(model.merged, 
					{
						model.D_X_tensorboard:D_X,
						model.D_Gen_tensorboard:D_Gen,
						model.D_value_tensorboard:D_value, 
						model.G_value_tensorboard:G_value,
					}
				)

	model.writer.add_summary(summary, epoch)



def gen_image(model, epoch):
	num_generate = 10
	noise = model.Generate_noise(num_generate) # noise = (num_generate, model.noise_dim)
	generated = sess.run(model.Gen, {model.noise_source:noise}) #num_generate, 784
	generated = np.reshape(generated, (-1, 28, 28)) #이미지 형태로. #num_generate, 28, 28
		
	fig, axes = plt.subplots(1, num_generate, figsize=(num_generate, 1))

	for i in range(num_generate):
		axes[i].set_axis_off()
		axes[i].imshow(generated[i])

	plt.savefig(make_image_path+str(epoch))
	plt.close(fig)	



def run(model, train_set, restore = 0):
	#weight 저장할 폴더 생성
	if not os.path.exists(saver_path):
		os.makedirs(saver_path)
	
	#생성된 이미지 저장할 폴더 생성
	if not os.path.exists(make_image_path):
		os.makedirs(make_image_path)

	#restore인지 체크.
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	

	#학습 진행
	for epoch in range(restore + 1, 10001):
		#Discriminator 입장에서 최대화 해야 하는 값, #Generator 입장에서 최대화 해야 하는 값, #학습데이터가 진짜일 확률, #노이즈로부터 생성된 데이터가 진짜일 확률  
		D_value, G_value, D_X, D_Gen = train(model, train_set)

		print("epoch : ", epoch, " D_value : ", D_value, " G_value : ", G_value, " 학습데이터 진짜일 확률 : ", D_X, " 생성데이터 진짜일 확률 : ", D_Gen)

		
		if epoch % 10 == 0:
			#tensorboard
			write_tensorboard(model, D_X, D_Gen, D_value, G_value, epoch)

			#weight save
			#save_path = model.saver.save(sess, saver_path+str(epoch)+".ckpt")
		
			#image 생성
			gen_image(model, epoch)




sess = tf.Session()

#model
model = GAN(sess) #noise_dim, input_dim

#get mnist data #이미지의 값은 0~1 사이임.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#train dset
train_set = mnist.train.images # 55000, 784

#run
run(model, train_set)

