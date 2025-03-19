'''
	用于独立训练某一个模块
'''


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from os.path import join

from model.tcn import TCN_IV_1D_C
from model.CNN2Layer import *
from model.RNN_CNN import inverse_model
from model.M2M_LSTM import LSTM_MM, GRU_MM, CNN_GRU_MM
from model.Unet_1D import Unet_1D

from model.Transformer import TransformerModel

from model.Forward import *
from model.geomorphology_classification import Facies_model_class


from setting import *
from utils.utils import extract, standardize
from utils.datasets import *
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score
import errno
import argparse


##------1维模型的选择-----------
model_name = TCN1D_train_p['model_name']
Forward_model = TCN1D_train_p['Forward_model']
Facies_model_C = TCN1D_train_p['Facies_model']

data_flag = TCN1D_train_p['data_flag']
get_F = TCN1D_train_p['get_F']
F = TCN1D_train_p['F']  # 地震数据的处理方式
cuda = True if torch.cuda.is_available() else False

if model_name == 'tcnc':
	choice_model = TCN_IV_1D_C      #  @@@论文4@@，TCN
if model_name == 'VishalNet':
	choice_model = VishalNet        #  @@@论文9@@，Conv_1d(1*81) + ReLU + Conv_1d(1*301),
if model_name == 'GRU_MM':
	choice_model = GRU_MM           #  @@@论文25@@，GRU模型： 2层GRU(dropout=0.75) + Linear(特征维度)
if model_name == 'Unet_1D':
	choice_model = Unet_1D          #  @@@论文24@@，unet模型的1D样式
if model_name == 'Transformer':
	choice_model = TransformerModel
print(choice_model)
# -------------------------------------------------正演模型选择-----
if Forward_model == 'cnn':
	forward = forward_model_0
if Forward_model == 'convolution':
	forward = forward_model_1  # 标准褶积模型
if Forward_model == 'cov_para':
	forward = forward_model_2  # 并联卷积

# ------------------------------------------------地震相模型---------
if Facies_model_C == 'Facies':
	Facies_class = Facies_model_class


def get_data(no_wells=10, data_flag='SEAM', get_F=get_F):
	'''
		no_well: 井数，
		输出的数据格式： (num_traces, channel, depth_samples)
	'''
	if data_flag == 'Fanny':
		# 地震波形(129600, 64)， GR反演效果明显好于Impedance。
		seismic = np.load(join('data',data_flag,'Seismic.npy')).squeeze()

		# model (129600, 64)
		GR = np.load(join('data',data_flag, 'GR.npy')).squeeze()       # 使用GR
		model = np.load(join('data',data_flag, 'Impedance.npy')).squeeze()  # 使用波阻抗

		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())

	if data_flag == 'Stanford_VI':
		# stanford_VI: 地震波形(200, 200, 150)
		seismic = np.load(join('data',data_flag,'synth_40HZ.npy'))
		H,inline, xline = seismic.shape
		seismic = seismic.reshape(H, inline*xline)
		seismic = np.transpose(seismic, (1, 0))

		# stanford_VI: 波阻抗(200, 200, 150) 
		model = np.load(join('data',data_flag,'AI.npy'))
		model = model.reshape(H, inline*xline)
		model = np.transpose(model, (1, 0))
		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean(), ) 

	# 1、为了满足unet下采样的长度要求，需要截取。
	s_L = seismic.shape[-1]
	n = int((s_L//8)*8)
	seismic = seismic[:,:n]
	model = model[:, :n]

	seismic_o, model_o = seismic[:, np.newaxis, :], model[:, np.newaxis, :]
	seismic, model = standardize(seismic, model, no_wells)

	# 2、增加地震数据的频率特征和动态特征。(num_traces, depth_samples) --> (num_traces, depth_samples, 5)
	if get_F:
		# 注意，此训练模块，通常设置get_F = 0
		seismic = seismic[1:2720, :]  # (2719,701), M2原始数据第一行和最后一行都是0。 注意这个值应该设置成变量？？
		seismic_feature = n_s_features(seismic, get_F//2)
		seismic = np.transpose(seismic_feature, (0,2,1))  # (num_traces, 5, depth_samples)
		model = model[1:2720, np.newaxis, :]                   # (num_traces, 1, depth_samples)
	else:
		# (num_traces, 1, depth_samples)
		seismic = seismic[:, np.newaxis, :]
		model = model[:, np.newaxis, :]

	print('train long: ',model.shape, seismic.shape)
	# 数据维度： （num_traces, channel=1或5, depth_samples）
	# plt.imshow(seismic[:,1,:])
	# plt.show()

	# 在重构损失，判别损失中，需要原始地震数据，所以输出-o版本（标准化之前的数据）
	return seismic, model, seismic_o, model_o

def train(train_p):
	"""Function trains paper4"""
	
	# 获取全部的数据，(num_traces, depth_samples)
	seismic, model, seismic_o, model_o = get_data(no_wells=train_p['no_wells'], data_flag=train_p['data_flag'])

	# 1、获取训练数据，需要注意的是，batch_size 是整个训练数据集的大小
	traces_train = np.linspace(0, len(model)-1, train_p['no_wells'], dtype=int)
	train_dataset = SeismicDataset1D(seismic, model, traces_train)
	# train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))  # 容易掉沟里出不来
	train_loader = DataLoader(train_dataset, batch_size=train_p['batch_size'])  # batch size越小，损失函数的下降随机性越大

	# 3、非监督训练数据获取，数量=unsupervised_seismic 
	unsupervised_traces_train = np.linspace(0, len(model)-1, train_p['unsupervised_seismic'], dtype=int)
	unsupervised_dataset = UnsupervisedSeismicDataset(seismic, unsupervised_traces_train)  # 非监督训练使用全部的地震数据
	unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=train_p['batch_size'])

	# 4、测试数据，3个井作为测试数据
	traces_validation = np.linspace(0, len(model)-1, 3, dtype=int)
	val_dataset = SeismicDataset1D(seismic, model, traces_validation)
	val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
	
	# define device for training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 反演模型和,正演模型,判别模块
	# inverse_model参数：input_dim=1, output_dim=1, num_heads=4, num_layers=3, dim_feedforward=128, dropout=0.1
	inverse_model = choice_model().to(device)               # (input:x,y)
	if Forward_model != '':
		forward_model = forward(resolution_ratio=1).to(device)
	
	# Set up loss
	criterion = torch.nn.MSELoss()  # MSE损失
	criterion2 = torch.nn.L1Loss()  # MAE损失，相对于平方的MSE损失，MAE更线性。

	criterion_D = torch.nn.MSELoss()  # pix2pix中使用的Least Squares GAN判别损失。******

	# 训练目标, 优化目标可能不同
	if Forward_model == '':
		optimizer = torch.optim.Adam(list(inverse_model.parameters()), 
									 weight_decay=0.0001, lr=train_p['lr'])
	else:
		optimizer = torch.optim.Adam(list(inverse_model.parameters())+list(forward_model.parameters()), 
								 	weight_decay=0.0001, lr=train_p['lr'])

	# optimizer_D = torch.optim.Adam(disc_model.parameters(), 
	# 							 weight_decay=0.0001, lr=0.0001)  # 只优化判别器

	train_unsup_loss = []  # 存放非监督学习的loss
	train_loss = []        # 存放监督训练的loss
	train_D_loss = []      # 存放监督训练判别器的loss
	val_loss = []

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	for epoch in range(train_p['epochs']):

		inverse_model.train()

		'''
			下面是训练的过程，监督训练和非监督训练。
			x,y: (B, 1, H); y_pred:(B, 1, H)
		'''

		# 2、真实地震数据，-----------非监督训练-->重构损失------------------------
		# for x in unsupervised_loader:
		# 	optimizer.zero_grad()
		# 	y_pred = inverse_model(x)
		# 	# y_pred = inverse_model(x, y)
		# 	x_rec = forward_model(y_pred)

		# 	loss_un_train = criterion(x, x_rec)
		# 	train_unsup_loss.append(loss_un_train.item())
		# 	loss_un_train.backward()
		# 	optimizer.step()


		# # 4、真实数据，----------混合训练-->监督损失 + 重构损失-------------
		# for x,y in train_loader:
		# 	optimizer.zero_grad()

		# 	# 前向传播
		# 	y_pred = inverse_model(x)
		# 	# y_pred = inverse_model(x,y)
		# 	x_rec = forward_model(y_pred)

		# 	# 波阻抗监督损失，地震重构损失
		# 	loss_train = criterion(y_pred, y) + criterion(x, x_rec)  # 监督中的重构损失负影响
		# 	# loss_train = criterion(y_pred, y)
		# 	train_loss.append(loss_train.item())
		# 	loss_train.backward()
		# 	optimizer.step()

		# 5、真实数据，------------监督损失---------------
		for x,y in train_loader:
			optimizer.zero_grad()

			# 前向传播
			y_pred = inverse_model(x)
			# y_pred = inverse_model(x,y)

			# 波阻抗监督损失
			loss_train = criterion(y_pred, y)
			train_loss.append(loss_train.item())
			loss_train.backward()
			optimizer.step()

		# 6、------测试------------------------------------
		for x, y in val_loader:
			# (3, 1, 701)
			inverse_model.eval()
			y_pred = inverse_model(x)
			# y_pred = inverse_model(x,y)
			loss_val = criterion(y_pred, y)
			val_loss.append(loss_val.item())

		print('Epoch: {} | Train Loss: {:0.4f} | Val Loss: {:0.4f} \
			'.format(epoch, loss_train.item(), loss_val.item()))

	# save trained models
	if not os.path.isdir('save_train_model'):  # check if directory for saved models exists
		os.mkdir('save_train_model')
	# torch.save(inverse_model, 'save_train_model/%s_%s_%s_sigle_s_uns.pth'%(model_name, Forward_model, data_flag))
	torch.save(inverse_model, 'save_train_model/%s_%s_sigle_s_%s.pth'%(model_name, Forward_model, data_flag))

	plt.plot(train_loss,'r')
	plt.plot(val_loss,'k')
	# plt.savefig('results/%s_%s_%s_sigle_s_uns.png'%(model_name, Forward_model, data_flag))
	plt.savefig('results/%s_%s_sigle_s_%s.png'%(model_name, Forward_model, data_flag))


if __name__ == '__main__':

	train(train_p=TCN1D_train_p)
