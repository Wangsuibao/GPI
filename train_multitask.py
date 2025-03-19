'''
*** 联合 {地震相分析} + {地震反演} + {地震正演}****
'''


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from os.path import join

from model.CNN2Layer import *

from model.tcn import TCN_IV_1D_C

from model.M2M_LSTM import LSTM_MM, GRU_MM, CNN_GRU_MM
from model.RNN_CNN import inverse_model
from model.Unet_1D import Unet_1D
from model.Transformer import TransformerModel

from model.Forward import forward_model_0, forward_model_1, forward_model_2
from model.geomorphology_classification import Facies_model_class

from setting import *
from utils.utils import *
from utils.datasets import SeismicDataset1D, UnsupervisedSeismicDataset, SeismicDataset1D_SPF, SeismicDataset1D_SPF_WS
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score
import errno
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 2、解决RuntimeError: CUDA error: unspecified launch failure
torch.cuda.device_count()

##------1维模型的选择-----------
model_name = TCN1D_train_p['model_name']
Forward_model = TCN1D_train_p['Forward_model']
Facies_model_C = TCN1D_train_p['Facies_model']
facies_n=4

data_flag = TCN1D_train_p['data_flag']
get_F = TCN1D_train_p['get_F']
F = TCN1D_train_p['F']  # 地震数据的处理方式


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


def get_data_SPF(no_wells=10, data_flag='Stanford_VI', get_F=get_F):
	'''
		no_well: 均匀采样的井数，
		data_flag: 数据集在深度域-时间域是完全对齐的。 stanford_VI,（深度，inline, xline）= (200, 200, 150)
		读取的数据维度是：（深度，inline, xline）-->（深度，inline*xline）--> (N, 深度)  --> (N, 1, 深度)
		输出的数据格式： (num_traces, channel, depth_samples)
		输出数据类型：地震，波阻抗，低频信息(地震相，构造，低频地震)
	'''
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

		# stanford_VI: 地震相(200, 200, 150)
		facies = np.load(join('data',data_flag,'Facies.npy'))
		facies = facies.reshape(H, inline*xline)
		facies = np.transpose(facies, (1, 0))

		print('original long: ', model.shape, seismic.shape, facies.shape)
		# 8800.779; -0.1263349; [0. 1. 2. 3.]
		print('model_Mean:',model.mean(),'seismic_mean:',seismic.mean(),'Facies_unique:',np.unique(facies))

	if data_flag == 'Fanny':
		seismic = np.load(join('data',data_flag, 'seismic.npy'))  # (129600, 64)
		model = np.load(join('data',data_flag, 'impedance.npy'))  # (129600, 64)
		GR = np.load(join('data',data_flag, 'GR.npy'))  # (129600, 64)
		facies = np.load(join('data',data_flag, 'facies.npy'))    # (129600, 64)

		print('original long: ', model.shape, seismic.shape, facies.shape)
		# 0.449; 0.4646; [0. 1.]
		print('model_Mean:', model.mean(),'seismic_mean:',seismic.mean(),'Facies_unique:', np.unique(facies)) 

	model = model  # 可以是GR

	seismic, model = standardize(seismic, model, no_wells)
	# 1、为了满足unet下采样的长度要求，需要截取。
	s_L = seismic.shape[-1]
	n = int((s_L//8)*8)
	seismic = seismic[:,:n]
	model = model[:, :n]
	facies = facies[:,:n]

	# 输出维度是（N, 1, H）
	return seismic[:, np.newaxis, :], model[:, np.newaxis, :], facies[:, np.newaxis, :]

def train(train_p):
	"""Function trains 1-D TCN as specified in the paper"""
	
	# 获取全部的数据，(num_traces, depth_samples)
	seismic, model, facies = get_data_SPF(no_wells=train_p['no_wells'], data_flag=train_p['data_flag'])

	# 获取训练数据，需要注意的是，batch_size 是整个训练数据集的大小
	traces_train = np.linspace(0, len(model)-1, train_p['no_wells'], dtype=int)
	train_dataset = SeismicDataset1D_SPF(seismic, model, facies, traces_train)
	train_loader = DataLoader(train_dataset, batch_size=train_p['batch_size'])  # batch size越小，损失函数的下降随机性越大

	# 弱监督训练,地震数据获取
	Wsupervised_traces_train = np.linspace(0, len(model)-1, train_p['unsupervised_seismic'], dtype=int)
	Wsupervised_dataset = SeismicDataset1D_SPF_WS(seismic, facies, Wsupervised_traces_train)  # 非监督训练使用全部的地震数据
	Wsupervised_loader = DataLoader(Wsupervised_dataset, batch_size=train_p['batch_size'])	

	# 测试数据
	traces_validation = np.linspace(0, len(model)-1, 3, dtype=int)
	val_dataset = SeismicDataset1D(seismic, model, traces_validation)
	val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
	
	# define device for training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 反演模型和正演模型
	# inverse_model = choice_model(1, resolution_ratio=1, nonlinearity='relu').to(device)
	inverse_model = choice_model().to(device)
	forward_model = forward().to(device)
	Facies_model = Facies_class(facies_n=facies_n).to(device)
	# Set up loss (先-output, 后-label)
	criterion = torch.nn.MSELoss()
	# 多分类-分割损失函数的输入： output：（B, n_class, H）, label:(B, H)
	criterion_facies = nn.CrossEntropyLoss()  # 损失函数输入的是没有经过softmax的原始输出。所以模型最后不用softmax
	
	# 优化目标中需要包含所有的模块：1)反演模块；2)正演模块；3)地震相分类模块

	optimizer = torch.optim.Adam(list(inverse_model.parameters())+
								list(forward_model.parameters())+
								list(Facies_model.parameters())
								, weight_decay=0.0001, lr=train_p['lr'])
	# optimizer = torch.optim.Adam(list(inverse_model.parameters())+
	# 							   list(forward_model.parameters()), 
	# 							  weight_decay=0.0001, lr=train_p['lr'])
	# optimizer = torch.optim.Adam(list(inverse_model.parameters())+
	# 							   list(Facies_model.parameters())
	# 							 , weight_decay=0.0001, lr=train_p['lr'])
	train_loss = []
	val_loss = []
	for epoch in range(train_p['epochs']):

		inverse_model.train()
		forward_model.train()
		Facies_model.train()

		'''
			下面是训练的过程，非监督训练和监督训练。
		'''
		for x, z in Wsupervised_loader:
			optimizer.zero_grad()
			y = inverse_model(x)
			# y = inverse_model(x,y)
			x_rec = forward_model(y)
			facie = Facies_model(y)
			# Wsupervised_loss = criterion_facies(facie, z)
			# Wsupervised_loss = criterion(x_rec, x)
			Wsupervised_loss = criterion_facies(facie, z) + criterion(x_rec, x)
			Wsupervised_loss.backward()
			# torch.nn.utils.clip_grad_norm_(list(inverse_model.parameters()) + list(forward_model.parameters()), train_p['grad_clip'])
			optimizer.step()

		for x, y, z in train_loader:
			# print('x,y: ',x.shape,y.shape,z.shape)
			# x, y, y_pred: (2, 1, 696), float ； z (2, 696) long
			optimizer.zero_grad()
			# y_pred = inverse_model(x,y)
			y_pred = inverse_model(x)
			x_rec = forward_model(y_pred)
			facie = Facies_model(y_pred)

			# print(facie.shape, z.shape)  # torch.Size([4, 4, 200]) torch.Size([4, 200])
			l1 = criterion(y_pred, y)
			l2 = criterion_facies(facie, z)
			l3 = criterion(x_rec, x)
			# print('l1: ', l1, 'l2: ', l2, 'l3: ', l3)  # 0.4, 0.08, 0.1, 波阻抗损失为主，是对的。
			loss_train = l1 + l2 + l3
			loss_train.backward()
			# torch.nn.utils.clip_grad_norm_(list(inverse_model.parameters()) + list(forward_model.parameters()), train_p['grad_clip'])
			optimizer.step()

			train_loss.append(loss_train.item())


		for x, y in val_loader:
			# (3, 1, 701)
			inverse_model.eval()
			# y_pred = inverse_model(x,y)
			y_pred = inverse_model(x)
			loss_val = criterion(y_pred, y)
			val_loss.append(loss_val.item())

		# print('Epoch: {} | Train Loss: {:0.4f} | Wsupervised Train Loss: {:0.4f} | Val Loss: {:0.4f} \
		# 	'.format(epoch, loss_train.item(), Wsupervised_loss.item(), loss_val.item()))
		print('Epoch: {} | Train Loss: {:0.4f} | Val Loss: {:0.4f} \
			'.format(epoch, loss_train.item(), loss_val.item()))

	# save trained models
	if not os.path.isdir('save_train_model'):  # check if directory for saved models exists
		os.mkdir('save_train_model')
	torch.save(inverse_model, 'save_train_model/%s_%s_%s_s_uns_%s.pth'%(model_name, Forward_model, Facies_model_C, data_flag))

	plt.plot(train_loss,'r')
	plt.plot(val_loss,'k')
	plt.savefig('results/%s_%s_%s_s_uns_%s.png'%(model_name, Forward_model, Facies_model_C, data_flag))



if __name__ == '__main__':

	train(train_p=TCN1D_train_p)
