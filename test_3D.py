'''
	data:
		用于可视化3D数据体，stanford VI， Fanny
		stanford VI： H=200, 分为三个层段，分别为80-40-80
		Fanny： H=64, 目标砂岩层位于中部。

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import os
import torch
from os.path import join
from scipy import ndimage

from model.tcn import TCN_IV_1D_C
from model.CNN2Layer import *
from model.M2M_LSTM import LSTM_MM
from model.Transformer import TransformerModel

from setting import *
from utils.utils import *
from utils.datasets import SeismicDataset1D
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from cv2 import PSNR

import errno
import argparse

##------1维模型的选择-----------

model_name = TCN1D_test_p['model_name']
data_flag = TCN1D_test_p['data_flag']
get_F = TCN1D_train_p['get_F']
F = TCN1D_train_p['F']  # 地震数据的处理方式

def get_data(no_wells=10, data_flag='Stanford_VI', get_F=get_F):
	'''
		no_well: 井数，
		data_flag: 数据集在深度域-时间域是完全对齐的。 stanford_VI
		读取的数据维度是：（深度，inline, xline）--> (N, H)  --> (N, 1, H)
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
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 8800.779; -0.1263349

	if data_flag == 'Fanny':
		# Fanny: 地震波形(360*360, 64)
		seismic = np.load(join('data',data_flag,'seismic.npy'))

		# Fanny: 波阻抗(360*360, 64) 
		GR  = np.load(join('data',data_flag,'GR.npy'))
		model= np.load(join('data',data_flag, 'Impedance.npy'))
		facies = np.load(join('data',data_flag,'facies.npy')) 

		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 8800.779; -0.1263349

	model = GR  # 可以是GR
	seismic, model = standardize(seismic, model, no_wells)  # 所有的数据都是0-1，经过标准化后，数据范围变成[-3,2]
	# 1、为了满足unet下采样的长度要求，需要截取。
	s_L = seismic.shape[-1]
	n = int((s_L//8)*8)
	seismic = seismic[:,:n]
	model = model[:, :n]

	# 输出维度是（N, 1, H）
	return seismic[:, np.newaxis, :], model[:, np.newaxis, :]

def show_Stanford_VI(AI_act, AI_pred, seismic, vmin, vmax):
	'''
		绘图，1、高分辨率图像，成图时设置dpi=400,默认为100，表是每英寸400个点
		set_aspect(60/30)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比

		3D 数据，（30000，1，200） --> (200,150,200) --> (inline, xline, depth)
		真实数据距离(5000m, 3750m, 200m)
	'''
	AI_act = AI_act.squeeze().reshape(200,150,200)
	AI_pred = AI_pred.squeeze().reshape(200,150,200)
	seismic = seismic.squeeze().reshape(200,150,200)


	# 地震中加噪音
	blurred = ndimage.gaussian_filter(seismic, sigma=1.1)
	seismic = blurred + 0.5 * blurred.std() * np.random.random(blurred.shape)

	depth_index = np.array(range(0, AI_act.shape[2], 1))*1
	inline_index = np.array(range(0,AI_act.shape[0]))*25
	xline_index = np.array(range(0,AI_act.shape[1]))*25

	# plt.imshow(AI_pred.T)
	# plt.show()

	# 1、预测xline剖面
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	img = ax1.imshow(AI_pred[:,50,:].T, vmin=vmin, vmax=vmax, extent=(0,5000, 200,0))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	ax1.set_aspect(80/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('Distance (m)', fontsize=20)
	ax1.set_ylabel('Depth (m)', fontsize=20)
	ax1.set_title('Inversion Profile', fontsize=20)
	# fig.colorbar(img, ax=ax1)  # 加色标的方法*************
	plt.savefig('results/%s_%s_test_Pred_xline_50.png'%(model_name, data_flag))
	plt.close()

	# 2、预测inline剖面
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_pred[100,:,:].T, vmin=vmin, vmax=vmax, extent=(0,3750, 200,0))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	ax1.set_aspect(45/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('Distance (m)', fontsize=20)
	ax1.set_ylabel('Depth (m)', fontsize=20)
	ax1.set_title('Inversion Profile', fontsize=20)
	plt.savefig('results/%s_%s_test_Pred_inline_100.png'%(model_name, data_flag))
	plt.close()

	# 3、预测切片（顺直河，曲流河，三角洲）
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_pred[:,:,40], vmin=vmin, vmax=vmax, extent=(0,3750, 0,5000))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	# ax1.set_aspect(10/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('x(m)', fontsize=20)
	ax1.set_ylabel('y(m)', fontsize=20)
	ax1.set_title('Inversion Slice', fontsize=20)
	plt.savefig('results/%s_%s_test_Pred_depth_40.png'%(model_name, data_flag))
	plt.close()

	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_pred[:,:,100], vmin=vmin, vmax=vmax, extent=(0,3750, 0,5000))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	# ax1.set_aspect(10/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('x(m)', fontsize=20)
	ax1.set_ylabel('y(m)', fontsize=20)
	ax1.set_title('Inversion Slice', fontsize=20)
	plt.savefig('results/%s_%s_test_Pred_depth_100.png'%(model_name, data_flag))
	plt.close()

	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_pred[:,:,160], vmin=vmin, vmax=vmax, extent=(0,3750, 0,5000))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	# ax1.set_aspect(10/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('x(m)', fontsize=20)
	ax1.set_ylabel('y(m)', fontsize=20)
	ax1.set_title('Inversion Slice', fontsize=20)
	plt.savefig('results/%s_%s_test_Pred_depth_160.png'%(model_name, data_flag))
	plt.close()

	# 4、真实xline剖面------------------------------------------------------------------------------
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_act[:,50,:].T, vmin=vmin, vmax=vmax, extent=(0,5000, 200,0))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	ax1.set_aspect(80/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('Distance (m)', fontsize=20)
	ax1.set_ylabel('Depth (m)', fontsize=20)
	ax1.set_title('Ground Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True_xline_50.png'%(model_name, data_flag))
	plt.close()

	# 5、真实inline剖面
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_act[100,:,:].T, vmin=vmin, vmax=vmax, extent=(0,3750, 200,0))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	ax1.set_aspect(45/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('Distance (m)', fontsize=20)
	ax1.set_ylabel('Depth (m)', fontsize=20)
	ax1.set_title('Ground Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True_inline_100.png'%(model_name, data_flag))
	plt.close()

	# 6、真实切片
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_act[:,:,40], vmin=vmin, vmax=vmax, extent=(0,3750, 0,5000))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	# ax1.set_aspect(10/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('x(m)', fontsize=20)
	ax1.set_ylabel('y(m)', fontsize=20)
	ax1.set_title('Ground Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True_depth_40.png'%(model_name, data_flag))
	plt.close()

	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_act[:,:,100], vmin=vmin, vmax=vmax, extent=(0,3750, 0,5000))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	# ax1.set_aspect(10/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('x(m)', fontsize=20)
	ax1.set_ylabel('y(m)', fontsize=20)
	ax1.set_title('Ground Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True_depth_100.png'%(model_name, data_flag))
	plt.close()

	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_act[:,:,160], vmin=vmin, vmax=vmax, extent=(0,3750, 0,5000))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	# ax1.set_aspect(10/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('x(m)', fontsize=20)
	ax1.set_ylabel('y(m)', fontsize=20)
	ax1.set_title('Ground Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True_depth_160.png'%(model_name, data_flag))
	plt.close()

	# 7、输入地震剖面（50）
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(seismic[:,50,:].T, cmap='seismic',extent=(0,5000, 200,0))
	ax1.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)
	ax1.set_aspect(80/10)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('Distance (m)', fontsize=20)
	ax1.set_ylabel('Depth (m)', fontsize=20)
	ax1.set_title('Seismic Profile', fontsize=20)
	plt.savefig('results/%s_%s_test_seismic_xline_50.png'%(model_name, data_flag))
	plt.close()

	# 三、具体道可视化
	'''
		可视乎道1660和480的原因是：
			1、0-17000米的地震道数据，480道对应透镜状河道，1660对应断层位置背斜
			2、2500-14500米有效地震范围，80和1260
	'''

	B_x, B_y = 100,100   # 有河道的位置
	H_x, H_y = 100,100    # 透镜状河道

	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, AI_pred[B_x, B_y, :], linestyle='--', label='Pred', color='blue', linewidth=3.0)  # 1660
	ax.plot(depth_index, AI_act[B_x, B_y, :], linestyle='-', label='True', color='red', linewidth=3.0)

	ax.xaxis.set_major_locator(MultipleLocator(20))  # 设置x轴（深度方向）刻度间隔为20
	ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度（阻抗值域）间隔为1000
	ax.tick_params(axis='both', labelsize=18)

	ax.set_xlabel("Depth(m)", fontsize=20)
	ax.set_ylabel('Impedance', fontsize=20)
	ax.set_title('Trace inline_%sxline_%s'%(B_x, B_y), fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_%s_test.png'%(model_name, data_flag, B_x, B_y))
	plt.close()

	# fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	# ax.plot(depth_index, AI_pred.T[:, H_T], linestyle='--', label='Pred', color='blue', linewidth=3.0)
	# ax.plot(depth_index, AI_act.T[:, H_T],  linestyle='-', label='True', color='red', linewidth=3.0)

	# ax.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	# ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为1000
	# ax.tick_params(axis='both', labelsize=18)

	# ax.set_xlabel("Depth(m)", fontsize=20)
	# ax.set_ylabel('Impedance', fontsize=20)
	# ax.set_title('Trace %s'%H_T, fontsize=20)
	# plt.legend(loc='upper left')
	# plt.savefig('results/%s_%s_T%s_test.png'%(model_name, data_flag, H_T))
	# plt.close()

def show_Fanny(AI_act, AI_pred, seismic, vmin, vmax):
	'''
		绘图，1、高分辨率图像，成图时设置dpi=400,默认为100，表是每英寸400个点
		set_aspect(3)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比

		3D 数据，（360*360，1，64） --> (360,360,64) --> (inline, xline, depth)

	'''
	model = AI_act.squeeze().reshape(360,360,-1)
	AI_pred = AI_pred.squeeze().reshape(360,360,-1)
	seismic = seismic.squeeze().reshape(360,360,-1)

	vm = np.percentile(model, 99)
	vs = np.percentile(seismic, 99)
	vmp =  np.percentile(AI_pred, 99)

	model_3D = model
	seismic_3D = seismic
	AI_pred_3D = AI_pred

	# 1、地震剖面   # T=35ms
	fig3, ax3 = plt.subplots(figsize=(16,8), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(seismic_3D[200,:,:].T, vmin=-vs, vmax=vs, extent=(0,360,35,0), aspect=3, interpolation='bilinear', cmap='seismic')
	ax3.xaxis.set_major_locator(MultipleLocator(50))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(5))   # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)

	ax3.set_xlabel('Seismic Trace', fontsize=20)
	ax3.set_ylabel('Time (ms)', fontsize=20)
	ax3.set_title('Seismic', fontsize=20)
	plt.savefig('results/%s_%s_show_seismic.png'%(model_name, data_flag))
	plt.close()

	# 1、model 模型剖面  # T=35ms
	colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]   # 红色到蓝色的渐变
	n_bins = 100  # 设定渐变的分段数
	cmap_name = 'my_custom_cmap'
	cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
	fig3, ax3 = plt.subplots(figsize=(16,8), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(model_3D[200,:,:].T, vmin=-vm, vmax=vm, extent=(0,360,35,0), aspect=3, interpolation='bilinear', cmap=cm)
	ax3.xaxis.set_major_locator(MultipleLocator(50))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(5))  # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)
	ax3.set_xlabel('Seismic Trace', fontsize=20)
	ax3.set_ylabel('Time (ms)', fontsize=20)
	ax3.set_title('Impedance', fontsize=20)
	plt.savefig('results/%s_%s_show_model.png'%(model_name, data_flag))
	plt.close()

	# 1、预测model 模型剖面  # T=35ms
	colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]   # 红色到蓝色的渐变
	n_bins = 100  # 设定渐变的分段数
	cmap_name = 'my_custom_cmap'
	cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
	fig3, ax3 = plt.subplots(figsize=(16,8), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(AI_pred_3D[200,:,:].T, vmin=-vmp, vmax=vmp, extent=(0,360,35,0), aspect=3, interpolation='bilinear', cmap=cm)
	ax3.xaxis.set_major_locator(MultipleLocator(50))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(5))  # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)
	ax3.set_xlabel('Seismic Trace', fontsize=20)
	ax3.set_ylabel('Time (ms)', fontsize=20)
	ax3.set_title('Impedance', fontsize=20)
	plt.savefig('results/%s_%s_show_model_pred.png'%(model_name, data_flag))
	plt.close()

	# 2、具体地震道-model道, 1D可视化
	B_T = 100
	SL = seismic_3D.shape[-1]
	depth_index = np.linspace(0, 35, SL)
	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, seismic_3D[B_T, B_T, :].T, linestyle='--', label='Seismic', color='blue', linewidth=3.0)
	ax.plot(depth_index, model_3D[B_T, B_T, :].T, linestyle='-', label='Impedance', color='red', linewidth=3.0)
	ax.xaxis.set_major_locator(MultipleLocator(5))  
	ax.yaxis.set_major_locator(MultipleLocator(0.3))
	ax.tick_params(axis='both', labelsize=18)
	ax.set_xlabel("Time(ms)", fontsize=20)
	ax.set_title('Trace %s'%B_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test.png'%(model_name, data_flag, B_T))
	plt.close()

	depth_index = np.linspace(0, 35, SL)
	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, AI_pred_3D[B_T, B_T, :].T, linestyle='--', label='Impedace_pred', color='blue', linewidth=3.0)
	ax.plot(depth_index, model_3D[B_T, B_T, :].T, linestyle='-', label='Impedance_True', color='red', linewidth=3.0)
	ax.xaxis.set_major_locator(MultipleLocator(5))  
	ax.yaxis.set_major_locator(MultipleLocator(0.3))
	ax.tick_params(axis='both', labelsize=18)
	ax.set_xlabel("Time(ms)", fontsize=20)
	ax.set_title('Trace %s'%B_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test_model.png'%(model_name, data_flag, B_T))
	plt.close()

	# 3、地震切片   # T=25ms
	fig3, ax3 = plt.subplots(figsize=(10,10), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(seismic_3D[:,:,42], vmin=-vs, vmax=vs, aspect=1, interpolation='bilinear', cmap='seismic')
	ax3.xaxis.set_major_locator(MultipleLocator(100))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)
	ax3.set_xlabel('line', fontsize=20)
	ax3.set_ylabel('trace', fontsize=20)
	ax3.set_title('Seismic', fontsize=20)
	plt.savefig('results/%s_%s_show_seismic_slice_%s.png'%(model_name, data_flag, '42'))
	plt.close()

	# 3、GR切片  # T=25ms
	colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]   # 红色到蓝色的渐变
	n_bins = 100  # 设定渐变的分段数
	cmap_name = 'my_custom_cmap'
	cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
	fig3, ax3 = plt.subplots(figsize=(10,10), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(model_3D[:,:,42], vmin=-vm, vmax=vm, aspect=1, interpolation='bilinear', cmap=cm)
	ax3.xaxis.set_major_locator(MultipleLocator(100))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)
	ax3.set_xlabel('line', fontsize=20)
	ax3.set_ylabel('trace', fontsize=20)
	ax3.set_title('Impedance', fontsize=20)
	plt.savefig('results/%s_%s_show_Impedance_slice_%s.png'%(model_name, data_flag, '42'))
	plt.close()

	# 3、预测GR切片  # T=25ms
	colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]   # 红色到蓝色的渐变
	n_bins = 100  # 设定渐变的分段数
	cmap_name = 'my_custom_cmap'
	cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
	fig3, ax3 = plt.subplots(figsize=(10,10), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(AI_pred_3D[:,:,42], vmin=-vmp, vmax=vmp, aspect=1, interpolation='bilinear', cmap=cm)
	ax3.xaxis.set_major_locator(MultipleLocator(100))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)
	ax3.set_xlabel('line', fontsize=20)
	ax3.set_ylabel('trace', fontsize=20)
	ax3.set_title('model', fontsize=20)
	plt.savefig('results/%s_%s_show_Impedace_pred_slice_%s.png'%(model_name, data_flag,'42'))
	plt.close()

def test(TCN1D_test_p):
	"""Function tests the trained network on SEAM and Marmousi sections and 
	prints out the results"""
	
	# 获取全部数据
	seismic, model = get_data(TCN1D_test_p['no_wells'], TCN1D_test_p['data_flag'])

	# define device for training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# 全部的数据用于最后测试，可视化 
	traces_test = np.arange(len(model), dtype=int)
	
	test_dataset = SeismicDataset1D(seismic, model, traces_test)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 按顺序逐个读取
	
	# 获取模型
	if not os.path.isdir('save_train_model/'):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'save_train_model/')
	print('test_model_name: %s, test_data_name: %s'%(model_name, data_flag))
	inver_model = torch.load('save_train_model/%s_%s.pth'%(model_name, data_flag)).to(device)
	# inver_model = torch.load('save_train_model/%s_%s_sigle.pth'%(model_name, data_flag)).to(device)
		
	# infer on 
	print("\nInferring ...")
	x, y = test_dataset[0]  # get a sample
	AI_pred = torch.zeros((len(test_dataset), y.shape[-1])).float().to(device)
	AI_act = torch.zeros((len(test_dataset), y.shape[-1])).float().to(device)
	
	mem = 0
	with torch.no_grad():
		inver_model.eval()
		for i, (x,y) in enumerate(test_loader):
			
			# y_pred  = inver_model(x, y)
			y_pred  = inver_model(x)

			AI_pred[mem:mem+len(x)] = y_pred.squeeze().data
			AI_act[mem:mem+len(x)] = y.squeeze().data
			mem += len(x)
			# del x, y, y_pred
	
	vmin, vmax = AI_act.min(), AI_act.max()  # 这个vmin和vmax，后续成图都使用。

	# 真实数据和预测数据的不同类型的统计误差，包含：
	AI_pred = AI_pred.detach().cpu().numpy()
	AI_act = AI_act.detach().cpu().numpy()

	np.save('result/pred_AI.npy', AI_pred)

	# R2_score(判别系数), PCC(皮尔逊相关系数), SSIM(结构相似性指数)， PSNR(峰值信噪比)， MSE
	print('r^2 score: {:0.4f}'.format(r2_score(AI_act.T, AI_pred.T)))  # 相似度
	pcc, _ = pearsonr(AI_act.T.ravel(), AI_pred.T.ravel())             # 相似性，相关性[-1,1]
	print('PCC: {:0.4f}'.format(pcc))

	print('ssim: {:0.4f}'.format(ssim(AI_act.T, AI_pred.T)))           # 结构相似度[-1,1]
	print('PSNR: {:0.4f}'.format(PSNR(AI_act.T, AI_pred.T)))           # 分贝（dB）单位，值越高图像质量越好（最大值361）

	print('MSE: {:0.4f}'.format(np.sum((AI_pred-AI_act).ravel()**2)/AI_pred.size))
	print('MAE: {:0.4f}'.format(np.sum(np.abs(AI_pred - AI_act)/AI_pred.size)))
	print('MedAE: {:0.4f}'.format(np.median(np.abs(AI_pred - AI_act))))

	# 保存可是化结果

	if data_flag == 'Stanford_VI': 
		show_Stanford_VI(AI_act, AI_pred, seismic, vmin, vmax)
		# show_Stanford_VI(AI_act, AI_pred, seismic, facies)
	if data_flag == 'Fanny': 
		show_Fanny(AI_act, AI_pred, seismic, vmin, vmax)	




if __name__ == '__main__':
	# get_data(no_wells=10, data_flag='M2')
	test(TCN1D_test_p)