import zipfile
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.fftpack import dct
from sklearn.decomposition import PCA

import torch
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, MaxNLocator

def extract(source_path, destination_path):
	"""从source_path 给定的zip文件，提取所有文件到destination_path"""
	
	with zipfile.ZipFile(source_path, 'r') as zip_ref:
		zip_ref.extractall(destination_path)
	
	
def standardize(seismic, model, no_wells):
	"""对数据进行标准化
	
	Parameters
	----------
	seismic : array_like, shape(num_traces, depth samples)
		2-D array containing seismic section
		
	model : array_like, shape(num_wells, depth samples)
		2-D array containing model section
		
	no_wells : int,
		no of wells (and corresponding seismic traces) to extract.
	"""

	seismic_normalized = (seismic - seismic.mean())/ seismic.std()
	train_indices = (np.linspace(0, len(model)-1, no_wells, dtype=np.int))  # 均匀采样的道索引
	# 标准化通过索引获取的训练数据，因为没有获取的数据默认是不知道的，所以不能引入均值
	model_normalized = (model - model[train_indices].mean()) / model[train_indices].std() 
	
	return seismic_normalized, model_normalized


def nor_1d(data):
	'''
		1D array （n, ）
		1、是否需要先归一化后标准化
		2、归一化是为了：去掉纲量
	'''
	dmax = data.max()
	dmin = data.min()
	nor_data = (data-dmin)/(dmax-dmin)
	return nor_data

def std_2d(data):
	'''
		2D array (n, features)

		1、标准化(每个样本一个均值，还是数组使用一个均值),====每个样本都要独立标准化=======
		2、标准化：是每个特征作为一个整体进行标准化。
	'''
	# 每个features一个均值和方差
	mu = np.mean(data, axis=0)[np.newaxis,:]
	sigma = np.std(data, axis=0)[np.newaxis,:]  # (n, feature) --> (feature,) --> (1, feature)
	std_data = (data - mu) / sigma    # (n, feature) 在第0维广播
	return std_data

def nor_2d(data):
	'''
		2D array (n, features)

		1、标准化(每个样本一个均值，还是数组使用一个均值),====每个样本都要独立标准化=======
		2、标准化：是每个特征作为一个整体进行标准化。
	'''
	# 每个features一个均值和方差
	dmax = np.max(data, axis=0)[np.newaxis,:]
	dmin = np.min(data, axis=0)[np.newaxis,:]  # (n, feature) --> (feature,) --> (1, feature)
	nor_data = (data-dmin)/(dmax-dmin)    # (n, feature) 在第0维广播
	return nor_data

def trace_F_P_D(s, n_components):
	'''
		完成一个地震道的特征提取,
		输入数据的维度：s (n,)
		输出数据的维度：s_features(n, 5)
	'''

	'''
	1、预加重，将语音信号通过一个高通滤波器。用于提升高频部分的能量。使信号的频谱变得平坦。
		y(t) = x(t) - a*x(t-1)
		y(t) = x(t) - x(t-1)  是一阶差分
		经测试，加重后，高频部分能量真的有提高
	'''
	pre_pha = 0.97
	pha_s = np.append(s[0], s[1:] - pre_pha * s[:-1])  # 如果pre_pha=1, 则结果就是一届差分

	'''
	2、小波变换: scipy.signal.cwt(data, wavelet, widths) ; data:(N,); return (len(widths), len(data)) = (channel, n)
		需要注意的是： 小波变换后的数组是（N/2, N）
		imshow, y轴是子波的扩大比例，和频率成反比
	'''
	M = len(pha_s)
	widths = np.arange(1, int(M//2)+1)  # 伸缩变换
	w = 5
	cwtm = signal.cwt(pha_s, signal.morlet2, widths, w=w)  # signal.ricker,signal.morlet 小波函数 ==> (n/2, n)
	cwtm = np.abs(cwtm)         # 0-0.7

	# plt.imshow(cwtm, extent=[1, 701, 1, 350], cmap='PRGn', aspect='auto')  # extent=(左,右,下，上)
	# plt.show()

	'''
	3、PCA
		1、标准化(每个样本一个均值，还是数组使用一个均值),====每个样本都要独立标准化=======
			标准化：是每个特征作为一个整体进行标准化。
			是否需要先归一化后标准化
			归一化是为了：去掉纲量
			标准化是为了：
		2、PCA 输入数据的样式：(n_samples, n_features)，返回(n_samples, n_components)
	'''
	cwtm = cwtm.T  # PCA的输入要求是（n, channel）
	# mu = np.mean(cwtm, axis=0)[np.newaxis,:]
	# sigma = np.std(cwtm, axis=0)[np.newaxis,:]  # (n, n/2) --> (n/2,) --> (1, n/2)
	# cwtm = (cwtm - mu) / sigma    # (n, n/2) 在第0维广播

	pca = PCA(n_components=n_components)
	pca.fit(cwtm)
	prime_f = pca.transform(cwtm) # 获取降维后的结果:去相关，压缩信息，获取主要信息。(n,2)
	# print(pca.explained_variance_ratio_)  # 主成分信息占比
	# print(prime_f.shape)  # （时间序列数，主成分数）

	'''
	4、一阶差分和二阶差分，获取动态信息
		数据的维度是（n, 主成分数）
	'''
	prime_f_1 = np.diff(prime_f, axis=0, prepend=0)
	prime_f_2 = np.diff(prime_f_1, axis=0, prepend=0)  # (n,2)

	'''
		组合三种模态的信息：时域地震数据s, 频域信息prime_s, 动态信息prime_f_2(一阶差分和二阶差分一致)
	'''
	s_ = s[:,np.newaxis]
	s_features = np.concatenate((nor_2d(s_), nor_2d(prime_f), nor_2d(prime_f_2)), axis=1)
	# print('seismic features shape (num_line, time, features): ', s_features.shape)

	return s_features


def n_s_features(seismic, n_components):
	'''
		输入地震数据的格式是(2719,701), 701是时间轴, 
		调用方式：
			seismic = seismic[1:2720, :]  # (2719,701), 原始数据第一行和最后一行都是0
			seismic_feature = n_s_features(seismic)
	'''
	s_features_all = []
	for i in range(seismic.shape[0]):
		s = seismic[i,:]
		s_features = trace_F_P_D(s, n_components)  # (701, 5)
		s_features_all.append(s_features)  # 2719个
	s_features_all_ndarray = np.stack(s_features_all, axis=0)  # (2719, 701, 5)
	return s_features_all_ndarray


