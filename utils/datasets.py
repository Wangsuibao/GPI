'''
	数据获取; seismic, model(地球物理参数)的数据格式是ndarray： (num_traces, channel, depth_samples) 等长
	数据样式是矩形。

	两种类型的数据，其一是浮点型seismic和Impedance，其二是整型facies类别。
	数据的处理过程：
		1、读入数据时，对浮点型数据seismic和Impedance数据的标准化。facies数据不变。
		2、浮点型seismic和Impedance数据是否需要归一化？
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

class SeismicDataset1D(Dataset):
	"""
		训练1D model 时的数据组织方式
		seismic, model的数据格式是： (num_traces, channel, depth_samples) 等长，矩形数据。
		返回：（channel, depth_samples）
	"""
	def __init__(self, seismic, model, trace_indices):
		# 直接把所有的数据读入
		self.seismic = seismic
		self.model = model
		self.trace_indices = trace_indices
	
		assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
	
	def __getitem__(self, index):
		# 随机获取index,提取数据，返回格式tensor, 这里返回只考虑（channel，depth_samples）,batch会自动添加
		# DataLoader自动循环，每次随机选取一个样本，最后形成一个batch
		trace_index = self.trace_indices[index]
		# 判断是否进行地震数据的类MFCC增强
		
		x = torch.tensor(self.seismic[trace_index, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		y = torch.tensor(self.model[trace_index, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		return x, y

	def __len__(self):
		return len(self.trace_indices)

class UnsupervisedSeismicDataset(Dataset):
	"""Dataset class for unsupervised loss in Motaz 
		监督模型中，输入是地震数据，输出是波阻抗。
		非监督训练的方式，输入是地震数据，经过反演模型-正演模型，输出的还是地震数据，即重构损失。
		非监督训练数据，只需获取地震数据即可
		地震数据的维度是：（B，1，T）。  M2：(B, 1, 696)
	"""
	def __init__(self, seismic, trace_indices):
		self.seismic = seismic
		self.trace_indices = trace_indices
	
		assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
	
	def __getitem__(self, index):
		trace_index = self.trace_indices[index]
		# x = torch.tensor(self.seismic[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		x = torch.tensor(self.seismic[trace_index], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

		return x

	def __len__(self):
		return len(self.trace_indices) 

class UnsupervisedSeismicDataset_unpaire(Dataset):
	"""Dataset class for unsupervised loss in Motaz 
		监督模型中，输入是地震数据，输出是波阻抗。
		非监督训练的方式，输入是地震数据，经过反演模型-正演模型，输出的还是地震数据，即重构损失。
		非监督训练数据，只需获取地震数据即可
		地震数据的维度是：（B，1，T）。  M2：(B, 1, 696)
	"""
	def __init__(self, seismic, model, trace_indices):
		self.seismic = seismic
		self.model = model
		self.trace_indices = trace_indices
	
	def __getitem__(self, index):
		'''
			seismic数据量大，model数据量小，cycleGAN训练使用的是非成对数据，
			**** 使用大数据集的索引，同时使用小数据集的索引循环。
		'''
		tindex = self.trace_indices[index]
		y = torch.tensor(self.model[tindex%len(self.model)], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		x = torch.tensor(self.seismic[tindex], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

		return x,y

	def __len__(self):
		return len(self.trace_indices) 

class SeismicDataset1D_SPF(Dataset):
	"""
		训练1D model 时的数据组织方式
		1、seismic, 2、model： (num_traces, channel, depth_samples) 等长，矩形数据。
		3、Face 的数据格式是： (num_traces, depth_samples)
		返回：（channel, depth_samples）
	"""
	def __init__(self, seismic, model, facie, trace_indices):
		# 直接把所有的数据读入
		self.seismic = seismic
		self.model = model
		self.facie = facie
		self.trace_indices = trace_indices
	
		assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
	
	def __getitem__(self, index):
		# 随机获取index,提取数据，返回格式tensor, 这里返回只考虑（channel，depth_samples）,batch会自动添加
		# DataLoader自动循环，每次随机选取一个样本，最后形成一个batch
		trace_index = self.trace_indices[index]
		# 判断是否进行地震数据的类MFCC增强
		
		x = torch.tensor(self.seismic[trace_index, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		y = torch.tensor(self.model[trace_index, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		z = torch.tensor(self.facie[trace_index, 0, :], dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		return x, y, z

	def __len__(self):
		return len(self.trace_indices)


class SeismicDataset1D_SPF_WS(Dataset):
	"""
		弱监督训练1D model 时的数据组织方式
		1、seismic 的数据格式是：  (num_traces, channel, depth_samples) 等长，矩形数据。
		2、Face 的数据格式是： (num_traces, depth_samples)
		返回：（channel, depth_samples）
	"""
	def __init__(self, seismic, facie, trace_indices):
		# 直接把所有的数据读入
		self.seismic = seismic
		self.facie = facie
		self.trace_indices = trace_indices
	
		assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
	
	def __getitem__(self, index):
		# 随机获取index,提取数据，返回格式tensor, 这里返回只考虑（channel，depth_samples）,batch会自动添加
		# DataLoader自动循环，每次随机选取一个样本，最后形成一个batch
		trace_index = self.trace_indices[index]
		# 判断是否进行地震数据的类MFCC增强
		
		x = torch.tensor(self.seismic[trace_index, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		z = torch.tensor(self.facie[trace_index, 0, :], dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		return x, z

	def __len__(self):
		return len(self.trace_indices)

class SeismicDataset1D_SGS(Dataset):
	"""
		训练1D model 时的数据组织方式
		seismic, model的数据格式是： (num_traces, channel, depth_samples) 等长，矩形数据。
		返回：（channel, depth_samples）
	"""
	def __init__(self, seismic, model):
		# 直接把所有的数据读入
		self.seismic = seismic
		self.model = model
		self.mean_seismic = np.mean(seismic)
		self.std_seismic = np.std(seismic)
		self.mean_model = np.mean(model)
		self.std_model = np.std(model)

	def __getitem__(self, index):
		# 随机获取index,提取数据，返回格式tensor, 这里返回只考虑（channel，depth_samples）,batch会自动添加
		# DataLoader自动循环，每次随机选取一个样本，最后形成一个batch
		x = self.seismic[index, :, :]
		y = self.model[index, :, :]
		x = (x - self.mean_seismic) / self.std_seismic
		y = (y - self.mean_model) / self.std_model

		x = torch.tensor(x, dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		y = torch.tensor(y, dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		return x, y

	def __len__(self):
		return len(self.model)

class SeismicDataset2D(Dataset):
	"""Dataset class for loading 2D seismic images for 2-D TCN"""
	def __init__(self, seismic, model, trace_indices, width):
		self.seismic = seismic
		self.model = model
		self.trace_indices = trace_indices
		self.width = width

		assert min(trace_indices) - int(width/2) >= 0 and max(trace_indices) + int(width/2) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"

	def __getitem__(self, index):
		offset = int(self.width/2)
		trace_index = self.trace_indices[index]
		x = torch.tensor(self.seismic[trace_index-offset:trace_index+offset+1].T[np.newaxis, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		y = torch.tensor(self.model[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		return x, y

	def __len__(self):
		return len(self.trace_indices)

