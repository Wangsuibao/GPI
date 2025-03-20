'''
序列到序列的预测
'''

import torch
import torch.nn as nn
import random
import numpy as np 


class LSTM_MM(nn.Module):
	def __init__(self, input_dim=1, hid_dim=256, output_dim=1, n_layers=2, dropout=0.75):
		super().__init__()
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		self.feature_dim = 32

		'''
			1、input_size: 输入x的特征，2、hidden_size: 隐层的特征，3、num_layers:LSTM的堆叠数量， 4、bias: 是否使用b
			5、batch_fist: 输入输出的数据格式，True:（batch,seq,H_in）, 默认False(seq, batch, H_in)
			6、dropout：除最后一个层的每个LSTM层后增加一个dropout层，默认为0
			7、bidirectional: 是否为双向LSTM.默认False
		'''

		# 在LSTM之前可以增加一个卷积层作为特征获取
		# self.cnn_feature = nn.Conv1d(input_dim, self.feature_dim, kernel_size=3, padding=1)
		# self.rnn = nn.LSTM(self.feature_dim, hid_dim, n_layers, batch_first=False, dropout=dropout, bidirectional=False)

		self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, batch_first=False, dropout=dropout, bidirectional=False)
		self.fc_out = nn.Linear(hid_dim, output_dim)  # 最后的拟合层：压缩feature部分

	def forward(self, src):
		'''
			所有的src:格式都是（Batch, channel, seq）

			seq:表示序列长度，H_in:输入特征数，H_cell:隐层特征数，H_out:输入特征数=隐层特征数, bi：双向为2，num_layer:层数
			输入：input:（seq, batch, H_in）; 
				  h_0:  (bi*num_layers, batch, H_out)； 
			      c_0:  (bi*num_layers, batch, H_cell);
			输出：output: (seq, batch, bi*H_out);
				  h_n:  (bi*num_layers, batch, H_out)； 输出信息
			      c_n:  (bi*num_layers, batch, H_cell);	记忆信息+更新信息 
		'''
		# src = self.cnn_feature(src)
		src = src.permute(2,0,1).contiguous()  # （Batch, channel, seq）=>(seq, batch,channel)

		outputs, (hidden, cell) = self.rnn(src)
		outputs = self.fc_out(outputs)    # 拟合,输出(seq, batch, channel)
		outputs = outputs.permute(1,2,0).contiguous()  # 处理成标准输出（Batch, channel, seq）
		
		# outputs:是叠置LSTM的最后一层输出
		return outputs

class GRU_MM(nn.Module):
	'''
		来自论文25
		2层GRU(hid=32)+gression
	'''
	def __init__(self, input_dim=1, hid_dim=256, output_dim=1, n_layers=2, dropout=0.75):
		super().__init__()
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		self.feature_dim = 32

		'''
			1、input_size: 输入x的特征，2、hidden_size: 隐层的特征，3、num_layers:LSTM的堆叠数量， 4、bias: 是否使用b
			5、batch_fist: 输入输出的数据格式，True:（batch,seq,H_in）, 默认False(seq, batch, H_in)
			6、dropout：除最后一个层的每个LSTM层后增加一个dropout层，默认为0
			7、bidirectional: 是否为双向LSTM.默认False
		'''

		# 在LSTM之前可以增加一个卷积层作为特征获取
		# self.cnn_feature = nn.Conv1d(input_dim, self.feature_dim, kernel_size=3, padding=1)
		# self.rnn = nn.GRU(self.feature_dim, hid_dim, n_layers, batch_first=False, dropout=dropout, bidirectional=False)

		self.rnn = nn.GRU(input_dim, hid_dim, n_layers, batch_first=False, dropout=dropout, bidirectional=False)
		self.fc_out = nn.Linear(hid_dim, output_dim)  # 最后的拟合层：压缩feature部分

	def forward(self, src):
		'''
			所有的src:格式都是（Batch, channel, seq）

			seq:表示序列长度，H_in:输入特征数，H_cell:隐层特征数，H_out:输入特征数=隐层特征数, bi：双向为2，num_layer:层数
			输入：input:（seq, batch, H_in）; 
				  h_0:  (bi*num_layers, batch, H_out)； 
			输出：output: (seq, batch, bi*H_out);
				  h_n:  (bi*num_layers, batch, H_out)； 输出信息

		'''
		# src = self.cnn_feature(src)
		src = src.permute(2,0,1).contiguous()  # （Batch, channel, seq）=>(seq, batch,channel)

		outputs, hidden = self.rnn(src)
		outputs = self.fc_out(outputs)    # 拟合,输出(seq, batch, channel)
		outputs = outputs.permute(1,2,0).contiguous()  # 处理成标准输出（Batch, channel, seq）
		
		# outputs:是叠置LSTM的最后一层输出
		return outputs

class CNN_GRU_MM(nn.Module):
	'''
		CNN + RNN + Linear
	'''
	def __init__(self, input_dim=1, hid_dim=256, output_dim=1, n_layers=2, dropout=0.75):
		super().__init__()
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		self.feature_dim = 32

		'''
			1、input_size: 输入x的特征，2、hidden_size: 隐层的特征，3、num_layers:LSTM的堆叠数量， 4、bias: 是否使用b
			5、batch_fist: 输入输出的数据格式，True:（batch,seq,H_in）, 默认False(seq, batch, H_in)
			6、dropout：除最后一个层的每个LSTM层后增加一个dropout层，默认为0
			7、bidirectional: 是否为双向LSTM.默认False
		'''

		# 在LSTM之前可以增加一个卷积层作为特征获取
		self.cnn_feature = nn.Conv1d(input_dim, self.feature_dim, kernel_size=3, padding=1)
		self.rnn = nn.GRU(self.feature_dim, hid_dim, n_layers, batch_first=False, dropout=dropout, bidirectional=False)
		self.fc_out = nn.Linear(hid_dim, output_dim)  # 最后的拟合层：压缩feature部分

	def forward(self, src):
		'''
			所有的src:格式都是（Batch, channel, seq）

			seq:表示序列长度，H_in:输入特征数，H_cell:隐层特征数，H_out:输入特征数=隐层特征数, bi：双向为2，num_layer:层数
			输入：input:（seq, batch, H_in）; 
				  h_0:  (bi*num_layers, batch, H_out)； 
			输出：output: (seq, batch, bi*H_out);
				  h_n:  (bi*num_layers, batch, H_out)； 输出信息

		'''
		src = self.cnn_feature(src)
		src = src.permute(2,0,1).contiguous()  # （Batch, channel, seq）=>(seq, batch,channel)

		outputs, hidden = self.rnn(src)
		outputs = self.fc_out(outputs)    # 拟合,输出(seq, batch, channel)
		outputs = outputs.permute(1,2,0).contiguous()  # 处理成标准输出（Batch, channel, seq）
		
		# outputs:是叠置LSTM的最后一层输出
		return outputs





if __name__ == '__main__':

	# 输入和输出的特征维度。当输入是地震多属性时，input_dim会增加，当输出是类别时，output_dim会增加
	INPUT_DIM = 1
	OUTPUT_DIM = 1

	# eocoder-decoder所有的隐层特征必须相等
	HID_DIM = 512

	# 基本参数
	N_LAYERS = 2
	ENC_DROPOUT = 0.75
	device = 'cpu'

	# 实例化模型
	model = LSTM_MM(INPUT_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, ENC_DROPOUT)

	# 数据的输入输出组织维度是（Batch, channel, seq）
	src = torch.normal(0.,1., size=(4, 1, 100))
	output = model(src)
	print('last ouput: ',output.shape)