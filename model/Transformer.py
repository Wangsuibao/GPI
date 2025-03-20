'''
	实现Transformer, 包含三部分，1、位置编码器，2、TransformerEncoder, 3、TransformerDecoder
	*** 体现目标序列建模的优势
	*** 体现长序列注意力机制的优势

	输入数据的维度是（B, C, T）; 输出数据的维度是（B, 1, T）
	*** Transformer中流动的数据维度是：（T, B, C）***
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random

class PositionalEncoding(nn.Module):
	# 输入和位置的加和
	def __init__(self, d_model, max_len=696):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return x

class TransformerModel(nn.Module):
	def __init__(self, input_dim=1, output_dim=1, num_heads=4, num_layers=3, dim_feedforward=128, dropout=0.1):
		super(TransformerModel, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.embedding_in = nn.Linear(input_dim, dim_feedforward)    # **input 特征维度的编码**
		self.embedding_out = nn.Linear(output_dim, dim_feedforward)  # **input 特征维度的编码**
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.pos_encoder = PositionalEncoding(dim_feedforward)
		
		encoder_layers = nn.TransformerEncoderLayer(dim_feedforward, num_heads, dim_feedforward, dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
		
		decoder_layers = nn.TransformerDecoderLayer(dim_feedforward, num_heads, dim_feedforward, dropout)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
		
		self.decoder = nn.Linear(dim_feedforward, output_dim)

	def add_noise(self, y, noise_level=0.5):
	    noise = noise_level * torch.randn_like(y)  # 生成与y同样形状的随机噪音
	    return y + noise

	def forward(self, src, tgt):
		# 将输入数据转换为Transformer期望的格式（T, B, C）
		src = src.permute(2, 0, 1)    # (701, 32, 3)
		tgt = tgt.permute(2, 0, 1)    # (701, 32, 1)
		tgt = self.add_noise(tgt)     # 向tgt中加入噪音，用于解决exposure bias问题。&&&&&&重点

		tgt_seq_len = tgt.shape[0]
		tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool().to(self.device)

		src = self.embedding_in(src)  # (T, B, input_dim) -> (T, B, dim_feedforward)
		src = self.pos_encoder(src)  # 加入位置编码
		memory = self.transformer_encoder(src)  # 通过Transformer编码器
		
		tgt = self.embedding_out(tgt)  # (T, B, output_dim) -> (T, B, dim_feedforward)
		tgt = self.pos_encoder(tgt)  # 加入位置编码
		output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # 通过Transformer解码器
		
		output = self.decoder(output)  # 解码
		output = output.permute(1,2,0)  # 返回和原始输入数据一致（B, C, T）
		return output

class TransformerModel_S(nn.Module):
	'''

	'''
	def __init__(self, input_dim=1, output_dim=1, num_heads=4, num_layers=3, dim_feedforward=128, dropout=0.1):
		super(TransformerModel_S, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.embedding_in = nn.Linear(input_dim, dim_feedforward)    # **input 特征维度的编码**
		self.embedding_out = nn.Linear(output_dim, dim_feedforward)  # **input 特征维度的编码**
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.pos_encoder = PositionalEncoding(dim_feedforward)
		
		encoder_layers = nn.TransformerEncoderLayer(dim_feedforward, num_heads, dim_feedforward, dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
		
		decoder_layers = nn.TransformerDecoderLayer(dim_feedforward, num_heads, dim_feedforward, dropout)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
		
		self.decoder = nn.Linear(dim_feedforward, output_dim)

	def forward(self, src, tgt, train=True, teacher_forcing_ratio=0.8):
		# 将输入数据转换为Transformer期望的格式（T, B, C）
		src = src.permute(2, 0, 1)    # (701, 32, 3)
		tgt = tgt.permute(2, 0, 1)    # (701, 32, 1)

		src = self.embedding_in(src)  # (T, B, input_dim) -> (T, B, dim_feedforward)
		src = self.pos_encoder(src)  # 加入位置编码
		memory = self.transformer_encoder(src)  # 通过Transformer编码器

		tgt_embedded = self.embedding_out(tgt[0, :].unsqueeze(0))
		tgt_embedded = self.pos_encoder(tgt_embedded)  # 加入位置编码

		outputs = []
		for t in range(1, tgt.size(0)):  # 从 t=1 开始解码
			tgt_seq_len = tgt_embedded.shape[0]
			tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool().to(self.device)

			output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
			output_step = self.decoder(output[-1, :, :])

			teacher_force = random.random() < teacher_forcing_ratio
			if teacher_force and train:
				input = tgt[t:t+1, :, :]
			else:
				input = output_step.unsqueeze(0)

			outputs.append(output_step.unsqueeze(0))

			tgt_embedded = torch.cat((tgt_embedded, self.embedding_out(input)), dim=0)

		output_ = torch.cat(outputs, dim=0)
		output_ = output_.permute(1,2,0)  # 返回和原始输入数据一致（B, C, T）
		return output_


class TransformerModel_R(nn.Module):
	'''
	'''
	def __init__(self, input_dim=1, output_dim=1, num_heads=4, num_layers=6, dim_feedforward=128, dropout=0.1):
		super(TransformerModel_R, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.embedding_in = nn.Linear(input_dim, dim_feedforward)    # **input 特征维度的编码**
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.pos_encoder = PositionalEncoding(dim_feedforward)
		
		encoder_layers = nn.TransformerEncoderLayer(dim_feedforward, num_heads, dim_feedforward, dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
		
		self.decoder = nn.Linear(dim_feedforward, output_dim)

	def forward(self, src):
		# 将输入数据转换为Transformer期望的格式（T, B, C）
		src = src.permute(2, 0, 1)    # (701, 32, 3)

		src = self.embedding_in(src)  # (T, B, input_dim) -> (T, B, dim_feedforward)
		src = self.pos_encoder(src)  # 加入位置编码
		output = self.transformer_encoder(src)  # 通过Transformer编码器
		
		output = self.decoder(output)  # 拟合目标
		output = output.permute(1,2,0)  # 返回和原始输入数据一致（B, C, T）
		return output



if __name__ == '__main__':
	# 定义输入输出维度和超参数
	input_dim = 1
	output_dim = 1
	num_heads = 4
	num_layers = 3
	dim_feedforward = 128

	# 创建模型实例
	model = TransformerModel(input_dim, output_dim, num_heads, num_layers, dim_feedforward)
