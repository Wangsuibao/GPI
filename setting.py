import numpy as np 
from scipy.spatial import distance

TCN1D_train_p = {'batch_size': 4,
				'no_wells': 401, 
				'unsupervised_seismic': 401,
				'SGS_data_n': 1000,
				'epochs': 1000, 
				'grad_clip': 1.0,
				'input_shape': (1, 200),
				'lr': 0.0001,

    # SOFT模型：1、VishalNet, 2、GRU_MM, 3、Unet_1D, 4、Unet_1D_convolution
    # 反演模型：Transformer_cov_para_geo：Transformer+cov_para+geo
    # 消融实验模型： 1、Tansformer_cov_para, 2、Tansformer_geo, 3、Tansformer_convolution_geo

				'model_name': 'VishalNet',
				'Forward_model': 'cov_para',       # '' 表示没有正演过程
				'Facies_model': 'Facies',

	# 'Stanford_VI', 'Fanny', 
				'data_flag': 'Stanford_VI',
				'get_F': 0,  #（0,2,4） 地震数据扩充了频率特征和动态特征，当0时表示只有时域地震波形
				'F': 'WE_PreSDM',  # 当："data_flag = M2_F
				}

TCN1D_test_p = {'no_wells':20,
				'data_flag':'Stanford_VI',
				'model_name': 'Tansformer_cov_para_Facies_s_uns',  # model-name_Forward-model_Facies-model_s_uns
				# 注意，如果是有正演模块，则：反演模块_正演模块
				}

