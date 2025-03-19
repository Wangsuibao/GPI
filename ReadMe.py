'''

=============================================================================================== 
Geophysics-Guided Multi-Stage Hybrid Learning for Post-Stack Seismic Impedance Inversion

Sui-Bao Wang
a. School of Earth Sciences, Northeast Petroleum University, Daqing 163318, China;
b. National Key Laboratory of Continental Shale Oil, Northeast Petroleum University, Daqing, Heilongjiang 163318, China

E-mail: wangsuibao@stu.nepu.edu.cn;


===============================================================================================
Module introduction：
	1、train_multitask.py test object:
		The GPI model was trained using different datasets.
		
	2、train_sigle.py test object:
		The models for the ablation study and advanced experiments of the GPI model were trained.

	3、test_3D.py
		The trained models were tested using different datasets.

	4、setting.py
		Hyperparameter setting
		
	5、utils/dataset.py
		Obtains the dataset.
		
	6、data:
		Stores the raw dataset.
		data/Stanford_VI/ {AI.npy, Facies.npy, synth_40HZ,npy}
		data: http://scrf.stanford.edu/resources.software.gslib.help.02.php


	7、model:
		Stores the model-related code.
		
	8、save_train_model:
		Stores the trained models.
		
	9、results:
		Stores seismic profiles,Impedance Inversion profiles and seismic facies profiles during the testing process.

SOFT model：           1、VishalNet, 2、GRU_MM, 3、Unet_1D, 4、Unet_1D_convolution
Inversion model：      1、Transformer_cov_para_geo, 2、Transformer+cov_para+geo
Ablation Study model： 1、Tansformer_cov_para, 2、Tansformer_geo, 3、Tansformer_convolution_geo


===============================================================================================
we propose a Geophysics-Guided Multi-stage Hybrid Learning P-wave Impedance Inversion Model (GPI). 
Specifically, we first incorporate the classical convolutional equation to 
design an explicit multi-frequency forward modeling module for the GPI inversion model. 
Second, we develop a seismic facies classification module within the GPI model, 
employing seismic facies data obtained through seismic interpretation as soft labels 
to enable weakly supervised learning. Finally, we integrate hard-label impedance data, 
soft-label seismic facies data, and unlabeled seismic data to achieve Multi-stage Hybrid Learning 
in the GPI model.

===============================================================================================
We thank The Stanford VI reservoir dataset, provided by Stanford University 

paper 《The Stanford VI-E Reservoir: A Synthetic Data Set for Joint Seismic-EM Time-lapse Monitoring Algorithms》

Jaehoon Lee and Tapan Mukerji
Department of Energy Resources Engineering
Stanford University