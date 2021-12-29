# Do group level stats and plots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
import glob
from scipy.ndimage.measurements import label
from scipy import stats
import mne
from scipy.stats import t

sns.set_context('talk', font_scale=1.1)
#sns.set_style('white')
sns.set_palette("colorblind")

included_subjects = ['128', '112', '108', '110', '120', '98', '86', '82', '115', '94', '76', '91', '80', '95', '121', '114', '125', '70',
'107', '111', '88', '113', '131', '130', '135', '140', '167', '145', '146', '138', '147', '176', '122', '118', '103', '142']

ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'
data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'



def compile_RT_TFR_stats():
	''' compile RT mixed effect regression model outputs'''
	data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'
	time = np.arange(0,226)
	times = np.load(data_path+'times.npy')
	freqs = np.arange(0,30)
	df = pd.DataFrame()
	for t in time:
		for f in freqs:
			pdf = pd.read_csv((ROOT+'RSA/regression_results/RSA_RT_t%s_f%s_results.csv' %(t, f)))
			pdf['time'] = np.round(times[t],3) #convert from second to ms
			pdf['frequency'] = np.round(np.load(data_path+'freqs.npy')[f],2)
			df = pd.concat([df, pdf])
	df.to_csv((ROOT+'RSA/regression_results/RSA_RT_TFR_compiled_results.csv'))
	return df


def compile_subj_TFR_stats():
	''' compile sub by subj RSA/GC regression outputs'''
	data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'
	time = np.arange(0,237)
	times = np.load(data_path+'times.npy')
	freqs = np.arange(0,30)
	df = pd.DataFrame()
	for t in time:
		for f in freqs:
			# pdf = pd.read_csv((data_path+'regression_results/RSA_GC_condition_t%s_f%s_results.csv' %(t, f)))
			# pdf['time'] = np.round(times[t],3) #convert from second to ms
			# pdf['frequency'] = np.round(np.load(data_path+'freqs.npy')[f],2)
			# df = pd.concat([df, pdf])
			pdf = pd.read_csv((data_path+'regression_results/RSA_GC_t%s_f%s_results.csv' %(t, f)))
			pdf['time'] = np.round(times[t],3) #convert from second to ms
			pdf['frequency'] = np.round(np.load(data_path+'freqs.npy')[f],2)
			df = pd.concat([df, pdf])
			df.loc[pd.isna(df.Condition), 'Condition'] = 'All'

	df.to_csv((ROOT+'RSA/regression_results/RSA_TFR_GC_condition_compiled_results.csv'))
	return df


def compile_subj_RT_regression_stats():
	''' compile sub by subj RT regression outputs'''
	data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'
	time = np.arange(0,226)
	times = np.load(data_path+'times.npy')
	freqs = np.arange(0,30)
	df = pd.DataFrame()
	for t in time:
		for f in freqs:
			pdf = pd.read_csv((data_path+'RT_regression_results/RSA_RT_t%s_f%s_results.csv' %(t, f)))
			pdf['time'] = np.round(times[t],3) #convert from second to ms
			pdf['frequency'] = np.round(np.load(data_path+'freqs.npy')[f],2)
			df = pd.concat([df, pdf])

	df.to_csv((ROOT+'RSA/RT_regression_results/RSA_TFR_RT_compiled_results.csv'))
	return df


def compile_RSA_TFR_matrices(df, accuracy, condition, var):
	mat = []
	for sub in included_subjects:
		mat.append(df.loc[(df['subject']==int(sub)) & (df['Accuracy']==accuracy) & (df['Condition']==condition)].reset_index().pivot('time','frequency', var).to_numpy())
	return np.array(mat)


def compile_GC_stats(window):
	''' read in GC matrices
	window can be 'All' or windowsize'''
	data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/GC/*%s*.csv' %window
	files = glob.glob(data_path)
	df = pd.DataFrame()
	for f in files:
		df = df.append(pd.read_csv(f))
	fn = 'cd compiled_GC_%s.csv' %window
	df.to_csv(fn)

	return df


def plot_GC(GCdf, source, target, accuracy, condition, var):
	df = GCdf.loc[(GCdf['source']==source) & (GCdf['target']==target) & (GCdf['accuracy']==accuracy) & (GCdf['condition']==condition)]
	mdf = df.groupby(['source_frequency', 'target_frequency']).mean().reset_index().pivot('source_frequency','target_frequency', var)
	sns.heatmap(mdf)
	plt.show()


def compile_GC_tfr_matrices(source, target, accuracy, condition='All', window_size = 'All', var ='diffF'):
	''' compile time by freq GC matrices, thse are GC estimaates from subj by subj'''

	mat = []
	for sub in included_subjects:
		try:
			fn = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/GC/%s_%s-%s_%s_%s_*%s_GC.csv' %(sub, source, target, condition, accuracy, window_size)
			fn = glob.glob(fn)[0]
			print(fn)
			mat.append(pd.read_csv(fn).pivot('time', 'source_frequency' , var).to_numpy())
		except:
			continue
	return np.array(mat) #sub by time by freq

def compile_GC_matrices(source, target, accuracy, condition='All', window_size = 'All', var ='F'):
	''' compile time by freq GC matrices, thse are GC estimaates from subj by subj'''

	mat = []
	for sub in included_subjects:
		try:
			fn = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/GC/%s_%s-%s_%s_%s_*%s_GC.csv' %(sub, source, target, condition, accuracy, window_size)
			fn = glob.glob(fn)[0]
			print(fn)
			mat.append(pd.read_csv(fn).pivot('source_frequency', 'target_frequency' , var).to_numpy())
		except:
			continue
	return np.array(mat) #sub by time by freq


def matrix_permutation(M1, M2, threshold, p, ttest=True):
	''' randomize permutation test for element wise comparison'''
	if M1.shape[0] != M2.shape[0]:
		print("check matrix dimesnion")
		return

	#this is for independent samples
	# num_sub = M1.shape[0]
	# mats = np.vstack([M1,M2])
	# rand_vec = np.random.permutation(np.arange(mats.shape[0])) #permute on the first dimension, subject
	# pM1 = mats[rand_vec[0:num_sub],:,:]
	# pM2 = mats[rand_vec[num_sub:],:,:]
	# dM = pM1 - pM2

	# related sample permutation test
	num_sub = M1.shape[0]
	dM = M1 - M2

	#get null_districution
	null_mass = np.zeros(5000)
	for iter in np.arange(5000):
		rand_vec = (2*np.random.randint(0,2,size=(num_sub))-1)
		perumted_M = np.zeros(np.shape(dM))

		for i, r in enumerate(rand_vec):
			perumted_M[i,:,:] = dM[i,:,:] * r

		if not ttest: # permutation test of mean diff
			mat = perumted_M.mean(axis=0)
		if ttest:
			mat = stats.ttest_1samp(perumted_M,0,0)[0]

		mass = 0
		#positive
		cmat, num_clust = label(mat>threshold)
		for c in np.arange(1,np.max(num_clust)+1):
			temp_mass = np.sum(mat[cmat==c])
			if temp_mass > mass:
				mass = temp_mass

		mass = 0
		#negative
		cmat, num_clust = label(mat<-1*(threshold))
		for c in np.arange(1,np.max(num_clust)+1):
			temp_mass = np.sum(mat[cmat==c])
			if abs(temp_mass) > mass: #flip the sign
				mass = abs(temp_mass)
		null_mass[iter] = mass

	pos_mass_thresh = np.quantile(null_mass, 1-0.5*p)
	neg_mass_thresh = np.quantile(null_mass, 1-0.5*p)

	# do the cluster formation and test
	if not ttest:
		mat = dM.mean(axis=0)
	if ttest:
		mat = stats.ttest_1samp(dM,0,0)[0]

	cmat, num_clust = label(mat>threshold)
	mask = np.zeros(np.shape(mat))
	for c in np.arange(1,np.max(num_clust)+1):
		if np.sum(mat[cmat==c]) > pos_mass_thresh:
			mask = mask + np.array(cmat==c)
	#negative
	cmat, num_clust = label(mat<-1*(threshold))
	for c in np.arange(1,np.max(num_clust)+1):
		if abs(np.sum(mat[cmat==c])) > neg_mass_thresh:
			mask = mask + np.array(cmat==c)

	return mask, mat


def calculate_TFR_classification_accuracy():
	classes = ['dcb', 'dcr', 'dpb', 'dpr', 'fcb', 'fcr', 'fpb', 'fpr']
	ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'
	mat = []
	for s, sub in enumerate(included_subjects):
		tfr = mne.time_frequency.read_tfrs((ROOT+'RSA/%s_tfr.h5' %sub))[0]
		y_data = tfr.metadata.cue.values.astype('str')
		data = np.load((ROOT+'/RSA/%s_tfr_prob.npy' %sub)) # posterior probability

		accu_mat = np.zeros((data.shape[1], data.shape[2]))
		#calcuate accuracy
		trials = data.shape[0]
		for t in np.arange(data.shape[1]):
			for f in np.arange(data.shape[2]):
				n_scores = data[:,t,f,:]
				i=0
				for ix, ip in enumerate(np.argmax(n_scores,axis=1)):
					if y_data[ix] == classes[ip]:
						i=i+1

				accu_mat[t,f] = (i/trials)*100
		mat.append(accu_mat)

	mean_acc = np.mean(mat,axis=0)
	df = pd.DataFrame(data =mean_acc, index = np.round(tfr.freqs,2), columns = np.round(tfr.times,3))
	return df, np.array(mat)


if __name__ == "__main__":


	########################################################################
	###### plot raw classification accuracy of individual cues. Figure 3
	##############################################################################
	#acc_df, _ = calculate_TFR_classification_accuracy()
	#acc_df.to_csv('Data/TFR_Accu.csv')
	acc_df = pd.read_csv('Data/TFR_Accu.csv')
	times = np.load(data_path+'times.npy')  #we want 31, 82, 134, 185. Which is -0.5, 0, 0.5, 1
	#freqs = np.round(np.load(data_path+'freqs.npy')[f],2)  
	ax = sns.heatmap(acc_df, xticklabels = 15, yticklabels = 3, vmin=12, vmax=16)
	ax.invert_yaxis()
	ax.set_xticks([31,82,134,185])
	ax.set_xticklabels([-0.5, 0, 0.5, 1])
	fn = 'Figures/tfracu.png'
	plt.savefig(fn)
	plt.close()


	########################################################################
	###### Plot RSA regression results and do cluster permutation. Figure 4
	#############################################################################
	##### compile TFR RSA sub by sub
	#df = compile_subj_TFR_stats()
	df = pd.read_csv(ROOT+'RSA/regression_results/RSA_TFR_GC_condition_compiled_results.csv')

	for cond in ['task_b', 'context_b', 'feature_b', 'identity_b']:
		D1 = compile_RSA_TFR_matrices(df, 'Correct', 'All', cond)
		D2 = np.zeros(np.shape(D1))
		mask, A = matrix_permutation(D1, D2, 2.53, 0.05, True)
		time = np.round(np.load('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/times.npy'),3)
		freqs = np.round(np.load('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/freqs.npy'),2)
		mdf = pd.DataFrame(data =A.T, index = freqs, columns =time)
		ax = sns.heatmap(mdf, center=0, vmin=-5, vmax=5, xticklabels = 15, yticklabels = 3, mask = 1- mask.T)
		ax.invert_yaxis()
		ax.set_xticklabels(ax.get_xticklabels(),rotation = 0)
		ax.set_yticklabels(ax.get_yticklabels(),rotation = 0)
		ax.set_xticks([31,82,134,185])
		ax.set_xticklabels([-0.5, 0, 0.5, 1])
		plt.tight_layout()
		fn = 'Figures/'+cond+'rsa.png'
		plt.savefig(fn)
		plt.close()


	########################################################################
	###### Plot GCA results. Figure 5
	##############################################################################
	# def plot_tfr_GC(source, target):
	# 	D1 = compile_GC_tfr_matrices(source, target, 'Correct', 'All', 50, 'F')
	# 	D2 = compile_GC_tfr_matrices(source, target, 'Correct', 'All', 50, 'reverseF')    
	# 	mask, A = matrix_permutation(D1, D2, 2.03, 0.05, True)
	# 	#time = np.arange(0,187) # need to create proper ones.
	# 	time = np.round(np.load('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/times.npy')[0:187],3)
	# 	freqs = np.round(np.load('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/freqs.npy'),2)
	# 	df = pd.DataFrame(data =A.T, index = freqs, columns =time)
	# 	ax = sns.heatmap(df, center = 0, vmin =-5, vmax=5, xticklabels = 15, yticklabels = 3, mask = 1- mask.T)
	# 	ax.invert_yaxis()
	# 	ax.set_xticks([31,82,134,185])
	# 	ax.set_xticklabels([-0.5, 0, 0.5, 1])
	# 	plt.tight_layout()
	# 	fn = 'Figures/GC_%s-%s.png' %(source, target)
	# 	plt.savefig(fn)
	# 	plt.close()

	sns.set_context('talk', font_scale=1)
	def plot_f2f_GC(source, target):
		D1 = compile_GC_matrices(source, target, 'Correct', 'All', 'All', 'F')
		D2 = compile_GC_matrices(source, target, 'Correct', 'All', 'All', 'reverseF')    
		mask, A = matrix_permutation(D1, D2, 2.03, 0.05, False)
		freqs = np.round(np.load('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/freqs.npy'),2)
		df = pd.DataFrame(data =A.T, index = freqs, columns =freqs)
		ax = sns.heatmap(df, center = 0, vmin =-4, vmax=4, xticklabels = 6, yticklabels = 4, square=True, mask = 1- mask.T)
		ax.invert_yaxis()
		ax.set_xticklabels(ax.get_xticklabels(),rotation = 0)
		ax.set_xticks([0,6,12,17,24,30])
		ax.set_xticklabels([1,2.15, 4.05,8.69,21.18, 40])
		ax.set_yticks([0,6,12,17,24,30])
		ax.set_yticklabels([1,2.15, 4.05,8.69,21.18, 40])
		plt.tight_layout()
		fn = 'Figures/GC_%s-%s.png' %(source, target)
		plt.savefig(fn)
		plt.close()

	plot_f2f_GC('context', 'feature')
	plot_f2f_GC('context', 'task')
	plot_f2f_GC('context', 'identity')
	plot_f2f_GC('feature', 'identity')
	plot_f2f_GC('feature', 'task')
	plot_f2f_GC('identity', 'task')





# end of line