# Do path model to look at structure of representations (Figure 5)
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import semopy

included_subjects = ['128', '112', '108', '110', '120', '98', '86', '82', '115', '94', '76', '91', '80', '95', '121', '114', '125', '70',
'107', '111', '88', '113', '131', '130', '135', '140', '167', '145', '146', '138', '147', '176', '122', '118', '103', '142']
ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'
data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'

####################################################################
# Trial by trial path model
####################################################################

def compile_single_trial_RSA():
	''' turns out the files are 30G, so needed to wrap this into a separate function'''
	''' compile trial by trial RSA'''
	#compile single trial dataframe
	time = np.arange(0,226)
	times = np.load(data_path+'times.npy')
	freqs = np.load(data_path+'freqs.npy')
	freq = np.arange(0,30)

	#define power and time epochs
	delta = np.where((4>=freqs) & (freqs>=1))[0]
	theta = np.where((8>=freqs) & (freqs>=4))[0]
	alpha = np.where((14>=freqs) & (freqs>=8))[0]
	beta = np.where((30>=freqs) & (freqs>=14))[0]
	delta_theta = np.where((8>=freqs) & (freqs>=1))[0]
	band_names = ['delta', 'theta', 'alpha', 'beta', 'delta_theta']

	cue = np.where((-0.5 <= times) & (times <=0))[0]
	resp = np.where((0 <= times) & (times <=0.5))[0]
	cue_resp = np.where((-0.5 <= times) & (times <=0.5))[0]
	iti = np.where((-1.5 <= times) & (times <=-0.5))[0]
	ep_names = ['cue', 'resp', 'iti', 'cue_resp']

	bands = [delta, theta, alpha, beta, delta_theta]
	epochs = [cue, resp, iti, cue_resp]

	df_long = pd.DataFrame()
	for b_idx, band in enumerate(bands):
		for e_idx, epoch in enumerate(epochs):
			bedf = pd.DataFrame()
			for t in epoch:
				for f in band:
					tdf = pd.read_csv(data_path+'singletrial_regression_results/RSA_GC_single_trial_t%s_f%s_results.csv' %(t, f))
					bedf = bedf.append(tdf)

			mdf = bedf.groupby(['subject','trial']).mean().reset_index()
			mdf['band'] = band_names[b_idx]
			mdf['epoch'] = ep_names[e_idx]
			df_long = df_long.append(mdf)

	df_long.to_csv(data_path+'singletrial_regression_results/RSA_GC_single_trial_longform_compiled_results.csv')

	return df_long


if __name__ == "__main__":

	#df_long = compile_single_trial_RSA()
	df_long = pd.read_csv(data_path+'singletrial_regression_results/RSA_GC_single_trial_longform_compiled_results.csv')

	df = df_long.loc[(df_long['band']=='delta') & (df_long['epoch'] == 'resp')] #this is to get all the delta data
	df.loc[:,'theta_task_b_resp'] = df_long.loc[(df_long['band']=='theta') & (df_long['epoch'] == 'resp')]['task_b'].values
	df.loc[:,'delta_task_b_resp'] = df_long.loc[(df_long['band']=='delta') & (df_long['epoch'] == 'resp')]['task_b'].values
	df.loc[:,'delta_theta_task_b_resp'] = df_long.loc[(df_long['band']=='delta_theta') & (df_long['epoch'] == 'resp')]['task_b'].values

	df.loc[:,'theta_identity_b_cue'] = df_long.loc[(df_long['band']=='theta') & (df_long['epoch'] == 'cue')]['identity_b'].values
	df.loc[:,'delta_identity_b_cue'] = df_long.loc[(df_long['band']=='delta') & (df_long['epoch'] == 'cue')]['identity_b'].values
	df.loc[:,'delta_theta_identity_b_cue'] = df_long.loc[(df_long['band']=='delta_theta') & (df_long['epoch'] == 'cue')]['identity_b'].values
	df.loc[:,'delta_theta_identity_b_cue_resp'] = df_long.loc[(df_long['band']=='delta_theta') & (df_long['epoch'] == 'cue_resp')]['identity_b'].values

	df.loc[:,'theta_context_b_cue'] = df_long.loc[(df_long['band']=='theta') & (df_long['epoch'] == 'cue')]['context_b'].values
	df.loc[:,'delta_context_b_cue'] = df_long.loc[(df_long['band']=='delta') & (df_long['epoch'] == 'cue')]['context_b'].values
	df.loc[:,'delta_theta_context_b_cue'] = df_long.loc[(df_long['band']=='delta_theta') & (df_long['epoch'] == 'cue')]['context_b'].values
	df.loc[:,'delta_theta_context_b_cue_resp'] = df_long.loc[(df_long['band']=='delta_theta') & (df_long['epoch'] == 'cue_resp')]['context_b'].values

	df.loc[:,'theta_feature_b_cue'] = df_long.loc[(df_long['band']=='theta') & (df_long['epoch'] == 'cue')]['feature_b'].values
	df.loc[:,'delta_feature_b_cue'] = df_long.loc[(df_long['band']=='delta') & (df_long['epoch'] == 'cue')]['feature_b'].values
	df.loc[:,'delta_theta_feature_b_cue'] = df_long.loc[(df_long['band']=='delta_theta') & (df_long['epoch'] == 'cue')]['feature_b'].values
	df.loc[:,'delta_theta_feature_b_cue_resp'] = df_long.loc[(df_long['band']=='delta_theta') & (df_long['epoch'] == 'cue_resp')]['feature_b'].values

	df.to_csv('Data/singel_trial_RSAbetas.csv')
	df = pd.read_csv('Data/singel_trial_RSAbetas.csv')


	########################################################################
	## SEM model using semopy, but functionality not as great as lavaan in R. See path_model.R for lavaan implementation.
	########################################################################
	# model = 'rt ~ task_b_delta \ntask_b_delta ~ feature_b_cue \nfeature_b_cue ~ context_b_cue'
	# sem_model = semopy.Model(model)
	# results = sem_model.fit(df, groups=['subject'])
	# print(sem_model.inspect())
	# stats = semopy.calc_stats(sem_model)
	# print(stats.T)

	# # null model
	# model = 'rt ~ task_b_delta \ntask_b_delta ~ identity_b_cue \nidentity_b_cue ~ feature_b_cue \nfeature_b_cue ~ context_b_cue'
	# sem_model = semopy.Model(model)
	# results = sem_model.fit(df, groups=['subject'])
	# print(sem_model.inspect())
	# stats = semopy.calc_stats(sem_model)
	# print(stats.T)

	# model = 'rt ~ task_b_delta \ntask_b_delta ~ identity_b_cue'
	# sem_model = semopy.Model(model)
	# results = sem_model.fit(df, groups=['subject'])
	# print(sem_model.inspect())
	# stats = semopy.calc_stats(sem_model)
	# print(stats.T)
	#end of line
