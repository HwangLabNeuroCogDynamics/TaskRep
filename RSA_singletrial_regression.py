####################################################################
# Script to obtain RSA model beta weights for each subject and EVERY TRIAL
# The purpuse is to do a SEM model including RT information
####################################################################
from gen_TFR_RSA import gen_tfr_indices_for_looping
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.formula.api as smf
import pathlib
from datetime import datetime
data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/' #path specific to Iowa Argon system.

def run_trial_regression(df):
	''' regression. sub by sub, trial by trial '''

	#### #### #### #### #### #### #### ####
	# read in df
	# drop long RT and missed trials.
	# drop error trials
	# drop outliers
	#### #### #### #### #### #### #### ####

	print(len(df))
	df = df.drop(df.loc[df.RT>5].index) #unrealistically slow
	df = df.drop(df.loc[np.isnan(df.RT)].index)  #missed responses
	df = df.drop(df.loc[df.Accuracy==0].index) # error
	print(len(df))

	subjects = df.subject.unique()
	results_df = pd.DataFrame()

	s=0
	for subject in subjects:
		#### #### #### #### #### #### #### ####
		#### RSA model regressors, fit the model trial by trial for each subject
		#### #### #### #### #### #### #### ####
		sdf = df.loc[df['subject']==subject]
		trials = sdf.trial.unique()

		for trial_idx, trial in enumerate(trials):
			dset = sdf.loc[sdf['trial']==trial]
			results_df.loc[s, 'subject'] = str(subject)
			results_df.loc[s, 'Accuracy'] = dset.Accuracy.values[0]
			results_df.loc[s, 'trial'] = trial
			results_df.loc[s, 'RT'] = dset.RT.min() #zscored
			results_df.loc[s, 'rt'] = dset.rt.min() #raw rt
			results_df.loc[s, 'Trial_type'] = dset.Trial_type.min()
			results_df.loc[s, 'Trial_n'] = dset.Trial_n.min()  
			
			model = "probability ~ 1 + context + feature + identity + task" 
			results = smf.ols(formula = model, data = dset).fit()
			for param in ["context",  "feature" , "identity" , "task"]:
				results_df.loc[s, param+'_b'] = results.params[param]
				results_df.loc[s, param+'_t'] = results.tvalues[param]
				results_df.loc[s, param+'_p'] = results.pvalues[param]
				results_df.loc[s, param+'_lower_ci'] = results.conf_int().loc[param][0]
				results_df.loc[s, param+'_upper_ci'] = results.conf_int().loc[param][1]
			s = s +1

	return results_df

if __name__ == "__main__":

	now = datetime.now()
	print("starting time: ", now)

	times = np.load(data_path+'times.npy')
	freqs = np.load(data_path+'freqs.npy')

	#define power and time epochs
	delta_theta = np.where((8>=freqs) & (freqs>=4))[0][0:-1:3]
	delta = np.where((4>=freqs) & (freqs>=1))[0][0:-1:3]
	All = np.where((40>=freqs) & (freqs>=1))[0][0:-1:3]
	cue = np.where((-0.5 <= times) & (times <=0))[0][0:-1:3]
	resp = np.where((0.0 <= times) & (times <=0.5))[0][0:-1:3]

	cue_df = pd.DataFrame()
	for t_ix in cue:
		for f in All:
			cue_df = cue_df.append(pd.read_csv((data_path+'model_regressors/RSA_t%s_f%s_data.csv' %(t_ix, f))))
	cue_df.to_csv(data_path+'singletrial_regression_results/RSA_GC_single_trial_cue_data.csv')

	probe_df = pd.DataFrame()
	for t_ix in resp:
		for f in All:
			probe_df = probe_df.append(pd.read_csv((data_path+'model_regressors/RSA_t%s_f%s_data.csv' %(t_ix, f))))
	probe_df.to_csv(data_path+'singletrial_regression_results/RSA_GC_single_trial_probe_data.csv')

	# get trial by trial model beta
	cue_results_df = run_trial_regression(cue_df)
	cue_results_df.to_csv(data_path+'singletrial_regression_results/RSA_GC_single_trial_cue_results.csv')

	probe_results_df = run_trial_regression(probe_df)
	probe_results_df.to_csv(data_path+'singletrial_regression_results/RSA_GC_single_trial_probe_results.csv')
	
	results_df = cue_results_df
	results_df['probe_task_b'] = probe_results_df['task_b']
	results_df.to_csv('Data/singel_trial_RSA_betas.csv')

	now = datetime.now()
	print("finished time: ", now)