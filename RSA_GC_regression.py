####################################################################
# Script to obtain RSA model beta weights for each subject.
# The purpuse is to create a "beta" timeseires that represent the time-varying model representation
# Which can be used to reveal the representational dynamics, and for GC analysis in GC_RSA_ts.py
# Please note this script was written to use UIowa's Argon cluster SGE system
####################################################################
from gen_TFR_RSA import gen_tfr_indices_for_looping
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.formula.api as smf
import pathlib
from datetime import datetime
data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/' #path specific to Iowa Argon system.

def run_regression(df):
	''' regression. sub by sub '''

	#### #### #### #### #### #### #### ####
	# read in df
	# drop long RT and missed trials.
	# drop incorrect trials
	# drop outliers
	#### #### #### #### #### #### #### ####

	print(len(df))
	#df.loc[df.Accuracy==0, 'ACC_contrast'] = -1
	df = df.drop(df.loc[df.RT>5].index) #unrealistically slow
	df = df.drop(df.loc[np.isnan(df.RT)].index)  #missed responses
	print(len(df))

	subjects = df.subject.unique()
	results_df = pd.DataFrame()

	s=0
	for subject in subjects:
		#### #### #### #### #### #### #### ####
		#### RSA model regressors, fit the model sub by sub
		#### #### #### #### #### #### #### ####
		sdf = df.loc[df['subject']==subject]
		dsets = [sdf.loc[sdf['Accuracy']==1], sdf.loc[sdf['Accuracy']==0]] #separate correct and incorrect, though too few incorrect trials so we ended up no analyzing those further
		dlabs = ['Correct', 'Incorrect']

		for d, dset in enumerate(dsets):
			model = "probability ~ 1 + RT*context + RT*feature + RT*identity + RT*task"
			results = smf.ols(formula = model, data = dset).fit()  
			print(results.summary())

			params = ["context" , "feature" , "identity" , "task", "RT", "RT:context", "RT:feature", "RT:identity", "RT:task"]
			for param in params:
				results_df.loc[s, 'subject'] = str(subject)
				results_df.loc[s, 'Accuracy'] = dlabs[d]
				results_df.loc[s, param+'_b'] = results.params[param]
				results_df.loc[s, param+'_t'] = results.tvalues[param]
				results_df.loc[s, param+'_p'] = results.pvalues[param]
				results_df.loc[s, 'time'] = t
				results_df.loc[s, param+'_lower_ci'] = results.conf_int().loc[param][0]
				results_df.loc[s, param+'_upper_ci'] = results.conf_int().loc[param][1]
			s = s +1

	return results_df


if __name__ == "__main__":

	now = datetime.now()
	print("starting time: ", now)


	#### The setup here is for the SGE system on UIowa's Argon cluster
	job_iter = int(input()) - 1 #sge job id is not zero based
	tfr_pair, tfr_index = gen_tfr_indices_for_looping()

	###
	#tfr_pair, tfr_index = gen_ftr_indices_for_looping()
	t = tfr_index[job_iter][0]
	time = tfr_pair[job_iter][0]
	f = tfr_index[job_iter][1]
	freq = tfr_pair[job_iter][1]
	print(('now running time %s' %time))
	print(('now running frequency %s' %freq))

	df = pd.read_csv((data_path+'model_regressors/RSA_t%s_f%s_data.csv' %(t, f)))
	df = df.reset_index()
	results_df, condition_results_df = run_regression(df)
	results_df.to_csv((data_path+'regression_results/RSA_GC_t%s_f%s_results.csv' %(t, f)))


	now = datetime.now()
	print("finished time: ", now)




# end of line
