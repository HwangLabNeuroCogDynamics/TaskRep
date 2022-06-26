####################################################################
# Script to generate RSA models and compile dataframe for regression
####################################################################
from gen_RSA import create_RSA_models
import pandas as pd
import numpy as np


included_subjects = ['128', '112', '108', '110', '120', '98', '86', '82', '115', '94', '76', '91', '80', '95', '121', '114', '125', '70',
'107', '111', '88', '113', '131', '130', '135', '140', '167', '145', '146', '138', '147', '176', '122', '118', '103', '142']
classes = ['dcb', 'dcr', 'dpb', 'dpr', 'fcb', 'fcr', 'fpb', 'fpr']
data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'

#########################################################################################################
##### conmpile dataframe and build regression model for each TFR bin
#########################################################################################################
def compile_fullTFR_RSA_model(t, time, f, freq):
	''' compile RSA model for each TFR bin'''

	all_df = pd.DataFrame() #save out all model
	for sub in included_subjects:

		trial_prob = np.load((data_path+'%s_tfr_prob.npy' %sub)) #note this is the full TFR version. Trial by freq by time by label
		metadata = pd.read_csv((data_path+'%s_metadata.csv' %sub))
		times = np.load((data_path+'%s_times.npy' %sub))

		metadata.loc[metadata.rt=='none', 'rt'] = np.nan
		metadata.rt = metadata.rt.astype('float')
		metadata['RT'] = (metadata.rt - np.nanmean(metadata.rt)) / np.nanstd(metadata.rt) #zcore
		metadata['Accuracy'] = metadata.trial_Corr

		#determine swap or not swap
		if 'version' in metadata.columns:  # the swapped version didnt have version column (new version)
			swapped = 1
			notswapped = 0
		else:
			notswapped = 1
			swapped = 0

		# compile dataframe by time, freq, trial by trial
		#for i, time in enumerate(times):

			# here add a loop for frequency
			#freqs = np.logspace(*np.log10([1, 35]), num=20)  # not ideal

			#for f, freq in enumerate(freqs):

		for j, trial in enumerate(np.arange(trial_prob.shape[0])):
			tdf = pd.DataFrame()
			tdf['probability'] = trial_prob[j, f, t, :]
			tdf['subject']  = sub
			tdf['trial'] = trial
			tdf['time'] = time
			tdf['time_idx'] = t
			tdf['freq'] = freq
			tdf['freq_idx'] = f
			tdf['block'] = metadata.loc[j, 'block']
			tdf['RT'] = metadata.loc[j, 'RT'] #zscored
			tdf['Accuracy'] = metadata.loc[j, 'Accuracy']
			tdf['rt'] = metadata.loc[j, 'rt'] #raw rt
			tdf['Trial_type'] = metadata.loc[j, 'Trial_type']
			#tdf['Rule_Switch'] = metadata.loc[j, 'Rule_Switch'] # not analyzed during revision
			tdf['Trial_n'] = metadata.loc[j, 'trial_n']

			trial_type = metadata.cue.values[j]
			if swapped:
				models = ['context', 'shape', 'color', 'identity', 'dimension', 'task', 'feature']
				for k, model in enumerate([context_model, shape_model, color_model, identity_model, swapped_dimension_model, swapped_task_model, swapped_feature_model]):
					tdf[models[k]] = model[classes.index(trial_type)]

			if notswapped:
				models = ['context', 'shape', 'color', 'identity', 'dimension', 'task', 'feature']
				for k, model in enumerate([context_model, shape_model, color_model, identity_model, nonswapped_dimension_model, nonswapped_task_model, nonswapped_feature_model]):
					tdf[models[k]] = model[classes.index(trial_type)]

			all_df = pd.concat([all_df, tdf])

			# #exiting trial loops
			# all_df = pd.concat([all_df, df])
	all_df = all_df.reset_index()
	#all_df.to_csv((ROOT+'RSA/%s_RSA_fullTFR_data.csv' %sub))

	return all_df

def gen_tfr_indices_for_looping():
	''' generate looping variables, time by freq'''
	times = np.load(data_path+'times.npy')
	freqs = np.load(data_path+'freqs.npy')
	tfr_pair = []
	tfr_index = []
	for i, t in enumerate(times):
		for j, f in enumerate(freqs):
			tfr_pair.append((t,f)) # create a tuple to save the time and freq of each iteration
			tfr_index.append((i,j)) # create a tuple to save the time and freq of each iteration
	return tfr_pair, tfr_index

def gen_ftr_indices_for_looping():
	''' generate looping variables, freq by time'''
	times = np.load(data_path+'times.npy')
	freqs = np.load(data_path+'freqs.npy')
	tfr_pair = []
	tfr_index = []
	for i, f in enumerate(freqs):
		for j, t in enumerate(times):
			tfr_pair.append((t,f)) # create a tuple to save the time and freq of each iteration
			tfr_index.append((i,j)) # create a tuple to save the time and freq of each iteration
	return tfr_pair, tfr_index


if __name__ == "__main__":

	context_model, shape_model, color_model, identity_model, swapped_dimension_model, nonswapped_dimension_model, swapped_task_model, nonswapped_task_model, swapped_feature_model, nonswapped_feature_model, alternative_model = create_RSA_models()
	for model in [context_model, shape_model, color_model, identity_model, swapped_dimension_model, nonswapped_dimension_model, swapped_task_model, nonswapped_task_model, swapped_feature_model, nonswapped_feature_model, alternative_model]:
		print(' ')
		print(model)
		print(' ')

	tfr_pair, tfr_index = gen_tfr_indices_for_looping()

	#time_pt = input()  #check how many timepoints, right now it is 237
	#print(time_pt)
	# unpack job array ID and find the tpt and freq to create models for
	job_iter = int(input()) - 1 #sge job id is not zero based

	t = tfr_index[job_iter][0]
	time = tfr_pair[job_iter][0]
	f = tfr_index[job_iter][1]
	freq = tfr_pair[job_iter][1]
	print(time)
	print(freq)

	df = compile_fullTFR_RSA_model(t, time, f, freq)
	df.to_csv((data_path+'model_regressors/RSA_t%s_f%s_data.csv' %(t, f)))

# end of line
