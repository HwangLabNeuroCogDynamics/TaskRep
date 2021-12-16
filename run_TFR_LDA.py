####################################################################
# Script to run time frequency decomp then linear discrimnation analysis
####################################################################
from ThalHiEEG import *
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, cross_val_predict, KFold
from scipy.stats import zscore
import statsmodels.formula.api as smf
from scipy.special import logit
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datetime import datetime
import numpy as np
from numpy import average, std
import pickle
from numpy.random import random, randint, normal, shuffle,uniform
import scipy
from scipy import sparse
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.stats.mstats import zscore
import seaborn as sns
import fnmatch
import os  # handy system and path functions
import sys  # to get file system encoding
import csv
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import mne
from mne.time_frequency import tfr_morlet

n_jobs = 2
included_subjects = input()  #wait for input to determine which subject to run

classes = ['dcb', 'dcr', 'dpb', 'dpr', 'fcb', 'fcr', 'fpb', 'fpr']
ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'

# included_subjects = ['128', '112', '108', '110', '120', '98', '86', '82', '115', '94', '76', '91', '80', '95', '121', '114', '125', '70',
# '107', '111', '88', '113', '131', '130', '135', '140', '167', '145', '146', '138', '147', '176', '122', '118', '103', '142']

def run_TFR(sub):
	''' run frequency decomp and save, sub by sub'''
	this_sub_path=ROOT+ 'eeg_preproc/' +str(sub)
	all_probe = {}
	for condition in ['IDS', 'EDS', 'stay']:
		all_probe[condition] =  mne.read_epochs(this_sub_path+'/probe_'+condition+'_events-epo.fif')
	all_probe = mne.concatenate_epochs([all_probe['IDS'], all_probe['EDS'], all_probe['stay']]) # can use trial_n in metadata to rank data.
	all_probe.baseline=None
	#all_probe = all_probe.resample(128,'auto')
	#all_probe.save(this_sub_path+'/all_epochs-epo.fif', overwrite=True)

	#merge meta data
	for index, row in all_probe.metadata.iterrows():
		try:
			all_probe.metadata.loc[index, 'Rule_Switch'] = all_cue.metadata.loc[(all_cue.metadata.block==row.block) & (all_cue.metadata.trial_n==row.trial_n), 'Rule_Switch'].values[0]
		except:
			all_probe.metadata.loc[index, 'Rule_Switch'] = 'No_Switch'

	all_probe.save(this_sub_path+'/all_probe_epochs-epo.fif', overwrite=True)
	all_probe.metadata.to_csv((ROOT+'RSA/%s_metadata.csv' %sub))

	freqs = np.logspace(*np.log10([1, 40]), num=30)
	n_cycles = np.logspace(*np.log10([3, 12]), num=30)

	tfr = tfr_morlet(mirror_evoke(all_probe), freqs=freqs,  n_cycles=n_cycles,
	average=False, use_fft=True, return_itc=False, decim=5, n_jobs=n_jobs)

	tfr = tfr.crop(tmin = -.8, tmax = 1.5) #trial by chn by freq by time, chop at 1.5s
	np.save((ROOT+'RSA/%s_times' %sub), tfr.times)
	tfr.save((ROOT+'RSA/%s_tfr.h5' %sub), overwrite=True)


def run_classification(x_data, y_data, tfr_data, permutation = False):
	''' clasification analysis with LDA, using all freqs as features, so this is temporal prediction analysis (Fig 3A)'''

	# do this 10 times then average?
	lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')

	n_scores = np.zeros((y_data.shape[0],8,10))
	for n in np.arange(10):
		cv = KFold(n_splits=4, shuffle=True, random_state=6*n)

		if permutation:
			permuted_order = np.arange(x_data.shape[0])
			np.random.shuffle(permuted_order)
			xp_data = x_data[permuted_order,:,:]  #x_data is trial by chn by freq, we permute the trial order
			#need to vectorize data. Feature space is trial by ch by freq = 4xx * 64 * 20, need to wrap it into 4xx * 1280 (collapsing chn by freq)
			xp_data = zscore(np.reshape(xp_data, (tfr_data.shape[0], tfr_data.shape[1]*tfr_data.shape[2])))
			scores = cross_val_predict(lda, xp_data, y_data, cv=cv, method='predict_proba', pre_dispatch = n_jobs) #logit the probability
		else:
			#need to vectorize data. Feature space is trial by ch by freq = 4xx * 64 * 20, need to wrap it into 4xx * 1280 (collapsing chn by freq)
			xp_data = zscore(np.reshape(x_data, (tfr_data.shape[0], tfr_data.shape[1]*tfr_data.shape[2])))
			scores = cross_val_predict(lda, xp_data, y_data, cv=cv, method='predict_proba', pre_dispatch = n_jobs) #logit the probability
			n_scores[:,:,n] = scores

	#logit transform prob.
	n_scores = np.mean(n_scores,axis=2) # average acroos random CV runs
	n_scores = logit(n_scores) #logit transform probability
	n_scores[n_scores==np.inf]=36.8 #float of .9999999999xx
	n_scores[n_scores==np.NINF]=-36.8 #float of -.9999999999xx

	##calcuate accuracy
	# i=0
	# for ix, ip in enumerate(np.argmax(n_scores,axis=1)):
	# 	if y_data[ix] == classes[ip]:
	# 		i=i+1
	# print("%0.2f accuracy for timepoint %0.2f" %(i/len(y_data)*100, tfr.times[t]))

	return n_scores


def run_full_TFR_classification(x_data, y_data, tfr_data):
	''' clasification analysis with LDA, inputing one frequency at a time. Time-frequency prediction (Figure 3B)
	Results then feed to RSA regression (Figure 4)
	'''

	# do this 10 times then average?
	lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')

	n_scores = np.zeros((y_data.shape[0],8,10))
	for n in np.arange(10):
		cv = KFold(n_splits=4, shuffle=True, random_state=6*n)

		#No need to vectorize data. Feature space is trial by ch
		#xp_data = zscore(np.reshape(x_data, (x_data.shape[0], x_data.shape[1])))
		scores = cross_val_predict(lda, x_data, y_data, cv=cv, method='predict_proba', pre_dispatch = n_jobs) #logit the probability
		n_scores[:,:,n] = scores

	#logit transform prob.
	n_scores = np.mean(n_scores,axis=2) # average acroos random CV runs
	n_scores = logit(n_scores) #logit transform probability
	n_scores[n_scores==np.inf]=36.8 #float of .9999999999xx
	n_scores[n_scores==np.NINF]=-36.8 #float of -.9999999999xx

	#calcuate accuracy
	i=0
	for ix, ip in enumerate(np.argmax(n_scores,axis=1)):
		if y_data[ix] == classes[ip]:
			i=i+1
	print("%0.2f accuracy for timepoint %0.2f at frequency %0.2f" %(i/len(y_data)*100, tfr.times[t], tfr.freqs[f]))

	return n_scores




for sub in [included_subjects]:

	# datetime object containing current date and time
	now = datetime.now()
	print("starting time: ", now)
	# dd/mm/YY H:M:S
	# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	# print("date and time =", dt_string)

	print('-------')
	print(('running subject %s' %sub))
	print('-------')

	run_TFR(sub) # already did
	tfr = mne.time_frequency.read_tfrs((ROOT+'RSA/%s_tfr.h5' %sub))[0]

	#average within bands? #(1-3 Hz for the delta-band, 4-7 Hz for the theta-band, 8-12 Hz for the alphaband, 13-30 Hz for the beta-band, 31-35 Hz for the gamma-band)
	# delta = [1, 3]
	# theta = [4, 7]
	# alpha = [8, 12]
	# beta = [13, 30]
	# gamma = [30, 35]
	# bands = [delta,theta,alpha,beta,gamma]
	# tfr_data = np.zeros((tfr.data.shape[0], tfr.data.shape[1], 5, tfr.data.shape[3]))
	# for i, band in enumerate(bands):
	# 	tfr_data[:,:,i,:] = np.mean(tfr.data[:,:, (freqs>=band[0]) & (freqs<=band[1]), ], axis=2)
	tfr_data = tfr.data

	#########################################################################################################
	##### linear discrimination analysis
	#########################################################################################################
	permutation = True # run permutation?
	full_TFR = True # run full TFR prediction (each TFR bin per prediction model)

	if permutation:
		num_permutations = 10
		trial_prob = np.zeros((tfr_data.shape[0],tfr_data.shape[3],8,num_permutations))
	elif not full_TFR:
		trial_prob = np.zeros((tfr_data.shape[0],tfr_data.shape[3],8)) #trial by time by labels (8)
	elif full_TFR:
		trial_prob = np.zeros((tfr_data.shape[0],tfr_data.shape[2], tfr_data.shape[3],8)) #trial by freq by time by labels (8)

	for t in np.arange(tfr.times.shape[0]):
		y_data = tfr.metadata.cue.values.astype('str')
		x_data = tfr_data[:,:,:,t]

		if permutation:
			for n_p in np.arange(num_permutations):
				n_scores = run_classification(x_data, y_data, tfr_data, permutation = True)
				trial_prob[:,t,:,n_p] = n_scores
		elif not full_TFR:
			n_scores = run_classification(x_data, y_data, tfr_data, permutation = False)
			trial_prob[:,t,:] = n_scores
		elif full_TFR:
			for f in np.arange(tfr_data.shape[2]):
				x_data = tfr_data[:,:,f,t]
				n_scores = run_full_TFR_classification(x_data, y_data, tfr_data)
				trial_prob[:,f,t,:] = n_scores

	#saving posterior prob into numpy array
	if permutation:
		np.save((ROOT+'/RSA/%s_prob_permutation' %sub), trial_prob)
	else:
		np.save((ROOT+'/RSA/%s_tfr_prob' %sub), trial_prob)

	now = datetime.now()
	print("Done at:", now)



#end of line
