## May 17 2022 re-preprocess thalhi data in response to reviewers
## Changes: only lowpass filtering, only processing probe epochs

import mne
from mne.datasets import sample
from mne import io
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import path
import pickle
import fnmatch
#import ipywidgets
#from ipywidgets import widgets

## MNE ver 0.18 i think
#import kai's test data
ROOT = '/home/dcellier/RDSS/ThalHi_data/EEG_data/'
ROOT_raw=ROOT+'eeg_raw/'
ROOT_behav=ROOT+'behavioral_data/'
ROOT_proc='/data/backed_up/shared/ThalHi_data/eeg_preproc_RespToReviewers/'
subject_files=[]
exclude_subs = ['308','137','143','3986','4041','4032','4036','4093','4072','73','96','200','201','203'] #subjects to exclude # 137 only temporarily skippied
    #note these are patients with brain lesions so not included in the study
for filename in os.listdir(ROOT_raw): #compiling the subjects downloaded from MIPDB
	if not filename=='realpeople' :#and "sub_130" in filename: #143
		subject_files.append(filename)
s_file_copy=subject_files[:]
for thisSub in s_file_copy:
	sub_name=thisSub.split('_')[1]
	print(sub_name)
	if os.path.exists(ROOT_proc+sub_name+'/') or (sub_name in exclude_subs):
		subject_files.remove(thisSub)
print(subject_files)

for sub in subject_files: # insert for loop through subject list here
	if 'sub_4032' in sub: # this subject is formatted a bit differently-- each run is 1 BDF, so they must be concatenated
		raws_list=[]
		for run in range(1,6):
			thisRun=ROOT_raw+'sub_4032_thalHi_07_16_20/4032_run'+str(run)+'.bdf'
			thisRunRaw=mne.io.read_raw_edf(thisRun,montage=mne.channels.read_montage('biosemi64'),preload=True)
			raws_list.append(thisRunRaw)
		raw=mne.concatenate_raws(raws_list,preload=True)
	else:
		raw_file=ROOT_raw+sub
		raw=mne.io.read_raw_bdf(raw_file,preload=True)
		
		#raw.plot(n_channels=72)
		#print(sub_name)
	sub_name=sub.split('_')[1]
	pattern=sub_name+'_00*_Task_THHS_2019_*_*_*.csv'#sub+'_alpha_pilot_01_20*_*_*_*.csv'
	pattern3=sub_name+'_00*_Task_THHS_2020_*_*_*.csv'
	pattern2=sub_name+'_00*_Task_THHS_swapped_2019_*_*_*.csv'
	behav_files=pd.DataFrame()
	for f in os.listdir(ROOT_behav):
		if fnmatch.fnmatch(f,pattern) or fnmatch.fnmatch(f,pattern2) or fnmatch.fnmatch(f,pattern3):
			print(f)
			behav_file=pd.read_csv(ROOT_behav+f,engine='python')
			behav_files=behav_files.append(behav_file,ignore_index=True)
	behav_files.dropna(axis=0,inplace=True,how='any')
	behav_files=behav_files.sort_values(['block','trial_n']).reset_index()
	print(behav_files)


	# # Re-reference, apply high and low pass filters (1 and 50) # # # # # # # # # #

	raw_f=raw.copy()
	raw_f.filter(None,50)
	raw_f.set_channel_types({'EXG1':'emg','EXG2':'emg','EXG3':'eog','EXG4':'eog','EXG5':'eog','EXG6':'eog',
                        'EXG7':'ecg','EXG8':'emg'})
	raw_f.set_montage(montage=mne.channels.make_standard_montage('biosemi64'))

	#raw_f.plot(n_channels=72)
	raw_f.plot_psd()

		# # # # # # selecting bad electrodes # # # # # #
	raw_fi=raw_f.copy()
	raw_DataCH=(raw_fi.copy()).drop_channels(['EXG1', 'EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8'])
	tryAgain=1
	while tryAgain:
		raw_DataCH.plot(block=True,scalings=0.0001) #pauses script while i visually inspect data and select which channels to delete
		bads=raw_DataCH.info['bads']
		text_msg2=input('The channels you marked as bad are: '+str(bads)+' \n Are you ready to interpolate? [y/n]: ')
		if text_msg2=='y':
			raw_fi.info['bads']=bads
			raw_fi=raw_fi.interpolate_bads()
			tryAgain=0
		elif text_msg2=='n':
			tryAgain=1
		else:
			print('invalid entry: '+text_msg2)
			tryAgain=1
	#raw_fi.plot(title='beforeRef',clipping=None)
	print(raw_fi.get_data(stop=20))
	all_eeg_elecs_list = raw_fi.copy().pick_types(eeg=True).info['ch_names']
	raw_fir= raw_fi.set_eeg_reference(ref_channels=all_eeg_elecs_list,ch_type='eeg')#ref_channels=all_eeg_elecs_list[:3], ch_type='eeg').apply_proj()
	print(raw_fir.get_data(stop=20))
	#raw_fi.copy().pick_types(eeg=True).info['ch_names'] 'average')#['EXG1', 'EXG2'])#,'EXG8'])#mastoids, nose -- nose we decided we didn't want to use to reref
	#raw_fir.plot(title='afterRef',clipping=None,block=True,)


	# # Finding Events (triggers) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	if (sub_name=='143') or (sub_name=='137'):
		events=mne.find_events(raw_fir,verbose=True,min_duration=(2/512))
	else:
		events = mne.find_events(raw_fir, verbose=True)

	raw_fe=raw_fir.copy() # raw_fe was originally the 2 epoched data
	#raw_fe.plot()

	# # Looping through conditions, epoching # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # #

	probe_event_id={'faceProbe_trig':151,'sceneProbe_trig':153}
	response_event_id={'subResp_trig':157,'subNonResp_trig':155}
	cue_event_id={'IDS_trig':101,'Stay_trig':105, 'EDS_trig':103}
	 ## triggers for emily's data: #cue_event_id={'Donut_Circle_blue_trig': 107,
 #'Donut_Circle_red_trig': 105,
 #'Donut_Polygon_blue_trig': 103,
 #'Donut_Polygon_red_trig': 101,
 #'Filled_Circle_blue_trig': 115,
 #'Filled_Circle_red_trig': 113,
 #'Filled_Polygon_blue_trig': 111,
 #'Filled_Polygon_red_trig': 109}

	if sub_name=='203':
		cue_tmin, cue_tmax = -0.8, 2 # cue is 2 seconds long for this sub
		probe_tmin, probe_tmax = -0.8,5.4 # probe is 2 seconds long +3.4 min ITI
		#response_tmin,response_tmax=-0.5,2
		baseline = (None, -0.3)
		epCond={}
		print('\n\n\n Epoching Conds \n ')
		epCond['cue_events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=cue_event_id, tmin=cue_tmin,tmax=cue_tmax,metadata=behav_files)
		epCond['probe events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=probe_event_id, tmin=probe_tmin, tmax=probe_tmax,metadata=behav_files)

	elif sub_name=='200':
		cue_tmin, cue_tmax = -0.8, 1 # cue is 1 seconds long for this sub
		probe_tmin, probe_tmax = -0.8,6.4 # probe is 3 secs long +3.4 ITI
		#response_tmin,response_tmax=-0.5,2
		baseline = (None, -0.3)
		epCond={}
		print('\n\n\n Epoching Conds \n ')
		epCond['cue_events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=cue_event_id, tmin=cue_tmin,tmax=cue_tmax,metadata=pd.read_csv(ROOT_behav+'sub200_allTrialsallBlocks_truncatedCUES.csv',engine='python'))
		epCond['probe events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=probe_event_id, tmin=probe_tmin,tmax=probe_tmax,metadata=pd.read_csv(ROOT_behav+'sub200_allTrialsallBlocks_truncatedPROBES.csv',engine='python'))

	elif sub_name=='201':
		cue_tmin, cue_tmax = -0.8, 2.5 # cue is 2 seconds long for this sub
		probe_tmin, probe_tmax = -0.8,8.4 # probe is 3 secs long in first block, second block 5 seconds long, +3.4 ITI
		#response_tmin,response_tmax=-0.5,2
		baseline = (None, -0.3)
		epCond={}
		print('\n\n\n Epoching Conds \n ')
		epCond['cue_events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=cue_event_id, tmin=cue_tmin,tmax=cue_tmax,metadata=behav_files)
		epCond['probe events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=probe_event_id, tmin=probe_tmin, tmax=probe_tmax,metadata=behav_files)

	else:
		cue_tmin, cue_tmax = -0.8, 1.5  # 800 ms before event, and then 2.5 seconds afterwards cue is 1 sec + 1.5 delay
		probe_tmin, probe_tmax = -0.8,3 # -800 and 2 second probe/response period
		response_tmin,response_tmax=-0.5,1.5 # probably won't analyze this but might as well have it
		baseline = (None, -0.3) #baseline correction applied with mne.Epochs, this is starting @ beginning of epoch ie -0.8
		epCond={}
		print('\n\n\n Epoching Conds \n ')
		#epCond['cue_events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=cue_event_id, tmin=cue_tmin,tmax=cue_tmax,metadata=behav_files)
		epCond['probe events']=mne.Epochs(raw_fe, events=mne.pick_events(events,include=[151,153]), baseline=baseline, event_id=probe_event_id, tmin=probe_tmin, tmax=probe_tmax,metadata=behav_files)
		#epCond['response events']=mne.Epochs(raw_fe, events=events, baseline=(0,None), event_id=response_event_id, tmin=response_tmin, tmax=response_tmax,metadata=behav_files)
		# changed the baseline correction for this one because it doesn't make a whole lot of sense to baseline correct to -500 w a motor response?


	# # Inspect and reject bad epochs # # # # # # # # # # # #  # # # # # # # # # #
	# # AND ICA on Raw data # # # # # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # #
	our_picks=mne.pick_types(raw_fe.info,meg=False,eeg=True,eog=False)#mne.pick_types(raw_fi.info,meg=False,eeg=True,eog=True,emg=True,stim=True,ecg=True)

	layout=raw_fe.get_montage()

	EOG_channels=['EXG3', 'EXG4', 'EXG5', 'EXG6']
	ECG_channels=['EXG7']

	for ev in epCond.keys():
		plotEpoch=1
		while plotEpoch:
			print('plotting ' +ev)
			keep_plotting=1
			while keep_plotting:
				thisEp=[]
				thisEp=epCond[ev].copy()
				thisEp.load_data()
				print(thisEp)
				thisEp.plot(block=True, title="SELECT BAD EPOCHS: "+ev, n_epochs=6,n_channels=15,scalings={'eeg':0.0002})
				bads=input('Are you sure you want to continue? [y/n]: ')
				if bads=='y':
					#epCond[ev]=thisEp
					keep_plotting=0
				elif bads=='n':
					continue
				else:
					print('oops, please indicate which epochs you would like to reject')
			### ICA ###
			thisCond=thisEp.copy()
			thisCond.set_montage(layout)
			ica=ICA(n_components=64,random_state=25)
			ica.fit(thisCond,picks=our_picks)
			eog_ic=[]
			for ch in EOG_channels:
				#find IC's attributable to EOG artifacts
				eog_idx,scores=ica.find_bads_eog(thisCond,ch_name=ch)
				eog_ic.append(eog_idx)
			ecg_ic=[]
			for ch in ECG_channels: # find IC's attributable to ECG artifacts
				ecg_idx,scores=ica.find_bads_ecg(thisCond,ch_name=ch)
				ecg_ic.append(ecg_idx)
			reject_ic=[]
			for eog_inds in eog_ic:
				for ele in eog_inds:
					if (ele not in reject_ic) and (ele <= 31):
						reject_ic.append(ele)
			for ecg_inds in ecg_ic:
				for ele in ecg_inds:
					if (ele not in reject_ic) and (ele <= 31):
						reject_ic.append(ele) #add these IC indices to the list of IC's to reject
			ica.exclude=[]
			#ica.exclude.extend(reject_ic)
			plotICA=1
			while plotICA:
				ica.plot_components(picks=range(12),ch_type=None,cmap='interactive',inst=thisCond)# changed to ch_type=None from ch_type='EEG'because this yielded an error
				#ica.plot_components(picks=range(32,64),ch_type=None,cmap='interactive',inst=thisCond)
				input('The ICs marked for exclusion are: '+str(ica.exclude)+ '\n Press enter.')
				thisCond.load_data()
				thisCond.copy().plot(title=ev+': BEFORE ICA',n_epochs=5,n_channels=30)
				thisCond_copy=thisCond.copy()
				thisCond_Ic=ica.apply(thisCond_copy,exclude=ica.exclude)
				thisCond_Ic.plot(block=True, title=ev+': AFTER ICA',n_epochs=5,n_channels=30)
				verification_ic=input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')
				if verification_ic=='y':
					plotICA=0
				else:
					print('Please select which ICs you would like to reject')
			save_ep=input('Save this epoch? Entering "no" will take you back to epoch rejection for this condition. [y/n]: ')
			if save_ep=='y':
				plotEpoch=0
		#ica.plot_components(picks=range(25),ch_type=None,inst=thisEp)
		thisCond_copy=thisCond.copy()
		ica.apply(thisCond_copy)
		thisCond_copy.drop_channels(['EXG1', 'EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8'])
		epCond[ev]=thisCond_copy

	print('\n\n\n\n SAVING OUT INFO ~~~~~~~ \n\n')
	save_path=ROOT_proc+sub_name+'/'
	os.mkdir(save_path)


	for event in epCond.keys():
		event_name=event.split('_')[0]
		thisEp=epCond[event]
		thisEp.save(save_path+event_name+'-epo.fif')
		os.chmod(save_path+event_name+'-epo.fif',0o660)
	os.chmod(save_path,0o2770)
	ask=1
	exit_loop=0
	while ask:
		end_msg=input('Continue on to next sub? [y/n]')
		if end_msg=='y':
			ask=0
		elif end_msg=='n':
			exit_loop=1
			ask=0
		else:
			print('Oops! Please answer y/n')

	if exit_loop:
		break
