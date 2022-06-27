####################################################################
# Script to do basic stats and regression on behavior data. Including transitional RT analysis
####################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import statsmodels.formula.api as smf
sns.set_context('paper', font_scale=1)
sns.set_style('white')
sns.set_palette("colorblind")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

included_subjects = ['128', '112', '108', '110', '120', '98', '86', '82', '115', '94', '76', '91', '80', '95', '121', '114', '125', '70',
'107', '111', '88', '113', '131', '130', '135', '140', '167', '145', '146', '138', '147', '176', '122', '118', '103', '142']
classes = ['dcb', 'dcr', 'dpb', 'dpr', 'fcb', 'fcr', 'fpb', 'fpr']

#these are paths for Argon cluster at UIowa, so specific to where we stored our data.
data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'
ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'  


def compile_behavior():
	''' conmpile dataframe for behavior data'''
	all_df = pd.DataFrame()
	for sub in included_subjects:

		metadata = pd.read_csv((ROOT+'RSA/%s_metadata.csv' %sub))
		metadata.loc[metadata.rt=='none', 'rt'] = np.nan  #recode no responses into nans
		metadata.loc[metadata.Subject_Respo=='none', 'Subject_Respo'] = np.nan #recode no responses into nans
		metadata.rt = metadata.rt.astype('float') #convert string to float
		metadata.Subject_Respo = metadata.Subject_Respo.astype('float') 
		metadata['RT_zscore'] = (metadata.rt - np.nanmean(metadata.rt)) / np.nanstd(metadata.rt) #zcore of RT, in case it is useful
		metadata['Accuracy'] = metadata.trial_Corr

		#determine swap or not swap version of context-feature mapping
		if 'version' in metadata.columns:  # only the swapped version has the version column
			metadata['swapped'] = 1
			### Dillan didnt code the Task info for the swapped version
			metadata.loc[metadata['cue'] == 'dcb', 'Task'] = 'Face'
			metadata.loc[metadata['cue'] == 'dcr', 'Task'] = 'Face'
			metadata.loc[metadata['cue'] == 'dpb', 'Task'] = 'Scene'
			metadata.loc[metadata['cue'] == 'dpr', 'Task'] = 'Scene'
			metadata.loc[metadata['cue'] == 'fcb', 'Task'] = 'Face'
			metadata.loc[metadata['cue'] == 'fcr', 'Task'] = 'Scene'
			metadata.loc[metadata['cue'] == 'fpb', 'Task'] = 'Face'
			metadata.loc[metadata['cue'] == 'fpr', 'Task'] = 'Scene'
		else:
			metadata['swapped'] = 0

			metadata['version'] = "Donut=Color,Filled=Shape"
		all_df = all_df.append(metadata)
	all_df = all_df.drop(columns=['pic_stim', 'img_path', 'index', 'cue_stim', 'trigs', 'Unnamed: 0'])  #clear book keeping vars.

	return all_df

#behavior_df = compile_behavior()
#behavior_df.to_csv('Data/behavior_data.csv')
behavior_df = pd.read_csv('Data/behavior_data.csv')

print('mean')
print(behavior_df.groupby(['sub']).mean().mean())
print('std')
print(behavior_df.groupby(['sub']).mean().std())


def rt_matrix(inputdf):
    ''' calcuate transtional RT matrix'''
    Subjects = inputdf['sub'].astype('int').unique()
    mat = np.zeros((8,8))
    subN = np.zeros((8,8))
    amat = np.zeros((8,8))
    all_mat = np.zeros((8,8, len(Subjects)))
    for idx, s in enumerate(Subjects):
        df = inputdf.loc[inputdf['sub'] == s]
        df = df.reset_index()

        #orgaznie matrix by texture, shape, color, return indx. Make sure the ordre here matches the "classes variable above"
        def get_positions(x):
            return {
                ('Filled', 'Polygon', 'red'): 7,
                ('Filled', 'Polygon', 'blue'): 6,
                ('Filled', 'Circle', 'red'): 5,
                ('Filled', 'Circle', 'blue'): 4,
                ('Donut', 'Polygon', 'red'): 3,
                ('Donut', 'Polygon', 'blue'): 2,
                ('Donut', 'Circle', 'red'): 1,
                ('Donut', 'Circle', 'blue'): 0,
            }[x]

        transitionRTs = np.zeros((8,8))
        trialN = np.zeros((8,8))
        for i in df.index:
            if i == 0:
                continue
            # if df.loc[i, 'cue']== df.loc[i-1, 'cue']: #remove cue repetition effect....?
            #     continue
            else:
                if df.loc[i, 'trial_Corr']==1: # only correct trials
                    if df.loc[i-1, 'trial_Corr']!=0: # exclude post error slowing trials
                        if df.loc[i, 'RT_zscore']<3:  #no overly slow trials
                            previous_condition = get_positions((df.loc[i-1, 'Texture'], df.loc[i-1, 'Shape'], df.loc[i-1, 'Color']))
                            current_condition = get_positions((df.loc[i, 'Texture'], df.loc[i, 'Shape'], df.loc[i, 'Color']))
                            diff_rt = df.loc[i,'rt'] #- df.loc[i-1, 'rt']
                            transitionRTs[previous_condition, current_condition] = transitionRTs[previous_condition, current_condition] + diff_rt
                            trialN[previous_condition, current_condition] =  trialN[previous_condition, current_condition] + 1

        smat = transitionRTs/trialN
        smat[np.isnan(smat)] = np.nanmean(smat)
        
        #here calcuate switch cost
        bmat = smat.copy()
        for k in np.arange(0,8):
            for j in np.arange(0,8):
                smat[k,j] = bmat[k,j] - bmat[j,j] #switch trial minus repeat trial

        #normalize
        smat = (smat - np.nanmean(smat)) / np.nanstd(smat)
        subaM = np.ones((8,8))
        subaM[np.isnan(smat)]=0
        subN = subN + subaM
        amat = amat + smat
        #stack subjects
        all_mat[:,:,idx] = smat
        #mat = np.hstack((mat,smat))

    amat = amat / subN #average across subjects
    mat = np.reshape(all_mat,(8,all_mat.shape[1]*all_mat.shape[2]))
    return mat, amat, all_mat

# swapdf = behavior_df.loc[behavior_df['swapped']==1]
# noswapdf = behavior_df.loc[behavior_df['swapped']==0]
# mat, amat, all_mat = rt_matrix(behavior_df)
#mat is stacked version
#amat is ave across subjects
#all_mat is the full 3D matrices
# for s in np.arange(np.shape(all_mat)[2]):
#     sns.heatmap(all_mat[:,:,s])
#     plt.show()


mat, amat, all_mat = rt_matrix(behavior_df)
diag_idx = np.eye(8)==1 # to get rid of eyes in the transition matrix which are repetition effects
amat[diag_idx] = np.nan 
sns.heatmap(amat, cmap = "bwr")
plt.savefig('Figures/behav.png')
#plt.show()

### create regression model
from gen_RSA import create_RSA_models
context_model, shape_model, color_model, identity_model, swapped_dimension_model, nonswapped_dimension_model, swapped_task_model, nonswapped_task_model, swapped_feature_model, nonswapped_feature_model, alternative_model = create_RSA_models()
diag_idx = np.eye(8)==0 # to get rid of eyes in the transition matrix which are repetition effects
rt_rsa_df = pd.DataFrame()
for s, subject in enumerate(behavior_df['sub'].astype('int').unique()):
    sdf = pd.DataFrame()
    sdf['TransitionRT'] = all_mat[:,:,s][diag_idx].flatten()
    sdf['Context'] = 1-context_model[diag_idx].flatten()
    sdf['Identity'] = 1-identity_model[diag_idx].flatten()
    sdf['alternative'] = 1-alternative_model[diag_idx].flatten()
    if behavior_df.loc[behavior_df['sub']==subject]['swapped'].unique()[0]==1:
        sdf['Feature'] = 1-swapped_dimension_model[diag_idx].flatten()
        sdf['Task'] = 1-swapped_task_model[diag_idx].flatten()
        sdf['Swap'] = 1
    else:
        sdf['Feature'] = 1-nonswapped_dimension_model[diag_idx].flatten()
        sdf['Task'] = 1-nonswapped_task_model[diag_idx].flatten()
        sdf['Swap'] = 0
    sdf['Subject'] = subject
    sdf['Steps'] = (sdf['Context'] + sdf['Feature'])
          
    rt_rsa_df = rt_rsa_df.append(sdf)

model = "TransitionRT ~ 0 + Identity + Task"  # identity is effectively intercept
base_model = smf.ols(formula = model, data = rt_rsa_df).fit() 
print(base_model.summary())

model = "TransitionRT ~ 0 + Identity + Task + Context + Feature"
test_model = smf.ols(formula = model, data = rt_rsa_df).fit()  
print(test_model.summary())

model = "TransitionRT ~ 0 + Identity + Task + Feature"
test_model2 = smf.ols(formula = model, data = rt_rsa_df).fit()  
print(test_model2.summary())

model = "TransitionRT ~ 0 + Identity + Task + Context + Feature + alternative"
test_model3 = smf.ols(formula = model, data = rt_rsa_df).fit()  
print(test_model3.summary())

model = "TransitionRT ~ 1 + alternative"
test_model4 = smf.ols(formula = model, data = rt_rsa_df).fit()  
print(test_model4.summary())

model = "TransitionRT ~ 1 + Task + alternative"
test_model4 = smf.ols(formula = model, data = rt_rsa_df).fit()  
print(test_model4.summary())

### nested model comparison
from statsmodels.stats.anova import anova_lm
print(anova_lm(base_model, test_model))
print(anova_lm(base_model, test_model2))
print(anova_lm(test_model, test_model2))



# end of script