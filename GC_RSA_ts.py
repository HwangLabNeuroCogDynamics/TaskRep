####################################################################
# Run GCA on the model betas ts.
# use statsmodel to run GC
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
####################################################################
import pandas as pd
import numpy as np
import os
import os.path
from statsmodels.tsa.stattools import grangercausalitytests

def compile_model_ts():
    ''' read in the regression outputs from RSA_GC_regression.py
    return two dfs, one with all trial by trial model paras, one subset trials by switching condition
    '''

    data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'
    time = np.arange(0,237)
    times = np.load(data_path+'times.npy')
    freqs = np.arange(0,30)
    df = pd.DataFrame()
    cdf = pd.DataFrame()
    for t in time:
        for f in freqs:
            pdf = pd.read_csv((data_path+'regression_results/RSA_GC_t%s_f%s_results.csv' %(t, f)))
            pdf['time'] = times[t]
            pdf['frequency'] = np.round(np.load(data_path+'freqs.npy')[f],2)
            pdf['Condition'] = 'All'
            df = pd.concat([df, pdf])
            # pdf = pd.read_csv((data_path+'regression_results/RSA_GC_condition_t%s_f%s_results.csv' %(t, f)))
            # pdf['time'] = times[t]
            # pdf['frequency'] = np.round(np.load(data_path+'freqs.npy')[f],2)
            # cdf = pd.concat([cdf, pdf])
    df.to_csv((data_path+'regression_results/RSA_TFR_GC_compiled_results.csv'))

    return df


def run_GCA(df, included_subjects, source, target, accuracy, condition='All', window_size = 'All'):
    ''' run GCA on the input dataframe
    Test "source" influencing "target" representation across all freq pair'''

    times = np.unique(df['time'])
    freqs = np.unique(df['frequency'])
    #subjects = np.unique(df['subject'])
    results = pd.DataFrame()

    if window_size == 'All':
        xy_data = np.zeros((len(times), 2))
        yx_data = np.zeros((len(times), 2)) # time in row, var in col
    else:
        xy_data = np.zeros((window_size, 2))
        yx_data = np.zeros((window_size, 2))

    i=0
    for subject in [included_subjects]:
        for ix, f2 in enumerate(freqs): # frequency of the source variable
            for iy, f1 in enumerate(freqs): #freuqecy of target var

                if window_size == 'All':
                    xy_data[:,0] = df.loc[(df['subject']==int(subject)) & (df['Accuracy']==accuracy) & (df['frequency']==f1) & (df['Condition']==condition), target+'_b'].values
                    xy_data[:,1] = df.loc[(df['subject']==int(subject)) & (df['Accuracy']==accuracy) & (df['frequency']==f2) & (df['Condition']==condition), source+'_b'].values #to test x cause y?
                    x2y = grangercausalitytests(xy_data,15) #max lag = 15, which is appx 150 ms.

                    #now flip to test reverse, y to x
                    yx_data[:,0] = xy_data[:,1] #now y cause x?
                    yx_data[:,1] = xy_data[:,0]
                    y2x = grangercausalitytests(yx_data,15)

                    #use BIC to determine optimal lag
                    bics = np.zeros(10)
                    for il, lag in enumerate(np.arange(1,11)):
                        bics[il] = x2y[lag][1][1].bic
                    opt_lag = np.where(bics==np.min(bics))[0][0]+1

                    results.loc[i, 'subject'] = subject
                    results.loc[i, 'condition'] = condition
                    results.loc[i, 'source'] = source
                    results.loc[i, 'target'] = target
                    results.loc[i, 'accuracy'] = accuracy
                    results.loc[i, 'source_frequency'] = f1
                    results.loc[i, 'target_frequency'] = f2
                    results.loc[i, 'model_order'] = opt_lag
                    results.loc[i, 'F'] = x2y[opt_lag][0]['params_ftest'][0] #grab f test, which is the ratio of residuals
                    results.loc[i, 'p'] = x2y[opt_lag][0]['params_ftest'][1]
                    results.loc[i, 'diffF'] = x2y[opt_lag][0]['params_ftest'][0] - y2x[opt_lag][0]['params_ftest'][0] # the diff score in F.
                    results.loc[i, 'reverseF'] = y2x[opt_lag][0]['params_ftest'][0]
                    i=i+1

                else:
                    if f1 ==f2: #only match freq
                        end_t = len(times) - window_size
                        for start_t in np.arange(0,end_t):
                            t_range = np.arange(start_t, start_t+window_size) #moving window
                            xy_data[:,0] = df.loc[(df['subject']==int(subject)) & (df['Accuracy']==accuracy) & (df['frequency']==f1) & (df['Condition']==condition), target+'_b'].values[t_range]
                            xy_data[:,1] = df.loc[(df['subject']==int(subject)) & (df['Accuracy']==accuracy) & (df['frequency']==f2) & (df['Condition']==condition), source+'_b'].values[t_range]
                            x2y = grangercausalitytests(xy_data,10)

                            #now flip to test reverse, y to x
                            yx_data[:,0] = xy_data[:,1]
                            yx_data[:,1] = xy_data[:,0]
                            y2x = grangercausalitytests(yx_data,10)

                            #use BIC to determine optimal lag
                            bics = np.zeros(10)
                            for il, lag in enumerate(np.arange(1,11)):
                                bics[il] = x2y[lag][1][1].bic
                            opt_lag = np.where(bics==np.min(bics))[0][0]+1

                            results.loc[i, 'subject'] = subject
                            results.loc[i, 'condition'] = condition
                            results.loc[i, 'source'] = source
                            results.loc[i, 'target'] = target
                            results.loc[i, 'accuracy'] = accuracy
                            results.loc[i, 'time'] = times[start_t]
                            results.loc[i, 'source_frequency'] = f1
                            results.loc[i, 'target_frequency'] = f2
                            results.loc[i, 'model_order'] = opt_lag
                            results.loc[i, 'F'] = x2y[opt_lag][0]['params_ftest'][0] #grab f test, which is the ratio of residuals
                            results.loc[i, 'p'] = x2y[opt_lag][0]['params_ftest'][1]
                            results.loc[i, 'diffF'] = x2y[opt_lag][0]['params_ftest'][0] - y2x[opt_lag][0]['params_ftest'][0] # the diff score in F.
                            results.loc[i, 'reverseF'] = y2x[opt_lag][0]['params_ftest'][0]
                            i=i+1

    return results



if __name__ == "__main__":

    #df = compile_model_ts() #save time by saving to disk.
    included_subjects = input()
    data_path = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/'
    df = pd.read_csv((data_path+'regression_results/RSA_TFR_GC_compiled_results.csv'))

    ### run through the models
    for condition in ['All']:
        for accu in ['Correct']:
            for window in ['All']:

                fn = data_path+'GC/%s_context-feature_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)
                if True: #not os.path.isfile(fn):
                    results = run_GCA(df, included_subjects, 'context', 'feature', accu, condition, window)
                    results.to_csv((data_path+'GC/%s_context-feature_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)))

                fn = data_path+'GC/%s_feature-task_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)
                if True: #not os.path.isfile(fn):
                    results = run_GCA(df, included_subjects, 'feature', 'task', accu, condition, window)
                    results.to_csv((data_path+'GC/%s_feature-task_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)))

                fn = data_path+'GC/%s_identity-task_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)
                if True: #not os.path.isfile(fn):
                    results = run_GCA(df, included_subjects, 'identity', 'task', accu, condition, window)
                    results.to_csv((data_path+'GC/%s_identity-task_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)))

                fn = data_path+'GC/%s_context-identity_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)
                if True: #not os.path.isfile(fn):
                    results = run_GCA(df, included_subjects, 'context', 'identity', accu, condition, window)
                    results.to_csv((data_path+'GC/%s_context-identity_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)))

                fn = data_path+'GC/%s_context-task_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)
                if True: #not os.path.isfile(fn):
                    results = run_GCA(df, included_subjects, 'context', 'task', accu, condition, window)
                    results.to_csv((data_path+'GC/%s_context-task_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)))

                fn = data_path+'GC/%s_feature-identity_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)
                if True: #not os.path.isfile(fn):
                    results = run_GCA(df, included_subjects, 'feature', 'identity', accu, condition, window)
                    results.to_csv((data_path+'GC/%s_feature-identity_%s_%s_%s_GC.csv' %(included_subjects, condition, accu, window)))

# end of line
