import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast, os
import bambi as bmb
import pymc as pm
import arviz as az
import scipy.stats as stat
from collections import Counter


def clean_data(filename):
    # print(filename)

    try:
        data = pd.read_csv('data/' + filename)

        # Drop instruction rows by dropping rows with missing data in column: 'blocks.thisRepN'
        data = data.dropna(subset=['blocks.thisRepN']).reset_index(drop=True)

        if len(data) < 1400:
            raise TypeError('Incomplete Data')

        #Compute accuracy as True if number of response values are the same as number of stim values
        # print(data['stim'])
        data['accuracy'] = [True if (len(ast.literal_eval(data['stim'][i])) == len(ast.literal_eval(data['key_resp.rt'][i]))) else False for i in range(len(data))]


        data['node_type'] = ['Boundary' if data['node idx'][i] in [0, 4, 5, 9, 10, 14] else 'Non Boundary' for i in range(len(data))]

        data['rt'] = [np.mean(ast.literal_eval(data['key_resp.rt'][i])) if data['accuracy'][i] else np.NaN for i in range(len(data))]
        data['transition_type'] = ['cross cluster' if (data['node_type'] == 'Boundary')[i] & (data['node_type'].shift() == 'Boundary')[i] else 'within cluster' for i in range(len(data))]

        if data['participant'][0]%4 == 0:
            data['walk_length'] = 0
        elif data['participant'][0]%4 == 1:
            data['walk_length'] = 3
        elif data['participant'][0]%4 == 2:
            data['walk_length'] = 6
        else:
            data['walk_length'] = 1400

        data['trial'] = np.arange(len(data))
        
        #Count the lag to indicate when the same stimulus was seen previously. Lag = 1 means the previous trial had the same stimulus.
        lag_counter_dict = Counter()
        lag = []

        for s in data['stim']:
            lag.append(lag_counter_dict[s])
            lag_counter_dict.update(lag_counter_dict.keys())
            lag_counter_dict[s] = 1
            # print(lag_counter_dict)
        data['lag'] = lag
        
    except:
        return None
    data['num_keypress'] = [len(ast.literal_eval(data['stim'][i])) for i in range(len(data))]

    return data[['participant', 'trial', 'blocks.thisRepN', 'accuracy', 'walk_length', 'node_type', 'transition_type', 'rt', 'stim', 'num_keypress', 'lag']]




# Reading data files
data_files = []
for f in os.listdir('data/'):
    if (f.startswith('3') & f.endswith('csv')):
        data_files.append(f)
        
        
#Cleaning data files
df_clean = pd.concat([clean_data(f) for f in data_files]).reset_index(drop = True)
df_clean['reset'] = 'False'
df_clean.loc[df_clean['trial'].values%(df_clean['walk_length'].values+1) == 0, 'reset'] = 'True'



df_clean_rt_outlier = df_clean[np.abs(stat.zscore(df_clean['rt'], nan_policy='omit')) < 3]
df_clean_rt_outlier['node_transition_type'] = df_clean_rt_outlier['node_type'] + ' ' + df_clean_rt_outlier['transition_type']

# df_clean_participant = df_clean_rt_outlier.groupby(['participant', 'blocks.thisRepN', 'walk_length', 'node_type', 'transition_type', 'num_keypress']).median().reset_index()

df_clean_rt_outlier['walk_length'] = df_clean_rt_outlier.walk_length.astype('str')
df_clean_rt_outlier['num_keypress'] = df_clean_rt_outlier['num_keypress'].astype(str)




#Removing the first trial because it may bias the reset parameter
df_clean_rt_outlier = df_clean_rt_outlier.loc[df_clean_rt_outlier['trial'] > 0].reset_index(drop=True)


#Grouping by trials
df_clean_rt_outlier_median = df_clean_rt_outlier.loc[df_clean_rt_outlier['trial'] > 0].reset_index(drop=True).groupby(['participant', 'trial', 'transition_type', 'walk_length', 'reset']).mean(numeric_only=True).reset_index()
df_clean_rt_outlier_median_cross = df_clean_rt_outlier_median.loc[df_clean_rt_outlier_median['transition_type'] == 'cross cluster'].reset_index(drop=True)
df_clean_rt_outlier_median_within = df_clean_rt_outlier_median.loc[df_clean_rt_outlier_median['transition_type'] == 'within cluster'].reset_index(drop=True)
diff = df_clean_rt_outlier_median_cross.merge(df_clean_rt_outlier_median_within, on = ['participant', 'blocks.thisRepN', 'walk_length'])
# df_clean_rt_outlier_median_diff['diff'] = df_clean_rt_outlier_median.loc[df_clean_rt_outlier_median['transition_type'] == 'cross cluster', 'rt'].values - df_clean_rt_outlier_median.loc[df_clean_rt_outlier_median['transition_type'] == 'Non Boundary', 'rt'].values
diff['rt_diff'] = diff['rt_x'] - diff['rt_y']

#Specifying the bayes model
model = bmb.Model("rt_diff ~ blocks.thisRepN*walk_length +  (1|participant)", data = diff)

# model = bmb.Model("diff ~ reset + lag + blocks.thisRepN*walk_length + walk_length*transition_type +  (transition_type|participant) + (blocks.thisRepN|participant) + (num_keypress|participant)", data = df_clean_rt_outlier)
model.build()
# model.graph()
sample = model.fit(inference_method = 'nuts_numpyro')


#Plotting. Specifying order of plots
coords = {'walk_length_dim': ['1400', '6', '3'], 
          # 'walk_length:transition_type_dim':['1400, within cluster', '6, within cluster', '3, within cluster'],
          'blocks.thisRepN:walk_length_dim':['1400', '6', '3'],
         }
#Plot posteriors
az.plot_posterior(sample, point_estimate='median', ref_val=0, hdi_prob=.95, var_names='~participant', filter_vars='like', coords = coords)
plt.tight_layout()
plt.savefig('results/posteriors_bayesianmodel_0_params_rt_diff_trialgrouped.png', dpi = 600)

#Plot ridge plot
g = az.plot_forest(sample, var_names='~participant', filter_vars='like', kind = 'ridgeplot', ridgeplot_quantiles = [0.5], ridgeplot_alpha=0.2, ridgeplot_overlap=1, 
              coords = coords)
g[0].axvline(0, ls = '--', color = 'black')
plt.savefig('results/posteriors_bayesianmodel_0_params_rt_diff_trialgrouped.png', dpi = 600)

#Save the model
az.to_netcdf(sample, 'results/bayesian_model0_rt_diff_trialgrouped')