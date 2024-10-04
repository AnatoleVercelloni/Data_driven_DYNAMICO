import numpy as np
import sys

path = '/gpfswork/rech/psl/upu87pm/_hyrid_climate_modelling_/data_processing/normalization_factors/all_low_res/'
nodes = 6
n_samples = 210240*384

col1_sum_GLOB = np.sum(np.array([np.load(path + '/col1_sum_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_sum2_GLOB = np.sum(np.array([np.load(path + '/col1_sum2_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_min_GLOB = np.min(np.array([np.load(path + '/col1_min_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_max_GLOB = np.max(np.array([np.load(path + '/col1_max_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_mean_GLOB = col1_sum_GLOB/(n_samples)
col1_std_GLOB  = np.sqrt((col1_sum2_GLOB/n_samples - col1_mean_GLOB**2))

col2_sum_GLOB = np.sum(np.array([np.load(path + '/col2_sum_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_sum2_GLOB = np.sum(np.array([np.load(path + '/col2_sum2_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_min_GLOB = np.min(np.array([np.load(path + '/col2_min_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_max_GLOB = np.max(np.array([np.load(path + '/col2_max_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_mean_GLOB = col2_sum_GLOB/(n_samples*60)
col2_std_GLOB  = np.sqrt((col2_sum2_GLOB/(n_samples*60) - col2_mean_GLOB**2))

path_ = '/gpfswork/rech/psl/upu87pm/_hyrid_climate_modelling_/data_processing/normalization_factors/glob_low_res/'

np.save(path_ + 'col1_sum_GLOB.npy', col1_sum_GLOB)
np.save(path_ + 'col1_sum2_GLOB.npy', col1_sum2_GLOB)
np.save(path_ + 'col1_min_GLOB.npy', col1_min_GLOB)
np.save(path_ + 'col1_max_GLOB.npy', col1_max_GLOB)
np.save(path_ + 'col1_mean_GLOB.npy', col1_mean_GLOB)
np.save(path_ + 'col1_std_GLOB.npy', col1_std_GLOB)

np.save(path_ + 'col2_sum_GLOB.npy', col2_sum_GLOB)
np.save(path_ + 'col2_sum2_GLOB.npy',col2_sum2_GLOB)
np.save(path_ + 'col2_min_GLOB.npy', col2_min_GLOB)
np.save(path_ + 'col2_max_GLOB.npy', col2_max_GLOB)
np.save(path_ + 'col2_mean_GLOB.npy',col2_mean_GLOB)
np.save(path_ + 'col2_std_GLOB.npy', col2_std_GLOB)





col1_sum_GLOB_o = np.sum(np.array([np.load(path + '/col1_sum_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_sum2_GLOB_o = np.sum(np.array([np.load(path + '/col1_sum2_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_min_GLOB_o = np.min(np.array([np.load(path + '/col1_min_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_max_GLOB_o = np.max(np.array([np.load(path + '/col1_max_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_mean_GLOB_o = col1_sum_GLOB_o/(n_samples)
col1_std_GLOB_o  = np.sqrt((col1_sum2_GLOB_o/n_samples - col1_mean_GLOB_o**2))

col2_sum_GLOB_o = np.sum(np.array([np.load(path + '/col2_sum_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_sum2_GLOB_o = np.sum(np.array([np.load(path + '/col2_sum2_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_min_GLOB_o = np.min(np.array([np.load(path + '/col2_min_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_max_GLOB_o = np.max(np.array([np.load(path + '/col2_max_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_mean_GLOB_o = col2_sum_GLOB_o/(n_samples*60)
col2_std_GLOB_o  = np.sqrt((col2_sum2_GLOB_o/(n_samples*60) - col2_mean_GLOB_o**2))

path_ = '/gpfswork/rech/psl/upu87pm/_hyrid_climate_modelling_/data_processing/normalization_factors/glob_low_res/'

np.save(path_ + 'col1_sum_GLOB_o.npy', col1_sum_GLOB_o)
np.save(path_ + 'col1_sum2_GLOB_o.npy', col1_sum2_GLOB_o)
np.save(path_ + 'col1_min_GLOB_o.npy', col1_min_GLOB_o)
np.save(path_ + 'col1_max_GLOB_o.npy', col1_max_GLOB_o)
np.save(path_ + 'col1_mean_GLOB_o.npy', col1_mean_GLOB_o)
np.save(path_ + 'col1_std_GLOB_o.npy', col1_std_GLOB_o)

np.save(path_ + 'col2_sum_GLOB_o.npy', col2_sum_GLOB_o)
np.save(path_ + 'col2_sum2_GLOB_o.npy',col2_sum2_GLOB_o)
np.save(path_ + 'col2_min_GLOB_o.npy', col2_min_GLOB_o)
np.save(path_ + 'col2_max_GLOB_o.npy', col2_max_GLOB_o)
np.save(path_ + 'col2_mean_GLOB_o.npy',col2_mean_GLOB_o)
np.save(path_ + 'col2_std_GLOB_o.npy', col2_std_GLOB_o)

