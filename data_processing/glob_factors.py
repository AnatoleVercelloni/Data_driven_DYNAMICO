import numpy as np
import sys
import glob


if len(sys.argv) < 3:
    print("missing arguments.. you should specify one argument: dataset_name n_samples")
    sys.exit(0)

dataset_name = sys.argv[1]

path = '/lustre/fswork/projects/rech/psl/upu87pm/Data_driven_DYNAMICO/saved/normalization_factors/lr/'+dataset_name+'/'

nodes = len(glob.glob(path+'col1_max_glob_o_*'))

n_samples = int(sys.argv[2])

print("found ", nodes, " nodes for ", n_samples, "samples")

col1_sum_GLOB = np.sum(np.array([np.load(path + '/col1_sum_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_sum2_GLOB = np.sum(np.array([np.load(path + '/col1_sum2_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_min_GLOB = np.min(np.array([np.load(path + '/col1_min_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_max_GLOB = np.max(np.array([np.load(path + '/col1_max_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col1_mean_GLOB = col1_sum_GLOB/(nodes*n_samples)
col1_std_GLOB  = np.sqrt((col1_sum2_GLOB/n_samples - col1_mean_GLOB**2))

col2_sum_GLOB = np.sum(np.array([np.load(path + '/col2_sum_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_sum2_GLOB = np.sum(np.array([np.load(path + '/col2_sum2_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_min_GLOB = np.min(np.array([np.load(path + '/col2_min_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_max_GLOB = np.max(np.array([np.load(path + '/col2_max_glob_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_mean_GLOB = col2_sum_GLOB/(nodes*n_samples*60)
col2_std_GLOB  = np.sqrt((col2_sum2_GLOB/(n_samples*60) - col2_mean_GLOB**2))


path_ = path

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
col1_mean_GLOB_o = col1_sum_GLOB_o/(n_samples*nodes)
col1_std_GLOB_o  = np.sqrt((col1_sum2_GLOB_o/n_samples - col1_mean_GLOB_o**2))

col2_sum_GLOB_o = np.sum(np.array([np.load(path + '/col2_sum_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_sum2_GLOB_o = np.sum(np.array([np.load(path + '/col2_sum2_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_min_GLOB_o = np.min(np.array([np.load(path + '/col2_min_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_max_GLOB_o = np.max(np.array([np.load(path + '/col2_max_glob_o_'+str(rank)+'.npy') for rank in range(nodes)]), axis = 0)
col2_mean_GLOB_o = col2_sum_GLOB_o/(nodes*n_samples*60)
col2_std_GLOB_o  = np.sqrt((col2_sum2_GLOB_o/(n_samples*60) - col2_mean_GLOB_o**2))

path_ = path
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

print("global normalization factors saved in ", path)