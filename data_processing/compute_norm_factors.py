import xarray as xr
import numpy as np
import glob
import os
import time
from multiprocessing import Pool, cpu_count, Manager
import sys

print("starting computing normalization factors")




if len(sys.argv) < 9:
    print("missing arguments.. you should specify 6 arguments: dataset_name resolution sride shift first_file last_file number_of_nodes rank")
    sys.exit(0)


name_processing = sys.argv[1]
resolution      = sys.argv[2]
stride          = int(sys.argv[3]) 
shift           = int(sys.argv[4])
first_file      = int(sys.argv[5])
last_file       = int(sys.argv[6])
nodes           = int(sys.argv[7])
rank            = int(sys.argv[8])    


if resolution != 'low' and resolution != 'high':
    print("wrong resolution! it is either low or high")
    sys.exit(0)

if stride < 1:
    print("stride should be a positive integer !")
    sys.exit(0)

if shift < 0:
    print("shift should be a non negative integer !")
    sys.exit(0)

if first_file < 0 or first_file > 210240:
    print("first_file should be between 0 and 210 240 !")
    sys.exit(0)

if last_file < 0 or last_file > 210240:
    print("last_file should be between 0 and 210 240 !")
    sys.exit(0)

if first_file >= last_file:
    print(" first_file should be lower than last_file !")
    sys.exit(0)

if nodes < 1:
    print("specified number of nodes is not valid! It has to be greater than 0!")
    sys.exit()
elif  nodes >=8:
    print("specified number of nodes is not valid! less than 8 is enough, keep it easy!")
    sys.exit()

if rank >= nodes:
    print("specified rank is too high ! Only ", nodes, " are availables")
    sys.exit()
    
print ("rank ", rank, " is running !")


#test0 => test stride
#test_all => test all
#all_stride_7 => all data stride 7 
#all_stride_97 => all data stride 97
#test_all_stride_set => with a shift of 3 
#all_stride_349_test => with a shift of  177
#first_dataset => 7 first years stride 7
#scoring_set: last year of data

#the fraction of file taken: 1/stride
#shift = 0
# stride = 1
# cut_s = 183960 #210240
# cut_e = 210240 #183960

#210 240 for the last file
#183 960 for the 7th tear

#the number of cpus -> number of files at the end
n_npy = 80


# if name_processing == 'first_dataset':
#     stride = 7     # one file over 7
#     cut_s = 0      # from the begining 
#     cut_e = 183960 # to the 7th year 

# elif name_processing == 'scoring_set':
#     stride = 1     # every file
#     cut_s = 183960 # from the 7th year 
#     cut_e = 210240 # to the end


#556 input scalars
vars_mli      = ['state_t','state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v',
                     'state_ps', 'pbuf_SOLIN','pbuf_LHFLX', 'pbuf_SHFLX',  'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS',
                     'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP',
                     'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND', 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']
                     
#368 output scalars
vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC',
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
n_var    = 556
n_scalar = 16
n_col    = 9
lev      = 60   

scalar_idx = list(range(6*60,6*60+16))
col_idx    = list(range(6*60)) + list(range(6*60+16, 556))

# print("for ", len(scalar_idx), " scalar values and ", len(col_idx), "col values")


# first normalization of the values (one per one)
col1_sum = np.zeros(n_var)
col1_sum2 = np.zeros(n_var)
col1_min  = np.zeros(n_var)
col1_max  = np.zeros(n_var)

# second normalization of the col values (one per 60)
col2_sum = np.zeros(n_col)
col2_sum2 = np.zeros(n_col)
col2_min   = np.zeros(n_scalar)
col2_max   = np.zeros(n_scalar)

# third normalization of the col values (log)





def task(i):
    #each worker write 'file_per_npy' amount of data in a npy file in parallel

    file_per_npy = N//n_npy
    if (N%n_npy) != 0 and i == 0 : print("missing some file because ", n_npy, " is not factor of ", N)

    if i == 0: print("loading ", file_per_npy*2, "files  (input + output):  ")
    
    start_time = time.time()

    # loading input files
    ds = [xr.open_dataset(list_file[j], engine='netcdf4') for j in range(i*file_per_npy, (i+1)*file_per_npy, 1)]

    # concatenate them along a time dimension
    ds = xr.concat(ds, dim = 'time_counter')
    ds  = ds[vars_mli]



    # loading output files
    dso = [xr.open_dataset(list_file[j].replace(".mli.", ".mlo.")) for j in range(i*file_per_npy, (i+1)*file_per_npy, 1)]

    # concatenate them along a time dimension
    dso = xr.concat(dso, dim = 'time_counter')

    end_time = time.time()

    if i == 0: print(f'loading  took {end_time - start_time} s')

    # we have to construct the tendency variables for the output data
    dso['ptend_t']     = (dso['state_t']     - ds['state_t'])/1200     # T tendency [K/s]
    dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_q0002'] = (dso['state_q0002'] - ds['state_q0002'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_q0003'] = (dso['state_q0003'] - ds['state_q0003'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_u']     = (dso['state_u']     - ds['state_u'])/1200     # Q tendency [kg/kg/s]
    dso['ptend_v']     = (dso['state_v']     - ds['state_v'])/1200     # Q tendency [kg/kg/s]
    dso = dso[vars_mlo]

    # if i == 0: print("first sample in ds = ", ds['state_t'][0][0])

    L_var_i = []
    for var in vars_mli:
        arr = np.array(ds[var])
        if len(arr.shape) == 3:
            arr = arr.transpose((2,0,1))
        else:
            arr = arr.transpose((1,0))

        arr = arr.reshape((file_per_npy*ncol, -1), order = 'F')

        L_var_i.append(arr)

    
    ds_np = np.concatenate(L_var_i, axis = 1)

    L_var_o = []
    for var in vars_mlo:
        arr = np.array(dso[var])
        if len(arr.shape) == 3:
            arr = arr.transpose((2,0,1))
        else:
            arr = arr.transpose((1,0))

        arr = arr.reshape((file_per_npy*ncol, -1), order = 'F')

        L_var_o.append(arr)

    
    dso_np = np.concatenate(L_var_o, axis = 1)



    # convert to numpy array
    ds = ds_np
    dso = dso_np

    #if i == 0: print("shape = ", ds.shape, " first sample in na = ", ds[:4,:4])


    #if i == 0: print("input shape  ", ds.shape)
    #if i == 0: print("output shpae ", dso.shape)

    ds1 = ds
    dso1 = dso

    #ds1 = ds.transpose((1,0)).reshape((-1,556),order='C')#.reshape(-1, 556,order='F')

    # ds1 = ds.reshape((-1,ncol,556),order='F').transpose((1,0,2)).reshape(-1, 556,order='F')
    # dso1 = dso.reshape((-1,ncol,368),order='F').transpose((1,0,2)).reshape(-1, 368,order='F')

    #if i == 0: print("shape  ds1 = ", ds1.shape, "first sample in na = ", ds1[:4,:4])


    i#f i == 0: print("shape before compute anything :", ds1.shape)

    #compute first normalization factors

    #input
    col1_sum = np.sum(ds1, axis = 0)
    col1_sum2 = np.sum(ds1**2, axis = 0)
    col1_min = np.min(ds1, axis = 0)
    col1_max = np.max(ds1, axis = 0)


    #output
    col1_sum_o = np.sum(dso1, axis = 0)
    col1_sum2_o = np.sum(dso1**2, axis = 0)
    col1_min_o = np.min(dso1, axis = 0)
    col1_max_o = np.max(dso1, axis = 0)

    #if i == 0: print("shape after first normalization : input = ", col1_sum.shape, "   output = ", col1_sum_o.shape)

    #compute second normalization factors

    ds_col = ds1[:,col_idx].reshape((-1, lev, n_col),order='F')
    ds_col_o = dso1[:,:360].reshape((-1, lev, 6),order='F')

    #if i == 0: print("shape ds_col : input = ",  ds_col.shape, " output = ", ds_col_o.shape)

    #input
    col2_sum = np.sum(ds_col, axis = (0,1))
    col2_sum2 = np.sum(ds_col**2, axis = (0,1))
    col2_min = np.min(ds_col, axis = (0,1))
    col2_max = np.max(ds_col, axis = (0,1))

    #if i == 0: print("mean t = ", col2_sum[0]/(60*384))

    #output
    col2_sum_o = np.sum(ds_col_o, axis = (0,1))
    col2_sum2_o = np.sum(ds_col_o**2, axis = (0,1))
    col2_min_o = np.min(ds_col_o, axis = (0,1))
    col2_max_o = np.max(ds_col_o, axis = (0,1))

    #if i == 0: print("shape after second normalization : input =  ", col2_sum.shape, "   output = ", col2_sum_o.shape)

    np.save(save_dir + '/input_'+str(rank*n_npy + i).rjust(3,'0')+'.npy', ds1)
    np.save(save_dir + '/target_'+str(rank*n_npy + i).rjust(3,'0')+'.npy', dso1)

    if i == 0: print("data files saved at ", save_dir, " from ", rank*n_npy, " to ", (rank + 1)*(n_npy))


    return col1_sum, col1_sum2, col1_min, col1_max, col2_sum, col2_sum2, col2_min, col2_max, col1_sum_o, col1_sum2_o, col1_min_o, col1_max_o, col2_sum_o, col2_sum2_o, col2_min_o, col2_max_o



def prepro():
    

    #we split the data processing between the n_npy procs available
    start_time = time.time()
    with Pool() as pool:
        result = pool.map(task, range(n_npy))

        print("pool finished, starting reduction..")
        print("output pool shape : ", len(result), result[0][0].shape)

        #rearranging the means to have two lists of dictionary 
        #input
        L_col1_sum  = np.array([result[i][0] for i in range(n_npy)])
        L_col1_sum2 = np.array([result[i][1] for i in range(n_npy)])
        L_col1_min  = np.array([result[i][2] for i in range(n_npy)])
        L_col1_max  = np.array([result[i][3] for i in range(n_npy)])

        L_col2_sum  = np.array([result[i][4] for i in range(n_npy)])
        L_col2_sum2 = np.array([result[i][5] for i in range(n_npy)])
        L_col2_min  = np.array([result[i][6] for i in range(n_npy)])
        L_col2_max  = np.array([result[i][7] for i in range(n_npy)])

        #output
        L_col1_sum_o  = np.array([result[i][8] for i in range(n_npy)])
        L_col1_sum2_o = np.array([result[i][9] for i in range(n_npy)])
        L_col1_min_o  = np.array([result[i][10] for i in range(n_npy)])
        L_col1_max_o  = np.array([result[i][11] for i in range(n_npy)])

        L_col2_sum_o = np.array([result[i][12] for i in range(n_npy)])
        L_col2_sum2_o = np.array([result[i][13] for i in range(n_npy)])
        L_col2_min_o = np.array([result[i][14] for i in range(n_npy)])
        L_col2_max_o  = np.array([result[i][15] for i in range(n_npy)])

        print("shape after rearranging : ", L_col1_sum.shape)

        #reduction

        #input
        col1_sum_glob = np.sum(L_col1_sum, axis = 0)
        col1_sum2_glob = np.sum(L_col1_sum2, axis = 0)
        col1_min_glob = np.min(L_col1_min, axis = 0)
        col1_max_glob = np.max(L_col1_max, axis = 0)

        col2_sum_glob = np.sum(L_col2_sum, axis = 0)
        col2_sum2_glob = np.sum(L_col2_sum2, axis = 0)
        col2_min_glob = np.min(L_col2_min, axis = 0)
        col2_max_glob = np.max(L_col2_max, axis = 0)

        #output
        col1_sum_glob_o = np.sum(L_col1_sum_o, axis = 0)
        col1_sum2_glob_o = np.sum(L_col1_sum2_o, axis = 0)
        col1_min_glob_o= np.min(L_col1_min_o, axis = 0)
        col1_max_glob_o = np.max(L_col1_max_o, axis = 0)

        col2_sum_glob_o = np.sum(L_col2_sum_o, axis = 0)
        col2_sum2_glob_o = np.sum(L_col2_sum2_o, axis = 0)
        col2_min_glob_o = np.min(L_col2_min_o, axis = 0)
        col2_max_glob_o = np.max(L_col2_max_o, axis = 0)

    print("shape after reduction : ", col1_sum_glob.shape, col2_sum_glob.shape)
    end_time = time.time()
    print(f'loading  everything took {end_time - start_time} s')

    # compute mean, std

    #input
    col1_mean_glob = col1_sum_glob/n_samples
    col2_mean_glob = col2_sum_glob/(n_samples*60)

    #output
    col1_mean_glob_o = col1_sum_glob_o/n_samples
    col2_mean_glob_o = col2_sum_glob_o/(n_samples*60)

    if resolution == 'high':
            path = '../saved/normalization_factors/hr/' + name_processing
    else:
        path = '../saved/normalization_factors/lr/' + name_processing

    if not os.path.exists(path):
        os.makedirs(path)

    #input    
    np.save(path + '/col1_sum_glob_'+str(rank)+'.npy', col1_sum_glob)
    np.save(path + '/col1_sum2_glob_'+str(rank)+'.npy',col1_sum2_glob)
    np.save(path + '/col1_min_glob_'+str(rank)+'.npy', col1_min_glob)
    np.save(path + '/col1_max_glob_'+str(rank)+'.npy', col1_max_glob)
    np.save(path + '/col1_mean_glob_'+str(rank)+'.npy',col1_mean_glob)

    np.save(path + '/col2_sum_glob_'+str(rank)+'.npy', col2_sum_glob)
    np.save(path + '/col2_sum2_glob_'+str(rank)+'.npy',col2_sum2_glob)
    np.save(path + '/col2_min_glob_'+str(rank)+'.npy', col2_min_glob)
    np.save(path + '/col2_max_glob_'+str(rank)+'.npy', col2_max_glob)
    np.save(path + '/col2_mean_glob_'+str(rank)+'.npy',col2_mean_glob)

    #output
    np.save(path + '/col1_sum_glob_o_'+str(rank)+'.npy', col1_sum_glob_o)
    np.save(path + '/col1_sum2_glob_o_'+str(rank)+'.npy',col1_sum2_glob_o)
    np.save(path + '/col1_min_glob_o_'+str(rank)+'.npy', col1_min_glob_o)
    np.save(path + '/col1_max_glob_o_'+str(rank)+'.npy', col1_max_glob_o)
    np.save(path + '/col1_mean_glob_o_'+str(rank)+'.npy',col1_mean_glob_o)

    np.save(path + '/col2_sum_glob_o_'+str(rank)+'.npy', col2_sum_glob_o)
    np.save(path + '/col2_sum2_glob_o_'+str(rank)+'.npy',col2_sum2_glob_o)
    np.save(path + '/col2_min_glob_o_'+str(rank)+'.npy', col2_min_glob_o)
    np.save(path + '/col2_max_glob_o_'+str(rank)+'.npy', col2_max_glob_o)
    np.save(path + '/col2_mean_glob_o_'+str(rank)+'.npy',col2_mean_glob_o)

    print("everything saved at ", path)



name_ = ''
if resolution == 'high':
    name_ = 'hr_'





if resolution == 'high':
    ncol = 21600
    save_dir = '/lustre/fsn1/projects/rech/psl/upu87pm/high_res_data/' + name_processing
    all_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_high-res/train/*/*.mli*.nc")

else:
    ncol = 384
    save_dir = '/lustre/fsn1/projects/rech/psl/upu87pm/low_res_data/' + name_processing
    all_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/*/*.mli*.nc")

if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)



all_path_list = sorted(all_path_list)
all_path_list = all_path_list[first_file:]
all_path_list = all_path_list[:last_file]
all_path_list = all_path_list[shift:]
all_path_list = all_path_list[::stride]





print("total amount of file processed: ", len(all_path_list))

list_file = all_path_list


N = len(list_file)//nodes
file_per_npy = N//n_npy

print("found ", N, " nc files from ",rank*N, " to ", (rank+1)*N, " and put ", file_per_npy, " of them in each npy file for node ", rank) 
list_file = list_file[rank*N:(rank+1)*N]
# print(" first file is ", list_file[0])


n_samples = n_npy*file_per_npy*ncol

print(n_samples, "samples !")

print("preprocessing all data")
prepro()

print("To Compute the GLOBAL normalizations factor, run python ../data_processing/glob_factors.py "  + name_processing + " " + str(n_samples))
