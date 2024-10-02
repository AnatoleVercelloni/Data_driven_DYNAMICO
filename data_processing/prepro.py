import xarray as xr
import numpy as np
import glob
import os
import random
import time
import math
from multiprocessing import Pool, cpu_count, Manager


verbose = False

#556 input scalars
vars_mli      = ['state_t','state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v',
                     'state_ps', 'pbuf_SOLIN','pbuf_LHFLX', 'pbuf_SHFLX',  'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS',
                     'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP',
                     'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND', 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']
                     
#368 output scalars
vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC',
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']


def task(i):
    #each worker write 'file_per_npy' amount of data in a npy file in parallel

    file_per_npy = N//n_npy
    if (N%n_npy) != 0 and verbose : print("missing some file because ", n_npy, " is not factor of ", N)

    if verbose: print("loading ", file_per_npy*2, "files  (input + output):  ")
    
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

    if verbose: print(f'loading  took {end_time - start_time} s')

    # we have to construct the tendency variables for the output data
    dso['ptend_t']     = (dso['state_t']     - ds['state_t'])/1200     # T tendency [K/s]
    dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_q0002'] = (dso['state_q0002'] - ds['state_q0002'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_q0003'] = (dso['state_q0003'] - ds['state_q0003'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_u']     = (dso['state_u']     - ds['state_u'])/1200     # Q tendency [kg/kg/s]
    dso['ptend_v']     = (dso['state_v']     - ds['state_v'])/1200     # Q tendency [kg/kg/s]
    dso = dso[vars_mlo]


    # stack all the variables together (1 lev = 1 var)
    ds = ds.stack({'batch':{'ncol', 'time_counter'}})
    ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
    dso = dso.stack({'batch':{'ncol', 'time_counter'}})
    dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')

    # convert to numpy array
    ds = np.array(ds)
    dso = np.array(dso)

    if verbose: print("input shape  ", ds.shape)
    if verbose: print("output shpae ", dso.shape)

    # keep only the variables that interessed us
    #ds = ds[:,var_idx]

    ds1 = ds.reshape((-1,ncol,556),order='F').transpose((1,0,2)).reshape(-1, 556,order='F')
    dso1 = dso.reshape((-1,ncol,368),order='F').transpose((1,0,2)).reshape(-1, 368,order='F')


    # normalize here with npy normalization files
    if  norm:
        if verbose: print("normalizing data")
        ds = (ds-mli_mean)/(mli_max-mli_min)
        dso = dso*mlo_scale 

    #saving the data into .npy files 
    np.save(save_dir.replace('training', set_) + '/input_'+str(i).rjust(2,'0')+'.npy', ds1)
    np.save(save_dir.replace('training', set_) + '/target_'+str(i).rjust(2,'0')+'.npy', dso1)

    return 


def prepro(idx):
    #preprocess the data represented by 'idx' of the files for 'set_'

    global N
    global list_file

    #get only the data needed
    list_file = [path_list[i] for i in idx]
    N = len(list_file)
    file_per_npy = N//n_npy
    print("found ", N, " nc files and put ", file_per_npy, " of them in each npy file") 

    if N == 0:
        print("no need for ", set_, "data")
        return
    #devide the work between the workers
    start_time = time.time()
    with Pool(n_proc) as pool:
        pool.map(task, range(n_npy))

    end_time = time.time()
    print(f'loading  everything took {end_time - start_time} s')

def preprocessing(train_idx, val_idx, test_idx, norm_path_, save_dir_, var_idx_ = list(range(556))):

    global set_
    global norm_path
    global mli_max
    global mli_min
    global mli_mean
    global mlo_scale
    global var_idx
    global save_dir

    norm_path = norm_path_
    var_idx = var_idx_



    save_dir = save_dir_


    os.makedirs(save_dir, exist_ok = True)
    os.makedirs(save_dir.replace('/training/', '/val/'), exist_ok = True)
    os.makedirs(save_dir.replace('/training/', '/test/'), exist_ok = True)

    print("creating directory "+ save_dir + "(resp. val and test) to save data")

    if norm:
   
        #normalization factor already computed from the ClimSim normalization factor but in np format and good shape
        l = np.shape(np.load(norm_path + 'input_max.npy'))[0]
        o = np.shape(np.load(norm_path + 'output_scale.npy'))[0]

        mli_max = np.load(norm_path + 'input_max.npy').reshape(1, l)
        mli_min = np.load(norm_path + 'input_min.npy').reshape(1, l)
        mli_mean = np.load(norm_path + 'input_mean.npy').reshape(1, l)
        mlo_scale = np.load(norm_path + 'output_scale.npy').reshape(1, o)


    set_ = 'training'
    prepro(train_idx)

    set_ = 'val'
    prepro(val_idx)

    set_ = 'test'
    prepro(test_idx)

    return save_dir


#amount of .npy to store the data
n_npy = 72
#amount of workers
n_proc = 72
#set True to normalize the data
norm = False

ncol = 21600

# get the list of all files
path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_high-res/train/*-*/*.mli*.nc")
path_list = sorted(path_list)
normalization_dir = ''
name = 'high_res_data_test'
save_dir = '/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/'+name+'/training/'
print('=================== PREPROCESSING =================\n')

n_file_per_year    = 26280
# 210240 files, 1 file every 20 minutes
training_data_idx  = list(range(72))
val_data_idx       = list()
test_data_idx      = list()

preprocessing_dir  = preprocessing(training_data_idx, val_data_idx, test_data_idx, normalization_dir, save_dir)


