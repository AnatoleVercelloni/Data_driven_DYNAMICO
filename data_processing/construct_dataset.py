import numpy as np
import xarray as xr
import sys
import psyplot
import psyplot.project as psy
from psy_maps.plotters import FieldPlotter




def npy_toxarray(file, grid_b, var, unscale = ""):
    #take either a list of .npy file or a numpy array for 'file', a netcdf file of the grid info for 'grid_b' 
    #if unscale is set to a netcdf file with the unscale factor, it also unscale the data
    #var is the variable which is load
    #==> it create a xarraydataset for this data and that can be used by psyplot

    #open the grid and get the dimensions
    grid = psyplot.open_dataset(grid_b)
    ncol = len(grid['ncol'])
    lev = 60#len(grid['lev'])
    nvertex = len(grid['nvertex'])

    #create a dictionnary of the available variables ('stat' is for any postprocessed data)
    VAR = {'ptend_t': range(lev), 'ptend_q0001': range(lev,2*lev,1), 'ptend_q0002': range(2*lev,3*lev,1), 'ptend_q0003': range(3*lev,4*lev,1), 
           'ptend_u': range(4*lev,5*lev,1), 'ptend_v': range(5*lev,6*lev,1),
           'cam_out_NETSW': [360], 'cam_out_FLWDS': [361],
           'cam_out_PRECSC': [362], 'cam_out_PRECC': [363], 'cam_out_SOLS': [364], 'cam_out_SOLL': [365], 'cam_out_SOLSD': [366],
           'cam_out_SOLLD': [367],
           
           'state_t': range(lev), 'state_q0001': range(lev,2*lev,1), 'state_q0002': range(2*lev,3*lev,1), 'state_q0003': range(3*lev,4*lev,1), 
           'state_u': range(4*lev,5*lev,1), 'state_v': range(5*lev,6*lev,1), 'state_ps': [6*lev], 'pbuf_SOLIN' : [6*lev + 1], 'pbuf_LHFLX' : [6*lev + 2],
           'pbuf_SHFLX' : [6*lev + 3], 'pbuf_TAUX' : [6*lev + 4], 'pbuf_TAUY' : [6*lev + 5], 'pbuf_COSZRS' : [6*lev + 6], 'cam_in_ALDIF' : [6*lev + 7],
           'cam_in_ALDIR' : [6*lev + 8], 'cam_in_ASDIF' : [6*lev + 9], 'cam_in_ASDIR' : [6*lev + 10], 'cam_in_LWUP' : [6*lev + 11], 'cam_in_ICEFRAC' : [6*lev + 12],
           'cam_in_LANDFRAC' : [6*lev + 13], 'cam_in_OCNFRAC' : [6*lev + 14], 'cam_in_SNOWHLAND' : [6*lev + 15], 'pbuf_ozone' : range(6*lev + 16, 7*lev + 16, 1),
           'pbuf_CH4' : range(7*lev + 16, 8*lev + 16, 1), 'pbuf_N2O' : range(8*lev + 16, 9*lev + 16, 1)}
                                                        

    N = 368
    

    #check var is set corectly 
    if var not in VAR:
        print("the variable is not available !")
        sys.exit(0)

    #get the length of the variable
    nvar = len(VAR[var])

    #check the input type (here .npy file list)
    if type(file[0]) is str:
        DATA = []

        #load just the data of var for each file
        for i in range(len(file)):  
            data = np.load(file[i], mmap_mode='r')
            data = data[:,VAR[var]]
            DATA.append(data)

        #shape the data to be (time*ncol)x(var)
        s = len(DATA[0])*len(file)
        data = np.array(DATA).reshape((s, nvar))

    #numpy array
    else:
        #check that the data is coherent with the variable: shape = (nsamples, nvar)
        if file.shape[1] == len(VAR[var]):
            data = file
        elif file.shape[1] == N or file.shape[1] == 556:
            data = file[:,VAR[var]]
        else:
            print("wrong file parameter! file size is ", file.shape[1])
            return 

    T = data.shape[0]//ncol
    print('time counter = ', T)

    #reshape the data 
    #if type(file[0]) is str:
    if nvar == lev:  data = data.reshape(T, ncol, lev).transpose((0,2,1))      #(T, ncol, lev)
    if nvar == 1: data = data.reshape(T, ncol)
    print("shape; ", data.shape)
    #The axis are inverted for stat    
    if 'stat' == var: data = data.reshape((T, ncol, lev)).transpose((0,2,1))

    #get the bpunds_lon(resp. lat) for the coordinates
    bounds_lon_ = np.array(grid['bounds_lon'])
    bounds_lat_ = np.array(grid['bounds_lat'])

    #if unscale is set, unscale the data
    if unscale != "" and 'stat' != var:
        sc = xr.open_dataset(unscale)
        if nvar == lev: 
            data = data/np.array(sc[var]).reshape((1,lev,1))
        else:
            data = data/np.array(sc[var]).reshape((1,1))

    #We can now create a xarraydataset for our data
    ds = xr.Dataset(

        #add the dimensions that are coordinates (for psyplot)
        coords=dict(
            lev=("lev", np.arange(0, lev, dtype=float)),
            bounds_lat = (("ncol", "nvertex") , bounds_lat_),
            bounds_lon = (("ncol", "nvertex"), bounds_lon_),
            time_counter=("time_counter", range(T)),
            ncol=("ncol", range(ncol))
        )

    )


    #add the data to the dataset
    if nvar == lev: ds[var]   = (['time_counter', 'lev', 'ncol'], data)
    elif nvar == 1: ds[var]   = (['time_counter', 'ncol'], data)


    #also set the coordinate attribute (for psyplot), redundant but doesn't work if it's not done
    for var in ds.variables:
        if str(var) == 'area': continue
        if str(var) == 'ncol': continue 
        if str(var) == 'lat' : ds[str(var)].attrs = {"coordinates": "bounds_lat"}
        elif str(var) == 'lon': ds[str(var)].attrs = {"coordinates": "bounds_lon"}
        elif (ds[str(var)].shape) == (ncol, ):
            ds[str(var)].attrs = {"coordinates": "lon lat"}
        elif (ds[str(var)].shape) == (ncol, nvertex) : ds[str(var)].attrs = {"coordinates": "lon lat"}
        elif (ds[str(var)].shape) == (T, ncol) : ds[str(var)].attrs = {"coordinates": "time_centered lon lat"}
        elif (ds[str(var)].shape) == (T, lev, ncol) : ds[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}


    #merge data with the grid
    ds = xr.merge([grid, ds])

    return ds