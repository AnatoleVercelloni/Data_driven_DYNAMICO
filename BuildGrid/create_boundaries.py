import netCDF4
from netCDF4 import Dataset
import xarray as xr
from construct_mesh import mesh, format_, matplotlib_plot
import numpy as np
import psyplot

def read_grid_info(filename, verbose = True):
    # take as input a netcdf grid info 'filename' and return
    # ncel the number of colums
    # POLY is the set of polygone representing the mesh
    # nvertex the nb of vertex per polygone
    # lon, lat are data read from the grid

    nc = xr.open_dataset(filename)
    
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    area = nc.variables['area'][:] 

    nvertex = 10
    ncel = lat.shape[0]

    #call mesh to construct the polygones
    POLY, nvertex = mesh(filename, nvertex, verbose)

    if verbose: print("read grid info from netcedf file ok !")
    
    return ncel, POLY, nvertex, lon, lat, area



def create_grid_info(filename, grid_info, verbose=True):
    # take as input 2 netcdf file names 'filename' (non existant) and 'gridinfo' that contains grid infos
    # write the grid with boundaries into 'filename'

    # get the informations from 'gridinfo'
    ncel, POLY, nvertex, lon_, lat_, area_ = read_grid_info(grid_info, verbose)

    # get the lon lat coordinates from cartesian coordinates
    bounds_lon_, bounds_lat_ = format_(POLY)

    # create a Dataset and fill it
    rootgrp = Dataset(filename, "w", format="NETCDF4")

    # dimensions
    nb = rootgrp.createDimension("axis_nbounds", 2)
    ilev = rootgrp.createDimension("ilev", 61)
    lev = rootgrp.createDimension("lev", 60)
    ncol = rootgrp.createDimension("ncol", ncel)
    nvertex = rootgrp.createDimension("nvertex", nvertex)
    
    # variables
    lat = rootgrp.createVariable("lat","f8",("ncol",))
    lon = rootgrp.createVariable("lon","f8",("ncol",))
    lev = rootgrp.createVariable("lev", "f8", ("lev",))
    ilev = rootgrp.createVariable("ilev", "f8", ("ilev",))
    bounds_lat = rootgrp.createVariable("bounds_lat","f8",("ncol","nvertex",))
    bounds_lon = rootgrp.createVariable("bounds_lon","f8",("ncol","nvertex",))
    area = rootgrp.createVariable("area","f8",("ncol",))


    # attributes
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.bounds = 'bounds_lat'

    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.bounds = 'bounds_lon'

    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'

    area.coordinates = "lon lat"

    rootgrp.description = "test_climsim"

    # fill the data
    lev[:] = range(60)
    ilev[:] = range(61)
    lat[:] = lat_
    lon[:] = lon_
    bounds_lat[:] = bounds_lat_
    bounds_lon[:] = bounds_lon_
    area[:] = area_
	
    rootgrp.close()

    return rootgrp

def load_ncfiles(filelist, grid_b, input_only = False):
    #take a list of netcdf file (one file = one snapshot) of data and a grid info netcdf file as input (with boundaries)
    #return the xarraydataset which merge all this data and add a time dimension and can be used by psyplot

    #open the grid and get the dimensions
    grid = psyplot.open_dataset(grid_b)
    ncol = len(grid['ncol'])
    lev = len(grid['lev'])
    T = len(filelist)
    print('time counter = ', T)
    nvert = len(grid['nvertex'])

    #open the data
    inputs = [xr.open_dataset(file) for file in filelist]

    #add the time dimension
    In = xr.concat(inputs, dim = 'time_counter')
    In['time_counter'] = range(T)

    #merge everything
    In = xr.merge([grid, In])

    print("input files loaded !")

    #add the coordinates attribute for psyplot
    for var in In.variables:
        if str(var) == 'area': continue
        if (In[str(var)].shape) == (ncol, ): In[str(var)].attrs = {"coordinates": "lon lat"}
        elif (In[str(var)].shape) == (ncol, nvert) : In[str(var)].attrs = {"coordinates": "lon lat"}
        elif (In[str(var)].shape) == (T, ncol) : In[str(var)].attrs = {"coordinates": "time_centered lon lat"}
        elif (In[str(var)].shape) == (T, lev, ncol) : In[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}


    #add the bounds attribute for psyplot
    In['lon'].attrs  = {"bounds": "bounds_lon", "units" : "degrees_east"}
    In['lat'].attrs  = {"bounds": "bounds_lat", "units" : "degrees_east"}
    In['ilev'].attrs = {"axis": "Z"}

    if input_only:
        Ou = None
    else:
        #same for the output files 
        outputs = [xr.open_dataset(file.replace('.mli.','.mlo.')) for file in filelist]
        Ou = xr.concat(outputs, dim = 'time_counter')
        Ou['time_counter'] = range(T)
        
    
        Ou = xr.merge([grid, Ou])

        print("output files loaded !")

    
        Ou['ptend_t'] = (Ou['state_t'] - In['state_t'])/1200
        Ou['ptend_q0001'] = (Ou['state_q0001'] - In['state_q0001'])/1200
    
        for var in Ou.variables:
            if str(var) == 'area': continue
            if (Ou[str(var)].shape) == (ncol, ): Ou[str(var)].attrs = {"coordinates": "lon lat"}
            elif (Ou[str(var)].shape) == (ncol, nvert) : Ou[str(var)].attrs = {"coordinates": "lon lat"}
            elif (Ou[str(var)].shape) == (T, ncol) : Ou[str(var)].attrs = {"coordinates": "time_centered lon lat"}
            elif (Ou[str(var)].shape) == (T, lev, ncol) : Ou[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}
    
        Ou['lon'].attrs = {"bounds": "bounds_lon", "units" : "degrees_east"}
        Ou['lat'].attrs = {"bounds": "bounds_lat", "units" : "degrees_east"}
        Ou['ilev'].attrs = {"axis": "Z"}




    return In, Ou




