# DESCRIPTION: Interpolates MRMS composite reflectivity to
# fv3 grid using scipy.interpolate.RegularGridInterpolator
# with neareststod interpolation method, prerequisite utility
# for enabling the obs-aware RTPS method in EnKF
#
# Author: OU MAP (N. Gasperoni. Y. Wang, X. Wang) 09-08-2023
#
# INPUTS: Only need the following files with assumed file and variable/dim names
#   Gridded_ref.nc             - MRMS 3d reflectivity, same as used for DA
#                                With dimensions ('height', 'latitude', 'longitude')
#                                and reflectivity variable named 'reflectivity'
#   fv3sar_tile1_grid_spec.nc  - FV3-LAM horizontal grid specs to read in 
#                                latitude  ('grid_latt') and 
#                                longitude ('grid_lont') arrays
#   fv3sar_tile1_akbk.nc       - FV3_LAM vertical coefficients
#                                only needed to define number of  
#                                vertical levels for output file
#   dbzthresh                  - Threshold for masking, default value 35.0
#
# OUTPUT: fv3sar_tile1_mask netcdf file with following sample header info
#   Note that for now EnKF reads in variable name 'obsmask'. 
#   And interpolated cref is copied to all vertical levels for EnKF code implementation
#
#   netcdf fv3sar_tile1_mask {
#   dimensions:
#	xaxis_1 = 1826 ;
#	xaxis_2 = 1827 ;
#	yaxis_1 = 1099 ;
#	yaxis_2 = 1098 ;
#	zaxis_1 = 65 ;
#	Time = UNLIMITED ; // (1 currently)
#   variables:
#	float Time(Time) ;
#	float obsmask(Time, zaxis_1, yaxis_2, xaxis_1) ;
#		obsmask:units = "dBZ" ;
#
#   // global attributes:
#		:description = "MRMS Composite reflectivity interpolated to FV3-LAM grid" ;
#		:intmethod = "scipy.interpolate.RegularGridInterpolator(method=nearest) " ;
#   }
#

import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
import argparse 

start_time = time.time()


# Create a parser object to handle command-line arguments
parser = argparse.ArgumentParser(description='''Create mask field based on observed MRMS composite reflectivity 
                                 with neareststod interpolation method, prerequisite utility
                                 for enabling the obs-aware RTPS method in EnKF''')

# Add an argument for the user-defined value with a default
parser.add_argument('dbzthresh', type=float, nargs='?', default=35.0, 
                    help='''threshold for masking composite reflectivity (default=35.0)''')

# Parse the command-line arguments
args      = parser.parse_args()
dbzthresh = args.dbzthresh

if dbzthresh == 35.0:
    print(f"Using the default value for dbzthresh ({dbzthresh})")
else:
    print(f"User-defined value provided for dbzthresh: {dbzthresh}")


# Open and read Gridded_ref.nc
file_mrms = 'Gridded_ref.nc'
nc_file = nc.Dataset(file_mrms, 'r')

# Access dimensions, variables, and attributes
nlat_mrms=len(nc_file.dimensions['latitude'])
nlon_mrms=len(nc_file.dimensions['longitude'])
print(f"\nDimensions of mrms reflectivity in {file_mrms}:")
for dim in nc_file.dimensions:
    print(f" - {dim}: {len(nc_file.dimensions[dim])}")

lat_mrms=nc_file.variables["latitude"][:]
lon_mrms=nc_file.variables["longitude"][:]
lon_mrms=np.where(lon_mrms > 180.0, lon_mrms-360.0, lon_mrms)
print(f"\nlat_mrms range: {np.amin(lat_mrms)}  {np.amax(lat_mrms)}")
print(f"lon_mrms range: {np.amin(lon_mrms)}  {np.amax(lon_mrms)}")

mrms_refl3d = nc_file.variables["reflectivity"][:]
nc_file.close()

mrms_cref = np.amax(mrms_refl3d, axis=0)
min_value = np.amin(mrms_cref)
max_value = np.amax(mrms_cref)
print(f"\nDimensions (nlat, nlon) of mrms_cref: {mrms_cref.shape}")
print(f"Min and Max values for mrms_cref: {np.amin(mrms_cref)}  {np.amax(mrms_cref)}")


# Read akbk file to get size of vertical dimension
nc_akbk = nc.Dataset("fv3sar_tile1_akbk.nc","r")
zaxis_1 = len(nc_akbk.dimensions['xaxis_1']) - 1
print(f"\nRead in fv3sar_tile1_akbk.nc, got zaxis_1 = {zaxis_1}")
nc_akbk.close()

# Read in gridspec file for latitude, longitude arrays
nc_gridspec = nc.Dataset("fv3sar_tile1_grid_spec.nc","r")
lat2d_fv3grid = nc_gridspec.variables["grid_latt"][:]
lon2d_fv3grid = nc_gridspec.variables["grid_lont"][:]
lon2d_fv3grid = np.where(lon2d_fv3grid > 180.0, lon2d_fv3grid-360.0, lon2d_fv3grid)
print(f"\nDimensions of fv3sar_tile1_grid_spec.nc for interpolation destination: {lat2d_fv3grid.shape}")
print(f"lat2d_fv3grid range: {np.amin(lat2d_fv3grid)}  {np.amax(lat2d_fv3grid)}")
print(f"lon2d_fv3grid range: {np.amin(lon2d_fv3grid)}  {np.amax(lon2d_fv3grid)}")
nc_gridspec.close()

#Define interpolating function
print("\nInterpolating to fv3grid using scipy.interpolate.RegularGridInterpolator(method='nearest')")
interp = RegularGridInterpolator((lon_mrms, lat_mrms), np.transpose(mrms_cref),
                                 bounds_error=False, fill_value=-999., method="nearest")


# Interpolate mrms cref to fv3 grid locations with bilinear interpolation
cref_fv3grid = interp((lon2d_fv3grid, lat2d_fv3grid))
cond         = np.logical_and(cref_fv3grid < dbzthresh, cref_fv3grid > -900)
cref_fv3grid = np.where(cond, -99., cref_fv3grid)
print(f"After interpolation, cref_fv3grid dimensions: {cref_fv3grid.shape}")
print(f"Min Max of interpolated cref_fv3grid: {np.amin(cref_fv3grid)} {np.amax(cref_fv3grid)}")


# Write to netcdf file
foutname="fv3sar_tile1_mask"
print(f'\nWriting to output netcdf file {foutname}')
yaxis_2 = cref_fv3grid.shape[0]
yaxis_1 = yaxis_2 + 1
xaxis_1 = cref_fv3grid.shape[1]
xaxis_2 = xaxis_1 + 1

ncout = nc.Dataset(foutname, 'w', format='NETCDF4_CLASSIC')
ncout.createDimension('xaxis_1', xaxis_1)
ncout.createDimension('xaxis_2', xaxis_2)
ncout.createDimension('yaxis_1', yaxis_1)
ncout.createDimension('yaxis_2', yaxis_2)
ncout.createDimension('zaxis_1', zaxis_1)
ncout.createDimension('Time', None)

# Create a variable for time (unlimited)
time_var = ncout.createVariable('Time', np.float32, ('Time',))

# Create a netcdf variable for output interpolated cref
cref_ncvar = ncout.createVariable('obsmask', np.float32, ('Time','zaxis_1','yaxis_2', 'xaxis_1'), zlib=True, complevel=4)

# Assign the data from  array to the NetCDF variable
# Note that entries are copied to each vertical level in the fv3grid
# since the EnKF code will use the 3d field for obsaware masking
cref_ncvar[0,:,:,:] = np.tile(cref_fv3grid[np.newaxis,:,:], (zaxis_1,1,1))

# Optionally, add attributes to the variable or the NetCDF file
cref_ncvar.units  = "dBZ"
ncout.description = "MRMS Composite reflectivity interpolated to FV3-LAM grid"
ncout.intmethod   = "scipy.interpolate.RegularGridInterpolator(method=nearest)"
ncout.dbzthresh   = dbzthresh

ncout.close()

print("\n--- mrms2fv3grid.py completed! Took %s seconds ---" % (time.time() - start_time))
