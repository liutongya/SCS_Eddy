from intake import open_catalog
import datetime
import numpy as np
import pandas as pd
import scipy.io
import gcsfs
import xarray as xr
import rclv
from copy import deepcopy
from skimage import measure

# load the time of eddy identification
fn = '/home/jovyan/RCLVs/SCS/Eddies_SCS_structure_olddata.mat'
data = scipy.io.loadmat(fn)

data1 = data['eds_28d_stru']
data2 = data['eds_28d_trk_stru']
data3 = data['t_eddy']

# load ssh eddy information
df = pd.read_pickle('ssh_eddy.pkl')

nyear = len(np.arange(1993, 2016, 1))
date_st = np.zeros((nyear*12,))

i = 0
for year in np.arange(1993, 2016, 1):
    for mon in np.arange(1, 13, 1):
        date_st[i] = datetime.date.toordinal(datetime.date(year, mon, 1))
        i += 1
        
gcs = gcsfs.GCSFileSystem(requester_pays=True)

x_left, x_right = 100, 132
y_south, y_north = 0, 30
dx = 32 * (x_right - x_left)
dy = 32 * (y_north - y_south)

cal_num = np.zeros((2394, 1)) # the number of the remain day 

# define vars to input data
leak_data = np.zeros((2394, 22))
leak_data[:, :] = np.NaN

intru_data = np.zeros((2394, 22))
intru_data[:, :] = np.NaN

for i in np.arange(2394):
    print('eddy id: ' + str(df.id[i]))
    leak_data[i, 0] = df.id[i]
    leak_data[i, 1] = df.duration[i]
    intru_data[i, 0] = df.id[i]
    intru_data[i, 1] = df.duration[i]
        
    eddy_time = df.time[i] # lifetime of ssh eddy
    
    date0_str = datetime.date.fromordinal(int(eddy_time[0]-366)).strftime('%Y-%m-%d')
    print('eddy_t0: ' + str(date0_str))
    
    duration = df.duration[i]
    print('lifespan: ' + str(duration))
    
    date_diff = (date_st - eddy_time[0] + 366).tolist()    # check the number!!!!!!
    
    var_tmp1 = [num for num in date_diff if num >= 0]
    
    if len(var_tmp1) > 0:
        min_diff = min(var_tmp1)
        print('min_diff: ' + str(min_diff))

        remain_days = df.duration[i] - min_diff # calculate the days left
        print('remain: ' + str(remain_days))
        cal_num[i] = int(remain_days / 10) # the output of lagrangian data is 10 days

        start_day = eddy_time[0] - 366 + min_diff
        start_str = datetime.date.fromordinal(int(start_day)).strftime('%Y-%m-%d')
        print('calculation_t0: ' + str(start_str))

        sd_id = np.argwhere(eddy_time == (start_day+366))
        
        if sd_id.shape[0] > 0:
            
            leak_data[i, 2] = df.area[i][sd_id]
            intru_data[i, 2] = df.area[i][sd_id]
            
            lg_fn = 'gs://pangeo-rclv-eddies/float_trajectories/' + start_str + '.zarr' # lagrangian particle file name
            ds = xr.open_zarr(gcs.get_mapper(lg_fn))
            ds_new = ds.sel(x0=slice(x_left, x_right), y0=slice(y_south, y_north))
            
            boundary = df.boundary[i]
            
            var_tmp2 = boundary[sd_id][0, 0]
        
            num2 = var_tmp2[1, :] / (y_north - y_south) * dy  # position of y
            num1 = (var_tmp2[0, :] - 100) / (x_right - x_left) * dx # position of x
    
            contour0 = np.zeros((var_tmp2.shape[1], 2))
            contour0[:, 1] = num1
            contour0[:, 0] = num2
            
            labels = rclv.label_points_in_contours(ds_new.x[0, :, :].shape, [contour0])
        
            mask1 = deepcopy(labels)
            mask2 = deepcopy(labels)
            mask2[mask1==0] = 1
            mask2[mask1==1] = 0
            
            for nday in np.arange(int(cal_num[i]+1)):
                
                tar_day = start_day + 366 + 10*nday
                sd_id2 = np.argmin(np.abs(tar_day - eddy_time + 0.01)) 
                # find the 10th 20th ... day
                # 0.01 is to aviod the double id
                
                pxin = ds_new.x[nday, :, :] * mask1
                pyin = ds_new.y[nday, :, :] * mask1
                pxout = ds_new.x[nday, :, :] * mask2
                pyout = ds_new.y[nday, :, :] * mask2

                pxin1d = np.reshape(pxin.values, (mask1.shape[0] * mask1.shape[1]))
                pyin1d = np.reshape(pyin.values, (mask1.shape[0] * mask1.shape[1]))
                pxout1d = np.reshape(pxout.values, (mask2.shape[0] * mask2.shape[1]))
                pyout1d = np.reshape(pyout.values, (mask2.shape[0] * mask2.shape[1]))

                boun_tmp = boundary[sd_id2]
                leak_data[i, nday+3] = np.sum(measure.points_in_poly(np.array([pxin1d, pyin1d]).T, np.array([boun_tmp[0, :], boun_tmp[1, :]]).T))
                intru_data[i, nday+3] = np.sum(measure.points_in_poly(np.array([pxout1d, pyout1d]).T, np.array([boun_tmp[0, :], boun_tmp[1, :]]).T))

df1 = pd.DataFrame(leak_data)
df1.to_pickle('leak_data.pkl')

df2 = pd.DataFrame(intru_data)
df2.to_pickle('intru_data.pkl')