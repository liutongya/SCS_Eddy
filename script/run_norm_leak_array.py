from intake import open_catalog
import datetime
import numpy as np
import pandas as pd
import scipy.io
import gsw
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

nyear = len(np.arange(1993, 2016, 1))
date_st = np.zeros((nyear*12,))

i = 0
for year in np.arange(1993, 2016, 1):
    for mon in np.arange(1, 13, 1):
        date_st[i] = datetime.date.toordinal(datetime.date(year, mon, 1))
        i += 1
        
# load ssh eddy information
df = pd.read_pickle('ssh_eddy.pkl')
df.tail()

gcs = gcsfs.GCSFileSystem(requester_pays=True)

lev1 = np.arange(-1, 1.01, 0.01)

x_left, x_right = 100, 130
y_south, y_north = 0, 28
dx = 32 * (x_right - x_left)
dy = 32 * (y_north - y_south)

cal_num = np.zeros((2394, 1)) # the number of the remain day 

xx, yy = np.mgrid[-5:5:0.2, -5:5:0.2]

for i in np.arange(2394):
    print('eddy id: ' + str(df.id[i]))
    
    eddy_time = df.time[i] # lifetime of ssh eddy
    
    date0_str = datetime.date.fromordinal(int(eddy_time[0]-366)).strftime('%Y-%m-%d')
    print('eddy_t0: ' + str(date0_str))
    
    duration = df.duration[i]
    print('lifespan: ' + str(duration))
    
    date_diff = (date_st - eddy_time[0] + 366).tolist()    # check the number!!!!!!
    
    min_diff = min([num for num in date_diff if num >= 0])
    print('min_diff: ' + str(min_diff))
    
    remain_days = df.duration[i] - min_diff # calculate the days left
    print('remain: ' + str(remain_days))
    cal_num[i] = int(remain_days / 10) # the output of lagrangian data is 10 days
    
    start_day = eddy_time[0] - 366 + min_diff
    start_str = datetime.date.fromordinal(int(start_day)).strftime('%Y-%m-%d')
    print('calculation_t0: ' + str(start_str))
    
    sd_id = np.argwhere(eddy_time == (start_day+366))
    
    lg_fn = 'gs://pangeo-rclv-eddies/float_trajectories/' + start_str + '.zarr' # lagrangian particle file name
    
    ds = xr.open_zarr(gcs.get_mapper(lg_fn))
    ds_new = ds.sel(x0=slice(x_left, x_right), y0=slice(y_south, y_north))

    if sd_id.shape[0] > 0:
                
        boundary = df.boundary[i]
        radius = df.radius[i]
        center = df.center_traj[i]
        area = df.area[i]
        cyc = df.cyc[i]
        
        var_tmp = boundary[sd_id][0, 0]
        
        num2 = var_tmp[1, :] / (y_north - y_south) * dy  # position of y
        num1 = (var_tmp[0, :] - 100) / (x_right - x_left) * dx # position of x
    
        contour0 = np.zeros((var_tmp.shape[1], 2))
        contour0[:, 1] = num1
        contour0[:, 0] = num2
        
        labels = rclv.label_points_in_contours(ds_new.x[0, :, :].shape, [contour0])
        
        mask1 = deepcopy(labels).astype('float')
        mask1[mask1 == 0] = np.nan
        
        kk = 0
        
        leak_array = np.zeros((2, xx.shape[0], xx.shape[0]))
        for nday in [0, int(cal_num[i])]:
        #for nday in [0, 0]:
            if nday >= 18:
                nday = 18
                
            tar_day = start_day + 10*nday
            sd_id = np.argmin(np.abs(tar_day + 366 - eddy_time + 0.01))

            pxin1 = ds_new.x[nday, :, :] * mask1
            pyin1 = ds_new.y[nday, :, :] * mask1

            dist1d = gsw.distance([center[sd_id][0], center[sd_id][0]+1], [center[sd_id][1], center[sd_id][1]])
            x0_norm = (pxin1.values - center[sd_id][0]) * dist1d / radius[sd_id]
            y0_norm = (pyin1.values - center[sd_id][1]) * dist1d / radius[sd_id]
            
            x0_norm_new = x0_norm[~np.isnan(x0_norm)].ravel()
            y0_norm_new = y0_norm[~np.isnan(y0_norm)].ravel()
            
            leak_array[kk, :, :] = np.histogram2d(x0_norm_new, y0_norm_new, bins=np.arange(-5, 5.1, 0.2))[0]
            
            #boun_tmp = boundary[sd_id]
            #lonx = (boun_tmp[0, :] - center[sd_id][0]) * dist1d / radius[sd_id]
            #laty = (boun_tmp[1, :] - center[sd_id][1]) * dist1d / radius[sd_id]

            #trajx = center[:, 0] - center[sd_id][0]
            #trajy = center[:, 1] - center[sd_id][1]

            kk = kk + 1

        #titlestr = 'Eddy ID: ' + str(i).zfill(4) + ', Lifespan: ' + str(int(duration)) + ', Remain: ' + str(int(remain_days))
        #plt.suptitle(titlestr, size=16)

        savename = './array_data/leak_' + str(i).zfill(4) + '.npy'
        np.save(savename, leak_array)
        
        