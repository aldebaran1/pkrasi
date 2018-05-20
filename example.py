# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:04:20 2018

@author: smrak
"""
import numpy as np
from datetime import datetime
from pkrasi import pkrasi as pa
from pkrasi import plotting as asiplot


# CFG:
plot = False
cfg = 'polar'
cfg = None
date = '20151007'
w2f = True
# Wavelength
wl = 558
# Folder
folder_root = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\TheMahali\\data\\allskydata\\'
#folder_root = 'G:\\Team Drive\\Semeter-Research in Progress\\DASC\\'
#h5fn = folder_root + date + '_' + str(wl) + '.h5'
folder = folder_root + date + '\\'
h5fn = folder + date + '_' + str(wl) + '.h5'
# ASI calibration files
as_cfg_folder = folder_root + 'cfg\\'
azfn = as_cfg_folder+'PKR_DASC_20110112_AZ_10deg.FITS'
elfn = as_cfg_folder+'PKR_DASC_20110112_EL_10deg.FITS'
#Interpolation grid
N =256
# Mapping altitude
mapping_alt = 220
# Timelim
timelim = [datetime(2015,10,7,6,16,0), datetime(2015,10,7,6,19,0)]
# Get data
if cfg == 'lla':
    t, xgrid, ygrid, im, [lon,lat]= pa.returnASLatLonAlt(folder, azfn=azfn, elfn=elfn, wl=wl, 
                                            timelim=timelim, alt=mapping_alt,
                                            Nim=N,asi=True)
    if plot:
        for i in range(t.shape[0]):
            tmp = im[i]
            tmp[tmp<=500] = np.nan
            asiplot.plotIMmap(xgrid,ygrid,tmp,title=t[i],lon0=lon,lat0=lat,alt=mapping_alt,
                      clim=[500,1000], cmap='Reds',norm_gamma=0.5)
if cfg == 'polar':
    t, xgrid, ygrid, im, [lon,lat]= pa.returnASpolar(folder, azfn=azfn, elfn=elfn, 
                         wl=wl, timelim=timelim, Nim=N,asi=True)
    if plot:
        for i in range(t.shape[0]):
            tmp = im[i]
            tmp[tmp<=500] = np.nan
            asiplot.plotIMpolar(xgrid,ygrid,tmp,title=t[i],lon0=lon,lat0=lat,
                      clim=[500,4000], cmap='Greens',norm_gamma=0.5)
            
if w2f:
    data = pa.returnRaw(folder, azfn=azfn,elfn=elfn,wl=wl,timelim=timelim)
    pa.write2HDF(data, h5fn=h5fn,wl=wl)

