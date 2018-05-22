# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:04:20 2018

@author: smrak
"""
import numpy as np
from datetime import datetime
from pkrasi import pkrasi as pa
from pkrasi import plotting as asiplot

from scipy.interpolate import griddata
from pymap3d import aer2geodetic
import matplotlib.pyplot as plt

# CFG:
el_filter = 20
plot = True
cfg = 'polar'
#cfg = None
w2f = False
steve = 1
folder_root = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\TheMahali\\data\\allskydata\\'
if steve:
    date = '20080326'
    wl= 0
    folder = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\steve\\data\\pkr\\'
    as_cfg_folder = folder_root + 'cfg\\'
    azfn = as_cfg_folder+'PKR_20111006_AZ_10deg.FITS'
    elfn = as_cfg_folder+'PKR_20111006_EL_10deg.FITS'
    timelim = [datetime(2008,3,26,11,43,0), datetime(2008,3,26,11,44,0)]
#    timelim = [datetime(2008,3,26,7,32,0), datetime(2008,3,26,7,33,0)]
else:
    date = '20151007'
    # Wavelength
    wl = 558
# Folder
#folder_root = 'G:\\Team Drive\\Semeter-Research in Progress\\DASC\\'
    #h5fn = folder_root + date + '_' + str(wl) + '.h5'
    folder = folder_root + date + '\\'
    
    h5fn = folder + date + '_' + str(wl) + '.h5'
    # ASI calibration files
    as_cfg_folder = folder_root + 'cfg\\'
    azfn = as_cfg_folder+'PKR_DASC_20110112_AZ_10deg.FITS'
    elfn = as_cfg_folder+'PKR_DASC_20110112_EL_10deg.FITS'
    # Timelim
    timelim = [datetime(2015,10,7,6,16,0), datetime(2015,10,7,6,20,0)]
#Interpolation grid
N = 512
# Mapping altitude
mapping_alt = 100


# Get data
if cfg == 'raw':
    data = pa.returnRaw(folder, azfn=azfn,elfn=elfn,wl=wl,timelim=timelim)
    T = data.time.values.astype(datetime)
    az = data.az[1]
    el = data.el[1]
    im_test = data[wl][0].values
    if el_filter is not None:
        el = np.where(el>=el_filter,el,np.nan)
        az = np.where(el>=el_filter,az,np.nan)
    # Reshape calibration files
    if im_test.shape != el.shape:
        el = pa.interpolateCoordinate(el,N=im_test.shape[0])
        az = pa.interpolateCoordinate(az,N=im_test.shape[0])
    # LLA
    # Map to altitude
    mapping_alt = 100000
    r = mapping_alt / np.sin(np.deg2rad(el))
    # Convert to WSG
    lat0 = data.lat
    lon0 = data.lon
    alt0 = data.alt_m
    lat, lon, alt = aer2geodetic(az,el,r,lat0,lon0,alt0)
    # Image
    for i in range(T.shape[0]):
        im = data[wl][i].values
        XG, YG, Zlla = pa.interpolateAS(lon,lat,im,N=N)
        asiplot.plotIMmap(XG,YG,Zlla,title=T[i],cmap='Greys_r',clim=[500,4000])

    
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
            if int(date[:4]) <= 2011:
                rot = 2
            else:
                rot = 0
            tmp = np.rot90(im[i],rot)
#            tmp[tmp<=200] = np.nan
            asiplot.plotIMpolar(xgrid,ygrid,tmp,title=t[i],lon0=lon,lat0=lat,
                      clim=[300,600], cmap='Greens',norm_gamma=0.5)
            
if w2f:
    data = pa.returnRaw(folder, azfn=azfn,elfn=elfn,wl=wl,timelim=timelim)
    pa.write2HDF(data, h5fn=h5fn,wl=wl)

