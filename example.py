# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:04:20 2018

@author: smrak
"""
import numpy as np
import h5py
from datetime import datetime
from pkrasi import pkrasi as pa
from pkrasi import plotting as asiplot
from pyGnss import gnssUtils as gu
from scipy.interpolate import griddata
from pymap3d import aer2geodetic
import matplotlib.pyplot as plt

import scipy.spatial.qhull as qhull

def interp_weights(xy, uv,d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

# CFG:
el_filter = 20
plot = True
cfg = 'polar'
cfg = 'lla'
read = 1
#cfg = 'testinterp'
#cfg = None
w2f = False
steve = 1
folder_root = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\TheMahali\\data\\allskydata\\'

if steve:
    date = '20080326'
    wl= 0
    folder = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\steve\\data\\pkr\\'
    h5fn = folder + 'raw_'+date + '_' + str(wl) + '.h5'
    savepolar = folder+'polar\\'
    as_cfg_folder = folder_root + 'cfg\\'
    azfn = as_cfg_folder+'PKR_20111006_AZ_10deg.FITS'
    elfn = as_cfg_folder+'PKR_20111006_EL_10deg.FITS'
    timelim = [datetime(2008,3,26,11,43,0), datetime(2008,3,26,11,44,0)]
    timelim = [datetime(2008,3,26,7,35,0), datetime(2008,3,26,7,36,0)]
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
if cfg == 'testinterp':
    data = pa.returnRaw(folder, azfn=azfn,elfn=elfn,wl=wl,timelim=timelim)
    T = data.time.values.astype(datetime)
    az = data.az[1]
    el = data.el[1]
    im_test = data[wl][0].values
    # Shrink the calibration file
    el = pa.interpolateCoordinate(el,N=im_test.shape[0])
    az = pa.interpolateCoordinate(az,N=im_test.shape[0])
    # Prepare a polar projection to cartesian
    rel = 90-el
    x = rel*np.cos(np.deg2rad(az))
    y = rel*np.sin(np.deg2rad(az))
    # Mask nans
    mask = np.ma.masked_invalid(x)
    X = x[~mask.mask]
    Y = y[~mask.mask]
    # Interpolation projection: Given grid
    xy=np.zeros((X.shape[0],2))
    xy[:,0] = X
    xy[:,1] = Y
    # Interpolation projection: New grid
    uv=np.zeros([N*N,2])
    xgrid, ygrid = np.mgrid[np.nanmin(x):np.nanmax(x):N*1j,
                            np.nanmin(y):np.nanmax(y):N*1j]
    uv[:,0] = xgrid.ravel()
    uv[:,1] = ygrid.ravel()
    # Make an interpolation frame
    vtx, wts = interp_weights(xy, uv)
    # Get an image
    image = np.rot90(data[wl][-1].values,1)
    image = image[~mask.mask]
    # Interpolate
    im=interpolate(image, vtx, wts)
    im=im.reshape(xgrid.shape[0],xgrid.shape[1])
    title = 'DASC: {} UT'.format(T[0])
    imgname = datetime.strftime(T[0],'%H%M%S') + 'a.png'
    fig = asiplot.plotIMpolar(xgrid,ygrid,im,clim=[300,600], title=title,
                        cmap='Greys_r',norm_gamma=0.5,savefn=savepolar+imgname)
    plt.show(fig)
    
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
    t, xgrid, ygrid, im, [lon,lat,alt]= pa.returnASLatLonAlt(folder, azfn=azfn, elfn=elfn, wl=wl, 
                                            timelim=timelim, alt=mapping_alt,
                                            Nim=N,asi=True)
    if plot:
        for i in range(t.shape[0]):
            tmp = im[i]
            tmp[tmp<=300] = np.nan
            asiplot.plotIMmap(xgrid,ygrid,tmp,title=t[i],lon0=lon,lat0=lat,alt=mapping_alt,
                      clim=[300,600], cmap='Greys_r',norm_gamma=0.5)
if cfg == 'polar':
    if read:
        t, xgrid, ygrid, im = pa.readPolarHDF(h5fn)
    else:
        t, xgrid, ygrid, im, [lon,lat,alt] = pa.returnASpolar(
                folder, azfn=azfn, elfn=elfn, wl=wl, timelim=timelim, 
                Nim=N,asi=True)
    
    if plot:
        for i in range(t.shape[0]):
            if int(date[:4]) <= 2011:
                rot = 2
            else:
                rot = 0
            tmp = np.rot90(im[i],rot)
            title = 'DASC: {} UT'.format(t[i])
            imgname = datetime.strftime(t[i],'%H%M%S') + '.png'
            tmp[tmp<=200] = np.nan
            fig = asiplot.plotIMpolar(xgrid,ygrid,tmp,title=title,figure=True,
                      clim=[300,600], cmap='Greys_r',norm_gamma=0.5,
                      savefn=savepolar+imgname)
            
    if w2f:
        ts = gu.datetime2posix(t)
        try:
            f = h5py.File(h5fn,'w')
            d = f.create_group('DASC')
            d.attrs[u'converted'] = datetime.now().strftime('%Y-%m-%d')
            d.attrs[u'wavelength'] = '{}'.format(wl)
            d.attrs[u'image resolution'] = '{}'.format(N)
            d.attrs[u'PKR camera lon'] = '{}'.format(lon)
            d.attrs[u'PKR camera lat'] = '{}'.format(lat)
            d.attrs[u'PKR camera alt'] = '{}'.format(alt)
            h5time = d.create_dataset('time', data=ts)
            h5time.attrs[u'time format'] = 'time format in POSIX time'
            d.create_dataset('xgrid', data=xgrid)
            d.create_dataset('ygrid', data=ygrid)
            h5img = d.create_dataset('img', data=im,compression=9)
            h5img.chunks
            h5img.attrs[u'Coordinates'] = 'Ntimes x Naz x Nel'
            # close file
            f.close()
        except Exception as e:
            raise (e)

