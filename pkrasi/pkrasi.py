# -*- coding: utf-8 -*-
"""
Created on Sun May 20 11:54:26 2018

@author: smrak
"""

from datetime import datetime
import dascutils.io as read_asi
import numpy as np
import h5py
from pyGnss import gnssUtils as gu


from pymap3d import aer2geodetic
from scipy.interpolate import griddata


def interpolateAS(x,y,im,N,method='linear'):
    mask = np.ma.masked_invalid(x)
    x1 = x[~mask.mask]
    y1 = y[~mask.mask]
    im = im[~mask.mask]
    xgrid, ygrid = np.mgrid[np.nanmin(x).min():np.nanmax(x).max():N*1j, 
                              np.nanmin(y).min():np.nanmax(y).max():N*1j]
    Zim = griddata((x1,y1), im.ravel(), (xgrid, ygrid), method=method,fill_value=0)
    
    return xgrid, ygrid, Zim

def interpolatePolar(x,y,im,N,bounds=[-80,80],method='linear'):
    mask = np.ma.masked_invalid(x)
    x1 = x[~mask.mask]
    y1 = y[~mask.mask]
    im = im[~mask.mask]
    xgrid, ygrid = np.mgrid[bounds[0]:bounds[1]:N*1j, 
                            bounds[0]:bounds[1]:N*1j]
    Zim = griddata((x1,y1), im.ravel(), (xgrid, ygrid), method=method,fill_value=0)
    
    return xgrid, ygrid, Zim
###############################################################################
def write2HDF(data,h5fn,wl):
    obstimes = data.time.values.astype(datetime)
    ts = gu.datetime2posix(obstimes)
    az = data['az'].values
    el = data['el'].values
    images = data[wl].values
    N = az.shape[0]
    lat0 = data.lat
    lon0 = data.lon
    alt0 = data.alt_m
    
    try:
        f = h5py.File(h5fn,'w')
        d = f.create_group('DASC')
        d.attrs[u'converted'] = datetime.now().strftime('%Y-%m-%d')
        d.attrs[u'wavelength'] = '{}'.format(wl)
        d.attrs[u'image resolution'] = '{}'.format(N)
        d.attrs[u'PKR camera lon'] = '{}'.format(lon0)
        d.attrs[u'PKR camera lat'] = '{}'.format(lat0)
        d.attrs[u'PKR camera alt'] = '{}'.format(alt0)
        h5time = d.create_dataset('time', data=ts)
        h5time.attrs[u'time format'] = 'time format in POSIX time'
        d.create_dataset('az', data=az)
        d.create_dataset('el', data=el)
        
        h5img = d.create_dataset('img', data=images,compression=9)
        h5img.chunks
        h5img.attrs[u'Coordinates'] = 'Ntimes x Naz x Nel'
        # close file
        f.close()
    except Exception as e:
        raise (e)

def returnASLatLonAlt(folder,azfn=None,elfn=None,wl=558, timelim=[], alt=130,
                      Nim=512, asi=False):
    # Mapping altitude to meters
    mapping_alt = alt * 1e3
    # Read in the data utliizing DASCutils
    print ('Reading the data')
    t1 = datetime.now()
    data = read_asi.load(folder,azfn=azfn,elfn=elfn, wavelenreq=wl)
    print ('Data loaded in {}'.format(datetime.now()-t1))
    # Get time vector as datetime
    obstimes = data.time.values.astype(datetime)
    # Timle limits
    if len(timelim) > 0:
        idt = np.where( (obstimes >= timelim[0]) & (obstimes <= timelim[1]))[0]
    else:
        idt = np.arange(obstimes.shape[0])
    T = obstimes[idt]
    print ('Data reducted from {0} to {1}'.format(obstimes.shape[0], T.shape[0]))
    # Get Az and El grids
    az = data['az'].values
    el = data['el'].values
    # Image size
    if Nim is None or (not isinstance(Nim,int)):
        Nim = az.shape[0]
    ########################## Convert into WSG84 #############################
    # Map to altitude
    r = mapping_alt / np.sin(np.deg2rad(el))
    # Convert to WSG
    lat0 = data.lat
    lon0 = data.lon
    alt0 = data.alt_m
    lat, lon, alt = aer2geodetic(az,el,r,lat0,lon0,alt0)
    # Make an empty image array
    imlla = np.nan * np.ones((T.shape[0],Nim,Nim))
    c = 0
    for i in idt:
        print ('Processing-interpolating {}/{}'.format(c+1,idt.shape[0]))
        # Read a raw image
        im = np.rot90(data[wl][i].values,0)
        #Interpolate Lat Lon to preset Alt
        xgrid, ygrid, Zim = interpolateAS(lon,lat,im,N=Nim)
        # Assign to array
        imlla[c,:,:] = Zim
        c += 1
    
    if asi:
        return T, xgrid, ygrid, imlla, [lon0, lat0]
    else:
        return T, xgrid, ygrid, imlla