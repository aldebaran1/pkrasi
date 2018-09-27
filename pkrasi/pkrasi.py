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
import matplotlib.pyplot as plt
from pymap3d import aer2geodetic
from scipy.interpolate import griddata

from  scipy.spatial import Delaunay

def interp_weights(xyz, uvw,d=3):
    tri = Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

def interpolateCoordinate(x,N=1024,method='linear'):
    x0,y0 = np.meshgrid(np.arange(x.shape[0]),
                        np.arange(x.shape[1]))
    mask = np.ma.masked_invalid(x)
    x0 = x0[~mask.mask]
    y0 = y0[~mask.mask]
    X = x[~mask.mask]
    x1,y1 = np.mgrid[0:x.shape[0]:N*1j, 
                     0:x.shape[1]:N*1j]
    z = griddata((x0,y0), X.ravel(), (x1, y1), method=method)
    return z

def interpolateAS(x,y,im,N,method='linear'):
    if x.shape[0] == im.shape[0]:
        mask = np.ma.masked_invalid(x)
        x1 = x[~mask.mask]
        y1 = y[~mask.mask]
        im = im[~mask.mask]
        xgrid, ygrid = np.mgrid[np.nanmin(x).min():np.nanmax(x).max():N*1j, 
                                np.nanmin(y).min():np.nanmax(y).max():N*1j]
        Zim = griddata((x1,y1), im.ravel(), (xgrid, ygrid), 
                       method=method,fill_value=0)
    else:
        N1 = im.shape[0]
        x1,y1 = np.mgrid[np.nanmin(x).min():np.nanmax(x).max():N1*1j, 
                         np.nanmin(y).min():np.nanmax(y).max():N1*1j]
        mask = np.ma.masked_invalid(im)
        x1 = x1[~mask.mask]
        y1 = y1[~mask.mask]
        im = im[~mask.mask]
        xgrid, ygrid = np.mgrid[np.nanmin(x).min():np.nanmax(x).max():N*1j, 
                                np.nanmin(y).min():np.nanmax(y).max():N*1j]
        Zim = griddata((x1,y1), im.ravel(), (xgrid, ygrid), method=method)
    return xgrid, ygrid, Zim

def interpolatePolar(x,y,im,N,bounds=[-80,80],method='linear'):
    if x.shape[0] == im.shape[0]:
        mask = np.ma.masked_invalid(x)
        x1 = x[~mask.mask]
        y1 = y[~mask.mask]
        im = im[~mask.mask]
        xgrid, ygrid = np.mgrid[bounds[0]:bounds[1]:N*1j, 
                                bounds[0]:bounds[1]:N*1j]
        Zim = griddata((x1,y1), im.ravel(), (xgrid, ygrid), 
                       method=method,fill_value=0)
    else:
        # Resize image to a size of the grid:
        N1 = im.shape[0]
        mask = np.ma.masked_invalid(im)
        x1,y1 = np.mgrid[bounds[0]:bounds[1]:N1*1j, 
                         bounds[0]:bounds[1]:N1*1j]
        x1 = x1[~mask.mask]
        y1 = y1[~mask.mask]
        im = im[~mask.mask]
        xgrid, ygrid = np.mgrid[bounds[0]:bounds[1]:N*1j, 
                                bounds[0]:bounds[1]:N*1j]
        Zim = griddata((x1,y1), im.ravel(), (xgrid, ygrid), method=method)
    return xgrid, ygrid, Zim
###############################################################################
def write2HDF(data,h5fn,wl):
    obstimes = data.time.values.astype(datetime)
    ts = gu.datetime2posix(obstimes)
    az = data.az[1]
    el = data.el[1]
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
###############################################################################
def writeInterpolated2HDF(t,lon,lat,h,images,h5fn,lon0=0,lat0=0,alt0=0,N=None,wl=0):
    if isinstance(t[0],datetime):
        t = gu.datetime2posix(t)
    if N == 0 or N is None:
        N = lon.shape[0]
    try:
        f = h5py.File(h5fn,'w')
        d = f.create_group('DASC')
        d.attrs[u'converted'] = datetime.now().strftime('%Y-%m-%d')
        d.attrs[u'wavelength'] = '{}'.format(wl)
        d.attrs[u'image resolution'] = '{}'.format(N)
        d.attrs[u'PKR camera lon'] = '{}'.format(lon0)
        d.attrs[u'PKR camera lat'] = '{}'.format(lat0)
        d.attrs[u'PKR camera alt'] = '{}'.format(alt0)
        
        h5time = d.create_dataset('time', data=t)
        h5time.attrs[u'time format'] = 'time format in POSIX time'
        
        d.create_dataset('lon', data=lon)
        d.create_dataset('lat', data=lat)
        H = d.create_dataset('alt', data=h)
        H.attrs[u'Mapping height'] = '{}'.format(h)
        
        h5img = d.create_dataset('img', data=images,compression=9)
        h5img.chunks
        h5img.attrs[u'Coordinates'] = 'Ntimes x Nlon x Nlat'
        # Close file
        f.close()
    except Exception as e:
        raise (e)
###############################################################################
def readtInterpolatedHDF(h5fn):
    f = h5py.File(h5fn, 'r')
    t = f['DASC/time'].value
    lon = f['DASC/lon'].value
    lat = f['DASC/lat'].value
    imstack = f['DASC/img'].value
    # Check the observation time instance. Change to datetime if necessary
    if not isinstance(t[0], datetime):
        t = np.array([datetime.utcfromtimestamp(ts) for ts in t])
    # Close the file
    f.close()
    return t, lon, lat, imstack

def readPolarHDF(h5fn):
    f = h5py.File(h5fn, 'r')
    t = f['DASC/time'].value
    xgrid = f['DASC/xgrid'].value
    ygrid = f['DASC/ygrid'].value
    imstack = f['DASC/img'].value
    # Check the observation time instance. Change to datetime if necessary
    if not isinstance(t[0], datetime):
        t = np.array([datetime.utcfromtimestamp(ts) for ts in t])
    # Close the file
    f.close()
    return t, xgrid, ygrid, imstack
###############################################################################
def returnRaw(folder,azfn=None,elfn=None,wl=558, timelim=[]):
    t1 = datetime.now()
    data = read_asi.load(folder,azfn=azfn,elfn=elfn, wavelenreq=wl, 
                         treq=timelim)
    print ('Data loaded in {}'.format(datetime.now()-t1))
    
    return data

def returnASLatLonAlt(folder,azfn=None,elfn=None,wl=558,timelim=[],alt=130,
                      Nim=512,el_filter=None,asi=False, verbose=False):
    # Mapping altitude to meters
    mapping_alt = alt * 1e3
    # Read in the data utliizing DASCutils
    print ('Reading the PKR asi images')
    t1 = datetime.now()
    data = read_asi.load(folder,azfn=azfn,elfn=elfn, wavelenreq=wl,
                         treq=timelim,verbose=verbose)
    print ('Data loaded in {}'.format(datetime.now()-t1))
    # Get time vector as datetime
    T = data.time.values.astype(datetime)
    # Get Az and El grids
    az = data.az[1]
    el = data.el[1]
    # Reshape image calibration if needed
    im_test = data[wl][0].values
    if im_test.shape[0] != el.shape[0]:
        el = interpolateCoordinate(el,N=im_test.shape[0])
        az = interpolateCoordinate(az,N=im_test.shape[0])
    # Elivation filter/mask
    if el_filter is not None and isinstance(el_filter,int):
        el = np.where(el>=el_filter,el,np.nan)
        az = np.where(el>=el_filter,az,np.nan)
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
#    c = 0
    for i in range(T.shape[0]):
        print ('Processing-interpolating {}/{}'.format(i+1,T.shape[0]))
        # Read a raw image
        im = data[wl][i].values
        #Interpolate Lat Lon to preset Alt
        xgrid, ygrid, Zim = interpolateAS(lon,lat,im,N=Nim)
        # Assign to an array
        imlla[i,:,:] = Zim
#        c += 1
    
    if asi:
        return T, xgrid, ygrid, imlla, [lon0, lat0, alt0]
    else:
        return T, xgrid, ygrid, imlla

def returnASpolar(folder,azfn=None,elfn=None, wl=558, 
                  timelim=[], Nim=512, asi=False, el_filter=None):
    # Read in the data utliizing DASCutils
    print ('Reading the data')
    t1 = datetime.now()
    data = read_asi.load(folder,azfn=azfn,elfn=elfn, wavelenreq=wl, treq=timelim)
    print ('Data loaded in {}'.format(datetime.now()-t1))
    # Get time vector as datetime
    T = data.time.values.astype(datetime)
    print ('Data reducted from {0} to {1}'.format(T.shape[0], T.shape[0]))
    # Get Az and El grids
    az = data.az[1]
    el = data.el[1]
    # Camera position
    lat0 = data.lat
    lon0 = data.lon
    alt0 = data.alt_m
    # Reshape image calibration if needed
    im_test = data[wl][0].values
    if im_test.shape[0] != el.shape[0]:
        el = interpolateCoordinate(el,N=im_test.shape[0])
        az = interpolateCoordinate(az,N=im_test.shape[0])
    # Elivation filter/mask
    if el_filter is not None and isinstance(el_filter,int):
        el = np.where(el>=el_filter,el,np.nan)
        az = np.where(el>=el_filter,az,np.nan)
    # Image size
    if Nim is None or (not isinstance(Nim,int)):
        Nim = az.shape[0]
    # Prepare a polar projection to cartesian
    rel = 90-el
    x = rel*np.cos(np.deg2rad(az))
    y = rel*np.sin(np.deg2rad(az))
    # Mask NaNs
#    mask = np.ma.masked_invalid(x)
#    x = x[~mask.mask]
#    y = y[~mask.mask]
    # Make an empty image array
    imae = np.nan * np.ones((T.shape[0],Nim,Nim))
    # Interpolation grid
    # Input grid
#    X,Y = np.mgrid[np.nanmin(x):np.nanmax(x):x.shape[0]*1j, 
#                   np.nanmin(y):np.nanmax(y):y.shape[0]*1j]
#    xy=np.zeros([X.shape[0] * x.shape[1],2])
#    xy[:,0]=X.flatten()
#    xy[:,1]=Y.flatten()
    
    # Output grid
#    uv=np.zeros([Nim*Nim,2])
#    Xi, Yi = np.mgrid[-80:80:Nim*1j, 
#                      -80:80:Nim*1j]
#    uv[:,0]=Yi.flatten()
#    uv[:,1]=Xi.flatten()
#    vtx, wts = interp_weights(xy, uv)
    for i in range(T.shape[0]):
        print ('Processing-interpolating {}/{}'.format(i+1,T.shape[0]))
        # Read a raw image
        im = np.rot90(data[wl][i].values,-1)
        #Interpolate Polar
        xgrid,ygrid,Zim = interpolatePolar(x,y,im,Nim)
#        Zim=interpolate(im.flatten(), vtx, wts)
        Zim=Zim.reshape(Nim,Nim)
        # Assign to array
        imae[i,:,:] = Zim
    if asi:
        return T, xgrid, ygrid, imae, [lon0, lat0, alt0]
    else:
        return T, xgrid, ygrid, imae
    