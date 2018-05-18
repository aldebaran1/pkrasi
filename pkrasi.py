# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:08:12 2018

@author: smrak
"""
from datetime import datetime
import dascutils.io as read_asi
import numpy as np
import xarray
import matplotlib.pyplot as plt
from pymap3d import aer2geodetic
from scipy.interpolate import griddata
from cartomap import geogmap as ggm
import cartopy.crs as ccrs

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

def plotIMmap(x,y,z, clim=[0,4000],title='',save=False):
    # Figure limits
    maplatlim = [58,72]
    maplonlim = [-162,-132]
    meridians = [-165,-155,-145,-135,-125]
    parallels = [55,60,65,70,75]
    # Plot
    ax = ggm.plotCartoMap(latlim=maplatlim,lonlim=maplonlim,
                      meridians=meridians, parallels=parallels,
                      projection='merc',title=title)
    plt.plot(lon0,lat0, 'xr',ms=10,transform=ccrs.PlateCarree())
    plt.pcolormesh(xgrid,ygrid,Zim,cmap='Greens',transform=ccrs.PlateCarree())
    plt.clim(clim)
    plt.colorbar()
    plt.show()
    

date = '20151007'
# Folder
folder_root = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\TheMahali\\data\\allskydata\\'
folder = folder_root + date + '\\'
# Wavelength
wl = 558
as_cfg_folder = folder_root + 'cfg\\'
azfn = as_cfg_folder+'PKR_DASC_20110112_AZ_10deg.FITS'
elfn = as_cfg_folder+'PKR_DASC_20110112_EL_10deg.FITS'
#Interpolation grid
N = 512
# Mapping altitude
mapping_alt = 120*1e3
# Read in the data
data = read_asi.load(folder,azfn=azfn,elfn=elfn, wavelenreq=wl)
az = data['az'].values
el = data['el'].values
# Map to altitude
r = mapping_alt / np.sin(np.deg2rad(el))
# Convert to WSG
lat0 = data.lat
lon0 = data.lon
alt0 = data.alt_m
lat, lon, alt = aer2geodetic(az,el,r,lat0,lon0,alt0)
# Conver to polar projection
rel = 90-el
x = rel*np.cos(np.deg2rad(az))
y = rel*np.sin(np.deg2rad(az))
# Circles:
azgrid = np.linspace(0,360,100)
elgrid = np.linspace(0,80,100)
rgrid = np.arange(0,80+1,20)
anglegrid = np.arange(0,360,45)
# Do -> until
c = 0
for i in range(data.time.shape[0]):
    if c < 1:
        im = data[558][i].values
        T = data.time.values.astype(datetime)[i]
        #Interpolate Lat Lon
#        xgrid, ygrid, Zim = interpolateAS(lon,lat,im,N=N)
        #Interpolate Polar
        xgrid,ygrid,Zim = interpolatePolar(x,y,im,N)# = griddata((x,y), im.ravel(), (polar_x,polar_y), method='linear')
        # Plotting
        fig = plt.figure(figsize=(8,8))
        plt.pcolormesh(xgrid,ygrid,im, cmap='Greens')
        # Plot rings
        for i in range(rgrid.shape[0]):
            xc = rgrid[i] * np.cos(np.deg2rad(azgrid))
            yc = rgrid[i] * np.sin(np.deg2rad(azgrid))
            plt.plot(xc,yc,'k',lw=1)
        for i in range(anglegrid.shape[0]):
            xc = elgrid * np.cos(np.deg2rad(anglegrid[i]))
            yc = elgrid * np.sin(np.deg2rad(anglegrid[i]))
            plt.plot(xc,yc,'k',lw=1)
        for i in range(anglegrid.shape[0]):
            xc = 85 * np.cos(np.deg2rad(anglegrid[i]))
            yc = 85 * np.sin(np.deg2rad(anglegrid[i]))
            plt.text(xc,yc,anglegrid[i],fontsize=12,fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
#        plotIMmap(xgrid,ygrid,Zim,title=T)
        c += 1
    else:
        break
