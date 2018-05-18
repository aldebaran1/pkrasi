# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:08:12 2018

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
###############################################################################
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
###############################################################################
def plotIMpolar(x,y,z,title='',cmap='Greens',save=None):
    # Circles:
    azgrid = np.linspace(0,360,100)
    elgrid = np.linspace(0,80,100)
    rgrid = np.arange(10,80+1,20)
    anglegrid = np.arange(0,360,45)
    #Plot Image
    fig = plt.figure(figsize=(8,8))
    plt.title(title,y=1.05)
    plt.pcolormesh(x,y,z, cmap=cmap)
    # Plot rings
    for i in range(rgrid.shape[0]):
        xc = rgrid[i] * np.cos(np.deg2rad(azgrid))
        yc = rgrid[i] * np.sin(np.deg2rad(azgrid))
        plt.plot(xc,yc,'k',lw=1)
        # Labels
        xl = rgrid[i] * np.cos(np.deg2rad(90))
        yl = rgrid[i] * np.sin(np.deg2rad(90))
        plt.text(xl,yl,90-rgrid[i],fontsize=12,fontweight='bold',color='blue')
    for i in range(anglegrid.shape[0]):
        xc = elgrid * np.cos(np.deg2rad(anglegrid[i]))
        yc = elgrid * np.sin(np.deg2rad(anglegrid[i]))
        plt.plot(xc,yc,'k',lw=1)
    for i in range(anglegrid.shape[0]):
        if i == 0 or i > 5:
            xc = -82 * np.sin(np.deg2rad(anglegrid[i]))
            yc = 82 * np.cos(np.deg2rad(anglegrid[i]))
            plt.text(xc,yc,(360-anglegrid[i])%360,fontsize=12,fontweight='bold')
        elif i == 4 or i == 1 or i == 5:
            xc = -85 * np.sin(np.deg2rad(anglegrid[i]))
            yc = 85 * np.cos(np.deg2rad(anglegrid[i]))
            plt.text(xc,yc,360-anglegrid[i],fontsize=12,fontweight='bold')
        else:
            xc = -90 * np.sin(np.deg2rad(anglegrid[i]))
            yc = 90 * np.cos(np.deg2rad(anglegrid[i]))
            plt.text(xc,yc,360-anglegrid[i],fontsize=12,fontweight='bold')
    plt.axis('off')
    cbaxes = fig.add_axes([1.02, 0.1, 0.03, 0.8]) 
    plt.clim([0,4000])
    plt.colorbar(cax=cbaxes)
    plt.tight_layout()

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

#def returnASLatLonAlt(folder, wl=558, timelim=[], alt=130):
#    

# CFG:
plot = False
cfg = 'polar'
date = '20151008'
w2f = True
# Wavelength
wl = 428
# Folder
#folder_root = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\TheMahali\\data\\allskydata\\'
folder_root = 'G:\\Team Drive\\Semeter-Research in Progress\\DASC\\'
#h5fn = folder_root + date + '_' + str(wl) + '.h5'
folder = folder_root + date + '\\'
h5fn = folder + 'h5\\' + date + '_' + str(wl) + '.h5'
# ASI calibration files
as_cfg_folder = folder_root + 'cfg\\'
azfn = as_cfg_folder+'PKR_DASC_20110112_AZ_10deg.FITS'
elfn = as_cfg_folder+'PKR_DASC_20110112_EL_10deg.FITS'
#Interpolation grid
N = 512
# Mapping altitude
mapping_alt = 120*1e3
# --------------------------------------------------------------------------- #
# Read in the data
# Time stamp
t1 = datetime.now()
#
data = read_asi.load(folder,azfn=azfn,elfn=elfn, wavelenreq=wl)
obstimes = data.time.values.astype(datetime)
ts = gu.datetime2posix(obstimes)
az = data['az'].values
el = data['el'].values
if cfg == 'lla':
    # Map to altitude
    r = mapping_alt / np.sin(np.deg2rad(el))
    # Convert to WSG
    lat0 = data.lat
    lon0 = data.lon
    alt0 = data.alt_m
    lat, lon, alt = aer2geodetic(az,el,r,lat0,lon0,alt0)
elif cfg == 'polar':
    # Convert to polar projection
    rel = 90-el
    x = rel*np.cos(np.deg2rad(az))
    y = rel*np.sin(np.deg2rad(az))
    
# Write to HDF:
if w2f and h5fn:
    write2HDF(data,h5fn,wl)
    print (datetime.now() - t1)
    
# Plot the data:  Do -> until
if plot == True:
    c = 0
    for i in range(data.time.shape[0]):
    #    i+=110
    #    if c < 5:
        T = obstimes[i]
        if cfg == 'lla':
            im = np.rot90(data[558][i].values,0)
            #Interpolate Lat Lon
            xgrid, ygrid, Zim = interpolateAS(lon,lat,im,N=N)
            plotIMmap(xgrid,ygrid,Zim,title=T)
        if cfg == 'polar':
            im = np.rot90(data[558][i].values,1)
            #Interpolate Polar
            xgrid,ygrid,Zim = interpolatePolar(x,y,im,N)
            plotIMpolar(xgrid,ygrid,Zim,title=T)
        c += 1