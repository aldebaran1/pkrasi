# -*- coding: utf-8 -*-
"""
Created on Sun May 20 11:51:18 2018

@author: smrak
"""
import numpy as np

from cartomap import geogmap as ggm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs


def plotIMmap(x,y,z,title='',save=False,lon0=None,lat0=None,alt=None,
              clim=[0,3000],cmap='Greens',norm_gamma=None,figure=False):
    # Figure limits
    maplatlim = [58,72]
    maplonlim = [-162,-132]
    meridians = [-165,-155,-145,-135,-125]
    parallels = [55,60,65,70,75]
    # Plot
    ax = ggm.plotCartoMap(latlim=maplatlim,lonlim=maplonlim,
                      meridians=meridians, parallels=parallels,
                      projection='merc',title=title)
    if lon0 is not None and lat0 is not None:
        plt.plot(lon0,lat0, 'xr',ms=10,transform=ccrs.PlateCarree())
    if alt is not None:
        plt.text(np.mean(maplonlim)+0.2*np.diff(maplonlim)[0], maplatlim[1], 
                 '{}km'.format(alt), fontsize=12,fontweight='bold',color='blue',
                 transform=ccrs.PlateCarree())
    gca = plt.pcolormesh(x,y,z,cmap=cmap,transform=ccrs.PlateCarree())
    if norm_gamma is not None:
        gca.set_norm(colors.PowerNorm(gamma=norm_gamma))
    plt.clim(clim)
    plt.colorbar()
    if figure:
        pass
#        return fig
    plt.show()
###############################################################################
def plotIMpolar(x,y,z,title='',clim=[0,4000],cmap='Greens',save=None,
                norm_gamma=None,figure=False,
                figsize=(8,8),savefn='',DPI=100):
    # Circles:
    azgrid = np.linspace(0,360,100)
    elgrid = np.linspace(0,80,100)
    rgrid = np.arange(10,80+1,20)
    anglegrid = np.arange(0,360,45)
    #Plot Image
    fig = plt.figure(figsize=figsize)
    plt.title(title,y=1.05)
    gca = plt.pcolormesh(x,y,z, cmap=cmap)
#    return fig
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
            plt.text(xc,yc,anglegrid[i],fontsize=12,fontweight='bold')
        elif i == 4 or i == 1 or i == 5:
            xc = -85 * np.sin(np.deg2rad(anglegrid[i]))
            yc = 85 * np.cos(np.deg2rad(anglegrid[i]))
            plt.text(xc,yc,anglegrid[i],fontsize=12,fontweight='bold')
        else:
            xc = -90 * np.sin(np.deg2rad(anglegrid[i]))
            yc = 90 * np.cos(np.deg2rad(anglegrid[i]))
            plt.text(xc,yc,anglegrid[i],fontsize=12,fontweight='bold')
    plt.axis('off')
    cbaxes = fig.add_axes([1.03, 0.1, 0.03, 0.8]) 
    if norm_gamma is not None:
        gca.set_norm(colors.PowerNorm(gamma=norm_gamma))
    gca.set_clim(clim)
    plt.colorbar(cax=cbaxes)
#    plt.tight_layout()
    if savefn is not None or savefn != False or savefn != '':
        plt.savefig(savefn,dpi=DPI)
        plt.close(fig)
    if figure:
        return fig
    else:
        plt.show()