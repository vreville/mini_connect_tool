""" Compute the source points of Solar Orbiter
for VSTP (very short term planning, some data may not be available) """

import os
import numpy as np
import pandas as pd
from mini_connect_tool import mini_connect_tool as mct
from mini_connect_tool import load_maps as load
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime, timedelta
from matplotlib.colors import to_rgb
import speasy as spz
from copy import deepcopy
from astropy.io import fits
import sunpy.visualization.colormaps

# Times                                                                                                                        
start = "2025-10-25T23:00:00"
stop  = "2025-10-30T23:59:59"

t1=datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
t2=datetime.strptime(stop,  "%Y-%m-%dT%H:%M:%S")

start_time, stop_time = t1, t2

try:
    b_rtn_slo = spz.get_data("amda/solo_b_rtn", start, stop)
    b_rtn_slo = b_rtn_slo.to_dataframe()
    bdata=True
except:
    bdata=False

try:
    v_rtn_slo = spz.get_data("amda/pas_momgr1_v_rtn", start, stop)
    v_rtn_slo = v_rtn_slo.to_dataframe().ffill()
except:
    dummy_speed=400
    hours=pd.date_range(t1,t2, freq="h")
    vr=np.ones(len(hours))+2*(np.random.rand(len(hours))-0.5)*0.1
    vr*=dummy_speed
    v_rtn_slo = pd.DataFrame({'Index':hours, 'vr':vr})
    v_rtn_slo = v_rtn_slo.set_index('Index')

adapt_dir=os.path.dirname(__file__)+"../data/adapt_carrington/"
    
t1_map = t1 - timedelta(days=5)
t2_map = t2

map_files, map_times = load.dl_adapt_carrington_maps(t1_map, t2_map, cadence_hours=24)
#maps_files=[maps_files[-1]] # To go faster
sdo_files, sdo_times = load.dl_sdo_carrington_maps(map_times)

rss=2.5
lmax=25

maps_time, theta, phi, rad, brr, btt, bpp = mct.build_mag_fields(adapt_dir, map_files, rss=rss, lmax=lmax, real=7)
coords1, vsws1, br_pfss1, fieldlines1, maps_indices1, sun_times = mct.connect(maps_time, v_rtn_slo.vr, rad, theta, phi, brr, btt, bpp, t1, t2, sc="Solar Orbiter")
lon_slo, lat_slo, frs, fexps, bsurfs = mct.compute_expansion(coords1, fieldlines1, maps_indices1, rad, theta, phi, brr, btt, bpp)

# Plot
for ii, dt in enumerate(coords1.dt):

    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(14,9))

    if bdata:
        ax1.scatter(b_rtn_slo.index, b_rtn_slo.br, s=0.5, color="black", alpha=0.1)

    ax1.plot(coords1.dt[:ii+1],br_pfss1[:ii+1],linewidth=2, color="C0", label="Solar Orbiter")
    ax1.plot(coords1.dt,np.zeros(len(br_pfss1)),color="black")
    ax1.set_xlim([coords1.dt[0],coords1.dt[-1]])
    ax1.set_yscale("symlog")
    ax1.set_ylabel(r"$B_r$ (nT) SolO", fontsize=15)
    ax1.legend(loc="best")

    ax3.plot(v_rtn_slo.index, v_rtn_slo.vr, color="black")
    ax3.set_ylabel(r"$V_r$ (km/s)", fontsize=15)
    ax3.set_ylim([0,v_rtn_slo.vr.max()*1.5])

    maxb=np.max(np.abs(brr[maps_indices1[ii],0,:,:]))
    ax2.pcolormesh(phi*180/np.pi,(np.pi/2-theta)*180/np.pi,brr[maps_indices1[ii],0,:,:],cmap="bwr", vmin=-maxb, vmax=maxb)
    ax2.contour(phi*180/np.pi,(np.pi/2-theta)*180/np.pi,brr[maps_indices1[ii],-1,:,:],levels=[0.0],colors="black")

    ax2.annotate("{}".format(maps_time[maps_indices1[ii]]), (20, 75), color="black")
    ax2.set_title("Suntime = {}".format(sun_times[ii]))

    sdoidx=np.argmin(np.abs(np.array(sdo_times)-maps_time[maps_indices1[ii]]))
    sdo_file=sdo_files[sdoidx]
    sdoimg=Image.open(sdo_file)
    sdoimg=sdoimg.convert("L")
    sdoimg=np.array(sdoimg)

    crlon=np.linspace(0,360,sdoimg.shape[1]+1)
    crlat=np.linspace(-90,90,sdoimg.shape[0]+1)

    ax4.pcolormesh(crlon, crlat, sdoimg[::-1,:], cmap="sdoaia193")
    ax4.set_ylim([-80,80])
    ax4.annotate("{}".format(sdo_times[sdoidx].isoformat(timespec="seconds")), (20, 65), color="white")
    ax4.set_ylim([-80,80])
    
    idx0=max(0,ii+1-10)
    r, g, b = to_rgb("C0")
    color=[(r,g,b,alpha) for alpha in np.linspace(0,1,len(lat_slo[idx0:ii+1]))]
    
    ax2.scatter(lon_slo[idx0:ii+1], lat_slo[idx0:ii+1], color=color)
    ax2.scatter(lon_slo[ii], lat_slo[ii], color="C0", edgecolors="black")

    ax4.scatter(lon_slo[idx0:ii+1], lat_slo[idx0:ii+1], color=color)
    ax4.scatter(lon_slo[ii], lat_slo[ii], color="C0", edgecolors="black")
    
    ax4.set_title(r"AIA 193 Carrington", fontsize=15)
    fig.suptitle("SC time = {}".format(dt))

    fig.autofmt_xdate()
    fig.savefig("mov_vstp_{}.png".format(dt), bbox_inches="tight")
    plt.close(fig)


    


