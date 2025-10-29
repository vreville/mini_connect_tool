import time
import numpy as np
from mini_connect_tool import fieldlines as sph
from mini_connect_tool import sph_harmonics as zdi
from mini_connect_tool import load_maps as mu
from mini_connect_tool import spacecraft_coords as sco
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy

def build_mag_fields(adapt_dir,list_files, r0=1, rss=2.5, lmax=15, real=0, v1=False, coeff=1):
    """
    Build 3D PFSS magnetic field cubes from a list of ADAPT/DeRosa maps.

    Parameters
    ----------
    adapt_dir : str
        ADAPT maps directory path.
    list_files : sequence of str
        Filenames (relative to adapt_dir) of map files to load.
    r0 : float, optional
        Inner radial boundary (Rsun units). Default 1.
    rss : float, optional
        Source surface radius (Rsun). Default 2.5.
    lmax : int, optional
        Maximum spherical harmonic degree for PFSS. Default 15.
    real : int, optional
        ADAPT Realisation index (when v1=False).
    v1 : bool, optional
        If True use DeRosa HDF5 format, if False use ADAPT FITS.
    coeff : float, optional
        Scaling coefficient applied when loading maps.

    Returns
    -------
    list
        [maps_time, theta, phi, rad, brr, btt, bpp] where
        - maps_time : array of datetimes for each input map
        - theta, phi : angular grids (radians)
        - rad : radial grid (Rsun)
        - brr, btt, bpp : arrays with shape (n_maps, n_rad, n_theta, n_phi)
          containing the PFSS mag field radial, colatitudinal and longitudinal components.
    """
    list_maps=[]
    tm=mu.tmap(adapt_dir+list_files[0],v1=v1,coeff=coeff,real=real)
    theta,phi=np.meshgrid(tm.theta,tm.phi,indexing='ij')
    SphHarm=zdi.mysph(lmax,theta,phi)
    tm.dmp_alm(SphHarm)
    list_maps.append(tm)

    for ll in list_files[1:]:
        tt=mu.tmap(adapt_dir+ll,v1=v1,coeff=coeff,real=real)
        tt.dmp_alm(SphHarm)
        list_maps.append(tt)

    maps_time=[tt.time for tt in list_maps]
    maps_time=np.array(maps_time)

    rad=np.linspace(1,rss,100)

    brr=[]
    btt=[]
    bpp=[]

    for ii,tt in enumerate(maps_time):
        br_time=[]
        bt_time=[]
        bp_time=[]

        for rr in rad:
            brss,btss,bpss=SphHarm.pfss3d(list_maps[ii].alm,rss,rb=1,rsph=rr)
            br_time.append(brss)
            bt_time.append(btss)
            bp_time.append(bpss)

        br_time=np.array(br_time)
        bt_time=np.array(bt_time)
        bp_time=np.array(bp_time)

        brr.append(br_time)
        btt.append(bt_time)
        bpp.append(bp_time)

    brr=np.array(brr)
    btt=np.array(btt)
    bpp=np.array(bpp)

    return [maps_time, tm.theta, tm.phi, rad, brr, btt, bpp]

def connect(maps_time, vr, rad, theta, phi, brr, btt, bpp, t1, t2, sc='Solar Orbiter'):
    """
    Connect spacecraft sampling times to photospheric maps and trace fieldlines.

    Parameters
    ----------
    maps_time : sequence of datetime
        Times corresponding to the magnetic maps used to build brr/btt/bpp.
    vr : pandas Series-like
        Solar wind speed time series indexed by datetimes (used to compute travel time).
    rad, theta, phi : arrays
        Grids used by the PFSS cubes (radial grid in Rsun, theta colatitude, phi longitude).
    brr, btt, bpp : arrays
        PFSS field component arrays shaped (n_maps, n_rad, n_theta, n_phi).
    t1, t2 : datetime
        Start and end times for spacecraft ephemeris query.
    sc : str, optional
        Spacecraft identifier passed to spacecraft_coords.spacecraft_coords (default 'Solar Orbiter').

    Returns
    -------
    list [coords, vsws, br_pfss, fieldlines, maps_indices, sun_times] where:
        - coords : spacecraft_coords.spacecraft_coords return object (with .dt, .rsc, .tsc, .psc, .pns, ...)
        - vsws : array of solar wind speeds used per sample
        - br_pfss : radial field at source surface mapped to spacecraft footpoints (G)
        - fieldlines : list of traced field lines (Cartesian arrays)
        - maps_indices : indices of maps used for each sample
        - sun_times : list of estimated photospheric times after travel_time correction
    """
    coords=sco.spacecraft_coords(t1,t2,sc=sc,delta=1)
    tms=[time.mktime(mm.timetuple()) for mm in maps_time]
    tms=np.array(tms)

    idx_vr=np.array([time.mktime(tt.timetuple()) for tt in vr.index])
    br_pfss=[]
    fieldlines=[]
    maps_indices=[]
    sun_times=[]
    pns=[]
    vsws=[]
    rss=0.99*rad[-1]

    for ii,dd in enumerate(coords.dt):

        ### Look for the closest map connected ###
        idx=np.argmin(np.abs(idx_vr-time.mktime(dd.timetuple())))
        Vsw=vr.values[idx]
        vsws.append(Vsw)

        travel_time=coords.rsc[ii]*6.957e5/Vsw # Constant wind speed
        sun_times.append(dd-timedelta(seconds=travel_time))
        tmd=time.mktime(dd.timetuple())-travel_time
        dist=tms-tmd
        if(tmd > tms[-1]):
            br = brr[-1,:,:,:]
            bt = btt[-1,:,:,:]
            bp = bpp[-1,:,:,:]
            maps_indices.append(len(maps_time)-1)

        elif(tmd < tms[0]):
            br = brr[0,:,:,:]
            bt = btt[0,:,:,:]
            bp = bpp[0,:,:,:]
            maps_indices.append(0)
        else:
            idx0=np.where(tms >= tmd)[0][0]
            br = brr[idx0,:,:,:]
            bt = btt[idx0,:,:,:]
            bp = bpp[idx0,:,:,:]
            maps_indices.append(idx0)

        ### Get coords in the rotating frame ###

        r=coords.rsc[ii]
        t=coords.tsc[ii]
        p=coords.psc[ii]
        pn=sco.ParkerSpiral(p,r,Rss=rss,Vsw=Vsw,Period=25.38)
        pns.append(pn)
        ibr=RegularGridInterpolator((rad, theta, phi), br)
        br_pfss.append(ibr((rss,t,pn))*(rss/r)**2)

        xfl=rss*np.sin(t)*np.cos(pn)
        yfl=rss*np.sin(t)*np.sin(pn)
        zfl=rss*np.cos(t)

        seed_points=np.array([[xfl,yfl,zfl]])
        fieldline=sph.compute_fieldlines(seed_points,rad,theta,phi,br,bt,bp,Sunward=True)
        fieldlines.append(fieldline)

    coords.pns = np.array(pns)
    br_pfss=np.array(br_pfss)*1e5

    return [coords, vsws, br_pfss, fieldlines, maps_indices, sun_times]

def compute_expansion(coords, fieldlines, maps_indices, rad, theta, phi, brr, btt, bpp):
    """
    Compute expansion factors and footpoint locations from traced fieldlines.

    Parameters
    ----------
    coords : spacecraft coords object
        The coords object returned by spacecraft_coords.spacecraft_coords (contains .tsc, .psc, .rsc, .pns).
    fieldlines : list
        List of fieldline traces produced by connect (each element is fieldline coordinates).
    maps_indices : sequence of int
        Map index used for each sample in fieldlines.
    rad, theta, phi : arrays
        Grids used by the PFSS cubes.
    brr, btt, bpp : arrays
        PFSS mag field component arrays shaped (n_maps, n_rad, n_theta, n_phi).

    Returns
    -------
    list [lon_path, lat_path, frs, fexps, bsurfs] where:
        - lon_path, lat_path : arrays of footpoint longitude/latitude (deg)
        - frs : list of radial distance arrays along each fieldline
        - fexps : list of expansion factor arrays along each fieldline
        - bsurfs : list/array of photospheric field strength values for each footpoint
    """
    lon_path=[]
    lat_path=[]
    frs=[]
    fexps=[]
    bsurfs=[]
    rss=0.99*rad[-1]

    for ii,fl in enumerate(fieldlines):
        fx=fl[0][0]
        fy=fl[0][1]
        fz=fl[0][2]
        fr=np.sqrt(fx**2+fy**2+fz**2)

        if fr[0] > fr[1]:
            fx=fx[::-1]
            fy=fy[::-1]
            fz=fz[::-1]
            fr=np.sqrt(fx**2+fy**2+fz**2)

        frcyl=np.sqrt(fx**2+fy**2)
        ft=np.pi/2.-np.arctan(fz/frcyl)
        fp=np.arctan2(fy,fx)%(2*np.pi)
        flon=(fp*180/np.pi)%(360)
        flat=(np.pi/2.-ft)*180/np.pi


        lon_path.append(flon[0])
        lat_path.append(flat[0])

        # compute fexp

        idm=maps_indices[ii]
        btot = np.sqrt(brr[idm,:,:,:]**2+btt[idm,:,:,:]**2+bpp[idm,:,:,:]**2)
        ibtot = RegularGridInterpolator((rad, theta, phi), btot)
        balong = ibtot((fr, ft, fp))
        bsurf = ibtot((fr[0], ft[0], fp[0]))
        bss = ibtot((rss, coords.tsc[ii], coords.pns[ii]))

        fexp= bsurf/balong*(fr[0]/fr)**2
        frs.append(fr)
        fexps.append(fexp)
        bsurfs.append(bsurf)

    return [np.array(lon_path), np.array(lat_path), frs, fexps, bsurfs]

