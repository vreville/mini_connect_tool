import os
import numpy as np
from datetime import datetime, timedelta
from sunpy import coordinates
from sunpy.data import cache
import astropy.units as u
from astropy.time import TimeDelta
cache._expiry = TimeDelta(100 * u.year)
from sunpy.coordinates import spice as sspice

Juno=True

def ParkerSpiral(phi,r,Rss=2.5,Vsw=4e2,Period=25.38,acceleration=False):
    """
    Compute the Parker spiral longitudinal displacement for a point at radius r.

    Parameters
    ----------
    phi : float or array_like
        Initial heliographic longitude (radians).
    r : float or array_like
        Radial distance in solar radii (Rsun).
    Rss : float, optional
        Source-surface radius in Rsun (default 2.5).
    Vsw : float, optional
        Solar wind speed in km/s (default 4e2).
    Period : float, optional
        Solar rotation period in days (used to compute angular speed, default 25.38).
    acceleration : bool, optional
        If True use an accelerating wind profile, otherwise use a constant wind speed.

    Returns
    -------
    float or array_like
        Longitude(s) corrected for Parker spiral, in range [0, 2*pi).
    """
    Rsunkm=6.957e5
    Om=2*np.pi/(Period*24*60*60)
    
    if acceleration==True:
        phi=phi-(1-r/Rss+np.log(r/Rss))*Rss*Rsunkm*Om/Vsw
    else:
        phi=phi+Om*Rsunkm/Vsw*(r-Rss)

    return (phi)%(2*np.pi)
    
Body_dict={0:'SOLAR SYSTEM BARYCENTER',
           1:'MERCURY BARYCENTER',
           2:'VENUS BARYCENTER',
           3:'EARTH-MOON BARYCENTER',
           4:'MARS BARYCENTER',
           5:'JUPITER BARYCENTER',
           6:'SATURN BARYCENTER',
           7:'URANUS BARYCENTER',
           8:'NEPTUNE BARYCENTER',
           9:'PLUTO BARYCENTER',
           10:'SUN',
           199:'MERCURY',
           299:'VENUS',
           399:'EARTH',
           301:'MOON',
           499:'MARS',
           599:'JUPITER',
           699:'SATURN',
           799:'URANUS',
           899:'NEPTUNE',
           -96:'PARKER SOLAR PROBE',
           -144:'SOLAR ORBITER',
           -234:'STEREO A',
           -61:'JUNO',
           -121:'BEPI COLOMBO MPO'}

def check_kernel():
    """
    Print information about loaded SPICE SPK kernels.

    The function queries the SPK kernel pool and prints the filename,
    the body IDs covered by each file and their coverage windows.
    Useful to verify which ephemeris kernels are currently loaded.
    """
    # Check the number of SPK type kernels loaded
    n_spk_ker = sspice.spiceypy.ktotal("SPK")

    for k in range(0,n_spk_ker):
        fn=sspice.spiceypy.kdata(k,"SPK")[0]
        ids = [int(i) for i in sspice.spiceypy.spkobj(fn)]        
        print(fn, ids)
        for bod in ids:
            cov = [sspice.spiceypy.et2utc(t,'ISOC',4) for t in sspice.spiceypy.spkcov(fn, bod)]
            print(cov, Body_dict[bod])                    

### Sunpy spiceypy wrapper ###

def sunspice_init():
    """
    Download (into SunPy cache) and initialize a minimal set of SPICE kernels.

    The function prepares a list of commonly used kernel URLs (planetary ephemerides,
    Solar Orbiter and Parker Solar Probe kernels and optionally Juno) and downloads
    them via sunpy.data.cache. The downloaded kernel file paths are then passed to
    sunpy.coordinates.spice.initialize() and the IAU_SUN frame is installed.

    Side effects
    ------------
    - Downloads kernel files into the SunPy cache (if not already present).
    - Calls sspice.initialize() and sspice.install_frame('IAU_SUN').
    """
    # Initialize all kernels useful for planets ephemerides, Solar Orbiter and Parker Solar Probe

    # Solar system planets
    kernel_urls = [f"https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp"]

    # Solar Orbiter
    so_dir = f"http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/"
    kernel_urls.append(so_dir+"lsk/naif0012.tls")
    kernel_urls.append(so_dir+"pck/pck00010.tpc")
    kernel_urls.append(so_dir+"spk/solo_ANC_soc-orbit-stp_20200210-20301120_280_V1_00288_V01.bsp")

    # Parker Solar Probe
    psp_dir=f"https://psp-gateway.jhuapl.edu/ancillary_data/ephemerides/"
    kernel_urls.append(psp_dir+"spp_nom_20180812_20251001_v041_RO8.bsp")

    # STEREO A
    sta_dir=f"https://sohowww.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/epm/ahead/"
    kernel_urls.append(sta_dir+"ahead_2017_061_5295day_predict.epm.bsp")

    # Bepi Colombo
    bepi_dir=f"https://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/spk/"
    kernel_urls.append(bepi_dir+"bc_mpo_fcp_00220_20181020_20270407_v01.bsp")
    
    # Juno
    if(Juno):
        juno_dir=f"https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/spk/"
        kernel_urls.append(juno_dir+"spk_ref_160226_180221_160226.bsp")
        kernel_urls.append(juno_dir+"juno_pred_orbit.bsp")
    
    # Download files from url into cache
    kernel_files = [cache.download(url) for url in kernel_urls]

    # Manually add kernel (Juno)
    #kernel_files.append(Path.joinpath(cache._cache_dir,"kernel.bsp"))
    
    sspice.initialize(kernel_files)
    sspice.install_frame('IAU_SUN')

sunspice_init()

class spacecraft_coords(object):
    """
    Minimal spacecraft ephemeris helper that samples SPICE positions between two times.

    Attributes
    ----------
    Rsunkm : float
        Solar radius in km.
    dt : ndarray
        Array of datetime objects sampled between t1 and t2 at `delta` hour intervals.
    xsc, ysc, zsc : array_like
        Cartesian coordinates (km) in the heliocentric inertial frame.
    rsc : array_like
        Radial distance in Rsun units.
    psc : array_like
        Longitude (radians, [0,2*pi)) in IAU_SUN-based frame.
    tsc : array_like
        Colatitude (radians).
    """
    def __init__(self,t1,t2,sc='SPP',delta=1):
        """
        Create sampled spacecraft coordinates.

        Parameters
        ----------
        t1, t2 : datetime
            Start and end datetimes for sampling.
        sc : str or int, optional
            Spacecraft identifier understood by sunpy.coordinates.spice.get_body
            (e.g. 'SPP', 'SOLAR ORBITER', or NAIF ID). Default 'SPP'.
        delta : int, optional
            Sampling cadence in hours (default 1).

        Notes
        -----
        After initialization the object has attributes .dt, .xsc, .ysc, .zsc, .rsc, .psc and .tsc.
        """
        self.Rsunkm=6.9570e5
        list_of_datetimes=[]
        
        dt_init=t1
        while dt_init < t2:
            list_of_datetimes.append(dt_init)
            dt_init+=timedelta(hours=delta)

        sc = sspice.get_body(sc, list_of_datetimes, spice_frame="IAU_SUN")
        sc_hci = sc.transform_to(coordinates.HeliocentricInertial())
        sc.radius=sc.distance
        
        self.dt=np.array(list_of_datetimes)

        self.xsc=sc_hci.cartesian.x.value
        self.ysc=sc_hci.cartesian.y.value
        self.zsc=sc_hci.cartesian.z.value
        
        self.rsc=sc.radius.value/self.Rsunkm
        self.psc=(sc.lon.value*np.pi/180)%(2*np.pi)
        self.tsc=((90-sc.lat.value)*np.pi/180)%(2*np.pi)
