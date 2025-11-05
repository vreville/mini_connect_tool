import os
import numpy as np
from subprocess import call
from datetime import datetime, timedelta
from astropy.io import fits
from copy import deepcopy
import requests

def find_outlist(dir,dtype="fts"):
    """
    Return a sorted list of filenames in `dir` that have extension `dtype`.

    Parameters
    ----------
    dir : str
        Directory to scan.
    dtype : str, optional
        Extension to match (without leading dot). Default is "fts".

    Returns
    -------
    list of str
        Sorted filenames in `dir` that end with ".{dtype}".
    """
    outlist=[]
    for ff in os.listdir(dir):
        i=len(dtype)+1
        if ff[-i:]=="."+dtype:
            outlist.append(ff)
    outlist.sort()
    return outlist

def get_adapt_maps_list(adapt_dir, t1, t2, forecast=False):
    """ Get list of ADAPT maps between t1 and t2.
    adapt_dir : str
        Directory where ADAPT maps are stored.
    t1 : datetime
        Start time.
    t2 : datetime
        End time.
    forecast : bool
        If True, include forecast maps (that will fill non-exisiting i-maps). Default is False.

    Returns the list of ADAPT map filenames and their corresponding times.
    """
    list_files=find_outlist(adapt_dir,dtype="fts")
    list_files2=deepcopy(list_files)
    map_times=[]

    for ll in list_files2:
        tmp=ll.split("_")[2]
        dtmap=datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]),int(tmp[10:12]))
        if dtmap < t1:
            list_files.remove(ll)
        elif dtmap > t2:
            list_files.remove(ll)
    list_files3=deepcopy(list_files)

    if forecast:
        dtmap_old=datetime(1,1,1,0)
        for ii,ll in enumerate(list_files3[::-1]):
            tmp=ll.split("_")[2]
            dtmap=datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]),int(tmp[10:12]))
            #print(dtmap, dtmap_old, dtmap==dtmap_old)
            if "_f" in ll and dtmap==dtmap_old:
                list_files.remove(ll)
            dtmap_old=dtmap
    else:
        for ii,ll in enumerate(list_files3):
            if "_f" in ll:
                list_files.remove(ll)

    for ii,ll in enumerate(list_files):
        tmp=ll.split("_")[2]
        dtmap=datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]),int(tmp[10:12]))
        map_times.append(dtmap)

    return list_files, map_times

def get_adapt_file_names(list_times):
    """
    Query the NSO/GONG ADAPT index pages for years present in `list_times`
    and return a list of ADAPT map filenames found on those pages.

    Parameters
    ----------
    list_times : sequence of datetime
        Times used to determine which yearly index pages to parse.

    Returns
    -------
    list of str
        Filenames (as listed on the GONG ADAPT index pages), e.g. 'adapt_...fts.gz'.
    """
    list_names=[]
    year_array=np.array([tt.year for tt in list_times])
    year_array=np.unique(year_array)

    for year in year_array:

        url="https://gong.nso.edu/adapt/maps/gong/{:04d}/".format(year)
        r=requests.get(url)
        s=str(r.content).split("href")
        for ss in s:
            if "fts.gz" in ss:
                list_names.append(ss.split('"')[1])

    return list_names

def match_names(names, list_times):
    # Match each requested time to the closest available filename in names
    matched_names = []
    for tt in list_times:
        # Find all filenames in names that contain the timestamp string
        tstr = "{:04d}{:02d}{:02d}{:02d}{:02d}".format(tt.year, tt.month, tt.day, tt.hour, tt.minute)
        candidates = [n for n in names if tstr in n]
        if candidates:
            matched_names.append(candidates[0])
        else:
            # fallback: pick the closest by date
            matched_names.append(names[0])  # or handle as needed

    return matched_names

def dl_adapt_carrington_maps(t1, t2, cadence_hours=24):
    """ Download ADAPT Carrington maps from NSO GONG between t1 and t2 with given cadence in hours.

    Parameters
    ----------
    t1 : datetime
        Start datetime for map selection.
    t2 : datetime
        End datetime for map selection.
    cadence_hours : int, optional
        Temporal cadence in hours for requested maps (default 24).

    Returns
    -------
    tuple
        (list_files, list_times) where list_files are downloaded filenames (without .gz)
        and list_times are the corresponding datetimes used to fetch them.
    """
    list_times=[]
    list_files=[]

    current_time=t1
    # Load noon maps
    current_time=current_time.replace(hour=12,minute=0,second=0,microsecond=0)

    while current_time <= t2:
        list_times.append(current_time)
        current_time+=timedelta(hours=cadence_hours)

    names=get_adapt_file_names(list_times)
    list_names=match_names(names,list_times)

    if not os.path.exists(os.path.dirname(__file__)+"/../data/adapt_carrington/"):
        call("mkdir -p {}/../data/adapt_carrington/".format(os.path.dirname(__file__)), shell=True)

    print("Downloading ADAPT Carrington maps from NSO/GONG...")
    for ii,tt in enumerate(list_times):
        filename="{}/../data/adapt_carrington/{}".format(os.path.dirname(__file__),list_names[ii])
        if os.path.exists(filename[:-3]):
            print("File {} already exists, skipping download.".format(filename[:-3]))
            pass
        else:
            url="https://gong.nso.edu/adapt/maps/gong/{:04d}/{}".format(tt.year, list_names[ii])
            response=requests.get(url)

            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)

                call("gzip -d {}".format(filename), shell=True)
        list_files.append(list_names[ii][:-3])

    print("Download complete.")
    return list_files, list_times

def get_sdo_maps_list(sdo_dir, t1, t2):
    """
    Select SDO Carrington map files in `sdo_dir` whose embedded timestamps
    lie between `t1` and `t2`.

    Parameters
    ----------
    sdo_dir : str
        Directory containing SDO Carrington .fits files.
    t1, t2 : datetime
        Start and end datetimes for selection.

    Returns
    -------
    tuple
        (list_sdo, sdo_times) where list_sdo are filenames matching the time window
        and sdo_times are the corresponding datetime objects.
    """
    list_sdo=find_outlist(sdo_dir,dtype="fits")
    list_sdo2=deepcopy(list_sdo)
    sdo_times=[]

    for ll in list_sdo2:
        tmp=ll.split("_")[3]
        dtmap=datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[9:11]),int(tmp[11:13]))
        if dtmap < t1:
            list_sdo.remove(ll)
        elif dtmap > t2:
            list_sdo.remove(ll)
        else:
            sdo_times.append(dtmap)

    return list_sdo, sdo_times

def dl_sdo_carrington_maps(list_times):
    """
    Download SDO AIA 193 Carrington PNG maps from the connect-tool API for each time in list_times.

    Parameters
    ----------
    list_times : sequence of datetime
        Times for which to request Carrington images.

    Returns
    -------
    tuple
        (list_sdo, list_times) where list_sdo is a list of saved local filenames and
        list_times is the input list_times (returned for convenience).
    """
    if not os.path.exists(os.path.dirname(__file__)+"/../data/sdo_carrington/"):
        call("mkdir -p {}/../data/sdo_carrington/".format(os.path.dirname(__file__)), shell=True)

    list_sdo=[]
    print("Downloading SDO AIA 193 Carrington maps from connect tool API...")
    for tt in list_times:
        url="https://connect-tool.irap.omp.eu/api_creation/PSP/ADAPT/SUNTIME/{:04d}{:02d}{:02d}T{:02d}{:02d}{:02d}/?background=euv193&frame=off&path=on".format(tt.year,
                                                                                                                                                                tt.month,
                                                                                                                                                                tt.day,
                                                                                                                                                                tt.hour,
                                                                                                                                                                tt.minute,
                                                                                                                                                                tt.second)
        filename="{}/../data/sdo_carrington/AIA193_{}.png".format(os.path.dirname(__file__),tt.isoformat())
        if os.path.exists(filename):
            print("File {} already exists, skipping download.".format(filename))
            list_sdo.append(filename)
            continue

        response=requests.get(url)

        if response.status_code == 200:
            response=requests.get(response.content)

            with open(filename, "wb") as f:
                f.write(response.content)

            list_sdo.append(filename)
    print("Download complete.")
    return list_sdo, list_times

class tmap(object):
    def __init__(self,filename,v1=True,coeff=1,real=0):
        """
        Load a magnetic map file into a tmap object.

        Parameters
        ----------
        filename : str
            Path to the map file. Format depends on `v1`.
        v1 : bool, optional
            If True, treat as "De Rosa" HDF5 format (default). If False, treat as ADAPT FITS.
        coeff : float, optional
            Scaling coefficient applied to magnetic data.
        real : int, optional
            Index to select a realisation in ADAPT FITS files (only used when v1 is False).

        Attributes created
        ------------------
        time : datetime
            Map timestamp parsed from filename.
        br, bt, bp : arrays
            Magnetic field components (format depends on source).
        n_r, n_t, n_p, theta, phi, slat, L0 : various
            Grid metadata used elsewhere in the code.
        """
        self.coeff=coeff

        # De Rosa Maps
        if(v1):
            f=h5py.File(filename,'r')
            a_group_key = list(f.keys())[0]

            tmp=(filename.split("/")[-1]).split("_")
            self.time=datetime(int(tmp[1][0:4]),int(tmp[1][4:6]),int(tmp[1][6:8]),int(tmp[2][0:2]),int(tmp[2][2:4]),int(tmp[2][4:6]))

            data = list(f[a_group_key])
            self.n_r=data[0][0]
            self.n_t=data[0][1]
            self.n_p=data[0][2]

            self.r=data[0][3]
            self.lat=data[0][4]
            self.lon=data[0][5]

            self.theta=data[0][6]
            self.phi=data[0][7]
            self.slat=np.sin(np.pi/2.-self.theta)

            self.br=data[0][8]
            self.bt=data[0][9]
            self.bp=data[0][10]

            self.br_ph=self.br[0]

            self.L0=(data[0][12]-180.)*np.pi/180.

        # Adapt Maps
        else:
            f=fits.open(filename)

            tmp=((filename.split("/")[-1]).split("_"))[2]
            self.time=datetime(int(tmp[0:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]),int(tmp[10:12]),0)

            # Get the data

            mag=f[0].data
            self.br=mag[real][::-1,:]
            self.br_ph=self.br

            self.n_t=self.br.shape[0]
            self.n_p=self.br.shape[1]

            self.theta=np.linspace(0,np.pi,self.n_t)
            self.phi=np.linspace(0,2*np.pi,self.n_p)
            self.slat=np.sin(np.pi/2.-self.theta)

            self.L0=0

    def dmp_alm(self,spharm):
        """
        Decompose the stored surface magnetic field (br_ph * coeff) into spherical harmonic coefficients.

        Parameters
        ----------
        spharm : object
            Spherical harmonic helper providing `spherical_harmonics_decomposition` method
            and attributes `theta` and `nl`.

        Side effects
        ------------
        Sets self.alm to the computed alm array.
        """
        Mlm,alm,all=spharm.spherical_harmonics_decomposition(self.br_ph*self.coeff,spharm.theta,lmax=spharm.nl)
        self.alm=alm
