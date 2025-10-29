import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.integrate import ode

def compute_fieldlines(seed_points,r,the,phi,Br,Bt,Bp,Sunward=True,max_it=10000):
    """
    Trace magnetic field lines from seed points in a spherical magnetic data cube.

    Parameters
    ----------
    seed_points : (N,3) array_like
        Cartesian starting points for each field line (same length-scale as r).
    r : 1D array_like
        Radial grid points (in Rsun).
    the : 1D array_like
        Colatitude grid points (radians, 0 = north pole).
    phi : 1D array_like
        Longitude grid points (radians).
    Br, Bt, Bp : 3D array_like
        Magnetic field components on the input grid with shape (nr, nthe, nphi).
    Sunward : bool, optional
        If True, choose tracing direction by the sign of Br at the seed (default True).
    max_it : int, optional
        Maximum number of integration steps per refinement loop (default 10000).

    Returns
    -------
    fieldlines : list of [x,y,z] arrays
        Each list element is a traced field line given as three 1D arrays (x,y,z).
    """
    #-------------------------------------------------------------------

    integ='vode'
    #integ='lsoda'
    #integ='dopri5'

    def bTrace(t, x):
        """
        Vector field callback for the ODE integrator: returns B/|B| at Cartesian x.

        Parameters
        ----------
        t : float
            Dummy time parameter required by scipy.integrate.ode (unused).
        x : array_like (length 3)
            Cartesian position where the field is evaluated.

        Returns
        -------
        ndarray (3,)
            Unit vector in the direction of the magnetic field at x.
        """
        # (ph,s,rh) coordinates of current point:
        ph = (np.arctan2(x[1], x[0]) + 2*np.pi) % (2*np.pi)
        r = np.sqrt(np.sum(x**2))
        s = x[2]/r
        rh = np.log(r)
        b1 = brgi( np.stack((ph, s, rh)) )
        return b1/np.linalg.norm(b1)
    
    def trace(x0, dtf=1e-2, tol=1e-2, nrefine=3,max_it=max_it,only_bwd=False,only_fwd=False):
        """
        Integrate a single field line starting at x0 using scipy.integrate.ode.

        The routine integrates along the normalized magnetic field vector both
        forward and backward (configurable), refining the step size near domain
        boundaries.

        Parameters
        ----------
        x0 : (3,1) ndarray
            Column vector with the starting Cartesian coordinates.
        dtf : float, optional
            Initial maximum step size for the integrator (default 1e-2).
        tol : float, optional
            Tolerance multiplier for integrator absolute tolerance (default 1e-2).
        nrefine : int, optional
            Number of refinement iterations to reduce step size near boundaries.
        max_it : int, optional
            Maximum integration steps per refinement loop.
        only_bwd : bool, optional
            If True, integrate only in the backward direction.
        only_fwd : bool, optional
            If True, integrate only in the forward direction.

        Returns
        -------
        (xl, yl, zl) : three 1D ndarrays
            Cartesian coordinate arrays of the traced field line.
        """
        xl = x0.copy()
    
        if not (only_fwd):
            # Backwards:
            t = 0.0
            dt = dtf
            count=0
            for j in range(nrefine):
                solver = ode(bTrace).set_integrator(integ, method='adams', atol=tol*dt)
                solver.set_initial_value(xl[:,0:1], t)
                while (count < max_it):
                    #Redirect error into null device.
                    #Error "capi_return is NULL. Call-back cb_f_in_dvode__user__routines failed" always send when reaching boundary even if exception caught.
                    #with module_toolbox_file.stdout_redirected(to=os.devnull,stdout=sys.stderr):
                    try:
                        solver.integrate(solver.t - dt)
                        xl = np.insert(xl, [0], solver.y, axis=1)
                    except ValueError: # reached boundary
                        break
                    count=count+1
                t = solver.t
                dt /= 10.0

        if not (only_bwd):
            # Forwards:
            t = 0.0
            dt = dtf
            count=0
            for j in range(nrefine):
                solver = ode(bTrace).set_integrator(integ, method='adams', atol=tol*dt)
                solver.set_initial_value(xl[:,-1:], t)
                while (count < max_it):
                    #Redirect error into null device.
                    #Error "capi_return is NULL. Call-back cb_f_in_dvode__user__routines failed" always send when reaching boundary even if exception caught.
                    #with module_toolbox_file.stdout_redirected(to=os.devnull,stdout=sys.stderr):
                    try:
                        solver.integrate(solver.t + dt)
                        xl = np.append(xl, solver.y, axis=1)
                    except ValueError: # reached boundary
                        break
                    count=count+1
                t = solver.t
                dt /= 10.0

        return xl[0,:], xl[1,:], xl[2,:]
    #-------------------------------------------------------------------
    
    # - read in magnetic field:
    # r    radial distance in Rsun 
    # the  colatitude in radian
    # phi  longitude in radian

    #Datacube has data as a matrix(nr,nthe,nphi), this routine need br(ph,th,r). 

    br   =  np.transpose(Br[:,::-1,:])  #radial magnetic component
    bthe =  np.transpose(Bt[:,::-1,:])  #colatitudinal magnetic component
    bphi =  np.transpose(Bp[:,::-1,:])  #longitudinal magnetic component

    # - (rho,s,phi) coordinates:
    rh = np.log(r)
    s = np.cos(the[::-1])
    brint=rgi((phi,s,rh),br,bounds_error=False,fill_value=0.0)

    # - convert to Cartesian components and make interpolator on (rho,s,phi) grid:
    ph3, s3, rh3 = np.meshgrid(phi, s, rh, indexing='ij')
    bx = np.sqrt(1-s3**2)*np.cos(ph3)*br + s3*np.cos(ph3)*bthe - np.sin(ph3)*bphi
    by = np.sqrt(1-s3**2)*np.sin(ph3)*br + s3*np.sin(ph3)*bthe + np.cos(ph3)*bphi
    bz = s3*br - np.sqrt(1-s3**2)*bthe

    del(br, bthe, bphi)
    bstack = np.stack((bx,by,bz),axis=3)
    del(bx, by, bz)
    brgi = rgi((phi, s, rh), bstack)#, bounds_error=False, fill_value=None)
    del(bstack)

    # - trace fieldlines from each seed point
    npoint = len(seed_points)
    fieldlines = []
    for j in range(npoint):
        x0 = np.empty((3,1))
        #print(x0)
        #Km -> Rsun conversion for seed_points
        x0[:,0] = [seed_points[j,0],seed_points[j,1],seed_points[j,2]]                

        if (Sunward):
            ph0=(np.arctan2(x0[1],x0[0]))%(2*np.pi)
            r0=np.sqrt(np.sum(x0**2))
            s0=x0[2]/r0
            rh0=np.log(r0)
            br0=brint((ph0,s0,rh0))

            if br0 >= 0:
                xl, yl, zl = trace(x0[:,0:1],only_bwd=True)
            else:
                xl, yl, zl = trace(x0[:,0:1],only_fwd=True)
        else:
            xl, yl, zl = trace(x0[:,0:1])
        fieldlines.append([xl,yl,zl])

    return fieldlines

class fieldline(object):
    """
    Lightweight container for a traced field line and derived diagnostics.

    Attributes
    ----------
    fx, fy, fz : 1D ndarrays
        Cartesian coordinates ordered from footpoint outward.
    fr : 1D ndarray
        Spherical radius for each point.
    frcyl : 1D ndarray
        Cylindrical radius sqrt(x^2+y^2).
    ft : 1D ndarray
        Colatitude (radians).
    fp : 1D ndarray
        Longitude (radians).
    flon, flat : 1D ndarray
        Longitude and latitude (degrees).
    """
    def __init__(self, fl):
        self.fx=fl[0]
        self.fy=fl[1]
        self.fz=fl[2]

        fr = np.sqrt(self.fx**2+self.fy**2+self.fz**2)

        if fr[0] > fr[1]:

            self.fx = fl[0][::-1]
            self.fy = fl[1][::-1]
            self.fz = fl[2][::-1]

        self.fr=np.sqrt(self.fx**2+self.fy**2+self.fz**2)
        self.frcyl=np.sqrt(self.fx**2+self.fy**2)
        self.ft=np.pi/2.-np.arctan(self.fz/self.frcyl)
        self.fp=np.arctan2(self.fy,self.fx)%(2*np.pi)

        self.flon=(self.fp*180/np.pi)%(360)
        self.flat=(np.pi/2 - self.ft)*180/np.pi

    def cmp_curv_s(self, qx, qy, qz):
        """
        Compute the curvilinear abscissa (arc length) along a discrete curve.

        Parameters
        ----------
        qx, qy, qz : 1D array_like
            Cartesian coordinates of the curve points.

        Returns
        -------
        curv_s : 1D ndarray
            Cumulative arc length starting at the first point.
        """
        rp=np.sqrt(qx**2+qy**2+qz**2)
        curv_s=np.array([rp[0]])

        for ii in range(1,len(rp)):
            curv_s=np.append(curv_s,curv_s[ii-1]+np.sqrt((qx[ii]-qx[ii-1])**2+(qy[ii]-qy[ii-1])**2+(qz[ii]-qz[ii-1])**2))

        return curv_s

    def cmp_fexp(self, ibtot):
        """
         Compute the expansion factor fexp along the field line.
         Parameters
         ----------
         ibtot : callable
             Interpolator or function that returns |B| given (r, theta, phi) tuples/arrays.

        Notes
         -----
         Sets attributes:
           - bsurf : |B| at the surface footpoint (r=1)
           - balong : |B| along the full fieldline
           - fexp : expansion factor array computed as bsurf / (balong * s^2)
         """
        self.bsurf = ibtot((1, self.ft[0], self.fp[0]))
        self.balong = ibtot((self.fr, self.ft, self.fp))
        curvs=self.cmp_curv_s(self.fx, self.fy, self.fz)
        self.fexp = self.bsurf/(self.balong*curvs**2)

