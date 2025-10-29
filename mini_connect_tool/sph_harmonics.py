#    This file is a legacy of starAML
#    See https://github.com/vreville/starAML
#    Developped by V. Réville & A. Strugarek

import numpy as np
from scipy.special import sph_harm
from timeit import default_timer as timer
from tqdm import tqdm

class mysph(object):
    """
    Class for magnetic field extrapolation using spherical harmonics.

    Precomputes and stores spherical harmonic functions for efficient reuse in
    magnetic field computations and transformations.

    Parameters
    ----------
    lmax : int
        Maximum degree l of spherical harmonics to use
    theta : array-like
        Colatitude grid (radians)
    phi : array-like
        Longitude grid (radians) 
    ndim : int, optional
        Number of dimensions in input grids (1 or 2, default 2)

    Attributes
    ----------
    nl : int
        Maximum degree l stored
    theta : array-like
        Colatitude grid
    phi : array-like
        Longitude grid
    plm : list
        Precomputed spherical harmonic functions
    """
    def __init__(self,lmax,theta,phi,ndim=2):
        #self.nl = int((-3+np.sqrt(9+8*len(alpha)))/2.)
        self.nl=lmax
        self.theta = theta ; self.phi = phi
        ## Initialize the coordinates
        self.plm=[]
        self.xx = [] ; self.yy = [] ; self.zz = []
        if ndim==1:
            theta1d=self.theta
            phi1d=self.phi*0.0
        else:
            theta1d=self.theta[:,0]
            phi1d=self.phi[:,0]*0.0

        self.plm.append(ylm(0,0,phi1d,theta1d))
        
        for l in tqdm(range(1,self.nl+1)):
            for m in range(l+1):
                self.plm.append(ylm(m,l,phi1d,theta1d))
                self.xx.append(xlm(m,l,self.phi,self.theta))
                self.yy.append(ylm(m,l,self.phi,self.theta))
                self.zz.append(zlm(m,l,self.phi,self.theta))

    def pfss3d(self, alpha, rss=5.0, rb=1.0, rsph=1.0):
        """
        Compute PFSS magnetic field components at specified radius.

        Parameters
        ----------
        alpha : array-like
            Spherical harmonic coefficients
        rss : float, optional
            Source surface radius in solar radii (default 5.0)
        rb : float, optional
            Inner boundary radius in solar radii (default 1.0)
        rsph : float, optional
            Radius at which to compute field in solar radii (default 1.0)

        Returns
        -------
        tuple
            (br, bt, bp) radial, theta and phi components of magnetic field
        """

        br = np.zeros(np.shape(self.theta))
        bt = np.zeros(np.shape(self.theta))
        bp = np.zeros(np.shape(self.theta))    
                
        if (rsph < rb):
            rop=rsph
            coeff=1.0
        elif(rsph < rss):
            rop=rsph
            coeff=1.0        
        else:
            rop=rss
            coeff=(rss/rsph)**2

        ii=0
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ell=float(l)

                myalpha=alpha[ii]*(ell*(rb/rss)**(2*ell+1)*(rop/rb)**(ell-1)+(ell+1)*(rop/rb)**(-ell-2))/(ell*(rb/rss)**(2*ell+1)+(ell+1))
                mybeta=(ell+1)*alpha[ii]*((rb/rss)**(2*ell+1)*(rop/rb)**(ell-1)-(rop/rb)**(-ell-2))/(ell*(rb/rss)**(2*ell+1)+(ell+1))
                tmp = myalpha*self.yy[ii]
                br = br + tmp.real
                tmp = mybeta*self.zz[ii]
                bt = bt + tmp.real
                tmp = mybeta*self.xx[ii]
                bp = bp + tmp.real
                ii = ii + 1 
        br*=coeff
        return br,bt,bp

    def multipolar_expansion(self, alpha, rb=1.0, rsph=1.0):
        """
        Compute multipolar expansion of magnetic field.

        Parameters
        ----------
        alpha : array-like 
            Spherical harmonic coefficients
        rb : float, optional
            Inner boundary radius (default 1.0)
        rsph : float, optional
            Radius at which to evaluate expansion (default 1.0)

        Returns
        -------
        tuple
            (br, bt, bp) real components of multipolar field
        """

        br = np.zeros(np.shape(self.theta))
        bt = np.zeros(np.shape(self.theta))
        bp = np.zeros(np.shape(self.theta))    

        ii=0
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ell=float(l)
                myalpha=alpha[ii]*(rsph/rb)**(-ell-2)
                mybeta=-myalpha
                tmp = myalpha*self.yy[ii]
                br = br + tmp.real
                tmp = mybeta*self.zz[ii]
                bt = bt + tmp.real
                tmp = mybeta*self.xx[ii]
                bp = bp + tmp.real
                ii = ii + 1 
        return br.real,bt.real,bp.real

    def reconstruct_field(self, alpha, av=0):
        """
        Reconstruct radial field from spherical harmonic coefficients.

        Parameters
        ----------
        alpha : array-like
            Spherical harmonic coefficients
        av : float, optional
            Constant term to add (default 0)

        Returns
        -------
        array-like
            Reconstructed radial field component
        """

        br = av*self.plm[0][0]*np.ones(self.yy[0].shape)
        ii=0
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ell=float(l)
                tmp = alpha[ii]*self.yy[ii]
                br = br + tmp.real
                ii = ii + 1
        return br.real

    def cmp_potential_vector(self, alpha, rb=1.0, rsph=1.0, rss=2.5):
        """
        Compute magnetic vector potential.

        Parameters
        ----------
        alpha : array-like
            Spherical harmonic coefficients
        rb : float, optional
            Inner boundary radius (default 1.0)
        rsph : float, optional
            Radius for evaluation (default 1.0)
        rss : float, optional
            Source surface radius (default 2.5)

        Returns
        -------
        tuple
            (Ar, At, Ap) vector potential components
        """

        Ar = np.zeros(np.shape(self.theta))
        At = np.zeros(np.shape(self.theta))
        Ap = np.zeros(np.shape(self.theta))
        ii=0
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ell=float(l)
                blm=alpha[ii]/(1+ell+ell*rss**(-2*ell-1))
                alm=-rss**(-2*ell-1)*blm
                tmp = (-alm/(ell+1)*rsph**(ell)+blm/(ell)*rsph**(-ell-1))*(ell+1)
                At=At+(tmp*self.xx[ii]).real
                Ap=Ap-(tmp*self.zz[ii]).real

                ii = ii + 1
                       
        return Ar,At,Ap


    def cmp_currents(self, alpha, rss=5.0, rsph=1.0, dr=0.01):
        """
        Compute current density components

        Parameters
        ----------
        alpha : array-like
            Spherical harmonic coefficients
        rss : float, optional
            Source surface radius (default 5.0)
        rsph : float, optional
            Evaluation radius (default 1.0)
        dr : float, optional
            Radial step for derivatives (default 0.01)

        Returns
        -------
        tuple
            (Jr, Jt, Jp, |J|, J_parallel, J_perpendicular)
        """

        br,bt,bp=self.pfss3d(alpha,rss=rss,rsph=rsph)
        br_p,bt_p,bp_p=self.pfss3d(alpha,rss=rss,rsph=rsph+dr)
        br_m,bt_m,bp_m=self.pfss3d(alpha,rss=rss,rsph=rsph-dr)
                
        dtheta=diff(self.theta,1)
        dphi=diff(self.phi,2)
        st=(np.sin(self.theta[1:-1,1:-1]))

        Jr=1/(rsph*st)*(1/dtheta*(diff(np.sin(self.theta)*bp,1))-(diff(bt,2))/dphi)
        Jt=1/(rsph*st)*(diff(br,2)/dphi)-1/rsph*(bp_p[1:-1,1:-1]*(rsph+dr)-bp_m[1:-1,1:-1]*(rsph-dr))/2/dr
        Jp=1/rsph*((bt_p[1:-1,1:-1]*(rsph+dr)-bt_m[1:-1,1:-1]*(rsph-dr))/2/dr-diff(br,1)/dtheta)

        jj=np.sqrt(Jr**2+Jt**2+Jp**2)
        bb=np.sqrt(br[1:-1,1:-1]**2+bt[1:-1,1:-1]**2+bp[1:-1,1:-1]**2)
        Jpar=(Jr*br[1:-1,1:-1]+Jt*bt[1:-1,1:-1]+Jp*bp[1:-1,1:-1])/bb
        Jperp=jj-Jpar

        return Jr,Jt,Jp,jj,Jpar,Jperp

    def spherical_harmonics_decomposition(self, A, theta, lmax=3, sym=None):
        """
        Project a field on the spherical harmonic basis.

        Parameters
        ----------
        A : array-like
            Field to decompose
        theta : array-like
            Colatitude grid
        lmax : int, optional
            Maximum degree l (default 3)
        sym : {None, "axis", int}, optional
            Symmetry assumption:
            - None: no symmetry (default)
            - "axis": axisymmetric
            - int: period in phi
        Returns
        -------
        list
            [B, alm, all] where:
            - B: coefficients array
            - alm: flattened coefficients 
            - all: coefficients by l
        """

        if(sym==None):
            period=1
            costheta=np.asarray(np.cos(theta[:,0]),dtype=float)
            Field=A
        elif(sym=="axis"):
            print("Field is axisymmetric")
            period=1
            costheta=np.asarray(np.cos(theta),dtype=float)
            Field=np.tile(A,(2*len(theta),1)).T
        else:
            period=int(sym)
            costheta=np.asarray(np.cos(theta[:,0]),dtype=float)
            Field=A

        n_theta,n_phi=Field.shape

        # Compute integration weights:  sin(theta)dtheta
        sintheta=np.sqrt(1-costheta**2)
        weights=np.ones(n_theta)*np.pi/n_theta*sintheta

        # Compute the Fourier transform phi->m
        t1=timer()
        ft=np.zeros(Field.shape,dtype=complex)
        for i in range(0,n_theta):
            ft[i,:]=2*np.pi/(n_phi)*np.fft.fft(Field[i,:])
        dt=timer()-t1
        #print("Fourier transform: {0} s".format(dt))
        B=np.zeros((lmax+1,lmax+1),dtype=complex)

        # Project on the (correctly normalized) Plm
        ii=0
        for l in range(0,lmax+1):
            B[l,0] = np.sum(ft[:,0]*self.plm[ii]*weights)
            ii=ii+1
            if sym=="axis":
                ii=ii+l
            else:
                # The 2-factor accounts for m < l
                for m in range(1,l+1):
                    B[l,m] = 2*np.sum(ft[:,m]*self.plm[ii]*weights)
                    ii=ii+1

        alm=np.array([],dtype=complex)
        all=np.array([],dtype=complex)

        ii=0
        for n in range(1,lmax+1):
            all=np.append(all,B[n,0])
            for m in range(0,n+1):
                alm=np.append(alm,B[n,m])
                ii=ii+1

        return [B,alm,all]

# Spherical harmonics base
def ylm(m, n, phi, theta):
    """
    Compute spherical harmonic Y_l^m.

    Parameters
    ----------
    m : int
        Order m
    n : int
        Degree l
    phi : array-like
        Longitude (radians)
    theta : array-like
        Colatitude (radians)

    Returns
    -------
    array-like
        Y_l^m values
    """

    ll = float(n) ; mm = float(m)
    if(abs(m) > n):
        return 0.0
    else:
        return sph_harm(m,n,phi,theta)
    
def xlm(m, n, phi, theta):
    """
    Compute vector spherical harmonic Z_l^m.

    Parameters
    ----------
    m, n : int
        Order m and degree l
    phi, theta : array-like
        Longitude and colatitude grids (radians)

    Returns
    -------
    array-like
        Modified spherical harmonic values
    """

    ll = float(n) ; mm = float(m)
    xlm=sph_harm(m,n,phi,theta)*1j*mm
    st=np.sin(theta)
    idz=np.where(st != 0)
    xlm[idz]/=(st[idz]*(ll+1))
    return xlm

def zlm(m, n, phi, theta):
    """
    Compute vector spherical harmonic Z_l^m.

    Parameters
    ----------
    m, n : int
        Order m and degree l
    phi, theta : array-like
        Longitude and colatitude grids (radians)

    Returns
    -------
    array-like
        Z_l^m values
    """

    ll = float(n) ; mm = float(m)
    cc = ((ll+1.-mm) / (ll+1.) )/np.sqrt( ((2.*(ll+1.)+1.0)*(ll+1-mm)) / ((2.*ll+1.0)*(ll+1.+mm)) )
    zlm=(-sph_harm(m,n,phi,theta)*np.cos(theta) + cc*sph_harm(m,n+1,phi,theta))
    st=np.sin(theta)
    idz=np.where(st != 0)
    zlm[idz]/=(st[idz])
    return zlm

def zlm2(m, n, phi, theta):
    """
    Compute alternative form of vector spherical harmonic Z_l^m.

    Similar to zlm() but uses a different recursion relation.
    """

    ll = float(n) ; mm = float(m)
    cc = ((ll+mm) / (ll+1.) )*np.sqrt( ((2.*ll+1.0)*(ll-mm)) / ((2.*ll-1.0)*(ll+mm)) )
    return (ll/(ll+1)*sph_harm(m,n,phi,theta)*np.cos(theta) - cc*ylm(m,n-1,phi,theta))/np.sin(theta)

def diff(field, dir=1):
    """
    Compute centered differences along specified axis.

    Parameters
    ----------
    field : array-like
        2D field array
    dir : {1, 2}, optional
        Direction for derivative:
        1: theta derivative (default)
        2: phi derivative

    Returns
    -------
    array-like
        Centered differences with reduced size in differentiation direction
    """

    if(dir==1):
        field1=field[:,1:-1]
        return field1[2:,:]-field1[:-2,:]
    if(dir==2):
        field1=field[1:-1,:]
        return field1[:,2:]-field1[:,:-2]

# Compute unsigned magnetic flux
def cmp_mag_flux(theta, phi, br, r, abs=True):
    """
    Compute total magnetic flux through a spherical surface.

    Parameters
    ----------
    theta : array-like
        Colatitude grid (radians)
    phi : array-like
        Longitude grid (radians)
    br : array-like
        Radial magnetic field component
    r : float
        Radius at which to compute flux
    abs : bool, optional
        If True return unsigned flux (default True)

    Returns
    -------
    float
        Total (signed or unsigned) magnetic flux
    """

    dtheta=np.tile(np.diff(theta[:,0]),(np.shape(phi)[1]-1,1)).transpose()
    dphi=np.tile(np.diff(phi[0,:]),(np.shape(theta)[0]-1,1))

    Opflux=0.5*(br[1:,1:]+br[:-1,:-1])*(np.sin(0.5*(theta[:-1,:-1]+theta[1:,1:])))
    Opflux*=dtheta*dphi

    if abs:
        Opflux=np.abs(Opflux)

    return np.sum(Opflux)*r**2
