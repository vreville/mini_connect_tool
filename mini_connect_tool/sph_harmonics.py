import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from packaging.version import Version
import scipy
if Version(scipy.__version__) < Version('1.15'): from scipy.special import sph_harm
else: from scipy.special import sph_harm_y as sph_harm
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    """

    def __init__(self,lmax,theta,phi,ndim=2):

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

        if Version(scipy.__version__) < Version('1.15'): version=13
        else: version=15

        for l in tqdm(range(0,self.nl+1)):
            for m in tqdm(range(l+1), leave=False):
                self.plm.append(ylm(m,l,phi1d,theta1d,version=version))
                self.xx.append(xlm(m,l,self.phi,self.theta,version=version))
                self.yy.append(ylm(m,l,self.phi,self.theta,version=version))
                self.zz.append(zlm(m,l,self.phi,self.theta,version=version))

    def pfss3d(self,alpha,rss=5.0,rb=1.0,rsph=1.0):
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

        ii=1
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

    def scs(self, alm, rss, r):
        """
        Compute the Schatten current sheet model between Rss and r
        from the decomposition of brss

        """

        br = np.zeros(np.shape(self.theta))
        bt = np.zeros(np.shape(self.theta))
        bp = np.zeros(np.shape(self.theta))

        ii=0
        for l in range(0,self.nl+1):
            for m in range(l+1):
                ell=float(l)

                gl = (rss/r)**(ell+2)
                tmp = gl*alm[ii]*self.yy[ii]
                br = br + tmp.real
                tmp = -gl*alm[ii]*self.zz[ii]
                bt = bt + tmp.real
                tmp = -gl*alm[ii]*self.xx[ii]
                bp = bp + tmp.real
                ii = ii + 1

        return br,bt,bp

    def multipolar_expansion(self,alpha,rb=1.0,rsph=1.0):
        """
        Compute multipolar expansion of magnetic field.
        Correspond to a pfss with rss -> infty
        """

        br = np.zeros(np.shape(self.theta))
        bt = np.zeros(np.shape(self.theta))
        bp = np.zeros(np.shape(self.theta))

        ii=1
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

    def reconstruct_field(self,alpha):
        """
        Reconstruct radial field from spherical harmonic coefficients.
        """
        br = np.zeros(np.shape(self.theta))
        ii=0
        for l in range(0,self.nl+1):
            for m in range(l+1):
                ell=float(l)
                tmp = alpha[ii]*self.yy[ii]
                br = br + tmp.real
                ii = ii + 1
        return br.real

    def cmp_potential_vector(self,alpha,rb=1.0,rsph=1.0,rss=2.5): # TO CHECK
        """
        Compute magnetic vector potential
        """

        Ar = np.zeros(np.shape(self.theta))
        At = np.zeros(np.shape(self.theta))
        Ap = np.zeros(np.shape(self.theta))
        ii=1
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


    def cmp_currents(self,alpha,rss=5.0,rsph=1.0,dr=0.01):
        """
        Compute current density components from PFSS
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

    def spherical_harmonics_decomposition(self,A,theta,lmax=3,sym=None,silent=True):
        """ Compute the spherical harmonics decomposition,
        through a optimized procedure using fft for toroidal modes."""

        if(sym==None):
            period=1
            costheta=np.asarray(np.cos(theta[:,0]),dtype=float)
            Field=A
        elif(sym=="axis"):
            prints("Field is axisymmetric",silent=silent)
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
        prints("Fourier transform: {0} s".format(dt),silent=silent)
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

        for l in range(0,lmax+1):
            all=np.append(all,B[l,0])
            for m in range(0,l+1):
                alm=np.append(alm,B[l,m])

        return [B,alm,all]

    def spherical_harmonics_decomposition2(self,A,theta,lmax=3,sym=None,silent=True):
        """ Compute the spherical harmonics decomposition,
        through a scalar product with each Y_l^m."""

        B=np.zeros((lmax+1,lmax+1),dtype=complex)

        # Project on the Ylms
        ii=0
        for l in range(0,lmax+1):
            B[l,0] = cmp_mag_flux2(self.theta, self.phi, A*np.conjugate(self.yy[ii]), 1, abs=False)
            ii=ii+1
            if sym=="axis":
                ii=ii+l
            else:
                # The 2-factor accounts for m < 0
                for m in range(1,l+1):
                    B[l,m] = 2*cmp_mag_flux2(self.theta, self.phi, A*np.conjugate(self.yy[ii]), 1, abs=False)
                    ii=ii+1

        alm=np.array([],dtype=complex)
        all=np.array([],dtype=complex)

        for l in range(0,lmax+1):
            all=np.append(all,B[l,0])
            for m in range(0,l+1):
                alm=np.append(alm,B[l,m])

        return [B,alm,all]


def dipole(theta,r):
    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))

    br=2*np.cos(theta)/r**3
    bt=np.sin(theta)/r**3

    return br,bt,bp

def diff(field,dir=1):
    if(dir==1):
        field1=field[:,1:-1]
        return field1[2:,:]-field1[:-2,:]
    if(dir==2):
        field1=field[1:-1,:]
        return field1[:,2:]-field1[:,:-2]

# Spherical harmonics base
def ylm(m,n,phi,theta,version=13):
    ll = float(n) ; mm = float(m)
    if(abs(m) > n):
        return 0.0
    elif version<15:
        return sph_harm(m,n,phi,theta)
    else:
        return sph_harm(n,m,theta,phi)

def xlm(m,n,phi,theta,version=13):
    ll = float(n) ; mm = float(m)
    if version<15: xlm=sph_harm(m,n,phi,theta)*1j*mm
    else:          xlm=sph_harm(n,m,theta,phi)*1j*mm
    st=np.sin(theta)
    idz=np.where(st != 0)
    xlm[idz]/=(st[idz]*(ll+1))
    return xlm

def zlm(m,n,phi,theta,version=13):
    ll = float(n) ; mm = float(m)
    cc = ((ll+1.-mm) / (ll+1.) )/np.sqrt( ((2.*(ll+1.)+1.0)*(ll+1-mm)) / ((2.*ll+1.0)*(ll+1.+mm)) )
    if version<15: zlm=(-sph_harm(m,n,phi,theta)*np.cos(theta) + cc*sph_harm(m,n+1,phi,theta))
    else:          zlm=(-sph_harm(n,m,theta,phi)*np.cos(theta) + cc*sph_harm(n+1,m,theta,phi))
    st=np.sin(theta)
    idz=np.where(st != 0)
    zlm[idz]/=(st[idz])
    return zlm

# Read ZDI map
def read_ZDImap(myfile,dir="./"):
    filename = dir+myfile
    f = open(filename,'r')
    tmp = f.readline()
    params = f.readline().split()
    nharms = int(params[0])+1 ; ncomps = params[1]
    nl = int((-3+np.sqrt(9+8*(nharms-1)))/2.)
    alpha = np.zeros(nharms,dtype=complex)
    ii = 0
    for n in range(1,nl+1):
        for m in range(n+1):
            vals = f.readline().split()
            alpha[ii+1] = complex(float(vals[2]),float(vals[3]))
            ii = ii + 1
    tmp=f.readline()
    beta = np.zeros(nharms,dtype=complex)
    ii = 0
    for n in range(1,nl+1):
        for m in range(n+1):
            vals = f.readline().split()
            beta[ii+1] = complex(float(vals[2]),float(vals[3]))
            ii = ii + 1
    tmp=f.readline()
    gamma = np.zeros(nharms,dtype=complex)
    ii = 0
    for n in range(1,nl+1):
        for m in range(n+1):
            vals = f.readline().split()
            gamma[ii+1] = complex(float(vals[2]),float(vals[3]))
            ii = ii + 1
    f.close()
    return alpha,beta,gamma,nl

def create_ZDImap(header,alm,mapname,path="./"):
    zdimap=path+mapname
    lmax = int((-3+np.sqrt(9+8*(len(alm)-1)))/2.)
    #if not os.path.exists(zdimap):
    if True:
        f=open(zdimap,'w')
        f.write(header+"\n")
        f.write("{:d} {:d} {:d}\n".format(len(alm)-1,0,0))
        k=0
        for i in range(1,lmax+1):
            for j in range(i+1):
                f.write("{:1d} {:1d} {:+.05e} {:+.05e}\n".format(i,j,alm[k+1].real,alm[k+1].imag))
                k=k+1
        f.write("\n")
        for i in range(1,lmax+1):
            for j in range(i+1):
                f.write("{:d} {:d} {:+.05e} {:+.05e}\n".format(i,j,0.,0.))
        f.write("\n")
        for i in range(1,lmax+1):
            for j in range(i+1):
                f.write("{:d} {:d} {:+.05e} {:+.05e}\n".format(i,j,0.,0.))
        f.write("\n")
        f.close()

# Compute unsigned magnetic flux (to improve)
def cmp_mag_flux(theta,phi,br,r,abs=True):
    dtheta=np.tile(np.diff(theta[:,0]),(np.shape(phi)[1]-1,1)).transpose()
    dphi=np.tile(np.diff(phi[0,:]),(np.shape(theta)[0]-1,1))

    Opflux=0.5*(br[1:,1:]+br[:-1,:-1])*(np.sin(0.5*(theta[:-1,:-1]+theta[1:,1:])))
    Opflux*=dtheta*dphi

    if abs:
        Opflux=np.abs(Opflux)

    return np.sum(Opflux)*r**2

# Improved version
def cmp_mag_flux2(theta,phi,br,r,abs=True):

    dtheta,dphi=np.meshgrid(np.gradient(theta[:,0]), np.gradient(phi[0,:]), indexing='ij')
    qr=br
    if abs:
        qr=np.abs(br)
    fl=((dphi*qr*dtheta*np.sin(theta)).sum(axis=1)).sum(axis=0)
    return fl*r**2

def prints(string,silent=True):
    if not silent:
        print(string)

