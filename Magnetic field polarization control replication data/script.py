import scipy.integrate
import numpy as np
import matplotlib.patches as mpatch
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def wrapper(a0, bx, density):
    shift = np.pi
    phase_y=0.0
    phase_z=phase_y+shift
    duration=4.0*2*np.pi
    thickness=0.01*2*np.pi
    alpha=density*thickness
    epsilon=alpha*0.5

    def pulse_y(xi):
        if((xi)>=0) and ((xi)<duration):
            toret = a0*np.sin(np.pi*(xi)/duration)**2*np.sin(xi-duration/2+phase_y)
        else:
            toret = 0.0
        return toret

    def pulse_z(xi):
        if((xi)>=0) and ((xi)<duration):
            toret = a0*np.sin(np.pi*(xi)/duration)**2*np.sin(xi-duration/2+phase_z)
        else:
            toret = 0.0
        return toret

    x=np.linspace(0,20*2*np.pi, 1000)
    pulsey=np.zeros(len(x))
    pulsez=np.zeros(len(x))

    for n,xx in enumerate(x):
        #print(n,xx)
        pulsey[n]=pulse_y(xx)
        pulsez[n]=pulse_z(xx)

    #plt.figure()
    #plt.plot(pulsey)
    #plt.plot(pulsez)
    # for RK4 we assume:
    # xi = t-x
    # y0 -> h(xi)
    # y1 -> x(xi)
    # y2 -> y(xi)
    # y3 -> z(xi)

    def f(y,xi):
        ay = pulse_y(xi)
        az = pulse_z(xi)
        u_perp_sqr = (ay-epsilon*y[2] - bx*y[3])**2+(az-epsilon*y[3] + bx*y[2])**2
        #restoring_field=sign(y[1])
        #restoring_field=density*y[1]
        restoring_field=np.tanh(y[1]/(thickness/4))

        f0 = epsilon*(restoring_field-u_perp_sqr/(1+u_perp_sqr))
        f1 = 0.5/y[0]**2*(1-y[0]**2+u_perp_sqr)
        f2 = 1./y[0]*(ay-epsilon*y[2] - bx*y[3])
        f3 = 1./y[0]*(az-epsilon*y[3] + bx*y[2])
        return [f0,f1,f2,f3]

    Time=25
    Npercycle=4096
    NofTS=Time*Npercycle
    xi_end=Time*2*np.pi

    xi = np.linspace(0, xi_end, NofTS)
    dxi=xi[1]-xi[0]

    y0=[1.0,0.0,0.,0.]

    sol = odeint(f, y0, xi)

    h=sol[:,0]
    x=sol[:,1]
    y=sol[:,2]
    z=sol[:,3]

    t=xi+x

    tdetector = t+x

    ux=np.gradient(x,xi)*h
    uy=np.gradient(y,xi)*h
    uz=np.gradient(z,xi)*h

    gamma=np.sqrt(1+ux**2+uy**2+uz**2)
    vx = ux/gamma
    gamma_x=1./np.sqrt(1-vx**2)

    Eydetector = epsilon*uy/gamma/(1+vx)
    Ezdetector = epsilon*uz/gamma/(1+vx)

    Ey_interp=np.interp(t, tdetector, Eydetector)
    Ez_interp=np.interp(t, tdetector, Ezdetector)

    sp_y=np.fft.fft(Ey_interp)
    sp_z=np.fft.fft(Ez_interp)
    w=np.fft.fftfreq(len(Ey_interp), d=t[1]-t[0])

    filter_center=15.0
    filter_width=5.
    bandgap_filter_minus=np.exp(-(w*2*np.pi-filter_center)**16/filter_width**16)
    bandgap_filter_plus=np.exp(-(w*2*np.pi+filter_center)**16/filter_width**16)
    bandgap_filter=bandgap_filter_minus+bandgap_filter_plus


    sp_y_filt = bandgap_filter*sp_y
    sp_z_filt = bandgap_filter*sp_z

    filtered_y=np.fft.ifft(sp_y_filt)
    filtered_z=np.fft.ifft(sp_z_filt)

    ### Stokes parameters and ellipticity
    spy=sp_y_filt
    spz=sp_z_filt

    S0=spy*np.conjugate(spy)+spz*np.conjugate(spz)
    S1=spy*np.conjugate(spy)-spz*np.conjugate(spz)
    S2=2*np.real(spy*np.conjugate(spz))
    S3=2*np.imag(spy*np.conjugate(spz))

    #V=sqrt(S1**2+S2**2+S3**2)/S0
    ind1 = np.where((w*2*np.pi)>10)[0][0]
    ind2 = np.where((w*2*np.pi)>=20)[0][0]
    #print(ind1, ind2)

    chi=0.5*np.arctan((S3+1e-200)/(np.sqrt(S1**2+S2**2)+1e-200))
    ellips = abs(np.tan(chi))[ind1:ind2]
    #figure(figsize=(12,5))
    #plot(w*2*pi, abs(tan(chi)))
    peaks = find_peaks(ellips)
    #plt.plot(w[ind1:ind2][peaks[0]]*2*pi, ellips[peaks[0]], 'x')
    #xlim(0,30)

    #print("peak ellipticity")
    #print(mean(ellips[peaks[0]]))
    #print("all ellipticity")
    #print(mean(abs(tan(chi))[ind1:ind2]))
    return [np.mean(ellips[peaks[0]]), np.mean(ellips)]

a0s = np.linspace(1, 100, 20)
bxs = np.linspace(1, 100, 20)
eps = np.linspace(200, 10000, 20)
print(a0s)
print(eps)
peak1 = np.zeros((20,20))
full1 = np.zeros((20,20))

peak2 = np.zeros((40,40))
full2 = np.zeros((40,40))

for i, a0 in tqdm(enumerate(a0s)):
    for j, bx in enumerate(bxs):
        peak1[i,j], full1[i,j] = wrapper(a0, bx, 800)

#for j, bx in tqdm(enumerate(bxs)):
#    for k, density in enumerate(eps):
#        peak2[j,k], full2[j,k] = wrapper(20, bx, density)

np.save("peak1", peak1)
#np.save("peak2", peak2)
np.save("full1", full1)
#np.save("full2", full2)
