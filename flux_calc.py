import numpy as np
from astropy.io import ascii
from astropy.table import Table
import pdb
from matplotlib import pyplot as plt

''' cutoff below 10 '''


n0 = 1.04 * 10**11 / 1000000    # cm-3
v0 = 3971 * 100000              # cm s-1
tau = 8.073 * 10**14     # s
R_J = 69911 * 1000              # m         why?
M_J = 1.898 * 10**27 * 1000     # g
n_J = 2.0 * 10**5 / 1000000     # cm-3
v_eff_J = 520 * 100000          # cm s-1
m_pr = 1.6726219 * 10**-27 * 1000      # g
kb = 1.3807 * 10**-16           # cm2 g s-2 K-1
m_el = 9.10938356 * 10**-31 * 1000     # g
Omega = 1.7                     # sr
Mom_J = 1.56 * 10**27 * 10000   # A cm2
sig_sb = 5.6704 * 10**-5        # g s-3 K-4
P_radio_J = 2.1 * 10**11 * 10**7       # erg s-1
omega_J = 1.77 * 10**-4         # s-1
alpha = 0.26                    # 1
Qp = 5 * 10**5                  # 1
G = 6.674 * 10**-8              # cm3 g-1 s-2
M_sol = 1.989 * 10**33          # g
R_sol = 6.957 * 10**10          # cm
R_earth = 6.371 * 10**8         # cm
charge = 1.6 * 10**-19 * 3 * 10**9  # statC
e0 = 1                          # 1 cgs
mu0 = 1                         # 1 cgs
Br0 = 2.6 * 10**-9 * 0.0001     # gauss
Bphi0 = 2.4 * 10**-9 * 0.0001   # gauss
d0 = 1.496 * 10**13             # cm


def plasma_frequency(n):
    return (1/(2*np.pi))*np.sqrt(n*charge**2 / (e0*m_el))  # s-1

def max_cyclotron_frequency(Mom, R):
    return 24.0 * 10**6 * (Mom / Mom_J) / (R / R_J)**3     # s-1
    # return (charge*mu0*Mom) * 10**-3 / (4* np.pi**2 * m_el * R**3)

def flux(P, s, deltaf):
    return (P/(Omega * s**2 * deltaf)) * 10**-20        # mJy

def particle_density(t):
    return n0 * (1+(t/tau))**-1.86

def stellar_wind_v(t):
    return v0 * (1+(t/tau))**-.43

def kin_input_power(n, v_eff, R_s):
    return P_radio_J * n * v_eff**3 * R_s**2

def mag_input_power(v_eff, B_perp, R_s):
    return P_radio_J * v_eff * B_perp**2 * R_s**2

def orbital_velocity(a, period):
    return 2*np.pi*a / period

def effective_velocity(v_orb, v):
    return np.sqrt(v_orb**2 + v**2)

def standoff_distance(Mom, n, v_eff, T):
    return 40 * R_J * ( ((Mom / Mom_J)**2) / ( ((n/n_J)*(v_eff/v_eff_J)) + ((2*(n/n_J)*kb*T)/(m_pr*(v_eff_J**2))) ) )**(1/6.)

def core_radius(M, R):
    test_rc_frac = np.arange(0.01, 1.01, 0.01)
    test_rho = []
    for i in range(len(test_rc_frac)):
        rc = test_rc_frac * R
        test_rho.append(((np.pi*M) / (4*(R**3))) * (np.sin(np.pi * rc / R) / (np.pi * rc / R)))
    test_rho = np.array(test_rho)
    dex = (np.abs(test_rho-0.7)).argmin()
    rc = test_rc_frac[dex] * R
    print ' '
    print M
    print R
    print rc / R
    print test_rho[0, dex]
    print ' '
    return rc

def magnetic_moment(M, R, omega):

    pj = (3*M_J) / (4*np.pi*(R_J**3))
    rj = 0.9 * R_J
    kc = rj / ((M_J**.75)*(R_J**-0.96))
    km = Mom_J / ((pj**.5) * omega_J * (rj**3.5))

    print ' '
    print Mom_J
    mu_check = km*(pj**.5)*omega_J*(rj**3.5)
    print mu_check

    rc = core_radius(M, R)
    p = ((np.pi*M) / (4*(R**3))) * (np.sin(np.pi * rc / R) / (np.pi * rc / R))
    # rc = kc * (M**.75) * (R**-0.96)
    mu = km*(p**.5)*omega*(rc**3.5)
    return mu



def temp_eq_at_planet_surface(A, R_star, Teff_star, a, e):
    if A == 0:
        A = 0.4
    L = 4 * np.pi * R_star**2 * sig_sb * Teff_star**4
    return (  ((1-A)*L) / (16*np.pi*sig_sb*(a**2) * (1+(e**2 / 2))**2)  )**0.25

def radius_irradiated(M, Teq):
    Rni = (1.47 * R_J * (M/M_J)**(1/3.)) / (1 + ((M/M_J)/3.16)**(2/3.))
    T0 = 764 * (M/M_J)**.28     # K
    gamma = 1.15 + .05*(0.59*M_J / M)**1.03
    return Rni * (1 + .05*(Teq/T0)**gamma)

def tau_sync(R, M, M_star, a):                  # being maybe problamatic
    return ((4/9.) * alpha * Qp * (R**3 / (G*M)) * omega_J * (M/M_star)**2. * (a/R)**6.) * 3.17098 * 10**-8 * 10**-6         # Myr

def omega_rot(tsync, P_orb):
    if tsync <= 100:
        return 2*np.pi / P_orb
    else:
        return omega_J

def B_imf_r(d):
    return Br0 * (d/d0)**-2

def B_imf_phi(d):
    return Bphi0 * (d/d0)**-2

def B_perpendicular(Bimfr, Bimfphi, vorb, v):
    a = np.arctan(Bimfphi/Bimfr)
    b = np.arctan(vorb/v)
    return np.sqrt(Bimfr**2 + Bimfphi**2) * np.abs(np.sin(a-b))

name = []
mass = []
radius = []
P_orb = []
a = []
e = []
albedo = []
star_dist = []
star_radius = []
star_mass = []
star_age = []
star_teff = []
temp_calc = []
mag_field = []

data = open('allplanets.csv', 'r')
data.readline()
for row in data:
    row = row.strip()
    name.append(row.split(',')[0])
    mass.append(float(row.split(',')[1]) * M_J)             # g
    radius.append(float(row.split(',')[3]) * R_J)           # cm
    P_orb.append(float(row.split(',')[4]) * 86400)          # s
    a.append(float(row.split(',')[5]) * 1.496*10**13)       # cm
    e.append(float(row.split(',')[6]))
    temp_calc.append(row.split(',')[11])
    albedo.append(float(row.split(',')[13]))
    star_dist.append(float(row.split(',')[23]) * 3.086*10**18)  # cm
    star_mass.append(float(row.split(',')[25]) * M_sol)         # g
    star_radius.append(float(row.split(',')[26]) * R_sol)       # cm
    star_age.append(float(row.split(',')[28]) * 3.16 * 10**16)  # s
    star_teff.append(float(row.split(',')[29]))
    mag_field.append(row.split(',')[31])

goodname = []
goodmm = []
goodplasma = []
f = []
phi_kin = []
phi_mag = []
temp = []
for i in range(len(name)):
    temp_eq = temp_eq_at_planet_surface(albedo[i], star_radius[i], star_teff[i], a[i], e[i])
    if radius[i] == 0:
        radius[i] = radius_irradiated(mass[i], temp_eq)
    tsync = tau_sync(radius[i], mass[i], star_mass[i], a[i])
    omega = omega_rot(tsync, P_orb[i])
    mag_mom = magnetic_moment(mass[i], radius[i], omega)
    print mag_mom/Mom_J
    temp.append(mag_mom/Mom_J)
    freq = max_cyclotron_frequency(mag_mom, radius[i])
    n = particle_density(star_age[i])
    v = stellar_wind_v(star_age[i])
    v_orb = orbital_velocity(a[i], P_orb[i])
    v_eff = effective_velocity(v_orb, v)
    Bimfr = B_imf_r(a[i])
    Bimfphi = B_imf_phi(a[i])
    B = B_perpendicular(Bimfr, Bimfphi, v_orb, v)
    R_s = standoff_distance(mag_mom, n, v_eff, temp_eq, radius[i])
    P_radio_kin = kin_input_power(n, v_eff, R_s)
    P_radio_mag = mag_input_power(v_eff, B, R_s)
    kin_flux = flux(P_radio_kin, star_dist[i], freq)
    mag_flux = flux(P_radio_mag, star_dist[i], freq)
    plasma = plasma_frequency(n)
    if plasma < freq:
        goodname.append(name[i])
        goodmm.append(mag_mom/Mom_J)
        f.append(freq*10**-6)
        phi_kin.append(kin_flux)
        phi_mag.append(mag_flux)
        goodplasma.append(plasma*10**-6)

import matplotlib.patches as patches
plt.figure()
ax = plt.subplot(121)
plt.xscale('log')
plt.yscale('log')
plt.plot(f, phi_kin, 'ro')
ax.add_patch(patches.Rectangle((50, 2.5614), 150, 100, hatch='/', fill=False))
plt.xlabel('frequency [MHz]')
plt.ylabel('flux [mJy]')
plt.ylim(min(phi_kin), 10)
plt.title('Kinetic Model')

plt.subplot(122)
plt.xscale('log')
plt.yscale('log')
plt.plot(f, phi_mag, 'ro')
# plt.bar(125, height = 10**9, width = 75, bottom = 10**-12, color='none', edgecolor='none', hatch="/")
plt.xlabel('frequency [MHz]')
plt.ylabel('flux [mJy]')
plt.title('Magnetic Model')
plt.show()

plt.figure()
plt.hist(goodmm, bins = 100)
plt.ylim(0, 10)
plt.show()
pdb.set_trace()