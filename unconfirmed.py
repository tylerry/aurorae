import numpy as np
from astropy.io import ascii
from astropy.table import Table
import pdb
from matplotlib import pyplot as plt

''' cutoff below 10 '''


n0 = 1.04 * 10**11 / 1000000    # cm-3
v0 = 3971 * 100000              # cm s-1
tau = 8.073 * 10**14            # s
R_J = 7.149 * 10**9             # cm
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
charge = 4.80320451 * 10**-10   # statC
e0 = 1                          # 1 cgs
mu0 = 1                         # 1 cgs
Br0 = 2.6 * 10**-9 * 0.0001     # gauss
Bphi0 = 2.4 * 10**-9 * 0.0001   # gauss
d0 = 1.496 * 10**13             # cm
a_J = 778.57*10**11             # cm
Rs_J = 63*R_J
AU = 1.496 * 10**13             # cm
f0 = 1.16

def plasma_frequency(n):
    return np.sqrt(4*np.pi*n*charge**2 / (m_el))  # s-1                         # changed from original

def max_cyclotron_frequency(Mom, R):
    return 24.0 * 10**6 * (Mom / Mom_J) / ((R / R_J)**3)     # s-1
    # return (charge*mu0*Mom) * 10**-3 / (4* np.pi**2 * m_el * R**3)

def flux(P, s, deltaf):
    return (P/(Omega * s**2 * deltaf)) * 10**23        # Jy

def particle_density(t, a):
    return n0 * (1+(t/tau))**-1.86 * (a/AU)**-2                                 # added distance dependence based on Johnstone+ 15

def stellar_wind_v(t):
    return v0 * (1+(t/tau))**-.43

def kin_input_power(n, v_eff, R_s):
    return ((n * v_eff**3 * R_s**2)/(n_J * v_eff_J**3 * Rs_J**2)) * P_radio_J

def mag_input_power(v_eff, B_perp, R_s):
    return ((B_perp**2 * v_eff * R_s**2)/(1**2 * v_eff_J * Rs_J**2)) * P_radio_J

def orbital_velocity(a, period):
    return 2*np.pi*a / period

def effective_velocity(v_orb, v):
    return np.sqrt(v_orb**2 + v**2)

def standoff_distance(Mom, n, v_eff, a, R):
    Rs = ((mu0 * f0**2 * Mom**2)/(8 * np.pi**2 * m_pr * n * v_eff**2))**(1/6.)
    if Rs < R:
        Rs = R
    return Rs

def core_radius(M, R):
    test_rc_frac = np.arange(0.01, 1.01, 0.01)
    test_rho = []
    for i in range(len(test_rc_frac)):
        rc = test_rc_frac * R
        test_rho.append(((np.pi*M) / (4*(R**3))) * (np.sin(np.pi * rc / R) / (np.pi * rc / R)))
    test_rho = np.array(test_rho)
    dex = (np.abs(test_rho-0.7)).argmin()
    rc = test_rc_frac[dex] * R
    return rc

moms = []
def magnetic_moment(M, R, omega):

    pj = (3*M_J) / (4*np.pi*(R_J**3))
    rj = 0.9 * R_J
    kc = rj / ((M_J**.75)*(R_J**-0.96))
    km = Mom_J / ((pj**.5) * omega_J * (rj**3.5))

    mu_check = km*(pj**.5)*omega_J*(rj**3.5)

    p = (3*M) / (4*np.pi*(R**3))
    rc = kc * (M**.75) * (R**-0.96)
    if rc > R:
        rc = R
    mu = km*(p**.5)*omega*(rc**3.5)

    if M == 1.45*M_J:
        moms.append(mu/Mom_J)
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

def tau_sync(R, M, M_star, a):
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
discovered = []
ra = []
dec = []

# data = open('allplanets_heraregime.csv', 'r')
# data = open('allplanets.csv', 'r')
data = open('unconfirmed_catalog.csv', 'r')
data.readline()
for row in data:
    if ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,," not in row:
        row = row.strip()
        name.append(row.split(',')[0])
        ra.append(row.split(',')[69])
        dec.append(float(row.split(',')[70]))
        mass.append(float(row.split(',')[2]) * M_J)             # g
        radius.append(float(row.split(',')[8]) * R_J)           # cm
        P_orb.append(float(row.split(',')[11]) * 86400.)          # s
        a.append(float(row.split(',')[14]) * 1.496*10**13)       # cm
        e.append(float(row.split(',')[17]))
        temp_calc.append(row.split(',')[53])
        albedo.append(float(row.split(',')[58]))
        star_dist.append(float(row.split(',')[76]) * 3.086*10**18)  # cm
        star_mass.append(float(row.split(',')[82]) * M_sol)         # g
        star_radius.append(float(row.split(',')[85]) * R_sol)       # cm
        star_age.append(float(row.split(',')[89]) * 3.16 * 10**16)  # s
        star_teff.append(float(row.split(',')[92]))
        mag_field.append(row.split(',')[96])                    # yes/blank
        discovered.append(float(row.split(',')[24]))



goodname = []
goodmass = []
goodradius = []
goodmm = []
goodplasma = []
f = []
phi_kin = []
phi_mag = []
temp = []
goodra = []
gooddec = []
goodstarage = []
for i in range(len(name)):
    temp_eq = temp_eq_at_planet_surface(albedo[i], star_radius[i], star_teff[i], a[i], e[i])
    if radius[i] == 0:
        radius[i] = radius_irradiated(mass[i], temp_eq)
    tsync = tau_sync(radius[i], mass[i], star_mass[i], a[i])
    omega = omega_rot(tsync, P_orb[i])
    mag_mom = magnetic_moment(mass[i], radius[i], omega)
    temp.append(mag_mom/Mom_J)
    freq = max_cyclotron_frequency(mag_mom, radius[i])
    n = particle_density(star_age[i], a[i])
    v = stellar_wind_v(star_age[i])
    v_orb = orbital_velocity(a[i], P_orb[i])
    v_eff = effective_velocity(v_orb, v)
    Bimfr = B_imf_r(a[i])
    Bimfphi = B_imf_phi(a[i])
    B = B_perpendicular(Bimfr, Bimfphi, v_orb, v)
    R_s = standoff_distance(mag_mom, n, v_eff, a[i], radius[i])
    P_radio_kin = kin_input_power(n, v_eff, R_s)
    P_radio_mag = mag_input_power(v_eff, B, R_s)
    kin_flux = flux(P_radio_kin, star_dist[i], freq)
    mag_flux = flux(P_radio_mag, star_dist[i], freq)
    plasma = plasma_frequency(n)
    if plasma < freq:
    # if plasma < freq and dec[i] > -38.7 and dec[i] < -22.7:
        goodname.append(name[i])
        goodra.append(ra[i])
        gooddec.append(dec[i])
        goodmass.append(mass[i]/M_J)
        goodradius.append(radius[i]/R_J)
        goodmm.append(mag_mom/Mom_J)
        f.append(freq*10**-6)
        phi_kin.append(kin_flux*1000)
        phi_mag.append(mag_flux*1000)
        goodplasma.append(plasma*10**-6)
        goodstarage.append(star_age[i]/ (3.16 * 10**16))

final = Table([goodname, goodra, gooddec, goodstarage, goodmass, goodradius, goodmm, goodplasma, f, phi_kin, phi_mag], names=['NAME', 'RA', 'DEC', 'AGE', 'MASS [Mj]', 'RADIUS [Rj]', 'MAG_MOM [MMj]', 'PLASMA_FRQ [MHz]', 'FRQ [MHz]', 'PHI_KIN [mJy]', 'PHI_MAG [mJy]'])
ascii.write(final, 'all_results_unconfirmed.csv', format='csv')



min = min(goodstarage)
max = max(goodstarage)
goodstaragecolor = goodstarage[:]
for i in range(len(goodstarage)):
    goodstaragecolor[i] = (goodstarage[i] - min) / (max-min)
colors = plt.cm.gray(goodstaragecolor)



import matplotlib.patches as patches
plt.figure()
plt.subplot(121)
plt.xscale('log')
plt.yscale('log')
plt.scatter(f, phi_kin, marker='o', c=colors)
plt.xlabel('frequency [MHz]')
plt.ylabel('flux [Jy]')
# plt.ylim(5*10**-8, 5*10)
# plt.xlim(1, 1000)
plt.fill_between([0,10],[10**-7,10**-1], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
plt.title('Kinetic Model')
plt.subplot(122)
plt.xscale('log')
plt.yscale('log')
plt.scatter(goodstarage, phi_kin, marker='o', c=colors)
plt.xlabel('Age [Gyr]')
plt.ylabel('flux [Jy]')
# plt.subplot(122)
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(f, phi_mag, 'ro')
# plt.fill_between([0,10],[10**-7,10**-1], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
# plt.xlabel('frequency [MHz]')
# plt.ylabel('flux [Jy]')
# plt.title('Magnetic Model')



print magnetic_moment(1.45*M_J, 1.23*R_J, 0.34*omega_J)/Mom_J

g7planet = []       # Planet name
g7mass = []         # Jupiter masses
g7radius = []       # Jupiter radii
g7mm = []           # Jupiter magnetic moment
g7fc = []           # MHz
g7fp = []           # MHz
g7phimag = []       # mJy
g7phikin = []       # mJy
g7phicme = []       # mJy
data2 = open('griessmeier07.list', 'r')
data2.readline()
data2.readline()
data2.readline()
for row in data2:
    row = row.strip()
    g7planet.append(row.split(';')[0].rstrip())
    g7mass.append(float(row.split(';')[1]))
    g7radius.append(float(row.split(';')[2]))
    g7mm.append(float(row.split(';')[3]))
    g7fc.append(float(row.split(';')[4]))
    g7fp.append(float(row.split(';')[5]))
    g7phimag.append(float(row.split(';')[6]))
    g7phikin.append(float(row.split(';')[7]))
    g7phicme.append(float(row.split(';')[8]))

plt.figure(figsize=(8, 9))
plt.subplot(321)
for i in range(len(g7planet)):
    for j in range(len(goodname)):
        if g7planet[i] == goodname[j]:
            plt.plot(g7mass[i], goodmass[j], 'ro')
plt.plot([0, 25], [0, 25], 'k--')
plt.ylabel('Richey-Yowell+ 19')
plt.title('Jupiter Masses')

plt.subplot(322)
for i in range(len(g7planet)):
    for j in range(len(goodname)):
        if g7planet[i] == goodname[j]:
            plt.plot(g7radius[i], goodradius[j], marker='o', color='orange')
plt.plot([0, 2], [0, 2], 'k--')
plt.xlim(0.5, 1.5)
plt.ylim(0.5, 1.5)
plt.title('Jupiter Radii')

plt.subplot(323)
for i in range(len(g7planet)):
    for j in range(len(goodname)):
        if g7planet[i] == goodname[j]:
            plt.plot(g7mm[i], goodmm[j], marker='o', color='yellow')
plt.ylabel('Richey-Yowell+ 19')
plt.plot([0, 6], [0, 6], 'k--')
plt.title('Jupiter MM')

plt.subplot(324)
for i in range(len(g7planet)):
    for j in range(len(goodname)):
        if g7planet[i] == goodname[j]:
            plt.plot(g7fc[i], f[j], marker='o', color='green')
plt.plot([0, 180], [0, 180], 'k--')
plt.title('Observed Frequency [MHz]')

plt.subplot(325)
for i in range(len(g7planet)):
    for j in range(len(goodname)):
        if g7planet[i] == goodname[j]:
            plt.plot(g7fp[i], goodplasma[j], marker='o', color='turquoise')
plt.plot([0, 1.4], [0, 1.4], 'k--')
plt.xlim(0, 1.4)
plt.ylabel('Richey-Yowell+ 19')
plt.xlabel('Griessmeier+ 07')
plt.title('Plasma Frequency [MHz]')

plt.subplot(326)
for i in range(len(g7planet)):
    for j in range(len(goodname)):
        if g7planet[i] == goodname[j]:
            plt.plot(g7phikin[i], phi_kin[j]*1000, marker='o', color='blue')
plt.plot([0, 10], [0, 10], 'k--')
plt.xlabel('Griessmeier+ 07')
plt.title('Phi_Kin [mJy]')

plt.subplots_adjust(left = 0.10, bottom = 0.06, right = 0.96, top = 0.96, wspace = 0.23, hspace = 0.26)

plt.show()
