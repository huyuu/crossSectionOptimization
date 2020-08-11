import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
from numpy import abs, sqrt, cos, sin, pi

from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1


# Constants

mu0 = 4*nu.pi*1e-7
I = 100


# Model

# phi_: coil phi, a: coil radius, z_: coil z position, lo: point p's radius, z: point p's z position
def _lo(phi_, a, z_, lo, z):
    return (z-z_)*cos(phi_) / (lo**2 + a**2 - 2*lo*a*cos(phi_) + (z-z_)**2)**1.5

# phi_: coil phi, a: coil radius, z_: coil z position, lo: point p's radius, z: point p's z position
def _z(phi_, a, z_, lo, z):
    return (a-lo*cos(phi_)) / (lo**2 + a**2 - 2*lo*a*cos(phi_) + (z-z_)**2)**1.5

# http://www.f-denshi.com/000TokiwaJPN/20vectr/cpx01.html
def BpFromBiosavart(I, coilRadius, coilZ, lo, z):
    Bp_r = mu0*I/4/pi*coilRadius * quadrature(_lo, 0, 2*pi, args=(coilRadius, coilZ, lo, z), maxiter=10000)[0]
    Bp_z = mu0*I/4/pi*coilRadius * quadrature(_z, 0, 2*pi, args=(coilRadius, coilZ, lo, z), maxiter=10000)[0]
    return (Bp_r, Bp_z)


# # phi_: coil phi, a: coil radius, z_: coil z position, lo: point p's radius, z: point p's z position
# def _absBp(z, lo, a, z_):
#     bp = BpFromBiosavart(I=I, coilRadius=a, coilZ=z_, lo=lo, z=z)
#     return bp[0]**2 + bp[1]**2

# def singleLossFromBiosavart(coilRadius, coilZ, Z0):
#     double_integrate = dblquad(_absBp, 0, 0.99*coilRadius, lambda z: 0, lambda z: Z0, args=(coilRadius, coilZ), tol=1e-8, maxiter=10000)[0]
#     return double_integrate


def calculateBnormFromLoop(I, coilRadius, coilZ, lo, z):
    bp = BpFromBiosavart(I=I, coilRadius=coilRadius, coilZ=coilZ, lo=lo, z=z)
    return sqrt(bp[0]**2 + bp[1]**2)

def calculateBnormFromCoil(I, r, l, N, lo, z):
    coilZPositions = nu.linspace(-l/2, l/2, N)
    return sum((calculateBnormFromLoop(I, r, coilZ, lo, z) for coilZ in coilZPositions))


def _f(phi, r1, r2, d):
    return r1 * r2 * nu.cos(phi) / nu.sqrt( r1**2 + r2**2 + d**2 - 2*r1*r2*nu.cos(phi) )

def MutalInductance(r1, r2, d):
    # return 0.5 * mu0 * quadrature(_f, 0, 2*nu.pi, args=(r1, r2, d), tol=1e-9, maxiter=100000)[0]
    # return 0.5 * mu0 * quadrature(_g, -1, 1, args=(r1, r2, d), tol=1e-12, maxiter=100000)[0]
    squaredK = 4*r1*r2/((r1+r2)**2+d**2)
    k = nu.sqrt(squaredK)
    if k < 0.99:
        result = mu0 * nu.sqrt(r1*r2) * ( (2/k-k)*ellipk(squaredK) - 2/k*ellipe(squaredK) )
    else:  # k around 1
        result = mu0 * nu.sqrt(r1*r2) * ( (2/k-k)*ellipkm1(squaredK) - 2/k*ellipe(squaredK) )

    if result >= 0:
        return result
    else:
        return 0.5 * mu0 * quadrature(_f, 0, 2*nu.pi, args=(r1, r2, d), tol=1e-6, maxiter=10000)[0]




if __name__ == '__main__':
    coilRadius = 1.5e-2
    coilZ = 0
    points = 100
    Z0 = coilRadius
    los = nu.linspace(0, 0.9*coilRadius, points)
    zs = nu.linspace(-Z0, Z0, points)

    # create args
    args = []
    for lo in los:
        for z in zs:
            args.append((coilRadius, coilZ, lo, z))
    # calculate bs for all points
    bs = []
    with mp.Pool(processes=min(mp.cpu_count()-1, 50)) as pool:
        bs = pool.starmap(BpFromBiosavart, args)
    bs_r = nu.array([ b[0] for b in bs ]).reshape((points, points))
    bs_z = nu.array([ b[1] for b in bs ]).reshape((points, points))

    pl.xlabel(r'$\rho$/coil_radius')
    pl.ylabel(r'$Z-Z_0$')
    X, Y = nu.meshgrid(los/coilRadius, zs-coilZ, indexing='ij')
    pl.quiver(X, Y, bs_r, bs_z)
    pl.show()
