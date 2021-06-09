import numpy as np
import matplotlib.pyplot as plt

def newt_sol(y, t, U0, w, nu, H):
    E = np.exp((1+1j)*H*np.sqrt(w/(2*nu)))
    def f():
        return (U0/2)*((1-E)*np.exp(-(1+1j)*y*np.sqrt(w/(2*nu))) \
            + (E**-1 - 1)*np.exp((1+1j)*y*np.sqrt(w/(2*nu))))/(E**-1 - E)
    return 2*np.real(np.exp((w*t)*1j)*f())

def old_b_sol(y, t, U0, w, mu, rho, tau, G, H):
    nu = mu/rho
    