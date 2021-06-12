import numpy as np
import cmath

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import odeint

def Ef(y, consts):
    denom = 1 + (consts["w"] ** 2) * (consts["tau"] ** 2)

    a1 = consts["tau"] / denom
    a2 = consts["w"] * (consts["w"] ** 2) / denom

    beta = 1j * consts["rho"] * consts["w"] / (consts["eta"] + consts["G"] * (a1 - a2 * 1j))

    return np.exp(y * np.sqrt(beta))

def oldroyd_b_sol(y, t, consts):
    E_H = Ef(consts["H"], consts)
    E_H_inv = Ef(-consts["H"], consts)

    f = ((1 - E_H_inv) * Ef(y, consts) + (E_H - 1) * Ef(-y, consts)) / (E_H - E_H_inv)

    return consts["u0"] * np.exp(1j * consts["w"] * t) * f

def d_oldroyd_b_dy(y, t, consts):
    E_H = Ef(consts["H"], consts)
    E_H_inv = Ef(-consts["H"], consts)

    denom = 1 + (consts["w"] ** 2) * (consts["tau"] ** 2)
    
    a1 = consts["tau"] / denom
    a2 = consts["w"] * (consts["w"] ** 2) / denom

    beta = 1j * consts["rho"] * consts["w"] / (consts["eta"] + consts["G"] * (a1 - a2 * 1j))

    dfdy = ((1 - E_H_inv) * Ef(y, consts) - (E_H - 1) * Ef(-y, consts)) / (E_H - E_H_inv)

    return consts["u0"] * np.sqrt(beta) * np.exp(1j * consts["w"] * t) * dfdy

def A12(y, t, consts):
    denom = 1 + (consts["w"] ** 2) * (consts["tau"] ** 2)
    
    a1 = consts["tau"] / denom
    a2 = consts["w"] * (consts["w"] ** 2) / denom

    return d_oldroyd_b_dy(y, t, consts) * (a1 - 1j * a2)
    
def fd_1(u, h):
    return (u[2] - u[0])/(2 * h)

def fd_2(u, h):
    return (u[2] - 2 * u[1] + u[0])/ (h * h)

def differential(u, t, N, consts):
    h = consts["H"]/N

    #i < N will be dudt while N <= i < 2N will be dA_11dt
    dudy = np.zeros(2 * N)

    dudy[0] = - consts["w"] * consts["u0"] * np.sin(consts["w"] * t)
    dudy[N - 1] = dudy[0]
    
    dudy[N] = - u[N]/ consts["tau"]
    dudy[2*N - 1] = dudy[N]
    
    for u_i in range(1, N - 1):
        a_i = u_i + N

        a_vals = u[a_i - 1: a_i + 2]
        u_vals = u[u_i - 1: u_i + 2]
        
        #Setting dA12dy = dudy - A12/tau
        dudy[a_i] = fd_1(u_vals, h) - u[a_i]/consts["tau"]

        #setting rho * dudy = eta * d2ydy + GdA12dy
        dudy[u_i] = (consts["eta"] * fd_2(u_vals, h) + consts["G"] * fd_1(a_vals, h)) / consts["rho"]

    return dudy

consts = {"u0": 2, "w": 1, "eta": 1, "rho": 1, "tau": 1, "G": 1, "H": 10}

N = 300
y = np.linspace(0, consts["H"], N)
y0 = np.real(oldroyd_b_sol(y, 0, consts))
A0 = np.real(A12(y, 0, consts))

N_t = 100
t = np.linspace(0, 10, N_t)

num_sol = odeint(differential, np.concatenate([y0, A0]), t, args = (N, consts))

#----------------------------Plot------------------------------------
fig, ax = plt.subplots()

ax.set_title("Oldroyd-B numerical residue: N = " + str(N))
ax.set_ylabel("Residue")
ax.set_xlabel("Time")

for y_sample in np.linspace(30, N - 30, 5):
    plt.plot(t, num_sol[:, int(y_sample)] - oldroyd_b_sol(y[int(y_sample)], t, consts), label = "y = " + "{:.2f}".format(y[int(y_sample)]))    

plt.legend()
plt.show()
