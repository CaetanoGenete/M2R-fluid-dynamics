import numpy as np
import math
import cmath

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import cm

from scipy.integrate import odeint

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

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

def sigma_12(y, t, consts):
    denom = 1 + (consts["w"] ** 2) * (consts["tau"] ** 2)
    
    a1 = consts["tau"] / denom
    a2 = consts["w"] * (consts["w"] ** 2) / denom

    return d_oldroyd_b_dy(y, t, consts) * (consts["eta"] +  consts["G"]*(a1 - 1j * a2))

def u(y, t, args):
    def Ef(y, args):
        return np.exp((1 + 1j)*y*((args["omega"]/(2 * (args["eta"]/args["rho"])))**0.5))
    
    E = Ef(args["H"], args)
    E_inv = Ef(-args["H"], args)

    y_comp = (E - 1) * Ef(-y, args) + (1 - E_inv)*Ef(y, args)

    complex_result = np.exp(1j * args["w"] * t) * y_comp / (E - E_inv)

    return args["u0"] * complex_result.real

def dudy(y, t, args):
    def Ef(y, args):
        return np.exp((1 + 1j)*y*((args["w"]/(2 * (args["eta"]/args["rho"])))**0.5))
    
    E = Ef(args["H"], args)
    E_inv = Ef(-args["H"], args)

    y_comp = (E - 1) * Ef(-y, args) - (1 - E_inv)*Ef(y, args)

    complex_result = np.exp(1j * args["w"] * t) * y_comp / (E - E_inv)

    return args["u0"]* ((args["w"]/(2 * (args["eta"]/args["rho"])))**0.5) * complex_result.real
    
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

        #setting dudy
        dudy[u_i] = (consts["eta"] * fd_2(u_vals, h) + consts["G"] * fd_1(a_vals, h)) / consts["rho"]

    return dudy

consts = {"u0": 2, "w": 1, "eta": 1, "rho": 1, "tau": 1, "G": 1, "H": 10}

N = 100
y = np.linspace(0, consts["H"], N)
y0 = np.real(oldroyd_b_sol(y, 0, consts))
A0 = np.real(A12(y, 0, consts))

N_t = 100
t = np.linspace(0, 2 * np.pi /consts["w"], N_t)
#num_sol = odeint(differential, np.concatenate([y0, A0]), t, args = (N, consts))

#----------------------------Plot------------------------------------
'''
fig, ax = plt.subplots()

ax.set_title("Oldroyd-B numerical residue: N = " + str(N))
ax.set_ylabel("Residue")
ax.set_xlabel("Time")

for y_sample in np.linspace(30, N - 30, 5):
    plt.plot(t, num_sol[:, int(y_sample)] - oldroyd_b_sol(y[int(y_sample)], t, consts), label = "y = " + "{:.2f}".format(y[int(y_sample)]))    

plt.legend()
plt.show()
'''

fig, (ax, ax2) = plt.subplots(1, 2)

ax.set_title("Magnitude of stress $\sigma_{12}$ on an Oldroyd-B fluid \n between oscillating plates")
ax.set_xlabel("Time $t$")
ax.set_ylabel("Height of fluid $y$")

ax2.set_title("Magnitude of stress $\sigma_{12}$ on a Newtonian fluid \n between oscillating plates")
ax2.set_xlabel("Time $t$")
ax2.set_ylabel("Height of fluid $y$")

'''
line, = ax.plot(y0, y)

def animate(i):    
    line.set_xdata(np.real(oldroyd_b_sol(y, i/10, consts)))
    return line,
    
ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=50)
'''
grid = np.zeros((100, 100))

for t_i in range(100):
    for y_i in range(100):

        grid[y_i, t_i] = abs(np.real(sigma_12(y[y_i], t[t_i], consts)))

        

T, Y = np.meshgrid(t, y)

t_tick_vals = np.linspace(0, 2 * np.pi /consts["w"], 9)
t_tick_labels = ["0"]

for i in range(1, 9):
    num = int(i / math.gcd(i, 4))
    denom = int(4 / math.gcd(i, 4))

    t_tick_labels.append("$\\frac{")

    if num != 1:
        t_tick_labels[i] += str(num)

    t_tick_labels[i] += "\pi}{"
    
    if denom > 1:
        t_tick_labels[i] += str(denom)
        
    t_tick_labels[i] += "\omega}$"

plt.sca(ax)
plt.xticks(t_tick_vals, t_tick_labels)
y_tick_vals = np.linspace(0, consts["H"], 6)
y_tick_labels = [0, 2, 4, 6, 8, "$H$ = " + str(consts["H"])]

plt.yticks(y_tick_vals, y_tick_labels)

graph = ax.pcolor(T, Y, grid, cmap = cm.hot)
fig.colorbar(graph, ax = ax)
graph.set_clim([0, 3])


grid_2 = np.zeros((100, 100))

for t_i in range(100):
    for y_i in range(100):

        grid_2[y_i, t_i] = consts["eta"] * abs(dudy(y[y_i], t[t_i], consts))

plt.sca(ax2)
plt.xticks(t_tick_vals, t_tick_labels)
y_tick_vals = np.linspace(0, consts["H"], 6)
y_tick_labels = [0, 2, 4, 6, 8, "$H$ = " + str(consts["H"])]

plt.yticks(y_tick_vals, y_tick_labels)

graph_2 = ax2.pcolor(T, Y, grid_2, cmap = cm.hot)
fig.colorbar(graph, ax = ax2)
graph_2.set_clim([0, 3])

plt.savefig('Stress_plots.pgf')
