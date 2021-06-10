import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("pgf")
#matplotlib.rcParams['axes.unicode_minus'] = False
#matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#})

def newt_sol(y, t, U0, w, nu, H):
    E = np.exp((1+1j)*H*np.sqrt(w/(2*nu)))
    def f():
        return (U0/2)*((1-E)*np.exp(-(1+1j)*y*np.sqrt(w/(2*nu))) \
            + (E**-1 - 1)*np.exp((1+1j)*y*np.sqrt(w/(2*nu))))/(E**-1 - E)
    return 2*np.real(np.exp((w*t)*1j)*f())

def old_b_sol(y, t, U0, w, mu, rho, tau, G, H):
    nu = mu/rho
    a1 = tau/(1 + (w**2)*(tau**2))
    a2 = w*(tau**2)/(1+(w**2)*(tau**2))
    b =1j*rho*w/(rho*nu + G*(a1 + 1j*a2))
    E = np.exp(H*np.sqrt(b))
    return U0*np.real((1-E**-1)*np.exp(w*t*1j + y*np.sqrt(b)) + (E -1)*np.exp(w*t*1j -y*np.sqrt(b))/(E - E**-1))

N = 100
H = 10
U0 = 1
w = 2
mu = 1
rho = 1
tau = 1
G = 1
y_array = np.linspace(0, H, num=N)
t_array = np.arange(0, 10, 0.01)
newt = [newt_sol(y_array, t, U0, w, mu/rho, H) for t in t_array]
old = [old_b_sol(y_array, t, U0, w, mu, rho, tau, G, H) for t in t_array]

fig, (newt_ax, old_ax) = plt.subplots(1, 2)
newt_ax.set_xlabel('u, velocity of fluid')
newt_ax.set_ylabel('y')
newt_ax.set_title('Newtonian fluid')
newt_ax.plot(newt[0], y_array, label="t=0")
newt_ax.plot(newt[199], y_array, label="t=2")
newt_ax.plot(newt[399], y_array, label="t=4")
newt_ax.plot(newt[599], y_array, label="t=6")
newt_ax.plot(newt[799], y_array, label="t=8")
newt_ax.plot(newt[999], y_array, label="t=10")

old_ax.set_xlabel('u, velocity of fluid')
old_ax.set_ylabel('y')
old_ax.set_title('Oldroyd-B fluid')
old_ax.plot(old[0], y_array, label="t=0")
old_ax.plot(old[199], y_array, label="t=2")
old_ax.plot(old[399], y_array, label="t=4")
old_ax.plot(old[599], y_array, label="t=6")
old_ax.plot(old[799], y_array, label="t=8")
old_ax.plot(old[999], y_array, label="t=10")
plt.show()
#plt.savefig('old_vs_newt.pgf')