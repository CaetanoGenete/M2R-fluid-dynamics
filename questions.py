import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.integrate import odeint
#matplotlib.use("pgf")
#matplotlib.rcParams['axes.unicode_minus'] = False
#matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#})
def fd2(u, t, U0, w, nu, N, H):
    h = H/N
    new_u = [-U0*w*np.sin(w*t)]
    for i in range(1, N-1):
        new_u.append(nu*(u[i+1]-2*u[i]+u[i-1])/h**2)
    new_u.append(-U0*w*np.sin(w*t))
    return np.array(new_u)

def a_sol(y, t, U0, w, nu, H):
    E = np.exp((1+1j)*H*np.sqrt(w/(2*nu)))
    def f():
        return (U0/2)*((1-E)*np.exp(-(1+1j)*y*np.sqrt(w/(2*nu))) + (E**-1 - 1)*np.exp((1+1j)*y*np.sqrt(w/(2*nu))))/(E**-1 - E)
    return 2*np.real(np.exp((w*t)*1j)*f())

H=1
U0 = 1
w = 1
nu = 1
t_array = np.linspace(0, 5)


#N_array = np.linspace(100, 1000, num=10)
#error_array = np.array([])

#for N2 in N_array:
    #y_array = np.linspace(0, H, num=int(N2))
    #init_sol = a_sol(y_array, 0, U0, w, nu, H)
    #num_sol = odeint(fd2, init_sol, t_array, args=(U0, w, nu, int(N2), H))
    #ac_sol = np.array([a_sol(y_array, t, U0, w, nu, H) for t in t_array])
    #error_array= np.append(error_array, np.mean(abs(num_sol - ac_sol)))
#fig = plt.figure()
#ax = fig.add_subplot()
#ax.set_ylabel("Residual")
#ax.set_xlabel("N")
#ax.set_title("Residual of odeint vs the analytical solution")
#line, = ax.plot(N_array, error_array)
#plt.savefig('residuals.pgf')
#plt.show()


def fd3(u, t, U0, w, nu, N, H, n):
    h = H/N
    def v():
        new_v = []
        for i in range(N-1):
            new_v.append(abs((u[i+1]-u[i])/h)**(n-1))
        new_v.append(new_v[-1])
        return np.array(new_v)
    new_v = v()
    new_u = np.array([-U0*w*np.sin(w*t)])
    for i in range(1, N-1):
        new_u = np.append(new_u, nu*(new_v[i]*(u[i+1]-2*u[i]+u[i-1])/h**2 + ((new_v[i+1]-new_v[i])/h)*(u[i+1]-u[i])/h))
    new_u = np.append(new_u, -U0*w*np.sin(w*t))
    return new_u


nu = 1
H = 1
N = 100
w = 1
y_array = np.linspace(0, H, num=N)
init_s = a_sol(y_array, 0, U0, w, nu, H)
pow_sol = odeint(fd3, init_s, t_array, args=(U0, w, nu, N, H, 1.5))
newt_sol = np.array([a_sol(y_array, t, U0, w, nu, H) for t in t_array])


#fig, (newt_ax, pow_ax) = plt.subplots(1, 2)
#newt_ax.set_xlabel('u, velocity of fluid')
#newt_ax.set_ylabel('y')
#newt_ax.plot(newt_sol[0], y_array, label='t=0')
#newt_ax.plot(newt_sol[9], y_array, label='t=1')
#newt_ax.plot(newt_sol[19], y_array, label='t=2')
#newt_ax.plot(newt_sol[29], y_array, label='t=3')

#pow_ax.set_xlabel('u, velocity of fluid')
#pow_ax.set_ylabel('y')
#pow_ax.plot(pow_sol[0], y_array, label='t=0')
#pow_ax.plot(pow_sol[5], y_array, label='t=1')
#pow_ax.plot(pow_sol[10], y_array, label='t=2')
#pow_ax.plot(pow_sol[25], y_array, label='t=3')
#plt.show()