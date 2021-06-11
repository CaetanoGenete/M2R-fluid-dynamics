import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint

pgf = False
if pgf:
    matplotlib.use("pgf")
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

def newt_sol(y, t, U0, w, nu, H):
    E = np.exp((1+1j)*H*np.sqrt(w/(2*nu)))
    def f():
        return (U0/2)*((1-E)*np.exp(-(1+1j)*y*np.sqrt(w/(2*nu))) \
            + (E**-1 - 1)*np.exp((1+1j)*y*np.sqrt(w/(2*nu))))/(E**-1 - E)
    return 2*np.real(np.exp((w*t)*1j)*f())

def u(y, t, **args):
    def Ef(y, **args):
        return np.exp((1 + 1j)*y*((args["omega"]/(2 * args["nu"]))**0.5))
    
    E = Ef(args["H"], **args)
    E_inv = Ef(-args["H"], **args)

    y_comp = (E - 1) * Ef(-y, **args) + (1 - E_inv)*Ef(y, **args)

    complex_result = np.exp(1j * args["omega"] * t) * y_comp / (E - E_inv)

    return args["u0"] * complex_result.real

def old_b_sol(y, t, U0, w, eta, rho, tau, G, H):
    nu = eta/rho
    a1 = tau/(1 + (w**2)*(tau**2))
    a2 = w*(tau**2)/(1+(w**2)*(tau**2))
    b =1j*rho*w/(rho*nu + G*(a1 - 1j*a2))
    E = np.exp(H*np.sqrt(b))
    return U0*np.real(((1-E**-1)*np.exp(w*t*1j + y*np.sqrt(b)) + (E -1)*np.exp(w*t*1j -y*np.sqrt(b)))/(E - E**-1))

N = 200
H = 10
U0 = 4
w = 1
mu = 1
rho = 1
tau = 5
G = 5
y_array = np.linspace(0, H, num=N)
t_array = np.arange(0, 10, 2)
newt = [u(y_array, t, u0=U0, omega=w, nu=mu/rho, H=H) for t in t_array]#[newt_sol(y_array, t, U0, w, mu/rho, H) for t in t_array]
old = [old_b_sol(y_array, t, U0, w, mu, rho, tau, G, H) for t in t_array]

fig, (newt_ax, old_ax) = plt.subplots(1, 2)
newt_ax.set_xlabel('u, velocity of fluid')
newt_ax.set_ylabel('y')
newt_ax.set_title('Newtonian fluid')
newt_ax.plot(newt[0], y_array, label="t=0")
newt_ax.plot(newt[1], y_array, label="t=2")
newt_ax.plot(newt[2], y_array, label="t=4")
newt_ax.plot(newt[3], y_array, label="t=6")
newt_ax.plot(newt[4], y_array, label="t=8")
#newt_ax.plot(newt[5], y_array, label="t=10")

old_ax.set_xlabel('u, velocity of fluid')
old_ax.set_ylabel('y')
old_ax.set_title('Oldroyd-B fluid')
old_ax.plot(old[0], y_array, label="t=0")
old_ax.plot(old[1], y_array, label="t=2")
old_ax.plot(old[2], y_array, label="t=4")
old_ax.plot(old[3], y_array, label="t=6")
old_ax.plot(old[4], y_array, label="t=8")
#old_ax.plot(old[5], y_array, label="t=10")
plt.legend()
#plt.show()
#plt.savefig('old_vs_newt.pgf')

#H_array = np.linspace(1, 50, 5)
#old = []
#for h in H_array:
#    y_array = (np.linspace(0, h, num=N))
#    old.append(old_b_sol(y_array, 8, U0, w, mu, rho, tau, G, h))
#y_array = np.linspace(0, 10, num=N)
#fig, ax = plt.subplots()
#ax.set_xlabel('u, velocity of fluid')
#ax.set_ylabel('y')
#ax.set_title('Oldroyd b fluid at differing heights')
#ax.plot(old[0], y_array, label="h=1")
#ax.plot(old[1], y_array, label=f"h={H_array[1]}")
#ax.plot(old[2], y_array, label=f"h={H_array[2]}")
#ax.plot(old[3], y_array, label=f"h={H_array[3]}")
#ax.plot(old[4], y_array, label=f"h={H_array[4]}")
#plt.legend()
#plt.savefig('old_at_heights.pgf')

def fd3(u, t, U0, w, nu, N, H, n, dt):
    h = H/N
    def v():
        new_v = []
        for i in range(N-1):
            new_v.append(abs((u[i+1]-u[i])/h)**(n-1)) #maybe here switch to a symettric finite difference?? 
        new_v.append(new_v[0])
        return np.array(new_v)
    new_v = v()
    new_u = [-U0*w*np.sin(w*(t-dt)) if t>0 else 0]
    for i in range(1, N-1):
        new_u.append(nu*(new_v[i]*(u[i+1]-2*u[i]+u[i-1])/h**2 + ((new_v[i+1]-new_v[i])/h)*(u[i+1]-u[i])/h))
    new_u.append(-U0*w*np.sin(w*(t-dt)) if t>0 else 0)
    return np.array(new_u)

nu = 1
H = 10
N = 200
w = 1
t_array = np.linspace(0, 10, num=1000)
dt = t_array[1] - t_array[0]
y_array = np.linspace(0, H, num=N)
init_s = old_b_sol(y_array, 0, U0, w, mu, rho, tau, G, H)
pow_s = odeint(fd3, init_s, t_array, args=(U0, w, nu, N, H, 1.5, dt))
old = [old_b_sol(y_array, t, U0, w, mu, rho, tau, G, H) for t in t_array]

ax, (old_ax, pow_ax) = plt.subplots(1,2)
old_ax.set(xlabel='u', ylabel='y', title='Oldroy-B')
pow_ax.set(xlabel='u', ylabel='y', title='Power law fluid')
old_ax.plot(old[0], y_array, label="t=0")
old_ax.plot(old[199], y_array, label="t=2")
old_ax.plot(old[399], y_array, label="t=4")
old_ax.plot(old[599], y_array, label="t=6")
old_ax.plot(old[799], y_array, label="t=8")
old_ax.plot(old[999], y_array, label="t=10")

pow_ax.plot(pow_s[0], y_array, label="t=0")
pow_ax.plot(pow_s[199], y_array, label="t=2")
pow_ax.plot(pow_s[399], y_array, label="t=4")
pow_ax.plot(pow_s[599], y_array, label="t=6")
pow_ax.plot(pow_s[799], y_array, label="t=8")
pow_ax.plot(pow_s[999], y_array, label="t=10")
plt.legend()
plt.show()
