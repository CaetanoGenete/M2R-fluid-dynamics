import cmath
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def u(y, t, **args):
    def Ef(y, **args):
        return np.exp((1 + 1j)*y*((args["omega"]/(2 * args["nu"]))**0.5))
    
    E = Ef(args["H"], **args)
    E_inv = Ef(-args["H"], **args)

    y_comp = (E - 1) * Ef(-y, **args) + (1 - E_inv)*Ef(y, **args)

    complex_result = np.exp(1j * args["omega"] * t) * y_comp / (E - E_inv)

    return args["u0"] * complex_result.real

fig, ax = plt.subplots()

omega = 3
v = 1
H = 1
u0 = 1

y = np.linspace(0, H, 100)

init_y = u(y, 0, omega = omega, nu = v, H = H, u0 = u0)

line, = ax.plot(y, init_y)
ax.set_ylim(-1, 1)

plt.ylabel("Velocity in x-direction")
plt.xlabel("y value of fluid")

def animate(i):    
    line.set_ydata(u(y, i/100, omega = omega, nu = v, H = H, u0 = u0))
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

plt.show()
