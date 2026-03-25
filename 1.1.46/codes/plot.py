import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

R, L, C = 100, 0.1, 100e-6
A, f = 10, 50
w = 2 * np.pi * f

# Transfer Function: H(s) = (R - s^2RLC) / (2s^2RLC + s(L + 4R^2C) + 2R)
num = [-R*L*C, 0, R]
den = [2*L*R*C, L + 4*R**2*C, 2*R]
sys = signal.TransferFunction(num, den)

t = np.linspace(0, 0.2, 200)
vin = A * np.sin(w * t)

tout, v0, _ = signal.lsim(sys, vin, t)

plt.plot(t, vin, label='Vin(t)', alpha=0.5, linestyle='--')
plt.plot(t, v0, label='V0(t)', linewidth=2)
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)
plt.title('Time Domain Analysis of AC Bridge')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.savefig('../figs/plot.jpg')
plt.show()
