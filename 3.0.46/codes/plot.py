import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, t = sp.symbols('x t', real=True)
M, N = sp.symbols('M N', real=True, positive=True)
pdf_sym = sp.Piecewise(
    (M * sp.exp(2 * x) + N * sp.exp(3 * x), x < 0),
    (M * sp.exp(-2 * x) + N * sp.exp(-3 * x), True)
)
total_prob = sp.integrate(pdf_sym, (x, -sp.oo, sp.oo))
M_sol = sp.solve(total_prob - 1, M)[0]

N_val = 0.6
M_val = M_sol.subs(N, N_val)
pdf_sub = pdf_sym.subs({M: M_val, N: N_val})

cdf_sym = sp.integrate(pdf_sub.subs(x, t), (t, -sp.oo, x))
pdf_func = sp.lambdify(x, pdf_sub, modules=['numpy'])
cdf_func = sp.lambdify(x, cdf_sym, modules=['numpy'])
x_vals = np.linspace(-4, 4, 1000)
pdf_vals = pdf_func(x_vals)
cdf_vals = cdf_func(x_vals)

plt.figure(1, figsize=(8, 6))
plt.plot(x_vals, pdf_vals, label=rf'PDF $P_X(x)$', color='blue', linewidth=2)
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)
plt.fill_between(x_vals, pdf_vals, color='blue', alpha=0.2)
plt.title('Probability Density Function')
plt.xlabel(r'$x$')
plt.ylabel('Probability Density')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('../figs/pdf_plot.jpg')

plt.figure(2, figsize=(8, 6))
plt.plot(x_vals, cdf_vals, label=r'CDF $F_X(x)$', color='red', linewidth=2)
plt.axhline(0, color='black', linewidth=2)
plt.axhline(1, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)
plt.title('Cumulative Distribution Function')
plt.xlabel(r'$x$')
plt.ylabel('Cumulative Probability')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('../figs/cdf_plot.jpg')
plt.show()
