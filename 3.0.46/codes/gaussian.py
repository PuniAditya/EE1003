import numpy as np
import matplotlib.pyplot as plt

N = 0.6
M = 1 - (2/3) * N  # Equals 0.6
num_samples = 5000

u = np.random.rand(num_samples)
is_term = u < M

# For lambda = 2, rate = 2 -> scale = 1/2
# For lambda = 3, rate = 4.5 -> scale = 1/4.5
V = np.zeros(num_samples)
V[is_term] = np.random.exponential(scale=1/2, size=np.sum(is_term))
V[~is_term] = np.random.exponential(scale=1/4.5, size=np.sum(~is_term))

samples = np.random.normal(loc=0, scale=np.sqrt(V))

plt.figure(figsize=(8, 6))
plt.hist(samples, bins=150, density=True, alpha=0.6, color='dodgerblue', 
         edgecolor='black', label='Sampled PDF')

x_vals = np.linspace(-4, 4, 1000)
pdf_theoretical = M * np.exp(-2 * np.abs(x_vals)) + N * np.exp(-3 * np.abs(x_vals))
plt.plot(x_vals, pdf_theoretical, color='red', linewidth=2, linestyle='--', 
         label=r'Theoretical: $P_X(x)$')
plt.title('PDF Generation using Conditional Gaussian (n = 5000)')
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('../figs/conditional_gaussian_5000.png')
plt.show()
