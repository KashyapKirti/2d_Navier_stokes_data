import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 256
h = (2 * np.pi) / n
x_grid = np.arange(h, 2 * np.pi + h/2, h)   # match MATLAB's (h:h:2*pi)
y_grid = np.arange(h, 2 * np.pi + h/2, h)
X, Y = np.meshgrid(x_grid, y_grid)

# ---- Reading binary data ----
with open("vor2500.dat", "rb") as f:
    dum1 = np.fromfile(f, dtype=np.float32, count=1)   # dummy float32
    omega = np.fromfile(f, dtype=np.float64, count=n*n)
    dum2 = np.fromfile(f, dtype=np.float32, count=n)   # dummy array
omega = omega.reshape((n, n))

# ---- Fourier wave numbers ----
kx = (2*np.pi/(n*h)) * np.fft.fftshift(np.arange(-n//2, n//2))
ky = kx.copy()
KX, KY = np.meshgrid(kx, ky)

k2 = KX**2 + KY**2
k2[0, 0] = 1  # avoid division by zero

# ---- FFT of vorticity ----
omega_hat = np.fft.fft2(omega)

# ---- Solve Poisson in Fourier space ----
psi_hat = -omega_hat / k2
psi_hat[0, 0] = 0   # force zero mean

# ---- Inverse FFT to get streamfunction ----
psi = np.real(np.fft.ifft2(psi_hat))

# ---- Derivatives ----
dpsi_dy, dpsi_dx = np.gradient(psi, h, h)  # careful: numpy returns [d/dy, d/dx]
u = dpsi_dy     # u = dψ/dy
v = -dpsi_dx    # v = -dψ/dx

# ================= PLOTTING =================
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 1. Vorticity field
im0 = axs[0].imshow(omega, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], 
                    origin="lower", aspect="equal")
plt.colorbar(im0, ax=axs[0])
axs[0].set_title(r"$\omega$ (vorticity)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

# 2. Streamfunction with streamlines
im1 = axs[1].imshow(psi, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], 
                    origin="lower", aspect="equal")
plt.colorbar(im1, ax=axs[1])
axs[1].streamplot(X, Y, u, v, color="w", linewidth=0.8, density=1.2)
axs[1].set_title(r"$\psi$ (streamfunction) with streamlines")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

# 3. Velocity magnitude with quiver
vel_mag = np.sqrt(u**2 + v**2)
im2 = axs[2].imshow(vel_mag, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], 
                    origin="lower", aspect="equal")
plt.colorbar(im2, ax=axs[2])

step = 10
axs[2].quiver(X[::step, ::step], Y[::step, ::step], 
              u[::step, ::step], v[::step, ::step], 
              scale=10, color="k", linewidth=3.0)

axs[2].set_title(r"$|v|$ with velocity vectors")
axs[2].set_xlabel("x")
axs[2].set_ylabel("y")

plt.tight_layout()
plt.show()

