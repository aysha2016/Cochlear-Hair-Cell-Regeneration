import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def simulate_resonance(mass, stiffness, damping, force_amplitude, duration, dt=0.001):
    n = int(duration / dt)
    x = np.zeros(n)
    v = np.zeros(n)
    for i in range(1, n):
        force = force_amplitude * np.sin(2 * np.pi * 100 * i * dt)
        a = (force - damping * v[i-1] - stiffness * x[i-1]) / mass
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt
    return np.linspace(0, duration, n), x

def simulate_resonance(scaffold, freq_range=np.linspace(100, 8000, 100)):
    # Simplified "finite-element"–like simulation:
    # – Treat scaffold density (mean) as a "stiffness" (or "resonance") factor.
    # – Assume "connectivity" (sum of material voxels / total voxels) also affects resonance.
    density = np.mean(scaffold)
    connectivity = np.sum(scaffold) / (scaffold.size)
    # (Example: resonance peak shifts with "stiffness" (density + connectivity)  
    resonance_freq = 1000 + 5000 * (density + connectivity) / 2  
    # (Simplified resonance curve – a Gaussian peak centered at resonance_freq)  
    response = np.exp(-((freq_range - resonance_freq) ** 2) / (2 * (500 ** 2)))
    return freq_range, response, resonance_freq

if __name__ == "__main__":
    mass = 0.01
    stiffness = 10
    damping = 0.1
    force_amplitude = 1
    t, x = simulate_resonance(mass, stiffness, damping, force_amplitude, 2)
    plt.plot(t, x)
    plt.title("Resonance Vibration at 100 Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.show()
