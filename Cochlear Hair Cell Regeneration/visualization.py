import matplotlib.pyplot as plt

def plot_scaffold(scaffold, slice_idx=None):
    if slice_idx is None:
        slice_idx = scaffold.shape[0] // 2
    # (Plot a 2‑D slice (heatmap) of the 3‑D scaffold)  
    plt.imshow(scaffold[slice_idx], cmap='gray')
    plt.title("Scaffold Structure (Slice)")
    plt.axis('off')
    plt.show()

def plot_resonance(freq_range, response):
    plt.plot(freq_range, response)
    plt.title("Resonance Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_cell_response(response):
    labels = ['Differentiation', 'Survival']
    plt.bar(labels, response)
    plt.title("Predicted Cell Outcomes")
    plt.show()
