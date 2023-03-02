import librosa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

# Read in the music file
# y, sr = librosa.load(librosa.ex('nutcracker'), offset=20, duration=20)
y, sr = librosa.load('Canon.wav')
save_animation = True
# Save path
save_path = 'Visualize Canon.mp4'

# CQT
C = np.abs(librosa.cqt(y, sr=sr))
# Convert to amplitude
logC = librosa.amplitude_to_db(abs(C))
# Chroma features
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

# Plot CQT
fig, ax = plt.subplots(figsize=(12, 4))
img = librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax, cmap='magma')
ax.set_title('Constant-Q Power Spectrum')
fig.colorbar(img, ax=ax, format='%+2.0f dB')
print(plt.get_cmap())
fig.show()

# Plot Chroma
fig, ax = plt.subplots(figsize=(12, 4))
img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='chroma_cqt')
yticklabels = ax.get_yticklabels()
fig.colorbar(img, ax=ax)
fig.show()

# Animation
if save_animation:
    palette = list(sns.color_palette("magma", 12).as_hex())
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.1])
    ax.set_xticks([0, 2, 4, 5, 7, 9, 11])
    ax.set_xticklabels(['C', 'D', 'E', 'F', 'G', 'A', 'B'])
    ax.set_xlabel('Pitch Class')
    ax.set_ylabel('Intensity')
    plt.style.use("seaborn")

    def animate(i):
        for bar in ax.containers:
            bar.remove()
        ax.bar(np.arange(12), chroma_cq[:, i], color=palette, width=1)

    anim = animation.FuncAnimation(fig, animate, frames=np.shape(C)[1])
    anim.save(save_path, fps=int(np.shape(C)[1]/(len(y)/sr)), extra_args=['-vcodec', 'libx264'])

    plt.show()