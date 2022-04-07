import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button



# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=0,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=0,
    orientation="vertical"
)
axamp = plt.axes([0.8, 0.25, 0.0225, 0.63])
p_slider = Slider(
    ax=axamp,
    label="l",
    valmin=0,
    valmax=10,
    valinit=0,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    print(val)


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.


plt.show()