#!/usr/bin/env python
# coding: utf-8

# # Exemple de bouton

#%% In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = plt.plot(t, s, lw=2)


class Index:
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

callback = Index()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)

# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.set_xlabel('Time [s]')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()


# # Imports de module

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# In[32]:


from typing import List

def dessiner_nuage(x:List[float], y:List[float], point_moyen:bool=True, a:float=0,b:float=0, path:str='figure.png')->None:
    
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))   
    
    #conversion en ndarray
    x = np.array(x)
    y = np.array(y)
    
    #dessin du nuage
    nuage, = ax.plot(x, y, ls='', marker='o', markersize=12)
    
    #fonction affine d'ajustement
    def f(x, a, b):
        return a * x + b

    droite, = ax.plot(x, f(x, 1, 1) , ls='', marker='x', markersize=10)

    xmin, xmax = min(x), max(x)
    ymin, ymax  = min(y), max(y)
    
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    axcolor = 'lightgoldenrodyellow'
    ax.margins(x=0)
    
    # Slider_horizontal
    
    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    a_slider = Slider(
        ax=ax_a,
        label="Coefficient directeur",
        valmin=-30,
        valmax=30,
        valinit=0,
    )

    # Make a vertically oriented slider to control the amplitude
    ax_b = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
    b_slider = Slider(
        ax=ax_b,
        label="Ordonnée à l'origine",
        valmin=-2 * ymin,
        valmax=2 * ymax,
        valinit=0,
        orientation="vertical"
    )


    # The function to be called anytime a slider's value changes
    def update(val):
        droite.set_ydata(f(x, a_slider.val, b_slider.val))
        fig.canvas.draw_idle()


    # register the update function with each slider
    a_slider.on_changed(update)
    b_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        a_slider.reset()
        b_slider.reset()
        
    button.on_clicked(reset)
    
    plt.show()
    fig.savefig(path)


# In[34]:


x = [k for k in range(1, 6)]
y = [10, 9.1, 8.5, 7.6, 6.8]
dessiner_nuage(x, y)


# In[ ]:




