# creating a neocortical nerve fiber with epileptic activity pattern

# imports
import signal

from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import seaborn as sns
# load run controls
h.load_file('stdrun.hoc')

# MODEL SPECIFICATION
# time params
h.tstop = 10000 # [ms]: simulation time
h.dt = 0.01 # [ms]: timestep, usually 0.001

# cell params (human layer 1 neocortical neuron)
n_nodes = 51 # []: (int) number of sections, enough to neglect end effects
nseg = 1 # []: (int)
Vo = -65 # [mV]: Vm @ rest for initializing membrane potential at start of simulation, seems accurate for layer 1 neurons
D = 12 # [um]: fiber diameter
inl = 100*D # [um]: internodal length
rhoa = 100 # [Ohm]: axoplasmic/axial resistivity
cm = 1 # [uF/cm**2]
L = 1 # [um], nodal length
g = 27.66e-3 # [S/cm**2]: Passive conductance in S/cm2

# material params
sigma_e = 0.4e-3 # [S/mm]: extracellular medium resistivity, here of white and gray matter

# stim params for intracellular electrode that simulates epileptic activity
epilep_delay = 0 # [ms]: start time of stim
epilep_dur = 3000 # [ms]: start time of stim
epilep_amp = 50 # [mA]: amplitude of (intracellular) stim object (negative cathodic, positive anodic), chosen amplitude
# that def. triggers AP, determined via XX TODO: explain whre we got that value from
# we want to have a rectangular stimulation pattern of a certain frequency:
epilep_f = 110*(10**-3)# [Hz * 10**-3 = ms^-1]: frequency of intracell. stim, set to 110 since firing rates above 100Hz
# indicate epileptic activity in a neuron
t = np.arange(0, h.tstop, h.dt)
epilep_stim_wave = epilep_amp/2 * signal.square(2 * np.pi * epilep_f * t, duty=0.5) + epilep_amp/2  # rectangular wave
epilep_stim_wave_vec = h.Vector(epilep_stim_wave)  # creating a vector with a rect. stim wave



# MODEL INITIALIZATION
# define nodes for cell
nodes = [h.Section(name=f'node[{i}]') for i in range(n_nodes)]

# insert extracellular/mechanisms part of the circuit
# connect the nodes
for node_ind, node in enumerate(nodes):
    node.nseg = nseg
    node.diam = 0.7*D  # using HH
    node.L = L
    node.Ra = rhoa*((L+inl)/L) # left this in here since it is a fn(*other params)
    node.cm = cm
    node.insert('hh')
    # node.insert('pas')
    node.insert('extracellular')

    for seg in node:
        # seg.pas.g = g
        # seg.pas.e = Vo  # setting this so there is no need to let membrane equilibriate for too long before stimulating
        seg.extracellular.e = 0  # extracellular voltages are 0

    if node_ind > 0:
        node.connect(nodes[node_ind-1](1))

# INSTRUMENTATION - STIMULATION/RECORDING
# create intracellular current stimulus
epilep_stim = h.IClamp(nodes[0](0))  # placing electrode at beginning of first node in fiber
epilep_stim.delay = epilep_delay
epilep_stim.dur = epilep_dur
# creating a hoc time vector for the vector play function
time_vec = h.Vector()
for step in np.arange(0, h.tstop, h.dt):
    time_vec.append(step)
epilep_stim_wave_vec.play(epilep_stim._ref_amp, time_vec, True)  # setting intracell. stimulation to rectangular wave

# stimulate and record
# set recording vectors
# create neuron "vector" for recording membrane potentials
epilep_vol_mem = h.Vector().record(nodes[25](0.5)._ref_v)  # record at middle of fiber
tvec = h.Vector().record(h._ref_t)
apc = h.APCount(nodes[0](0.5))
# run stimulation
h.finitialize(Vo)


# this is somewhat of a "hack" to change the default run procedure in HOC
#h(r"""
#proc advance() {
#    nrnpython("my_advance()")
#}""")

# run until tstop
h.continuerun(h.tstop)

# calculate the firing rate
firing_rate = apc.n/epilep_dur*1000
print("The firing rate is {} Hz.".format(firing_rate))

# DATA POST PROCESSING / OUTPUT
# plot the first 100 ms
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(tvec.as_numpy()[:10000], epilep_vol_mem.as_numpy()[:10000])
ax1.set(ylabel="membrane voltage (mV)",
        title="Transmembrane potential of an intracellularly activated \nnerve fiber simulating epileptic activity")
ax2.plot(t[:10000], epilep_stim_wave[:10000])
ax2.set(xlabel="time (ms)", ylabel="stimulus voltage (mV)")
plt.tight_layout()
plt.show()

# plot all
plt.plot(tvec, epilep_vol_mem)
plt.xlabel('time (ms)')
plt.ylabel('membrane voltage (mV)')
plt.title("Transmembrane potential of an intracellularly activated \nnerve fiber simulating epileptic activity")
plt.show()

# Plot the power spectrum
sf = 1000
win = 2 * sf
freqs, psd = signal.welch(epilep_vol_mem, sf, nperseg=win)

plt.loglog(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
# plt.ylim([0, psd.max() * 1.1])
plt.title("Power Spectrum")
plt.xlim([0, freqs.max()])

powerlaw = np.linspace(1, 100, 100)
plt.loglog(powerlaw, 1000*powerlaw ** -2, linestyle='--', label=f'f^-2')
plt.loglog(powerlaw, 10*powerlaw ** -4, linestyle='--', label=f'f^-4')
# exponents = np.array([-3, -5])
# for exponent in exponents:
#     y_powerlaw = 0.1*powerlaw ** exponent
#     plt.loglog(powerlaw, y_powerlaw, linestyle='--', label=f'f^{exponent}')
    
plt.legend()
plt.show()

print('============ DONE ============')
