# creating a neocortical nerve fiber with epileptic activity pattern

# imports
import signal

from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# load run controls
h.load_file('stdrun.hoc')

# MODEL SPECIFICATION
# time params
h.tstop = 10 # [ms]: simulation time
h.dt = 0.001 # [ms]: timestep, usually 0.001

# cell params
n_nodes = 51 # []: (int) number of sections, enough to neglect end effects
nseg = 1 # []: (int)
# TODO: fill in correct parameters
Vo = -65 # [mV]: Vm @ rest for initializing membrane potential at start of simulation
D =  # [um]: fiber diameter
inl = 100*D # [um]: internodal length
rhoa =  # [Ohm]: axoplasmic/axial resistivity
cm =  # [uF/cm**2]
L =  # [um], nodal length
g =  # [S/cm**2]: Passive conductance in S/cm2

# material params
sigma_e = 2e-4 # [S/mm]: extracellular medium resistivity # TODO: update this with general brain tissue resistivity

# stim params for intracellular electrode that simulates epileptic activity
epilep_delay = 0 # [ms]: start time of stim
epilep_dur = h.tstop # [ms]: start time of stim
epilep_amp = 50 # [mA]: amplitude of (intracellular) stim object (negative cathodic, positive anodic), chosen amplitude
# that def. triggers AP, determined via XX TODO: explain whre we got that value from
# we want to have a rectangular stimulation pattern of a certain frequency:
epilep_f =  110*(10**-3)# [Hz * 10**-3 = ms^-1]: frequency of intracell. stim, set to 110 since firing rates above 100Hz
# indicate epileptic activity in a neuron TODO: insert citation
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
    node.insert('extracellular')

    for seg in node:
        seg.pas.g = g
        seg.pas.e = Vo  # setting this so there is no need to let membrane equilibriate for too long before stimulating
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

# run stimulation
h.finitialize(Vo)

# this is somewhat of a "hack" to change the default run procedure in HOC
h(r"""
proc advance() {
    nrnpython("my_advance()")
}""")

# run until tstop
h.continuerun(h.tstop)


# DATA POST PROCESSING / OUTPUT
# plot things
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1,1,1)
ax.plot(tvec, epilep_vol_mem)
ax.set(xlabel="time (ms)", ylabel="membrane voltage (mV)",
       title="Transmembrane potential of an intracellularly activated \nnerve fiber simulating epileptic activity")
plt.tight_layout()
plt.show()

print('============ DONE ============')