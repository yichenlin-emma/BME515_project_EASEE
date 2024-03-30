# creating a neocortical nerve fiber with epileptic activity pattern
# 1 extracellular electrode

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
epilep_dur = 3000 # [ms]: duration of stim
epilep_amp = 0.2 # [mA]: amplitude of (intracellular) stim object (negative cathodic, positive anodic), chosen amplitude
# that def. triggers AP, determined via experiments
# we want to have a rectangular stimulation pattern of a certain frequency:
epilep_f = 110*(10**-3)# [Hz * 10**-3 = ms^-1]: frequency of intracell. stim, set to 110 since firing rates above 100Hz
# indicate epileptic activity in a neuron
t = np.arange(0, h.tstop, h.dt)
epilep_stim_wave = epilep_amp/2 * signal.square(2 * np.pi * epilep_f * t, duty=0.5) + epilep_amp/2  # rectangular wave
epilep_stim_wave_vec = h.Vector(epilep_stim_wave)  # creating a vector with a rect. stim wave

# stim parameters for extracellular stimulation
e2f_1 = 6.25# [mm]: electrode 1 to fiber distance, skull thickness at temporal lobe plus cortex
e1_delay = 0  # [ms]: start time of stim
e1_dur = 500 # [ms]: duration of stim, 0.5 sec for high frequency stimulation
e1_amp = 4 # [mA]: amplitude of stim object (negative cathodic, positive anodic)
e1_f = 100*(10**-3)# [Hz * 10**-3 = ms^-1]: 100Hz high frquency stim
e1_pw = 0.16 # [ms]: total rectangular pulse width
duty_cyc_1 = 0.5*e1_pw / (1/e1_f - 0.5*e1_pw) # duty cycle of half rect. pulse
e1_stim_wave_pos = e1_amp/2 * signal.square(2 * np.pi * e1_f * t, duty=duty_cyc_1) + e1_amp/2 # positive portion
e1_stim_wave_neg = -e1_amp/2 * signal.square(2 * np.pi * e1_f * (t - 0.5*e1_pw), duty=duty_cyc_1) - e1_amp/2 # negative portion
e1_stim_wave = e1_stim_wave_pos + e1_stim_wave_neg
e1_stim_wave_vec = h.Vector(e1_stim_wave)  # creating a vector with a rect. stim wave

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

# create extracellular current stimulus
dummy = h.Section(name='dummy')
electrode = h.IClamp(dummy(0.5)) # puts the stim halfway along the length
electrode.delay = e1_delay
electrode.dur = e1_dur
e1_stim_wave_vec.play(electrode._ref_amp, time_vec, True)  # setting extracell. stimulation to rectangular wave
# we have 5 of these electrodes. but rather than creating 5 different stimulus objects, we can just use the same one and
# calculate the field based on if this electrode was shifted

# STIMULATE AND RECORD
# set recording vectors
# create neuron "vector" for recording membrane potentials
epilep_vol_mem = h.Vector().record(nodes[25](0.5)._ref_v)  # record at middle of fiber
tvec = h.Vector().record(h._ref_t)
apc = h.APCount(nodes[0](0.5))

# compute extracellular potentials from point current source (call this from my_advance to update at each timestep)
def update_field():
    phi_e = []
    for node_ind, node in enumerate(nodes):
        # x distances
        x_loc_l = 1e-3 * (-(n_nodes-1)/2*inl + inl*node_ind - 15) # 1e-3 [um] -> [mm], for left electrodes
        x_loc_r = 1e-3 * (-(n_nodes-1)/2*inl + inl*node_ind + 15)  # 1e-3 [um] -> [mm], for right electrodes
        x_loc_c = 1e-3 * (-(n_nodes-1)/2*inl + inl*node_ind)  # 1e-3 [um] -> [mm], for center electrode
        # y distance is 15mm for left and right, and 0mm for central
        # radii
        r_l = np.sqrt(x_loc_l ** 2 + 15 ** 2 + e2f_1 ** 2)  # [mm], same for both front and back left electrodes
        r_r = np.sqrt(x_loc_r ** 2 + 15 ** 2 + e2f_1 ** 2)  # [mm], same for both front and back right electrodes
        r_c = np.sqrt(x_loc_c ** 2 + e2f_1 ** 2)  # [mm], y=0 here

        # e-field for 5 electrodes positioned around fiber
        node(0.5).e_extracellular = (2*electrode.i)/(4*sigma_e*np.pi*r_l) + (2*electrode.i)/(4*sigma_e*np.pi*r_r) + \
                                    (electrode.i)/(4*sigma_e*np.pi*r_c)


# time integrate with constant time step - this just defines method, called by proc advance() below
def my_advance():
    update_field()
    h.fadvance()

# RUN
# run stimulation
h.finitialize(Vo)

# this is somewhat of a "hack" to change the default run procedure in HOC
h(r"""
proc advance() {
    nrnpython("my_advance()")
}""")

# run until tstop
h.continuerun(h.tstop)

# calculate the firing rate
firing_rate = apc.n/epilep_dur*1000
print("The firing rate is {} Hz.".format(firing_rate))


# DATA POST PROCESSING / OUTPUT
# plot the first 100 ms
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
ax1.plot(tvec.as_numpy()[:10000], epilep_vol_mem.as_numpy()[:10000])
ax1.set(ylabel="membrane \nvoltage (mV)",
        title="Transmembrane potential of an extracellularly \nactivated epileptic nerve fiber")
ax2.plot(t[:10000], epilep_stim_wave[:10000])
ax2.set(xlabel="time (ms)", ylabel="intracell. stimulus \nvoltage (mV)")
ax3.plot(t[:10000], e1_stim_wave[:10000])
ax3.set(xlabel="time (ms)", ylabel="extracell. stimulus \nvoltage (mV)")
plt.tight_layout()
plt.show()

# plot all
plt.plot(tvec, epilep_vol_mem)
plt.xlabel('time (ms)')
plt.ylabel('membrane voltage (mV)')
plt.title("Transmembrane potential of an extracellularly \nactivated epileptic nerve fiber")
plt.show()

print('============ DONE ============')
