# creating a neocortical nerve fiber with epileptic activity pattern and 5 extracellular electrodes using Precisis' 
# stimulus parameters

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
h.tstop = 500 # [ms]: simulation time
h.dt = 0.01 # [ms]: timestep, usually 0.001

# cell params (human layer 1 neocortical neuron)
n_nodes = 51 # []: (int) number of sections, enough to neglect end effects
nseg = 11 # []: (int)
Vo = -65 # [mV]: Vm @ rest for initializing membrane potential at start of simulation, seems accurate for layer 1 neurons
D = 12 # [um]: fiber diameter
inl = 100*D # [um]: internodal length
rhoa = 100 # [Ohm]: axoplasmic/axial resistivity
cm = 1 # [uF/cm**2]
L = 1 # [um], nodal length
g = 27.66e-3 # [S/cm**2]: Passive conductance in S/cm2

# material params
sigma_e = 9.234e-5 # [S/mm]: extracellular medium resistivity, here of white and gray matter

# stim params for intracellular electrode that simulates epileptic activity
epilep_delay = 0 # [ms]: start time of stim
epilep_dur = h.tstop # [ms]: duration of stim
epilep_amp = 0.2 # [mA]: amplitude of (intracellular) stim object (negative cathodic, positive anodic), chosen amplitude
# that def. triggers AP, determined via experiments
# we want to have a rectangular stimulation pattern of a certain frequency:
epilep_f = 50*(10**-3)# [Hz * 10**-3 = ms^-1]: frequency of intracell. stim, firing rates above 100Hz
# TODO: epilepsy_f needs to be refined to account for exact diseases Precisis is targeting 
# indicate epileptic activity in a neuron
t = np.arange(0, h.tstop+h.dt, h.dt)
epilep_stim_wave = epilep_amp/2 * signal.square(2 * np.pi * epilep_f * t, duty=0.05) + epilep_amp/2 # rectangular wave
epilep_stim_wave_vec = h.Vector(epilep_stim_wave)  # creating a vector with a rect. stim wave

# stim parameters for extracellular stimulation
e2f_1 = 5.519 # [mm]: electrode 1 to fiber distance, skull thickness at temporal lobe plus cortex
e1_delay = 100  # [ms]: start time of stim

# LFS (low frequency stimulation mode): 20ms -2mA, then 100ms +0.4mA, then 5ms off
e1_dur = 500 # [ms]: duration of stim, let's say 1min at a time
e1_amp_neg = 2 # [mA]: amplitude of stim object (negative cathodic, positive anodic)
e1_amp_pos = 0.4 # [mA]: amplitude of stim object (negative cathodic, positive anodic)
e1_f = 8*(10**-3)# [Hz * 10**-3 = ms^-1]: 8Hz low frquency stim (with 125ms length, 8 pulses per second)
e1_stim_wave_neg = -e1_amp_neg/2 * signal.square(2*np.pi*0.008 * (t+150), duty=20/125) - e1_amp_neg/2 # negative portion
e1_stim_wave_pos = e1_amp_pos/2 * signal.square(2*np.pi*0.008 * (t+5), duty=100/125) + e1_amp_pos/2 # positive portion


e1_stim_wave = e1_stim_wave_neg+e1_stim_wave_pos
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
# epilep_stim.delay = e1_pw * 1000  # [ms], added delay here to see if hyperpolarization before intracellular stimulation 
epilep_stim.delay = epilep_delay
# changes activation pattern # TODO: play with this delay to investigate effect on firing rate 
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
epilep_vol_mem = h.Vector().record(nodes[n_nodes-1](0.5)._ref_v)  # record at end of fiber
tvec = h.Vector().record(h._ref_t)
apc = h.APCount(nodes[n_nodes-1](0.5))  # looking at AP generation at end of fiber

# compute extracellular potentials from point current source (call this from my_advance to update at each timestep)
def update_field():
    phi_e = []
    for node_ind, node in enumerate(nodes):
        # x_loc_c = 1e-3 * (-(n_nodes-1)/2*inl + inl*node_ind)
        # r_c = np.sqrt(x_loc_c ** 2 + e2f_1 ** 2)
        # node(0.5).e_extracellular = electrode.i//(4*sigma_e*np.pi*r_c)
        
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

h.continuerun(e1_delay)  # Run until the time of extracellular stimulation

# Count action potentials before stimulation
n_spikes_before = apc.n

# Run the simulation until the end
h.continuerun(h.tstop)

# Count action potentials after stimulation
n_spikes_after = apc.n - n_spikes_before  # Subtract the count before stimulation

# Calculate firing rates before and after stimulation
firing_rate_before = n_spikes_before / (e1_delay * 0.001)  # Convert ms to seconds
firing_rate_after = n_spikes_after / ((h.tstop - e1_delay) * 0.001) 

# firing_rate = apc.n/h.tstop * 1000  # note: this is calculated over h.tstop time - but there are temporal differences 
# in effect of extracell stim on fiber potential!! # TODO: investigate effect of different delays on firing rate - 
#  TODO: take into account that this is a mean and there are temporal differences"
print("The original firing rate is {} Hz.".format(firing_rate_before))
print("The firing rate after stimulation is {} Hz.".format(firing_rate_after))


# Reconstruct stim for plotting (add delay)
epilep_stim_wave_plot = epilep_stim_wave
epilep_stim_wave_plot[int(epilep_dur/h.dt):-1] = 0
e1_stim_wave_plot = e1_stim_wave
e1_stim_wave_plot[int(e1_dur/h.dt):-1] = 0
if epilep_delay > 0:
    epilep_stim_wave_plot[0:int(epilep_delay/h.dt)] = 0
if e1_delay > 0:
    e1_stim_wave_plot[0:int(e1_delay/h.dt)] = 0


# DATA POST PROCESSING / OUTPUT
# plot the first 100 ms
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
ax1.plot(tvec.as_numpy(), epilep_vol_mem.as_numpy())
ax1.set(ylabel="membrane \nvoltage (mV)",
        title="Transmembrane potential of an extracellularly \nactivated epileptic nerve fiber")
ax2.plot(t, epilep_stim_wave_plot)
ax2.set(xlabel="time (ms)", ylabel="intracell. stimulus \nvoltage (mV)")
ax3.plot(tvec.as_numpy(), e1_stim_wave_plot)
ax3.set(xlabel="time (ms)", ylabel="extracell. stimulus \nvoltage (mV)")
plt.tight_layout()
plt.show()

print('============ DONE ============')
