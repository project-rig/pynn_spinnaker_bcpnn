import numpy, pylab, random, sys

import spynnaker.pyNN as sim
import bcpnn

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

#-------------------------------------------------------------------
# General Parameters
#-------------------------------------------------------------------
# Population parameters
model = sim.IF_curr_exp
cell_params = {
    'cm' : 0.25, # nF
    'i_offset'  : 0.0,
    'tau_m'     : 10.0,
    'tau_refrac': 2.0,
    'tau_syn_E' : 2.5,
    'tau_syn_I' : 2.5,
    'v_reset'   : -70.0,
    'v_rest'    : -65.0,
    'v_thresh'  : -55.4
}


#-------------------------------------------------------------------
# Creation of neuron populations
#-------------------------------------------------------------------
def spike_vector_to_times(vector, timestep, delay):
    return list(numpy.multiply(numpy.where(vector != 0)[0], timestep) + delay)
        
# Neuron populations
pre_pop = sim.Population(1, model(**cell_params))
post_pop = sim.Population(1, model(**cell_params))

# Load reference spike trains
pre_spike_vector = numpy.load("pre_spikes_1.npy")
post_spike_vector = numpy.load("post_spikes_1.npy")

# Convert to time format
pre_spike_times = spike_vector_to_times(pre_spike_vector, 1.0, 1.0)
post_spike_times = spike_vector_to_times(post_spike_vector, 1.0, 0.0)

# From these calculate sim time
sim_time = max(max(pre_spike_times), max(post_spike_times))

# Stimulating populations
pre_stim = sim.Population(1, sim.SpikeSourceArray(spike_times=pre_spike_times))
post_stim = sim.Population(1, sim.SpikeSourceArray(spike_times=post_spike_times))

#-------------------------------------------------------------------
# Creation of connections
#-------------------------------------------------------------------
# Connection type between noise poisson generator and excitatory populations
ee_connector = sim.OneToOneConnector(weights=2, delays=1)

sim.Projection(pre_stim, pre_pop, ee_connector, target='excitatory')
sim.Projection(post_stim, post_pop, ee_connector, target='excitatory')

# Plastic Connections between pre_pop and post_pop
bcpnn_model = bcpnn.BCPNNMechanism(
    tau_zi=10.0,                  # ms
    tau_zj=10.0,                  # ms
    tau_eligibility=1000.0,       # ms
    max_firing_frequency=50.0,    # Hz
    phi=0.05,                     # nA
    w_max=1.0)                    # nA / uS for conductance

sim.Projection(pre_pop, post_pop, sim.OneToOneConnector(), 
    synapse_dynamics = sim.SynapseDynamics(slow=bcpnn_model)
)

# Record spikes
pre_pop.record()
post_pop.record()

# Run simulation
sim.run(sim_time)

def plot_spikes(spikes, axis, title):
  if spikes != None:
      axis.set_xlim([0, sim_time])
      axis.plot([i[1] for i in spikes], [i[0] for i in spikes], ".") 
      axis.set_xlabel('Time/ms')
      axis.set_ylabel('spikes')
      axis.set_title(title)
     
  else:
      print "No spikes received"

pre_spikes = pre_pop.getSpikes(compatible_output=True)
post_spikes = post_pop.getSpikes(compatible_output=True)

figure, axisArray = pylab.subplots(2)

plot_spikes(pre_spikes, axisArray[0], "Pre-synaptic neurons")
plot_spikes(post_spikes, axisArray[1], "Post-synaptic neurons")

pylab.show()


# End simulation on SpiNNaker
sim.end(stop_on_board=False)
