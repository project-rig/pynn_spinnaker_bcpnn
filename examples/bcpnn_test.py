import logging, numpy, pylab, random, sys, copy

import pynn_spinnaker as sim
import pynn_spinnaker_bcpnn as bcpnn

logger = logging.getLogger("pynn_spinnaker")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# SpiNNaker setup
sim.setup(timestep=1.0, max_delay=8.0,
          spinnaker_hostname="192.168.1.1",
          stop_on_spinnaker=True)

#-------------------------------------------------------------------
# General Parameters
#-------------------------------------------------------------------
# Population parameters
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

bcpnn_cell_params = copy.deepcopy(cell_params)
bcpnn_cell_params.update({
    "tau_z": 10.0,       # ms
    "tau_p": 1000.0,      # ms
    "f_max": 50.0,        # Hz
    "phi": 1.0,          # nA
    "bias_enabled": False,
    "plasticity_enabled": True,
})

#-------------------------------------------------------------------
# Creation of neuron populations
#-------------------------------------------------------------------
def spike_vector_to_times(vector, timestep, delay):
    return numpy.multiply(numpy.where(vector != 0)[0], timestep) + delay
        
# Neuron populations
pre_pop = sim.Population(1, sim.IF_curr_exp(**cell_params))
post_pop = sim.Population(1, bcpnn.IF_curr_exp(**bcpnn_cell_params))

pre_pop.spinnaker_config.flush_time = 100.0

post_pop.record("bias")

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
sim.Projection(pre_stim, pre_pop, sim.OneToOneConnector(),
               sim.StaticSynapse(weight=2.0),
               receptor_type="excitatory")
sim.Projection(post_stim, post_pop, sim.OneToOneConnector(),
               sim.StaticSynapse(weight=2.0),
               receptor_type="excitatory")

# Plastic Connections between pre_pop and post_pop
bcpnn_synapse = bcpnn.BCPNNSynapse(
    tau_zi=10.0,   # ms
    tau_zj=10.0,   # ms
    tau_p=1000.0,  # ms
    f_max=50.0,    # Hz
    w_max=1.0,     # nA / uS for conductance
    weights_enabled=False,
    plasticity_enabled=True)

proj = sim.Projection(pre_pop, post_pop, sim.OneToOneConnector(), bcpnn_synapse)

# Run simulation
# **TODO** run for 10ms, get weights, rinse and repeat
sim.run(sim_time)

post_data = post_pop.get_data().segments[0].filter(name="bias")[0]
weight_data = proj.get("weight", format="list", with_address=False)
print "Weight:", weight_data


figure, axis = pylab.subplots()
axis.plot(numpy.arange(sim_time), post_data)
pylab.show()

# End simulation
sim.end()
