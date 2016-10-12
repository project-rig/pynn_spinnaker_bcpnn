# Import modules
import itertools
import logging
import math
import numpy
import pickle
import pylab
import random

# Import classes
from pyNN.random import NumpyRNG, RandomDistribution

# Import simulator
import pynn_spinnaker as sim
import pynn_spinnaker_bcpnn as bcpnn

#------------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------------
downscale           = 200      # scale number of neurons down by this factor
                      # scale synaptic weights up by this factor to
                      # obtain similar dynamics independent of size
order               = 50000  # determines size of network:
                      # 4*order excitatory neurons
                      # 1*order inhibitory neurons
epsilon             = 0.1     # connectivity: proportion of neurons each neuron projects to
    
# Parameters determining model dynamics, cf Brunel (2000), Figs 7, 8 and Table 1
eta                 = 1.3
g                   = 5.0

J                   = 0.1     # synaptic weight [mV]
delay               = 1.0     # synaptic delay, all connections [ms]

# single neuron parameters
tauMem              = 20.0    # neuron membrane time constant [ms]
tau_syn_ampa_gaba   = 5.0     # synaptic time constant [ms]
tau_syn_nmda        = 150.0
tauRef              = 2.0     # refractory time [ms]
U0                  = -70.0     # resting potential [mV]
theta               = -50.0    # threshold

# simulation-related parameters  
dt                  = 1.0     # simulation step length [ms]

#------------------------------------------------------------------------------
# Calculate derived parameters
#------------------------------------------------------------------------------
# scaling: compute effective order and synaptic strength
order_eff = int(float(order)/downscale)
J_eff     = J*downscale
  
# compute neuron numbers
NE = int(4 * order_eff)  # number of excitatory neurons
NI = int(1 * order_eff)  # number of inhibitory neurons

# compute synapse numbers
CE = int(epsilon * NE)  # number of excitatory synapses on neuron
CI = int(epsilon * NI)  # number of inhibitory synapses on neuron
Cext = CE               # number of external synapses on neuron  

# synaptic weights, scaled for alpha functions, such that
# for constant membrane potential, charge J would be deposited
# **NOTE** multiply by 250 to account for larger membrane capacitance
fudge = 0.00041363506632638 * 250.0 # ensures dV = J at V=0

# Fixed excitatory and inhibitory weights
JE = (J_eff / tau_syn_ampa_gaba) * fudge
JI = -g * JE
logger = logging.getLogger("pynn_spinnaker")

# threshold, external, and Poisson generator rates:
nu_thresh = (theta - U0) / (J_eff * CE * tauMem)
nu_ext    = eta * nu_thresh     # external rate per synapse
p_rate    = 1000 * nu_ext * Cext  # external input rate per neuron (Hz)

# put cell parameters into a dict
cell_params = {"tau_m"      : tauMem,
               "tau_syn_E"  : tau_syn_ampa_gaba,
               "tau_syn_I"  : tau_syn_ampa_gaba,
               "tau_refrac" : tauRef,
               "v_rest"     : U0,
               "v_reset"    : U0,
               "v_thresh"   : theta,
               "cm"         : 0.25}     # (nF)

#-------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------
# Convert weights in format returned by getWeights into a connection list
def convert_weights_to_list(filename, delay, weight_scale=1.0):
    # Load dense matrix
    matrix = numpy.load(filename)

    def build_list(indices):
        # Extract weights from matrix using indices
        weights = matrix[indices]

        # Scale weights
        weights = numpy.multiply(weights, weight_scale)

        # Build numpy array of delays
        delays = numpy.repeat(delay, len(weights))

        # Zip x-y coordinates of non-zero weights with weights and delays
        return zip(indices[0], indices[1], weights, delays)

    # Get indices of non-nan i.e. connected weights
    connected_indices = numpy.where(~numpy.isnan(matrix))

    # Return connection lists
    return build_list(connected_indices)

# Generate poisson noise of given rate between start and stop times
def poisson_generator(rate, t_start, t_stop):
    n = (t_stop - t_start) / 1000.0 * rate
    number = numpy.ceil(n + 3 * numpy.sqrt(n))
    if number < 100:
        number = min(5 + numpy.ceil(2 * n),100)

    if number > 0:
        isi = numpy.random.exponential(1.0/rate, number)*1000.0
        if number > 1:
            spikes = numpy.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = numpy.array([])

    spikes += t_start
    i = numpy.searchsorted(spikes, t_stop)

    extra_spikes = []
    if len(spikes) == i:
        # ISI buf overrun

        t_last = spikes[-1] + numpy.random.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += numpy.random.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = numpy.concatenate((spikes, extra_spikes))
    else:
        spikes = numpy.resize(spikes,(i,))

    # Return spike times, rounded to millisecond boundaries
    return [round(x) for x in spikes]

def generate_discrete_hcu_stimuli(stim_minicolumns, num_mcu_neurons):
    # Calculate number of minicolumn in each hypercolumn
    num_mcu_per_hcu = NE // num_mcu_neurons

    # Loop through minicolumn indices to stimulate
    spike_times = [[] for _ in range(NE)]
    for m, start_time, frequency, duration in stim_minicolumns:
        logger.debug("Stimulating minicolumn %u at %f Hz for %f ms from %f ms" % (m, frequency, duration, start_time))
        # Loop through neurons in minicolumn and add a block of noise to their spike times
        for n in range(m, NE, num_mcu_per_hcu):
            spike_times[n].extend(poisson_generator(frequency, start_time, start_time + duration))
    return spike_times

def euclidean_hcu_delay(i_pre, i_post, grid_size, distance_scale, velocity):
    # Convert HCU indices into coordinate
    x_pre = i_pre % grid_size
    y_pre = i_pre // grid_size
    x_post = i_post % grid_size
    y_post = i_post // grid_size

    # Calculate euclidian distance
    distance = math.sqrt((x_post - x_pre) ** 2 + (y_post - y_pre) ** 2)

    # Calculate delay from this
    return int(round(((distance_scale * distance) / velocity) + 1.0))

def constant_hcu_delay(i_pre, i_post, delay):
    return delay

#-------------------------------------------------------------------
# HCU
#-------------------------------------------------------------------
class HCU(object):
    def __init__(self, name, sim, rng,
                 e_cell_model, i_cell_model,
                 e_cell_params, i_cell_params,
                 e_cell_flush_time, e_cell_mean_firing_rate,
                 stim_spike_times, wta, background_weight, stim_weight, simtime,
                 record_bias, record_spikes, record_membrane):

        # Cache recording flags
        self.record_bias = record_bias
        self.record_spikes = record_spikes
        self.record_membrane = record_membrane
        self.wta = wta

        logger.info("Creating HCU:%s" % name)

        logger.debug("Membrane potentials uniformly distributed between %g mV and %g mV." % (-80, U0))
        membrane_voltage_distribution = RandomDistribution("uniform", low=-80.0, high=U0, rng=rng)

        logger.debug("Background noise rate %g Hz." % (p_rate))

        logger.debug("Creating excitatory population with %d neurons." % (NE))
        self.e_cells = sim.Population(NE, e_cell_model(**e_cell_params),
                                      label="%s - e_cells" % name)
        self.e_cells.initialize(v=membrane_voltage_distribution)

        # Set e cell mean firing rate
        self.e_cells.spinnaker_config.mean_firing_rate = e_cell_mean_firing_rate

        # **HACK** issue #28 means plastic version needs clustering hack
        self.e_cells.spinnaker_config.max_neurons_per_core = 256

        # **HACK** issue #18 means that we end up with 1024 wide clusters
        # which needs a lot of 256-wide neuron and synapse cores
        self.e_cells.spinnaker_config.max_cluster_width = 512

        # Set flush time
        self.e_cells.spinnaker_config.flush_time = e_cell_flush_time

        # **YUCK** record spikes actually entirely ignores
        # sampling interval but throws exception if it is not set
        if self.record_spikes:
            self.e_cells.record("spikes", sampling_interval=100.0)

        if self.record_bias:
            self.e_cells.record("bias", sampling_interval=100.0)

        if self.record_membrane:
            self.e_cells.record("v", sampling_interval=100.0)

        e_poisson = sim.Population(NE, sim.SpikeSourcePoisson(rate=p_rate, duration=simtime),
                                   label="%s - e_poisson" % name)

        logger.debug("Creating background->E AMPA connection weight %g nA." % (background_weight))
        sim.Projection(e_poisson, self.e_cells,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=background_weight, delay=delay),
                       receptor_type="excitatory")

        if self.wta:
            logger.debug("Creating inhibitory population with %d neurons." % (NI))
            self.i_cells = sim.Population(NI, i_cell_model, i_cell_params,
                                          label="%s - i_cells" % name)
            self.i_cells.initialize(v=membrane_voltage_distribution)

            # Inhibitory cells generally fire at a low rate
            self.i_cells.spinnaker_config.mean_firing_rate = 5.0

            if self.record_spikes:
                self.i_cells.record("spikes")

            i_poisson = sim.Population(NI, sim.SpikeSourcePoisson(rate=p_rate, duration=simtime),
                                       label="%s - i_poisson" % name)

            logger.debug("Creating I->E GABA connection with connection probability %g, weight %g nA and delay %g ms." % (epsilon, JI, delay))
            I_to_E = sim.Projection(self.i_cells, self.e_cells,
                                    sim.FixedProbabilityConnector(p_connect=epsilon, rng=rng),
                                    sim.StaticSynapse(weight=JI, delay=delay),
                                    receptor_type="inhibitory")

            logger.debug("Creating E->I AMPA connection with connection probability %g, weight %g nA and delay %g ms." % (epsilon, JE, delay))
            sim.Projection(self.e_cells, self.i_cells,
                           sim.FixedProbabilityConnector(p_connect=epsilon, rng=rng),
                           sim.StaticSynapse(weight=JE, delay=delay),
                           receptor_type="excitatory")

            logger.debug("Creating I->I GABA connection with connection probability %g, weight %g nA and delay %g ms." % (epsilon, JI, delay))
            sim.Projection(self.i_cells, self.i_cells,
                           sim.FixedProbabilityConnector(p_connect=epsilon, rng=rng),
                           sim.StaticSynapse(weight=JI, delay=delay),
                           receptor_type="inhibitory")

            logger.debug("Creating background->I AMPA connection weight %g nA." % (background_weight))
            sim.Projection(i_poisson, self.i_cells,
                           sim.OneToOneConnector(),
                           sim.StaticSynapse(weight=background_weight, delay=delay),
                           receptor_type="excitatory")

        # Create a spike source capable of stimulating entirely excitatory population
        stim_spike_source = sim.Population(NE, sim.SpikeSourceArray(spike_times=stim_spike_times))

        # Connect one-to-one to excitatory neurons
        sim.Projection(stim_spike_source, self.e_cells,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=stim_weight, delay=delay),
                       receptor_type="excitatory")


    #-------------------------------------------------------------------
    # Public methods
    #-------------------------------------------------------------------
    def read_results(self):
        results = ()

        if self.record_spikes:
            e_spikes_writer = lambda filename: self.e_cells.write_data(filename)
            results += (e_spikes_writer,)

            if self.wta:
                i_spikes_writer = lambda filename: self.i_cells.write_data(filename)
                results += (i_spikes_writer,)

        return results

    #-------------------------------------------------------------------
    # Class methods
    #-------------------------------------------------------------------
    # Create an HCU suitable for testing:
    # Uses adaptive neuron model and doesn't record biases
    @classmethod
    def testing_adaptive(cls, name, sim, rng,
                         bias, tau_ca2, i_alpha,
                         e_cell_mean_firing_rate,
                         simtime, stim_spike_times,
                         record_membrane):

        # Copy base cell parameters
        e_cell_params = cell_params.copy()
        e_cell_params["tau_syn_E2"] = tau_syn_nmda
        e_cell_params["tau_ca2"] = tau_ca2
        e_cell_params["i_alpha"] = i_alpha
        e_cell_params["i_offset"] = bias
        e_cell_params["bias_enabled"] = False
        e_cell_params["plasticity_enabled"] = False
        assert len(stim_spike_times) == NE

        # Build HCU
        return cls(name=name, sim=sim, rng=rng,
                   e_cell_model=bcpnn.IF_curr_ca2_adaptive_dual_exp, i_cell_model=sim.IF_curr_exp,
                   e_cell_params=e_cell_params, i_cell_params=cell_params,
                   e_cell_flush_time=None, e_cell_mean_firing_rate=e_cell_mean_firing_rate,
                   stim_spike_times=stim_spike_times, wta=True, background_weight=JE,
                   stim_weight=4.0, simtime=simtime, record_bias=False,
                   record_spikes=True, record_membrane=record_membrane)

    # Create an HCU suitable for training
    # Uses a non-adaptive neuron model and records biaseses
    @classmethod
    def training(cls, name, sim, rng, intrinsic_tau_z, intrinsic_tau_p,
                 simtime, e_cell_mean_firing_rate, stim_spike_times):
        # Copy base cell parameters
        e_cell_params = cell_params.copy()
        e_cell_params["tau_syn_E2"] = tau_syn_nmda
        e_cell_params["phi"] = 0.05
        e_cell_params["f_max"] = 20.0
        e_cell_params["tau_z"] = intrinsic_tau_z
        e_cell_params["tau_p"] = intrinsic_tau_p
        e_cell_params["bias_enabled"] = False
        e_cell_params["plasticity_enabled"] = True
        assert len(stim_spike_times) == NE
        
        # Build HCU
        return cls(name=name, sim=sim, rng=rng,
                   e_cell_model=bcpnn.IF_curr_dual_exp, i_cell_model=sim.IF_curr_exp,
                   e_cell_params=e_cell_params, i_cell_params=cell_params,
                   e_cell_flush_time=500.0, e_cell_mean_firing_rate=e_cell_mean_firing_rate,
                   stim_spike_times=stim_spike_times, wta=False, background_weight=0.2,
                   stim_weight=2.0, simtime=simtime, record_bias=True,
                   record_spikes=True, record_membrane=False)

#------------------------------------------------------------------------------
# HCUConnection
#------------------------------------------------------------------------------
class HCUConnection(object):
    def __init__(self, sim,
                 pre_hcu, post_hcu,
                 ampa_connector, nmda_connector,
                 ampa_synapse, nmda_synapse,
                 record_ampa, record_nmda):

        self.record_ampa = record_ampa
        self.record_nmda = record_nmda

        # Create connection
        self.ampa_connection = sim.Projection(pre_hcu.e_cells, post_hcu.e_cells,
                                              ampa_connector, ampa_synapse,
                                              receptor_type="excitatory",
                                              label="%s->%s (AMPA)" % (pre_hcu.e_cells.label, post_hcu.e_cells.label))

        self.nmda_connection = sim.Projection(pre_hcu.e_cells, post_hcu.e_cells,
                                              nmda_connector, nmda_synapse,
                                              receptor_type="excitatory2",
                                              label="%s->%s (NMDA)" % (pre_hcu.e_cells.label, post_hcu.e_cells.label))

    #-------------------------------------------------------------------
    # Public methods
    #-------------------------------------------------------------------
    def read_results(self):
        results = ()
        if self.record_ampa:
            ampa_writer = lambda filename: numpy.save(filename, self.ampa_connection.get("weight", format="array"))
            results += (ampa_writer,)

        if self.record_nmda:
            nmda_writer = lambda filename: numpy.save(filename, self.nmda_connection.get("weight", format="array"))
            results += (nmda_writer,)

        return results


    #-------------------------------------------------------------------
    # Class methods
    #-------------------------------------------------------------------
    # Creates an HCU connection for training
    @classmethod
    def training(cls, sim,
                 pre_hcu, post_hcu,
                 ampa_synapse, nmda_synapse, rng):
        # Build connector
        ampa_connector = sim.FixedProbabilityConnector(p_connect=epsilon, rng=rng)
        nmda_connector = sim.FixedProbabilityConnector(p_connect=epsilon, rng=rng)

        return cls(sim=sim,
                   pre_hcu=pre_hcu, post_hcu=post_hcu,
                   ampa_connector=ampa_connector, nmda_connector=nmda_connector,
                   ampa_synapse=ampa_synapse, nmda_synapse=nmda_synapse,
                   record_ampa=True, record_nmda=True)

    # Creates an HCU connection for testing:
    # AMPA and NMDA connectivity, reconstructed from matrices
    @classmethod
    def testing(cls, sim,
                pre_hcu, post_hcu,
                ampa_gain, nmda_gain,
                ampa_synapse, nmda_synapse,
                connection_weight_filename, delay):
        # Build connectors
        ampa_connector = sim.FromListConnector(convert_weights_to_list(connection_weight_filename[0], delay, ampa_gain))
        nmda_connector = sim.FromListConnector(convert_weights_to_list(connection_weight_filename[1], delay, nmda_gain))

        return cls(sim=sim,
                   pre_hcu=pre_hcu, post_hcu=post_hcu,
                   ampa_connector=ampa_connector, nmda_connector=nmda_connector,
                   ampa_synapse=ampa_synapse, nmda_synapse=nmda_synapse,
                   record_ampa=False, record_nmda=False)

#------------------------------------------------------------------------------
# Train
#------------------------------------------------------------------------------
def train_discrete(ampa_tau_zi, ampa_tau_zj, nmda_tau_zi, nmda_tau_zj, tau_p,
                   minicolumn_indices, training_stim_time, training_interval_time,
                   delay_model, num_hcu, num_mcu_neurons, timer, **setup_kwargs):
    # Setup simulator and seed RNG
    sim.setup(timestep=dt, min_delay=dt, max_delay=7.0 * dt, **setup_kwargs)
    rng = sim.NativeRNG(host_rng=NumpyRNG(seed=1))

    # Determine length of each epoch
    epoch_duration = training_stim_time + training_interval_time

    # Stimulate minicolumns in sequence
    stim_minicolumns = [(m, float(i * epoch_duration), 20.0, training_stim_time)
                        for i, m in enumerate(minicolumn_indices)]

    # Calculate length of training required
    training_duration = float(len(stim_minicolumns)) * epoch_duration

    # Calculate mean firing rate
    e_cell_mean_firing_rate = (num_mcu_neurons / NE) * 20.0

    # Build HCUs configured for training
    hcus = [HCU.training(name="%u" % h, sim=sim, rng=rng, simtime=training_duration,
                         intrinsic_tau_z=ampa_tau_zj, intrinsic_tau_p=tau_p,
                         e_cell_mean_firing_rate=e_cell_mean_firing_rate,
                         stim_spike_times=generate_discrete_hcu_stimuli(stim_minicolumns, num_mcu_neurons)) for h in range(num_hcu)]

    # Loop through all hcu products
    connections = []
    for (i_pre, hcu_pre), (i_post, hcu_post) in itertools.product(enumerate(hcus), repeat=2):
        # Use delay model to calculate delay
        hcu_delay = delay_model(i_pre, i_post)

         # Build BCPNN models
        ampa_synapse = bcpnn.BCPNNSynapse(
            tau_zi=ampa_tau_zi,
            tau_zj=ampa_tau_zj,
            tau_p=tau_p,
            f_max=20.0,
            w_max=JE,
            weights_enabled=False,
            plasticity_enabled=True,
            weight=0.0,
            delay=hcu_delay)

        nmda_synapse = bcpnn.BCPNNSynapse(
            tau_zi=nmda_tau_zi,
            tau_zj=nmda_tau_zj,
            tau_p=tau_p,
            f_max=20.0,
            w_max=JE,
            weights_enabled=False,
            plasticity_enabled=True,
            weight=0.0,
            delay=hcu_delay)

        logger.info("Connecting HCU %u->%u with delay %ums" % (i_pre, i_post, hcu_delay))
        connections.append(HCUConnection.training(
            sim=sim, pre_hcu=hcu_pre, post_hcu=hcu_post,
            ampa_synapse=ampa_synapse, nmda_synapse=nmda_synapse, rng=rng))

    logger.info("Build time %gs", timer.diff())

    # Run simulation
    sim.run(training_duration)

    logger.info("Load and run time %gs", timer.diff())

    # Read results from HCUs
    hcu_results = [hcu.read_results() for hcu in hcus]

    # Read results from inter-hcu connections
    connection_results = [c.read_results() for c in connections]

    return hcu_results, connection_results, sim.end

#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------
def test_discrete(connection_weight_filenames, hcu_biases,
                  ampa_gain, nmda_gain, tau_ca2, i_alpha,
                  stim_minicolumns, testing_simtime, delay_model,
                  num_hcu, num_mcu_neurons, record_membrane, timer, **setup_kwargs):

    assert len(hcu_biases) == num_hcu, "An array of biases must be provided for each HCU"
    assert len(connection_weight_filenames) == (num_hcu ** 2), "A tuple of weight matrix filenames must be provided for each HCU->HCU product"

    # Setup simulator and seed RNG
    sim.setup(timestep=dt, min_delay=dt, max_delay=7.0 * dt, **setup_kwargs)
    rng = sim.NativeRNG(host_rng=NumpyRNG(seed=1))

    # Calculate mean firing rate
    e_cell_mean_firing_rate = (num_mcu_neurons / NE) * 20.0

    # Build HCUs configured for testing
    hcus = [HCU.testing_adaptive(name="%u" % i, sim=sim, rng=rng,
                                 bias=bias, tau_ca2=tau_ca2, i_alpha=i_alpha,
                                 e_cell_mean_firing_rate=e_cell_mean_firing_rate,
                                 simtime=testing_simtime, record_membrane=record_membrane,
                                 stim_spike_times=generate_discrete_hcu_stimuli(stim_minicolumns, num_mcu_neurons)) for i, bias in enumerate(hcu_biases)]

    # **HACK** not actually plastic - just used to force signed weights
    bcpnn_synapse = bcpnn.BCPNNSynapse(
        tau_zi=tau_syn_ampa_gaba,
        tau_zj=tau_syn_ampa_gaba,
        tau_p=1000.0,
        f_max=20.0,
        w_max=JE,
        weights_enabled=True,
        plasticity_enabled=False)

    # Loop through all hcu products and their corresponding connection weight
    for connection_weight_filename, ((i_pre, hcu_pre), (i_post, hcu_post)) in zip(connection_weight_filenames, itertools.product(enumerate(hcus), repeat=2)):
        # Use delay model to calculate delay
        hcu_delay = delay_model(i_pre, i_post)

        logger.info("Connecting HCU %u->%u with delay %ums" % (i_pre, i_post, hcu_delay))

        # Build connections
        HCUConnection.testing(
            sim=sim,
            pre_hcu=hcu_pre, post_hcu=hcu_post,
            ampa_gain=ampa_gain, nmda_gain=nmda_gain,
            ampa_synapse=bcpnn_synapse, nmda_synapse=bcpnn_synapse,
            connection_weight_filename=connection_weight_filename, delay=hcu_delay)

    logger.info("Build time %gs", timer.diff())

    # Run simulation
    sim.run(testing_simtime)

    logger.info("Load and run time %gs", timer.diff())
    
    # Read results from HCUs
    results = [hcu.read_results() for hcu in hcus]

    return results, sim.end
