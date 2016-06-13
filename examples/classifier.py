import itertools
import logging
import numpy
import pylab

# Import simulator
import pynn_spinnaker as sim
import pynn_spinnaker_bcpnn as bcpnn

from copy import deepcopy


logger = logging.getLogger("pynn_spinnaker")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

#-------------------------------------------------------------------
# General Parameters
#-------------------------------------------------------------------
# Network dimensions
INPUT_NAMES = ["X", "Y"]
CLASS_NAMES = ["X'", "Y'"]

CLASS_POP_SIZE = 30

WEIGHT_GAIN = 6.0

# Weight of connection between stimuli input and class and input populations
PRE_STIMULI_WEIGHT = 2.0
POST_STIMULI_WEIGHT = 2.0

# Firing frequency that corresponds to certainty
MAX_FREQUENCY = 20.0    # Hz
MIN_FREQUENCY = 0.001

#-------------------------------------------------------------------
# Training parameters
#-------------------------------------------------------------------
# Experiment configuration
TRAINING_TIME = 20 * 1000
TRAINING_STIMULUS_TIME = 100

BCPNN_TAU_PRIMARY = 10.0       # ms
BCPNN_TAU_ELIGIBILITY = 1000.0 # ms
BCPNN_PHI = 0.045                # nA

# Maximum weight multiplied by Wij value calculated by BCPNN rule
BCPNN_MAX_WEIGHT = 0.012       # uS for conductance

#-------------------------------------------------------------------
# Testing parameters
#-------------------------------------------------------------------
TESTING_STIMULUS_TIME = 5000
TESTING_TIME = 4 * TESTING_STIMULUS_TIME


cell_params = {
    'cm'        : 0.25, # nF
    'tau_m'     : 20.0,
    'tau_refrac': 2.0,
    'tau_syn_E' : 5.0,
    'tau_syn_I' : 5.0,
    'v_reset'   : -70.0,
    'v_rest'    : -70.0,
    'v_thresh'  : -55.0
}
#-------------------------------------------------------------------
# Generate poisson noise of given rate between start and stop times
#-------------------------------------------------------------------
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

#-------------------------------------------------------------------
# Convert weights in format returned by getWeights into a connection list 
# **NOTE** this requires signed weight support
#-------------------------------------------------------------------
# Convert weights in format returned by getWeights into a connection list
def convert_weights_to_list(matrix, delay, weight_scale=1.0):
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

#-------------------------------------------------------------------
# Convert list of stimuli rates and a duration into blocks of 
# Poisson noise in format to load into spike source array
#-------------------------------------------------------------------
def generate_stimuli_spike_times(stimuli_rates, stimuli_duration, population_size):
    # Build spike source array format spike trains for each neuron in population
    population_spike_times = []
    for n in range(population_size):
        # Loop through all stimuli and add poisson noise to this neuron's spike times
        neuron_spike_times = []
        for i, r in enumerate(stimuli_rates):
            start_time = i * stimuli_duration
            end_time = start_time + stimuli_duration
            neuron_spike_times.extend(poisson_generator(r, start_time, end_time))
        
        # Add neuron spike times to population
        population_spike_times.append(neuron_spike_times)
    return population_spike_times

def create_input_population(size, name, record, sim):
    # Population parameters
    p = sim.Population(size, sim.IF_curr_exp(**cell_params), label=name)
    if record:
        # **YUCK** record spikes actually entirely ignores
        # sampling interval but throws exception if it is not set
        p.record("spikes", sampling_interval=100.0)
    
    return p

def create_class_population(size, name, record, ioffset, train, sim):
    params = deepcopy(cell_params)
    params["bias_enabled"] = False
    params["plasticity_enabled"] = train
    params["i_offset"] = ioffset

    # Population parameters
    p = sim.Population(size, bcpnn.IF_curr_exp(**params), label=name)
    if record:
        # **YUCK** record spikes actually entirely ignores
        # sampling interval but throws exception if it is not set
        p.record("spikes", sampling_interval=100.0)

    if train:
        p.record("bias", sampling_interval=100.0)

    return p

#-------------------------------------------------------------------
# Build basic classifier network
#-------------------------------------------------------------------
def build_basic_network(input_stimuli_rates, input_stimuli_duration, 
                        class_stimuli_rates, class_stimuli_duration, 
                        record, ioffset, train, sim):
    # Create main input and class populations
    input_populations = [create_input_population(CLASS_POP_SIZE, i, record, sim) for i in INPUT_NAMES]
    
    if isinstance(ioffset, list):
        class_populations = [create_class_population(CLASS_POP_SIZE, c, record, o, train, sim) for i, (o, c) in enumerate(zip(ioffset, CLASS_NAMES))]
    else:
        print ioffset
        class_populations = [create_class_population(CLASS_POP_SIZE, c, record, ioffset, train, sim) for c in CLASS_NAMES]
    
    # Create pre-synaptic stimuli populations
    pre_stimuli_connector = sim.OneToOneConnector()
    pre_stimuli_synapse = sim.StaticSynapse(weight=PRE_STIMULI_WEIGHT)
    for i, (rate, input_pop) in enumerate(zip(input_stimuli_rates, input_populations)):
        # Convert stimuli into spike times
        spike_times = generate_stimuli_spike_times(rate, input_stimuli_duration, CLASS_POP_SIZE)
        
        # Build spike source array with these times
        stim_pop = sim.Population(CLASS_POP_SIZE, sim.SpikeSourceArray(spike_times=spike_times),
                                  label="pre_stimuli_%u" % i)
        
        # Connect spike source to input
        sim.Projection(stim_pop, input_pop, pre_stimuli_connector, pre_stimuli_synapse, receptor_type="excitatory",
                       label="%s-%s" % (stim_pop.label, input_pop.label))
    
    # Create training spike source array populations
    post_stimuli_connector = sim.OneToOneConnector()
    post_stimuli_synapse = sim.StaticSynapse(weight=POST_STIMULI_WEIGHT)
    for i, (rate, class_pop) in enumerate(zip(class_stimuli_rates, class_populations)):
        # Convert stimuli into spike times
        spike_times = generate_stimuli_spike_times(rate, class_stimuli_duration, CLASS_POP_SIZE)
        
        # Build spike source array with these times
        stim_pop = sim.Population(CLASS_POP_SIZE, sim.SpikeSourceArray, {"spike_times": spike_times},
                                  label="post_stimuli_%u" % i)
        
        # Connect spike source to input
        sim.Projection(stim_pop, class_pop, post_stimuli_connector, post_stimuli_synapse, receptor_type="excitatory",
                       label="%s-%s" % (stim_pop.label, class_pop.label))
    
    # Return created populations
    return input_populations, class_populations

#-------------------------------------------------------------------
# Train network and return weights
#-------------------------------------------------------------------
def train():
    # SpiNNaker setup
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0, spinnaker_hostname="192.168.1.1")
    
    # Generate orthogonal input stimuli
    orthogonal_stimuli_rates = []
    num_inputs = len(INPUT_NAMES)
    for i in range(num_inputs):
        input_stimuli = []
        for s in range(TRAINING_TIME / TRAINING_STIMULUS_TIME):
            input_stimuli.append(MIN_FREQUENCY if (s % num_inputs) == i else MAX_FREQUENCY)
        orthogonal_stimuli_rates.append(input_stimuli)
    
    # Build basic network with orthogonal stimulation of both populations
    input_populations, class_populations = build_basic_network(orthogonal_stimuli_rates, TRAINING_STIMULUS_TIME,
                                                               orthogonal_stimuli_rates, TRAINING_STIMULUS_TIME,
                                                               False, 0.0, True, sim)

        
    # Create BCPNN model with weights disabled
    bcpnn_synapse = bcpnn.BCPNNSynapse(
            tau_zi=BCPNN_TAU_PRIMARY,
            tau_zj=BCPNN_TAU_PRIMARY,
            tau_p=BCPNN_TAU_ELIGIBILITY,
            f_max=MAX_FREQUENCY,
            w_max=BCPNN_MAX_WEIGHT,
            weights_enabled=False,
            plasticity_enabled=True,
            weight=0.0)
    
    # Create all-to-all conector to connect inputs to classes
    input_class_connector = sim.AllToAllConnector()
    
    # Loop through all pairs of input populations and classes
    plastic_connections = []
    for (i, c) in itertools.product(input_populations, class_populations):
        # Connect input to class with all-to-all plastic synapse
        connection = sim.Projection(i, c, input_class_connector, bcpnn_synapse,
                                    receptor_type="excitatory", label="%s-%s" % (i.label, c.label))
        plastic_connections.append(connection)
    
    # Run simulation
    sim.run(TRAINING_TIME)
    
    # Plot bias evolution
    num_classes = len(CLASS_NAMES)
    #bias_figure, bias_axes = pylab.subplots()
    
    # **HACK** Extract learnt biases from gsyn channel
    learnt_biases = []
    plotting_times = range(TRAINING_TIME)
    for i, c in enumerate(class_populations):
        # Read bias from class
        bias = c.get_data().segments[0].filter(name="bias")[0]
        '''
        # Loop through plotting times to get mean biases
        mean_pj = []
        for t in plotting_times:
            # Slice out the rows for all neurons at this time
            time_rows = gsyn[t::TRAINING_TIME]
            time_bias = zip(*time_rows)[2]
            mean_pj.append(numpy.average(numpy.exp(numpy.divide(time_bias,BCPNN_PHI))))
        
        bias_axes.plot(plotting_times, mean_pj, label=c.label)
        '''
        # Add final bias column to list
        # **HACK** investigate where out by 1000 comes from!
        learnt_biases.append(bias[-1,:] * 0.001)
    '''
    bias_axes.set_title("Mean final bias")
    bias_axes.set_ylim((0.0, 1.0))
    bias_axes.set_ylabel("Pj")
    bias_axes.set_xlabel("Time/ms")
    bias_axes.legend()
    '''
    # Plot weights
    weight_figure, weight_axes = pylab.subplots(num_inputs, num_classes)
    
    # Loop through plastic connections
    learnt_weights = []
    for i, c in enumerate(plastic_connections):
        # Extract weights and calculate mean
        weights = c.get("weight", format="array")
        mean_weight = numpy.average(weights)
        
        # Add weights to list
        learnt_weights.append(weights)
        
        # Plot mean weight in each panel
        axis = weight_axes[i % num_inputs][i / num_classes]
        axis.matshow([[mean_weight]], cmap=pylab.cm.gray)
        #axis.set_title("%s: %fuS" % (c.label, mean_weight))
        axis.set_title("%u->%u: %f" % (i % num_inputs, i / num_classes, mean_weight))
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    
    # Show figures
    pylab.show()
        
    # End simulation on SpiNNaker
    sim.end()
    
    # Return learnt weights
    return learnt_weights, learnt_biases

#-------------------------------------------------------------------
# Test trained network
#-------------------------------------------------------------------
def test(learnt_weights, learnt_biases):
    # SpiNNaker setup
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0, spinnaker_hostname="192.168.1.1")
    
    # Generate testing stimuli patters
    testing_stimuli_rates = [
        [MAX_FREQUENCY, MIN_FREQUENCY, MAX_FREQUENCY, MIN_FREQUENCY],
        [MIN_FREQUENCY, MAX_FREQUENCY, MAX_FREQUENCY, MIN_FREQUENCY],
    ]
    
    # Generate uncertain class stimuli pattern
    uncertain_stimuli_rates = [
        [MAX_FREQUENCY * 0.5],
        [MAX_FREQUENCY * 0.5],
    ]
    
    # Build basic network
    input_populations, class_populations = build_basic_network(testing_stimuli_rates, TESTING_STIMULUS_TIME, 
                                                               uncertain_stimuli_rates, TESTING_TIME,
                                                               True, learnt_biases, False, sim)

    # Create BCPNN model with weights disabled
    bcpnn_synapse = bcpnn.BCPNNSynapse(
            tau_zi=BCPNN_TAU_PRIMARY,
            tau_zj=BCPNN_TAU_PRIMARY,
            tau_p=BCPNN_TAU_ELIGIBILITY,
            f_max=MAX_FREQUENCY,
            w_max=BCPNN_MAX_WEIGHT,
            weights_enabled=True,
            plasticity_enabled=False)
    
    for ((i, c), w) in zip(itertools.product(input_populations, class_populations), learnt_weights):
        # Convert learnt weight matrix into a connection list
        connections = convert_weights_to_list(w, 1.0, 7.0)
        
        # Create projections 
        sim.Projection(i, c, sim.FromListConnector(connections), bcpnn_synapse,
                       receptor_type="excitatory", label="%s-%s" % (i.label, c.label))
        
    # Run simulation
    sim.run(TESTING_TIME)
    
    # Read spikes from input and class populations
    input_data = [i.get_data() for i in input_populations]
    class_data = [c.get_data() for c in class_populations]

    # End simulation on SpiNNaker
    sim.end()
    
    # Return spikes
    return input_data, class_data

def plot_spiketrains(axis, segment, offset, **kwargs):
    for spiketrain in segment.spiketrains:
        y = numpy.ones_like(spiketrain) * (offset + spiketrain.annotations["source_index"])
        axis.scatter(spiketrain, y, **kwargs)

def calculate_rate(segment, rate_bins, population_size):
    population_histogram = numpy.zeros(len(rate_bins) - 1)
    for spiketrain in segment.spiketrains:
        population_histogram += numpy.histogram(spiketrain, bins=rate_bins)[0]

    return population_histogram * (1000.0 / 500.0) * (1.0 / float(population_size))
#-------------------------------------------------------------------
# Experiment
#-------------------------------------------------------------------
# Train model and get weights
'''
learnt_weights, learnt_biases = train()


numpy.save("learnt_weights.npy", learnt_weights)
numpy.save("learnt_biases.npy", learnt_biases)

'''

learnt_weights = numpy.load("learnt_weights.npy")
learnt_biases = list(numpy.load("learnt_biases.npy"))

input_data, class_data = test(learnt_weights, learnt_biases)

'''
for l, s in input_spike_lists:
    s.save("input_spikes_%s.dat" % l)
    
for l, s in class_spike_lists:
    s.save("class_spikes_%s.dat" % l)
'''

figure, axes = pylab.subplots(len(INPUT_NAMES) + len(CLASS_NAMES))

rate_bins = numpy.arange(0, TESTING_TIME + 1, 500)

for d, n, a in zip(input_data, INPUT_NAMES, axes[:len(INPUT_NAMES)]):
    rates = calculate_rate(d.segments[0], rate_bins, CLASS_POP_SIZE)
    #plot_spiketrains(a, d.segments[0], 0.0)
    a.plot(rate_bins[:-1], rates, color="red")
    a.set_ylim((0, 40.0))
    a.axhline(MAX_FREQUENCY, color="grey", linestyle="--")
    a.set_title(n)

for d, n, a in zip(class_data, CLASS_NAMES, axes[len(INPUT_NAMES):]):
    rates = calculate_rate(d.segments[0], rate_bins, CLASS_POP_SIZE)
    #plot_spiketrains(a, d.segments[0], 0.0)
    a.plot(rate_bins[:-1], rates, color="blue")
    a.set_ylim((0, 40.0))
    a.axhline(MAX_FREQUENCY, color="grey", linestyle="--")
    a.set_title(n)

# Show figures
pylab.show()
