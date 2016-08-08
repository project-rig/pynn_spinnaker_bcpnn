import csv
import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt

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
BCPNN_TAU_ELIGIBILITY = 2000.0 # ms
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
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + numpy.ceil(2 * n),100)

    if number > 0:
        isi = np.random.exponential(1.0/rate, number)*1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if len(spikes) == i:
        # ISI buf overrun

        t_last = spikes[-1] + np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes,(i,))

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
        weights = np.multiply(weights, weight_scale)

        # Build numpy array of delays
        delays = np.repeat(delay, len(weights))

        # Zip x-y coordinates of non-zero weights with weights and delays
        return zip(indices[0], indices[1], weights, delays)

    # Get indices of non-nan i.e. connected weights
    connected_indices = np.where(~np.isnan(matrix))

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

def calc_unit_mean_sd(param, num_mc):
    # Calculate parameter range
    min_param = np.amin(param)
    max_param = np.amax(param)

    print("\tMin:%f, Max:%f" % (min_param, max_param))

    # Evenly space units across parameter space
    spacing = (max_param - min_param) / float(num_mc - 1)
    unit_means = np.arange(min_param, max_param + 0.00001, spacing)

    # Return unit means
    return min_param, max_param, spacing / 2.35482, unit_means

def calc_response(value, unit_mean, unit_sd):
    return np.exp(-((value - unit_mean) ** 2) / (2.0 * (unit_sd ** 2)))

def plot_response(mean_sd, axis, title):
    axis.set_title(title)
    axis.set_xlabel("Parameter value")
    axis.set_ylabel("Firing rate [Hz]")
    axis.set_xlim((mean_sd[0], mean_sd[1]))

    param_range = np.arange(mean_sd[0], mean_sd[1], 0.01)

    for unit_mean in mean_sd[3]:
        axis.plot(param_range, calc_response(param_range, unit_mean, mean_sd[2]))

def calculate_rates(param_values, unit_means, stimuli_rates, f_max):
    # Loop through units representing parameter
    for unit_mean in unit_means[3]:
        # Calculate rate for each parameter value in training data
        input_stimuli = [f_max * calc_response(value, unit_mean, unit_means[2])
                            for value in param_values]
        # Add list of rates to list
        stimuli_rates.append(input_stimuli)

#-------------------------------------------------------------------
# Entry point
#-------------------------------------------------------------------
with open("iris.csv", "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")

    # Read columns
    columns = zip(*csv_reader)
    sepal_length = np.asarray(columns[0], dtype=float)
    sepal_width = np.asarray(columns[1], dtype=float)
    petal_length = np.asarray(columns[2], dtype=float)
    petal_width = np.asarray(columns[3], dtype=float)
    num_samples = len(sepal_length)

    # Find unique species and hence index each species
    unique_species = list(set(columns[4]))
    species = np.asarray([unique_species.index(s) for s in columns[4]], dtype=int)

    # Get min and max param values
    print("Sepal length")
    sepal_length_unit_mean_sd = calc_unit_mean_sd(sepal_length, 11)

    print("Sepal width")
    sepal_width_unit_mean_sd = calc_unit_mean_sd(sepal_width, 11)

    print("Petal length")
    petal_length_unit_mean_sd = calc_unit_mean_sd(petal_length, 11)

    print("Petal width")
    petal_width_unit_mean_sd = calc_unit_mean_sd(petal_width, 11)

    # Plot minicolumn responses
    figure, axes = plt.subplots(4)
    plot_response(sepal_length_unit_mean_sd, axes[0], "Sepal length")
    plot_response(sepal_width_unit_mean_sd, axes[1], "Sepal width")
    plot_response(petal_length_unit_mean_sd, axes[2], "Petal length")
    plot_response(petal_width_unit_mean_sd, axes[3], "Petal width")
    plt.show()

    # Split data into training and test
    permute_indices = np.random.permutation(num_samples)
    num_training = int(0.8 * num_samples)
    training_indices = permute_indices[:num_training]
    testing_indices = permute_indices[num_training:]

    # Calculate input rates
    input_rates = []
    calculate_rates(sepal_length[training_indices], sepal_length_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_rates(sepal_width[training_indices], sepal_width_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_rates(petal_length[training_indices], petal_length_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_rates(petal_width[training_indices], petal_width_unit_mean_sd, input_rates, MAX_FREQUENCY)

    # Calculate class rates
    class_rates = []
    for u, _ in enumerate(unique_species):
        class_rates.append(list((species[training_indices] == u) * MAX_FREQUENCY))

    # SpiNNaker setup
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0, spinnaker_hostname="192.168.1.1")

    # Build basic network with orthogonal stimulation of both populations
    input_populations, class_populations = build_basic_network(input_rates, TRAINING_STIMULUS_TIME,
                                                               class_rates, TRAINING_STIMULUS_TIME,
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
