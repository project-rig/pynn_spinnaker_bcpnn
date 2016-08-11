import csv
import itertools
import logging
import numpy as np
import matplotlib.patches as mpatches
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
STIMULUS_TIME = 75.0

BCPNN_TAU_PRIMARY = 10.0       # ms
BCPNN_TAU_ELIGIBILITY = 2000.0 # ms
BCPNN_PHI = 0.045                # nA

# Maximum weight multiplied by Wij value calculated by BCPNN rule
BCPNN_MAX_WEIGHT = 0.012       # uS for conductance

TRAIN = True

#-------------------------------------------------------------------
# Testing parameters
#-------------------------------------------------------------------
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
        number = min(5 + np.ceil(2 * n),100)

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

def create_input_population(size, record, sim):
    # Population parameters
    p = sim.Population(size, sim.IF_curr_exp(**cell_params))
    if record:
        # **YUCK** record spikes actually entirely ignores
        # sampling interval but throws exception if it is not set
        p.record("spikes", sampling_interval=100.0)

    return p

def create_class_population(size, record, ioffset, train, sim):
    params = deepcopy(cell_params)
    params["bias_enabled"] = False
    params["plasticity_enabled"] = train
    params["i_offset"] = ioffset

    # Population parameters
    p = sim.Population(size, bcpnn.IF_curr_exp(**params))
    if record:
        # **YUCK** record spikes actually entirely ignores
        # sampling interval but throws exception if it is not set
        p.record("spikes", sampling_interval=100.0)

    if train:
        p.record("bias", sampling_interval=100.0)

    return p

def calculate_spike_rate(segment, rate_bins, population_size):
    population_histogram = np.zeros(len(rate_bins) - 1)
    for spiketrain in segment.spiketrains:
        population_histogram += np.histogram(spiketrain, bins=rate_bins)[0]

    return population_histogram * (1000.0 / 500.0) * (1.0 / float(population_size))

#-------------------------------------------------------------------
# Build basic classifier network
#-------------------------------------------------------------------
def build_basic_network(input_stimuli_rates, input_stimuli_duration,
                        class_stimuli_rates, class_stimuli_duration,
                        record, ioffset, train, sim):
    # Create main input and class populations
    input_populations = [create_input_population(CLASS_POP_SIZE, record, sim)
                         for _ in input_stimuli_rates]

    if isinstance(ioffset, list) or isinstance(ioffset, np.ndarray):
        assert len(ioffset) == len(class_stimuli_rates)
        class_populations = [create_class_population(CLASS_POP_SIZE, record, o, train, sim)
                             for i, o in enumerate(ioffset)]
    else:
        print ioffset
        class_populations = [create_class_population(CLASS_POP_SIZE, record, ioffset, train, sim)
                             for _ in class_stimuli_rates]

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

def calculate_stim_rates(param_values, unit_means, stimuli_rates, f_max):
    # Loop through units representing parameter
    for unit_mean in unit_means[3]:
        # Calculate rate for each parameter value in training data
        input_stimuli = [f_max * calc_response(value, unit_mean, unit_means[2])
                            for value in param_values]
        # Add list of rates to list
        stimuli_rates.append(input_stimuli)

def train(sepal_length, sepal_length_unit_mean_sd,
          sepal_width, sepal_width_unit_mean_sd,
          petal_length, petal_length_unit_mean_sd,
          petal_width, petal_width_unit_mean_sd,
          unique_species, species):
    # SpiNNaker setup
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0,
            spinnaker_hostname="192.168.1.1")

    # Calculate input rates
    input_rates = []
    calculate_stim_rates(sepal_length, sepal_length_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_stim_rates(sepal_width, sepal_width_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_stim_rates(petal_length, petal_length_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_stim_rates(petal_width, petal_width_unit_mean_sd, input_rates, MAX_FREQUENCY)

    # Calculate class rates
    class_rates = []
    for u, _ in enumerate(unique_species):
        class_rates.append(list((species == u) * MAX_FREQUENCY))

    # Build basic network with orthogonal stimulation of both populations
    input_populations, class_populations = build_basic_network(input_rates, STIMULUS_TIME,
                                                               class_rates, STIMULUS_TIME,
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

    # Create all-to-all connector to connect inputs to classes
    input_class_connector = sim.AllToAllConnector()

    # Loop through all pairs of input populations and classes
    plastic_connections = []
    for (i, c) in itertools.product(input_populations, class_populations):
        # Connect input to class with all-to-all plastic synapse
        connection = sim.Projection(i, c, input_class_connector, bcpnn_synapse,
                                    receptor_type="excitatory", label="%s-%s" % (i.label, c.label))
        plastic_connections.append(connection)

    # Run simulation
    sim.run(STIMULUS_TIME * len(sepal_length))

    # Read biases
    # **HACK** investigate where out by 1000 comes from!
    learnt_biases = [c.get_data().segments[0].filter(name="bias")[0][-1,:] * 0.001
                    for c in class_populations]

    # Read plastic weights
    learnt_weights = [p.get("weight", format="array") for p in plastic_connections]

    return learnt_biases, learnt_weights

def test(sepal_length, sepal_length_unit_mean_sd,
         sepal_width, sepal_width_unit_mean_sd,
         petal_length, petal_length_unit_mean_sd,
         petal_width, petal_width_unit_mean_sd,
         num_species,
         learnt_biases, learnt_weights):
    # SpiNNaker setup
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0,
              spinnaker_hostname="192.168.1.1")

    # Calculate input rates
    input_rates = []
    calculate_stim_rates(sepal_length, sepal_length_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_stim_rates(sepal_width, sepal_width_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_stim_rates(petal_length, petal_length_unit_mean_sd, input_rates, MAX_FREQUENCY)
    calculate_stim_rates(petal_width, petal_width_unit_mean_sd, input_rates, MAX_FREQUENCY)

    # Generate uncertain class pattern
    uncertain_class_rates = [[MAX_FREQUENCY * (1.0 / num_species)]
                             for s in range(num_species)]

    # Build basic network
    testing_time = STIMULUS_TIME * len(sepal_length)
    input_populations, class_populations = build_basic_network(input_rates, STIMULUS_TIME,
                                                               uncertain_class_rates, testing_time,
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
    sim.run(testing_time)

    # Read spikes from input and class populations
    input_data = [i.get_data() for i in input_populations]
    class_data = [c.get_data() for c in class_populations]

    # End simulation on SpiNNaker
    sim.end()

    # Return spikes
    return input_data, class_data

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
    species = np.asarray([unique_species.index(s)
                          for s in columns[4]], dtype=int)

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
# **YUCK** use fixed seed so split is the same after reloading
np.random.seed(0x12345678)
permute_indices = np.random.permutation(num_samples)
num_training = int(0.8 * num_samples)
training_indices = permute_indices[:num_training]
testing_indices = permute_indices[num_training:]

if TRAIN:
    # Train
    learnt_biases, learnt_weights = train(sepal_length[training_indices], sepal_length_unit_mean_sd,
                                          sepal_width[training_indices], sepal_width_unit_mean_sd,
                                          petal_length[training_indices], petal_length_unit_mean_sd,
                                          petal_width[training_indices], petal_width_unit_mean_sd,
                                          unique_species, species[training_indices])

    # Save trained data
    np.save("learnt_weights.npy", learnt_weights)
    np.save("learnt_biases.npy", learnt_biases)
else:
    learnt_weights = np.load("learnt_weights.npy")
    learnt_biases = np.load("learnt_biases.npy")

# Test
input_data, class_data = test(sepal_length[testing_indices], sepal_length_unit_mean_sd,
                              sepal_width[testing_indices], sepal_width_unit_mean_sd,
                              petal_length[testing_indices], petal_length_unit_mean_sd,
                              petal_width[testing_indices], petal_width_unit_mean_sd,
                              len(unique_species),
                              learnt_biases, learnt_weights)

figure, axes = plt.subplots(len(unique_species), sharex=True)

# Calculate rates for each class for each stimulus
rate_bins = np.arange(0, (STIMULUS_TIME * len(testing_indices)) + 1, STIMULUS_TIME)
rates = np.vstack([calculate_spike_rate(d.segments[0], rate_bins, CLASS_POP_SIZE)
                   for d in class_data])

# Calculate largest rate in each bin
winner = np.argmax(rates, axis=0)

# Determine whether winner is correct in each bin
correct = (winner == species[testing_indices])

# Loop through rows of rates, axis and the name of the species they represent
colours = ["gray", "red", "gray", "green"]
for i, (r, n, a) in enumerate(zip(rates, unique_species, axes)):
    # Lookup bar colour based on whether this class is
    # the winner and whether that is correct
    bar_colour = [colours[i]
                  for i in (winner == i) + (correct * 2)]

    # Plot bar showing rate
    a.bar(rate_bins[:-1], r, STIMULUS_TIME, color=bar_colour)

    # Mark certainty on axis, make all axis limits match and label axis
    a.axhline(MAX_FREQUENCY, color="grey", linestyle="--")
    a.set_ylim((0, MAX_FREQUENCY * 1.5))
    a.set_ylabel("Firing rate [Hz]")

    # Show class name as axis title
    a.set_title(n)

axes[-1].set_xlabel("Time [ms]")

# Build legend
figure.legend([mpatches.Patch(color="red"), mpatches.Patch(color="green")],
              ["Incorrect classification", "Correct classification"])

# Calculate classification accuracy
print "Classification accuracy = %f%%" % (100.0 * (float(np.sum(correct)) / float(len(correct))))

plt.show()