import enum
import functools
import itertools
import logging
import numpy
import os
import pickle
import sys

import network

# Set PyNN spinnaker log level
logger = logging.getLogger("pynn_spinnaker")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class Mode(enum.Enum):
    train_asymmetrical = 1
    train_symmetrical  = 2
    test_asymmetrical  = 3
    test_symmetrical   = 4

mode = Mode.test_asymmetrical

hcu_grid_size = 1
num_hcu = hcu_grid_size ** 2
num_mcu_neurons = 100

record_membrane = True

spinnaker_kwargs = {"spalloc_num_boards": 1,
                    "stop_on_spinnaker": True}

tau_p = 2000

folder = "sequence_%u_%u" % (num_hcu, tau_p)
if not os.path.exists(folder):
    os.makedirs(folder)

# Bind parameters to euclidean HCU delay model
delay_model = functools.partial(network.euclidean_hcu_delay,
                                grid_size=hcu_grid_size, distance_scale=0.75, velocity=0.2)
#delay_model = functools.partial(network.euclidean_hcu_delay,
#                                grid_size=hcu_grid_size, distance_scale=0.6, velocity=0.2)
#delay_model = functools.partial(network.euclidean_hcu_delay,
#                                grid_size=hcu_grid_size, distance_scale=0.5, velocity=0.2)

# If we're training
if mode == Mode.train_asymmetrical or mode == Mode.train_symmetrical:
    # Training parameters
    training_stim_time = 100.0
    training_interval_time = 0.0
    num_training_epochs = 50

    # Repeat sequences of sequential minicolumn activation for each epoch
    minicolumn_indices = itertools.chain(*itertools.repeat(range(10), num_training_epochs))

    # Determine tau_zj for NMDA synapses depending on mode
    nmda_tau_zj = network.tau_syn_ampa_gaba if mode == Mode.train_asymmetrical else network.tau_syn_nmda

    # Simulate
    hcu_results, connection_results, end_simulation = network.train_discrete(network.tau_syn_ampa_gaba, network.tau_syn_ampa_gaba,
                                                                             network.tau_syn_nmda, nmda_tau_zj, tau_p,
                                                                             minicolumn_indices, training_stim_time, training_interval_time,
                                                                             delay_model, num_hcu, num_mcu_neurons, **spinnaker_kwargs)

    # Save weights for all connections
    for i, (ampa_weight_writer, nmda_weight_writer) in enumerate(connection_results):
        # Write AMPA weights
        ampa_weight_writer("%s/connection_%u_e_e_ampa.npy" % (folder, i))

        # Write NMDA weights to correct folder
        if mode == Mode.train_asymmetrical:
            nmda_weight_writer("%s/connection_%u_e_e_nmda_asymmetrical.npy" % (folder, i))
        else:
            nmda_weight_writer("%s/connection_%u_e_e_nmda_symmetrical.npy" % (folder, i))

    # Loop through the HCU results and save data to pickle format
    for i, (hcu_e_data_writer,) in enumerate(hcu_results):
        hcu_e_data_writer("%s/hcu_%u_e_data.pkl" % (folder, i))

    # Once data is read, end simulation
    end_simulation()
else:
    # Testing parameters
    testing_simtime = 6000.0   # simulation time [ms]

    ampa_nmda_ratio = 4.795918367
    tau_ca2 = 300.0

    if mode == Mode.test_symmetrical:
        i_alpha = 0.7
        gain_per_hcu = 1.3
    else:
        i_alpha = 0.15
        gain_per_hcu = 0.546328125

    # Calculate gain
    gain = gain_per_hcu / float(num_hcu)

    # Load biases for each HCU
    hcu_biases = []
    for i in range(num_hcu):
        # Open pickle file
        with open("%s/hcu_%u_e_data.pkl" % (folder, i), "rb") as f:
            # Load pickled data
            pickled_data = pickle.load(f)

            # Filter out bias
            hcu_bias = pickled_data.segments[0].filter(name="bias")[0]

            # Add final recorded bias to list
            # **HACK** investigate where out by 1000 comes from!
            hcu_biases.append(hcu_bias[-1,:] * 0.001)

    # Build correct filename format string for weights
    nmda_weight_filename_format = ("%s/connection_%u_e_e_nmda_asymmetrical.npy"
                                   if mode == Mode.test_asymmetrical
                                   else "%s/connection_%u_e_e_nmda_symmetrical.npy")

    # Load weights for each connection
    connection_weights = []
    for i in range(num_hcu ** 2):
        connection_weights.append((
            "%s/connection_%u_e_e_ampa.npy" % (folder, i),
            nmda_weight_filename_format % (folder, i)
        ))

    # Stimulate the first minicolumn for 50ms, 100ms into simulation
    stim_minicolumns = [(0, 100.0, 20.0, 50.0)]


    hcu_results, end_simulation = network.test_discrete(connection_weights, hcu_biases,
                                                        gain, gain / ampa_nmda_ratio, tau_ca2, i_alpha,
                                                        stim_minicolumns, testing_simtime, delay_model,
                                                        num_hcu, num_mcu_neurons, record_membrane,
                                                        **spinnaker_kwargs)

    # Build correct filename format string for data
    e_filename_format = ("%s/hcu_%u_e_testing_data_asymmetrical.pkl"
                         if mode == Mode.test_asymmetrical
                         else "%s/hcu_%u_e_testing_data_symmetrical.pkl")
    i_filename_format = ("%s/hcu_%u_i_testing_data_asymmetrical.pkl"
                         if mode == Mode.test_asymmetrical
                         else "%s/hcu_%u_i_testing_data_symmetrical.pkl")

    # Loop through the HCU results and save spikes data
    for i, (hcu_e_data_writer, hcu_i_data_writer) in enumerate(hcu_results):
        hcu_e_data_writer(e_filename_format % (folder, i))
        hcu_i_data_writer(i_filename_format % (folder, i))

    # Once data is read, end simulation
    end_simulation()
