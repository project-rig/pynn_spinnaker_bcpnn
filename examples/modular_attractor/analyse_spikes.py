import argparse
import numpy
import pickle
import pylab
import sys

from network import (NI, NE)

def display_spikes_interleaved(e_spikes, i_spikes, raster_axis, rate_axis, num_hcus, num_mcu_neurons,
                               sim_start_time, sim_end_time, cmap, scatter_size=0.1):
    num_mcus = (NE * num_hcus) // num_mcu_neurons
    num_mcu_per_hcu = NE // num_mcu_neurons
    num_cells_per_mcu = num_mcu_neurons * num_hcus

    # Determine which minicolumn excitatory spikes have originated
    e_minicolumn = numpy.floor(e_spikes[:,0] / num_cells_per_mcu)

    # Plot spike raster, colouring excitatory spikes based on minicolumn
    raster_axis.scatter(e_spikes[:,1], e_spikes[:,0], c=e_minicolumn / float(num_mcu_per_hcu), cmap=cmap, linewidths=0.0, s=scatter_size)
    if i_spikes is not None:
        raster_axis.scatter(i_spikes[:,1], i_spikes[:,0] + (NE * num_hcus), color="gray", linewidths=0.0, s=scatter_size)

    # Plot excitatory and inhibitory rate
    binsize = 10.0
    rate_bins = numpy.arange(sim_start_time, sim_end_time + 1, binsize)

    # Loop through minicolumns
    for m in range(num_mcu_per_hcu):
        # Mask out spikes that come from minicolumn
        minicolumn_spikes = e_spikes[e_minicolumn == m]

        # Calculate firing rates based on these and plot in correct colour
        minicolumn_histogram = numpy.histogram(minicolumn_spikes[:,1], bins=rate_bins)
        minicolumn_rate = minicolumn_histogram[0] * (1000.0 / binsize) * (1.0 / float(num_cells_per_mcu))

        rate_axis.plot(minicolumn_histogram[1][:-1], minicolumn_rate, color=cmap(float(m) / float(num_mcu_per_hcu)))

    if i_spikes is not None:
        i_histogram = numpy.histogram(i_spikes[:,1], bins=rate_bins)
        i_rate = i_histogram[0] * (1000.0 / binsize) * (1.0 / float(NI * num_hcus))
        rate_axis.plot(i_histogram[1][:-1], i_rate, color="gray")

def load_spikes(filename, segment):
    hcu_spikes = None
    with open(filename, "rb") as f:
        # Load data
        spiketrains = pickle.load(f).segments[segment].spiketrains

        # Loop through spike trains
        for s in spiketrains:
            # Build empty numpy array to hold data in old PyNN format
            neuron_spikes = numpy.empty((len(s), 2), dtype=float)
            neuron_spikes[:,0] = s.annotations["source_index"]
            neuron_spikes[:,1] = s

            # Stack HCU's spikes onto networks
            if hcu_spikes is None:
                hcu_spikes = neuron_spikes
            else:
                hcu_spikes = numpy.vstack((hcu_spikes, neuron_spikes))

    return hcu_spikes


def combine_e_spikes(filenames, segment, num_mcu_neurons):
    num_mcus = (NE * len(filenames)) // num_mcu_neurons

    # Loop through filenames
    e_spikes = None
    for i, f in enumerate(filenames):
        # Load e spikes
        hcu_e_spikes = load_spikes(f, segment)

        # Determine which minicolumn each spike originates from [0, num_mcus)
        hcu_e_minicolumn = numpy.remainder(hcu_e_spikes[:,0], num_mcus)

        # Determine which neuron within the minicolumn each spike originates from [0, num_mcu_neurons)
        hcu_e_index = numpy.floor(hcu_e_spikes[:,0] / num_mcus)

        # Offset index
        hcu_e_index += (i * num_mcu_neurons)

        # Interleave neuron IDs so neurons so minicolumns are sequential across HCUs
        #hcu_e_spikes[:,0] += (i * num_mcu_neurons)
        #hcu_e_spikes[:,0] += (hcu_e_minicolumn * (num_mcu_neurons * (len(filenames) - 1)))
        hcu_e_spikes[:,0] = (hcu_e_minicolumn * num_mcu_neurons * len(filenames)) + hcu_e_index

        # Stack HCU's spikes onto networks
        if e_spikes is None:
            e_spikes = hcu_e_spikes
        else:
            e_spikes = numpy.vstack((e_spikes, hcu_e_spikes))


    return e_spikes

def combine_i_spikes(filenames, segment):
    # Loop through filenames
    i_spikes = None
    for i, f in enumerate(filenames):
        # Load i spikes
        hcu_i_spikes = load_spikes(f, segment)

        hcu_i_spikes[:,0] += (i * NI)

        if i_spikes is None:
            i_spikes = hcu_i_spikes
        else:
            i_spikes = numpy.vstack((i_spikes, hcu_i_spikes))
    return i_spikes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine together spikes recorded from multiple HCUs and display raster")
    parser.add_argument("--num_hcus", type=int, default=9, help="How many HCUs is data for")
    parser.add_argument("--num_mcu_neurons", type=int, default=100, help="How many neurons make up an MCU")
    parser.add_argument("folder", nargs=1, help="Folder to search for spike files in")
    parser.add_argument("e_filename", nargs=1, help="Filenames of excitatory spike files are of the form hcu_X_YYYY.npy where filename specified YYYY")
    parser.add_argument("i_filename", nargs="?", help="Filenames of inhibitory spike files are of the form hcu_X_YYYY.npy where filename specified YYYY")
    args = parser.parse_args()

    # Combine e spikes
    e_filenames = ["%s/hcu_%u_%s.pkl" % (args.folder[0], i, args.e_filename[0]) for i in range(args.num_hcus)]
    e_spikes = combine_e_spikes(e_filenames, 0, args.num_mcu_neurons)

    figure, axes = pylab.subplots(2, sharex=True)

    # If i spikes filename was specified
    if args.i_filename is not None:
        i_filenames = ["%s/hcu_%u_%s.pkl" % (args.folder[0], i, args.i_filename) for i in range(args.num_hcus)]
        i_spikes = combine_i_spikes(i_filenames, 0)

        axes[0].set_ylim((0, (NE + NI) * args.num_hcus))
    else:
        i_spikes = None
        axes[0].set_ylim((0, NE * args.num_hcus))

    sim_start_time = 0
    sim_end_time = numpy.amax(e_spikes[:,1])

    axes[0].set_ylabel("Neuron ID")
    axes[0].set_xlim((sim_start_time, sim_end_time))
    axes[1].set_ylabel("Rate/Hz")
    axes[1].set_xlabel("Time/ms")

    display_spikes_interleaved(e_spikes, i_spikes, axes[0], axes[1], args.num_hcus, args.num_mcu_neurons,
                                sim_start_time, sim_end_time, pylab.cm.rainbow, 1.0)

    pylab.show()
