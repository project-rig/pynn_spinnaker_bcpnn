import argparse
import copy
import itertools
import numpy
import pylab
import sys

from network import (NI, NE)

def load_masked_weights(filename):
    # **YUCK** older scripts output 3D rather than 2D weight data
    data = numpy.load(filename)
    if len(data.shape) == 3:
        data = data[0]

    weight = numpy.ma.array(data, fill_value=0.0)
    return numpy.ma.fix_invalid(weight)

def display_raw_weights(masked_weights, figure, axis):
    image = axis.imshow(masked_weights, cmap="jet", interpolation="none")
    axis.set_xlabel("Post neuron ID")
    axis.set_ylabel("Pre neuron ID")
    figure.colorbar(image)

def display_mean_weights(masked_weights, figure, axis, num_mcu_neurons, palette):
    # Calculate number of minicolumns
    num_minicolumns = masked_weights.shape[0] // num_mcu_neurons
    mean_weights = numpy.zeros((num_minicolumns, num_minicolumns))
    for mi, mj in itertools.product(range(num_minicolumns), repeat=2):
        slice_i = slice(mi, masked_weights.shape[0], num_minicolumns)
        slice_j = slice(mj, masked_weights.shape[0], num_minicolumns)
        sub_weights = masked_weights[slice_i, slice_j]

        mean_weights[mi, mj] = numpy.ma.mean(sub_weights)

    image = axis.imshow(mean_weights, cmap=palette, interpolation="none")
    return image

def display_single_attractor(masked_weights, mi, axis, num_mcu_neurons):
    # Calculate number of minicolumns
    num_minicolumns = masked_weights.shape[0] / num_mcu_neurons

    slice_i = slice(mi, masked_weights.shape[0], num_minicolumns)

    mean_weights = [numpy.ma.mean(masked_weights[slice_i, slice(mj, masked_weights.shape[0], num_minicolumns)])
                    for mj in range(num_minicolumns)]

    axis.plot(mean_weights, marker="x")
    axis.axhline(linestyle="--", color="gray")
    axis.axvline(mi, linestyle="--", color="gray")
    axis.set_xlabel("Post minicolumn ID")
    axis.set_ylabel("Mean weight")

def combine_connection_weights(filenames):
    # Initialise combined weights to nan
    combined_masked_weights = numpy.ma.empty((NE, NE), fill_value=0.0)
    combined_masked_weights[:] = numpy.ma.masked

    # Loop through all connectors, 256
    for f in filenames:
        # Load weights
        masked_weights = load_masked_weights(f)
        combined_masked_weights = numpy.ma.array(combined_masked_weights.data + masked_weights.data,
                                                 mask=numpy.logical_and(combined_masked_weights.mask, masked_weights.mask),
                                                 fill_value=0.0)
    return combined_masked_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine together weights recorded from multiple HCUs and display")
    parser.add_argument("--num_hcus", type=int, default=9, help="How many HCUs is data for")
    parser.add_argument("--num_mcu_neurons", type=int, default=100, help="How many neurons make up an MCU")
    parser.add_argument("--selected_attractor", type=int, default=7, help="Which attractor to plot")
    parser.add_argument("folder", nargs=1, help="Folder to search for weight files in")
    parser.add_argument("filename", nargs=1, help="Filenames of weight files are of the form connection_X_YYYY.npy where filename specified YYYY")
    args = parser.parse_args()

    figure, axes = pylab.subplots(1, 2)

    # Build filenames from command line
    filenames = ["%s/connection_%u_%s.npy" % (args.folder[0], i, args.filename[0]) for i in range(args.num_hcus ** 2)]
    combined_masked_weights = combine_connection_weights(filenames)

    # Display weights
    display_single_attractor(combined_masked_weights, args.selected_attractor, axes[0], args.num_mcu_neurons)
    mean_image = display_mean_weights(combined_masked_weights, figure, axes[1], args.num_mcu_neurons, "jet")
    axes[1].set_xlabel("Post-synaptic attractor number")
    axes[1].set_ylabel("Pre-synaptic attractor number")

    figure.colorbar(mean_image, shrink=0.75)
    #display_raw_weights(combined_masked_weights, figure, axes[2])
    pylab.show()
