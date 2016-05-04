# Import modules
import lazyarray as la
import math
import numpy as np
from pynn_spinnaker.spinnaker import lazy_param_map
from pynn_spinnaker.spinnaker import regions

# Import classes
from pyNN.standardmodels.cells import StandardCellType
from pyNN.standardmodels.synapses import StandardSynapseType
from pynn_spinnaker.spinnaker.utils import LazyArrayFloatToFixConverter
from pynn_spinnaker.standardmodels.cells import IF_curr_exp
from pynn_spinnaker_if_curr_dual_exp import IF_curr_dual_exp
from pynn_spinnaker_if_curr_ca2_adaptive import IF_curr_ca2_adaptive_exp

# Import functions
from copy import deepcopy
from functools import partial
from pyNN.standardmodels import build_translations
from pynn_spinnaker.spinnaker.utils import get_homogeneous_param

# Import globals
from pynn_spinnaker.simulator import state
from pynn_spinnaker_if_curr_ca2_adaptive import (
    if_curr_ca2_adaptive_neuron_translations,
    if_curr_ca2_adaptive_neuron_immutable_param_map,
    if_curr_ca2_adaptive_neuron_mutable_param_map)
from pynn_spinnaker_if_curr_dual_exp import (
    dual_exp_synapse_translations,
    dual_exp_synapse_immutable_param_map,
    dual_exp_synapse_curr_mutable_param_map)

# Create a converter functions to convert from float
# to various fixed-point formats used by BCPNN
float_to_s1813_no_copy = LazyArrayFloatToFixConverter(True, 32, 13, False)
float_to_s69_no_copy = LazyArrayFloatToFixConverter(True, 16, 9, False)

# Fixed-point conversion wrapper for parameter mapping
def s1813(values, **kwargs):
    return float_to_s1813_no_copy(deepcopy(values))

# Generate a LUT of ln(x) for x in (1.0, 2.0]
def ln_lut(input_shift, float_to_fixed):
    # Calculate the size of the LUT
    size = (1 << float_to_fixed.n_frac) >> input_shift

    # Build a lazy array of x values to calculate log for
    x = la.larray(np.arange(1.0, 2.0, 1.0 / float(size)))

    # Take log and convert to fixed point
    return float_to_fixed(la.log(x))

# Partially bound exponent decay LUT generator for S6.9 fixed-point
s69_exp_decay_lut = partial(lazy_param_map.exp_decay_lut,
                            float_to_fixed=float_to_s69_no_copy)
s1813_exp_decay = partial(lazy_param_map.exp_decay,
                          float_to_fixed=float_to_s1813_no_copy)
s1813_ln_lut = partial(ln_lut, float_to_fixed=float_to_s1813_no_copy)

# ----------------------------------------------------------------------------
# Intrinsic plasticity default parameters
# ----------------------------------------------------------------------------
intrinsic_plasticity_default_parameters = {
    "tau_z": 5.0,               # Time constant of primary trace (ms)
    "tau_p": 1000.0,            # Time constant of probability trace (ms)
    "f_max": 20.0,              # Firing frequency representing certainty (Hz)
    "phi": 0.05,                # Scaling of intrinsic bias current from probability to current domain (nA)
    "bias_enabled": True,       # Are the learnt biases passed to the neuron
    "plasticity_enabled": True  # Is plasticity enabled
}

# ----------------------------------------------------------------------------
# Intrinsic plasticity translations
# ----------------------------------------------------------------------------
intrinsic_plasticity_translations = build_translations(
    ("tau_z",               "tau_z"),
    ("tau_p",               "tau_p"),

    ("f_max",               "minus_a",  "1000.0 / (f_max * (tau_p - tau_z))", ""),
    ("phi",                 "phi"),
    ("bias_enabled",        "epsilon",  "1000.0 / (f_max * tau_p)", ""),

    ("plasticity_enabled",  "mode",     "bias_enabled + (plasticity_enabled * 2)", "")
)

# ----------------------------------------------------------------------------
# Intrinsic plasticity region map
# ----------------------------------------------------------------------------
intrinsic_plasticity_param_map = [
    ("minus_a", "i4", s1813),
    ("phi", "i4", lazy_param_map.s1615),
    ("epsilon", "i4", s1813),
    ("tau_z", "i4", s1813_exp_decay),
    ("tau_p", "i4", s1813_exp_decay),
    ("mode", "u4", lazy_param_map.integer),
    (s1813_ln_lut(6), "128i2"),
]

# ------------------------------------------------------------------------------
# IF_curr_exp
# ------------------------------------------------------------------------------
class IF_curr_exp(IF_curr_exp):
    # Update translations to handle intrinsic plasticity parameters
    translations = deepcopy(IF_curr_exp.translations)
    translations.update(intrinsic_plasticity_translations)

    # Update default parameters to handle intrinsic plasticity parameters
    default_parameters = deepcopy(IF_curr_exp.default_parameters)
    default_parameters.update(intrinsic_plasticity_default_parameters)

    # Set intrinsic plasticity parameter map
    intrinsic_plasticity_param_map = intrinsic_plasticity_param_map

    # Add bias to list of recordables
    recordable = IF_curr_exp.recordable + ["bias"]

    # Add units for bias
    units = deepcopy(IF_curr_exp.units)
    units["bias"] = "nA"

# ------------------------------------------------------------------------------
# IF_curr_dual_exp
# ------------------------------------------------------------------------------
class IF_curr_dual_exp(IF_curr_dual_exp):
    # Update translations to handle intrinsic plasticity parameters
    translations = deepcopy(IF_curr_dual_exp.translations)
    translations.update(intrinsic_plasticity_translations)

    # Update default parameters to handle intrinsic plasticity parameters
    default_parameters = deepcopy(IF_curr_dual_exp.default_parameters)
    default_parameters.update(intrinsic_plasticity_default_parameters)

    # Set intrinsic plasticity parameter map
    intrinsic_plasticity_param_map = intrinsic_plasticity_param_map

    # Add bias to list of recordables
    recordable = IF_curr_dual_exp.recordable + ["bias"]

    # Add units for bias
    units = deepcopy(IF_curr_dual_exp.units)
    units["bias"] = "nA"

# ------------------------------------------------------------------------------
# IF_curr_ca2_adaptive_dual_exp
# ------------------------------------------------------------------------------
class IF_curr_ca2_adaptive_dual_exp(StandardCellType):
    default_parameters = {
        "v_rest"     : -65.0,   # Resting membrane potential in mV.
        "cm"         : 1.0,     # Capacity of the membrane in nF
        "tau_m"      : 20.0,    # Membrane time constant in ms.
        "tau_refrac" : 0.1,     # Duration of refractory period in ms.
        "tau_ca2"    : 50.0,    # Time contant of CA2 adaption current in ms
        "tau_syn_E"  : 5.0,     # Decay time of excitatory synaptic current in ms.
        "tau_syn_E2" : 5.0,    # Decay time of second excitatory synaptic current in ms.
        "tau_syn_I"  : 5.0,     # Decay time of inhibitory synaptic current in ms.
        "i_offset"   : 0.0,     # Offset current in nA
        "i_alpha"    : 0.1,     # Influx of CA2 caused by each spike in nA
        "v_reset"    : -65.0,   # Reset potential after a spike in mV.
        "v_thresh"   : -50.0,   # Spike threshold in mV.
    }
    default_parameters.update(intrinsic_plasticity_default_parameters)
    
    recordable = ["spikes", "v", "bias"]
    conductance_based = False
    default_initial_values = {
        "v": -65.0,  # 'v_rest',
        "isyn_exc": 0.0,
        "isyn_exc2": 0.0,
        "isyn_inh": 0.0,
        "i_ca2": 0.0,
    }
    units = {
        "v": "mV",
        "isyn_exc": "nA",
        "isyn_exc2": "nA",
        "isyn_inh": "nA",
        "i_ca2": "nA",
        "bias": "nA",
    }
    receptor_types = ("excitatory", "inhibitory", "excitatory2")

    # How many of these neurons per core can
    # a SpiNNaker neuron processor handle
    max_neurons_per_core = 1024

    neuron_region_class = regions.Neuron

    directly_connectable = False

    translations = deepcopy(if_curr_ca2_adaptive_neuron_translations)
    translations.update(dual_exp_synapse_translations)
    translations.update(intrinsic_plasticity_translations)

    neuron_immutable_param_map = if_curr_ca2_adaptive_neuron_immutable_param_map
    neuron_mutable_param_map = if_curr_ca2_adaptive_neuron_mutable_param_map

    synapse_immutable_param_map = dual_exp_synapse_immutable_param_map
    synapse_mutable_param_map = dual_exp_synapse_curr_mutable_param_map

    # Set intrinsic plasticity parameter map
    intrinsic_plasticity_param_map = intrinsic_plasticity_param_map

# ------------------------------------------------------------------------------
# BCPNNSynapse
# ------------------------------------------------------------------------------
class BCPNNSynapse(StandardSynapseType):
    """
    BCPNN synapse

    Arguments:
        `tau_zi`:
            Time constant of presynaptic primary trace (ms).
        `tau_zj`:
            Time constant of postsynaptic primary trace (ms).
        `tau_p`:
            Time constant of probability trace (ms).
        `f_max`:
            Firing frequency representing certainty (Hz).
        `w_max`:
            Scaling of weights from probability to current domain (nA/uS).
        `weights_enabled`:
            Are the learnt or pre-loaded weights passed to the ring-buffer.
        `plasticity_enabled`:
            Is plasticity enabled.

    .. _`Knight, Tully et al (2016)`: http://journal.frontiersin.org/article/10.3389/fnana.2016.00037/full
    """
    default_parameters = {
        "weight": 0.0,
        "delay": None,
        "tau_zi": 5.0,              # Time constant of presynaptic primary trace (ms)
        "tau_zj": 5.0,              # Time constant of postsynaptic primary trace (ms)
        "tau_p": 1000.0,            # Time constant of probability trace (ms)
        "f_max": 20.0,              # Firing frequency representing certainty (Hz)
        "w_max": 2.0,               # Scaling of weights from probability to current domain (nA/uS)
        "weights_enabled": True,    # Are the learnt or pre-loaded weights passed to the ring-buffer
        "plasticity_enabled": True, # Is plasticity enabled

        # **YUCK** translation requires the same number of PyNN parameters
        # as native parameters so these make up the numbers
        "_placeholder1": None,
        "_placeholder2": None,
        "_placeholder3": None,
    }


    translations = build_translations(
        ("weight",              "weight"),
        ("delay",               "delay"),

        ("tau_zi",              "tau_zi"),
        ("tau_zj",              "tau_zj"),
        ("tau_p",               "tau_p"),

        ("f_max",               "a_i",              "1000.0 / (f_max * (tau_zi - tau_p))", ""),
        ("weights_enabled",     "a_j",              "1000.0 / (f_max * (tau_zj - tau_p))", ""),
        ("plasticity_enabled",  "a_ij",             "(1000000.0 / (tau_zi + tau_zj)) / ((f_max ** 2) * ((1.0 / ((1.0 / tau_zi) + (1.0 / tau_zj))) - tau_p))", ""),

        ("_placeholder1",       "epsilon",          "1000.0 / (f_max * tau_p)", ""),
        ("_placeholder2",       "epsilon_squared",  "(1000.0 / (f_max * tau_p)) ** 2", ""),

        ("w_max",               "w_max"),

        ("_placeholder3",       "mode",             "weights_enabled + (plasticity_enabled * 2)", ""),
    )

    plasticity_param_map = [
        ("a_i", "i4", s1813),
        ("a_j", "i4", s1813),
        ("a_ij", "i4", s1813),

        ("epsilon", "i4", s1813),
        ("epsilon_squared", "i4", s1813),

        ("w_max", "i4", lazy_param_map.s32_weight_fixed_point),

        ("mode", "u4", lazy_param_map.integer),

        ("tau_zi", "262i2", partial(s69_exp_decay_lut,
                                    num_entries=262, time_shift=2)),
        ("tau_zj", "262i2", partial(s69_exp_decay_lut,
                                    num_entries=262, time_shift=2)),
        ("tau_p", "1136i2", partial(s69_exp_decay_lut,
                                    num_entries=1136, time_shift=4)),
        (s1813_ln_lut(6), "128i2"),
    ]

    comparable_param_names = ("tau_zi", "tau_zj", "tau_p", "f_max", "w_max",
                              "weights_enabled", "plasticity_enabled")

    # How many post-synaptic neurons per core can a
    # SpiNNaker synapse_processor of this type handle
    max_post_neurons_per_core = 256

    # Assuming relatively long row length, at what rate can a SpiNNaker
    # synapse_processor of this type process synaptic events (hZ)
    max_synaptic_event_rate = 1E6

    # BCPNN requires a synaptic matrix region
    # with support for extra per-synapse data
    synaptic_matrix_region_class = regions.ExtendedPlasticSynapticMatrix

    # How many timesteps of delay can DTCM ring-buffer handle
    # **NOTE** only 7 timesteps worth of delay can be handled by
    # 8 element delay buffer - The last element is purely for output
    max_dtcm_delay_slots = 7

    # Static weights are unsigned
    signed_weight = True

    # BCPNN synapses require post-synaptic
    # spikes back-propagated to them
    requires_back_propagation = True

    # Presynaptic state consists of a uint32 containing
    # time of last update and an int16 for Zi and Pi
    pre_state_bytes = 8

    # Each synape has an additional 16-bit trace: Pij
    synapse_trace_bytes = 2

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == "auto":
            d = state.dt
        return d

    def update_weight_range(self, weight_range):
        # If plasticity is enabled, maximum weight can be calculated with
        #             Pij
        # w_max * ln(-----)
        #             PiPj
        #
        # Therefore, maximum value is:
        #
        #                 1.0
        # w_max * ln(-------------)
        #             Epsilon ^ 2
        if get_homogeneous_param(self.parameter_space, "plasticity_enabled"):
            # Read parameters from parameter space
            f_max = get_homogeneous_param(self.parameter_space, "f_max")
            tau_p = get_homogeneous_param(self.parameter_space, "tau_p")
            w_max = get_homogeneous_param(self.parameter_space, "w_max")

            # Calculate epsilon and hence maximum weight
            # **HACK** double to take into account signedness of BCPNN weights
            epsilon = 1000.0 / (f_max * tau_p)
            weight_range.update(2.0 * w_max * math.log(1.0 / (epsilon ** 2)))