# Import modules
from pynn_spinnaker.spinnaker import lazy_param_map

# Import classes
from pyNN.standardmodels.synapses import StandardSynapseType
from pynn_spinnaker.spinnaker.utils import LazyArrayFloatToFixConverter

# Import functions
from functools import partial

# Create a converter functions to convert from float
# to various fixed-point formats used by BCPNN
float_to_s1813_no_copy = LazyArrayFloatToFixConverter(True, 32, 13, False)
float_to_s69_no_copy = LazyArrayFloatToFixConverter(True, 16, 9, False)

# Fixed-point conversion wrapper for parameter mapping
def s1813(values, **kwargs):
    return float_to_s1813_no_copy(deepcopy(values))

# Generate a LUT of ln(x) for x in (1.0, 2.0]
def s1813_ln_lut(input_shift):
    # Calculate the size of the LUT
    size = (1 << 13) >> input_shift

    # Build a lazy array of x values to calculate log for
    x = la.larray(np.arange(1.0, 2.0, 1.0 / float(size)))

    # Take log and convert to fixed point
    return float_to_s1813_no_copy(la.log(x))

# Partially bound exponent decay LUT generator for S6.9 fixed-point
s69_exp_decay_lut = partial(lazy_param_map.exp_decay_lut,
                            float_to_fixed=float_to_s69_no_copy)

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
        `max_firing_frequency`:
            Firing frequency representing certainty (Hz).
        `phi`:
            Scaling of intrinsic bias current from probability to current domain (nA).
        `w_max`:
            Scaling of weights from probability to current domain (nA/uS).
        `weights_enabled`:
            Are the learnt or pre-loaded weights passed to the ring-buffer.
        `plasticity_enabled`:
            Is plasticity enabled.
        `bias_enabled`:
            Are the learnt biases passed to the neuron.

    .. _`Knight, Tully et al (2016)`: http://journal.frontiersin.org/article/10.3389/fnana.2016.00037/full
    """
    default_parameters = {
        "weight": 0.0,
        "delay": None,
        "tau_zi": 5.0,                # Time constant of presynaptic primary trace (ms)
        "tau_zj": 5.0,                # Time constant of postsynaptic primary trace (ms)
        "tau_p": 1000.0,              # Time constant of probability trace (ms)
        "max_firing_frequency": 20.0, # Firing frequency representing certainty (Hz)
        "phi": 0.05,                  # Scaling of intrinsic bias current from probability to current domain (nA)
        "w_max": 2.0,                 # Scaling of weights from probability to current domain (nA/uS)
        "weights_enabled": True,      # Are the learnt or pre-loaded weights passed to the ring-buffer
        "plasticity_enabled": True,   # Is plasticity enabled
        "bias_enabled": True,         # Are the learnt biases passed to the neuron
    }


    translations = build_translations(
        ("weight", "weight"),
        ("delay",  "delay"),

        ("tau_zi", "tau_zi"),
        ("tau_zj", "tau_zj"),
        ("tau_p",  "tau_p"),

        ("",       "a_i",             "1000.0 / (max_firing_frequency * (tau_zi - tau_p))", ""),
        ("",       "a_j",             "1000.0 / (max_firing_frequency * (tau_zj - tau_p))", ""),
        ("",       "a_ij",            "1000.0 / (max_firing_frequency * (tau_zj - tau_p))", ""), #**FIXME**

        ("",       "epsilon",         "1000.0 / (max_firing_frequency * tau_p)", ""),
        ("",       "epsilon_squared", "(1000.0 / (max_firing_frequency * tau_p)) ** 2", ""),

        ("phi",    "phi"),
        ("w_max",  "w_max"),

        ("",       "mode",            "(1 << 0) if weights_enabled else 0) | ((1 << 1) if plasticity_enabled else 0) | ((1 << 2) if bias_enabled else 0"),
    )

    plasticity_param_map = [
        ("a_i", "i4", s1813),
        ("a_j", "i4", s1813),
        ("a_ij", "i4", s1813),

        ("epsilon", "i4", s1813),
        ("epsilon_squared", "i4", s1813),

        ("phi", "i4", lazy_param_map.s1615),
        ("w_max", "i4", lazy_param_map.s32_weight_fixed_point),

        ("mode", "u4", lazy_param_map.integer),

        ("tau_zi", "128i2", partial(s69_exp_decay_lut,
                                    num_entries=128, time_shift=0)),
        ("tau_zj", "128i2", partial(s69_exp_decay_lut,
                                    num_entries=128, time_shift=0)),
        ("tau_p", "1136i2", partial(s69_exp_decay_lut,
                                    num_entries=1136, time_shift=3)),
        (s1813_ln_lut(6), "128i2"),
    ]

    comparable_param_names = ("tau_zi", "tau_zj", "tau_p", "max_firing_frequency",
                              "phi", "w_max", "weights_enabled", "plasticity_enabled", "bias_enabled")

    # How many post-synaptic neurons per core can a
    # SpiNNaker synapse_processor of this type handle
    max_post_neurons_per_core = 256

    # Assuming relatively long row length, at what rate can a SpiNNaker
    # synapse_processor of this type process synaptic events (hZ)
    max_synaptic_event_rate = 1E6

    synaptic_matrix_region_class = regions.StaticSynapticMatrix

    # How many timesteps of delay can DTCM ring-buffer handle
    # **NOTE** only 7 timesteps worth of delay can be handled by
    # 8 element delay buffer - The last element is purely for output
    max_dtcm_delay_slots = 7

    # Static synapses don't require post-synaptic
    # spikes back-propagated to them
    requires_back_propagation = True

    # Pre trace consists of two 16-bit traces: Zi and Pi
    pre_trace_bytes = 4

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == "auto":
            d = state.dt
        return d