#pragma once

// Common includes
#include "common/spike_input_buffer.h"
namespace SynapseProcessor
{
  typedef Common::SpikeInputBufferBase<1024> SpikeInputBuffer;
}

#include "common/key_lookup_binary_search.h"
namespace SynapseProcessor
{
  typedef Common::KeyLookupBinarySearch<10> KeyLookup;
}

// BCPNN synapses using 16-bit control words with 3 delay bits and 10 index bits;
// previously configured timing dependence, weight dependence and synapse structure;
// and a post-synaptic event history with 10 entries
#include "../../bcpnn_synapse.h"
namespace SynapseProcessor
{
  typedef BCPNN::BCPNNSynapse<int16_t, 3, 10,
                              9,        // Fixed point position for Z* and P*
                              13,       // Fixed point format for Z and P
                              262, 2,   // Exponential decay LUT config for TauZi
                              262, 2,   // Exponential decay LUT config for TauZj
                              1136, 4,  // Exponential decay LUT config for TauP
                              6, 10> SynapseType;
}

// Ring buffer with 32-bit signed entries, large enough for 256 neurons
#include "synapse_processor/ring_buffer.h"
namespace SynapseProcessor
{
  typedef RingBufferBase<int32_t, 3, 8> RingBuffer;
}

#include "synapse_processor/delay_buffer.h"
namespace SynapseProcessor
{
  typedef DelayBufferBase<10> DelayBuffer;
}
