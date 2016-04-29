#pragma once

// Model includes
#include "neuron_processor/input_buffer.h"
#include "neuron_processor/neuron_models/if_curr.h"
#include "neuron_processor/synapse_models/exp.h"

// BCPNN includes
#include "../../bcpnn_intrinsic.h"

namespace NeuronProcessor
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef NeuronModels::IFCurr Neuron;
typedef SynapseModels::Exp Synapse;
typedef BCPNN::BCPNNIntrinsic<9, 13, 6> IntrinsicPlasticity;

typedef InputBufferBase<uint32_t> InputBuffer;
};