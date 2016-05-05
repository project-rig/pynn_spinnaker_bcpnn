#pragma once

// Model includes
#include "neuron_processor/input_buffer.h"
#include "neuron_processor/neuron_models/if_curr.h"

// IF curr dual exp module includes
#include "dual_exp.h"

#include "../../bcpnn_intrinsic.h"

namespace NeuronProcessor
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef NeuronModels::IFCurr Neuron;
typedef ExtraModels::DualExp Synapse;
typedef BCPNN::BCPNNIntrinsic<13, 6> IntrinsicPlasticity;

typedef InputBufferBase<int32_t> InputBuffer;
};