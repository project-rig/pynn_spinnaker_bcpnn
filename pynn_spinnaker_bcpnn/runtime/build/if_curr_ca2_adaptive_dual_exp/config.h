#pragma once

// Model includes
#include "neuron_processor/input_buffer.h"

// IF curr ca2 adaptive module includes
#include "ca2_adaptive.h"
#include "dual_exp.h"

#include "../../bcpnn_intrinsic.h"

namespace NeuronProcessor
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef ExtraModels::CA2Adaptive Neuron;
typedef ExtraModels::DualExp Synapse;
typedef BCPNN::BCPNNIntrinsic<13, 6> IntrinsicPlasticity;

typedef InputBufferBase<uint32_t> InputBuffer;
};