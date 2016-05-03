#pragma once

// Common includes
#include "common/fixed_point_number.h"
#include "common/utils.h"

// BCPNN includes
#include "bcpnn.h"
#include "ln_lut.h"

// Namespaces
using namespace Common::FixedPointNumber;
using namespace Common::Utils;

//-----------------------------------------------------------------------------
// BCPNN::BCPNNIntrinsic
//-----------------------------------------------------------------------------
namespace BCPNN
{
template<unsigned int TraceFixedPoint, unsigned int LnLUTShift>
class BCPNNIntrinsic
{
private:
  //-----------------------------------------------------------------------------
  // Typedefines
  //-----------------------------------------------------------------------------
  typedef Pair Trace;

  //-----------------------------------------------------------------------------
  // Enumerations
  //-----------------------------------------------------------------------------
  enum Mode
  {
    ModeBiasEnabled       = (1 << 0),
    ModePlasticityEnabled = (1 << 1),
  };

public:
  //-----------------------------------------------------------------------------
  // Enumerations
  //-----------------------------------------------------------------------------
  enum RecordingChannel
  {
    RecordingChannelBias,
    RecordingChannelMax,
  };

  BCPNNIntrinsic() : m_Traces(NULL)
  {
  }

  //-----------------------------------------------------------------------------
  // Public methods
  //-----------------------------------------------------------------------------
  S1615 GetIntrinsicCurrent(unsigned int neuron)
  {
    // If plasticity is enabled
    if((m_Mode & ModePlasticityEnabled) != 0)
    {
      // Calculate new trace values
      const int32_t newZjStar = ShiftDownRound(__smulbb(m_ZjStarDecay, m_Traces[neuron].m_Word));
      const int32_t newPjStar = ShiftDownRound(__smulbt(m_PjStarDecay, m_Traces[neuron].m_Word));
      m_Traces[neuron] = Trace(newZjStar, newPjStar);

      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tZj*:%d, Pj*:%d",
                newZjStar, newPjStar);

      // If bias is enabled, return newly calculated bias
      if((m_Mode & ModeBiasEnabled) != 0)
      {
        return CalculateBias(newZjStar, newPjStar);
      }
    }
    // Otherwise, if bias is enabled, return current bias
    else if((m_Mode & ModeBiasEnabled) != 0)
    {
      return CalculateBias(m_Traces[neuron].m_HalfWords[0],
                           m_Traces[neuron].m_HalfWords[1]);
    }

    // No bias should be provideds
    return 0;
  }

  S1615 GetRecordable(RecordingChannel channel, unsigned int neuron) const
  {
    if(channel == RecordingChannelBias)
    {
      // Calculate bias and return
      return CalculateBias(m_Traces[neuron].m_HalfWords[0], m_Traces[neuron].m_HalfWords[1]);
    }
    else
    {
      LOG_PRINT(LOG_LEVEL_WARN, "Attempting to get data from non-existant recording channel %u",
                channel);
      return 0;
    }
  }

  void ApplySpike(unsigned int neuron, bool spiked)
  {
    // If neuron has spiked and plasticity is enabled, add Aj to Zj* and Pj* traces
    if(spiked && (m_Mode & ModePlasticityEnabled) != 0)
    {
      m_Traces[neuron].m_HalfWords[0] += m_MinusAj;
      m_Traces[neuron].m_HalfWords[1] += m_MinusAj;
    }
  }

  bool ReadSDRAMData(uint32_t *region, uint32_t, unsigned int numNeurons)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "BCPNN::BCPNNIntrinsic::ReadSDRAMData");

    // Copy plasticity region data from region
    m_MinusAj = *reinterpret_cast<int32_t*>(region++);
    m_Phi = *reinterpret_cast<int32_t*>(region++);
    m_Epsilon = *reinterpret_cast<int32_t*>(region++);
    m_ZjStarDecay = *reinterpret_cast<int32_t*>(region++);
    m_PjStarDecay = *reinterpret_cast<int32_t*>(region++);
    m_Mode = *region++;

    LOG_PRINT(LOG_LEVEL_INFO, "\tAj:%d, Phi:%k, Epsilon:%d, Zj* decay:%d, Pj* decay:%d, Mode:%08x",
              -m_MinusAj, m_Phi, m_Epsilon, m_ZjStarDecay, m_PjStarDecay, m_Mode);

    // Copy LUTs from subsequent memory
    m_LnLUT.ReadSDRAMData(region);

    // Allocate a trace for each neuron
    m_Traces = (Trace*)spin1_malloc(numNeurons * sizeof(Trace));
    if(m_Traces == NULL)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate intrinsic plasticity traces");
      return false;
    }

    // Initially zero traces
    for(unsigned int i = 0; i < numNeurons; i++)
    {
      m_Traces[i].m_Word = 0;
    }
    
    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Private methods
  //-----------------------------------------------------------------------------
  S1615 CalculateBias(int32_t zjStar, int32_t pjStar) const
  {
    // From these calculate Pj
    // **NOTE** because both PJ* and Zj* are multiplied by -Aj this is negated
    const int32_t pj = pjStar - zjStar;

    // Calculate log
    const int32_t logPj = m_LnLUT.Get(pj + m_Epsilon);

    // Multiply by phi to scale into S1615
    const S1615 bias = Mul<int32_t, int32_t, TraceFixedPoint>(logPj, m_Phi);

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\tPj:%d, log(Pj):%d, bias:%k",
              pj, logPj, bias);

    return bias;
  }

  static int32_t ShiftDownRound(int32_t valueShifted)
  {
    const int32_t value = valueShifted >> (TraceFixedPoint - 1);
    return (value >> 1) + (value & 0x1);
  }

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Postsynaptic scaling factor
  int32_t m_MinusAj;

  // Intrinsic bias multiplier
  S1615 m_Phi;

  // Epsilon for eliminating ln(0)
  int32_t m_Epsilon;

  // 1 timestep of decay for Zj* and Pj*
  int32_t m_ZjStarDecay;
  int32_t m_PjStarDecay;

  // What mode should intrinsic plasticity operate in
  uint32_t m_Mode;

  // Per neuron traces
  Trace *m_Traces;

  // Natural log lookup table
  LnLUT<LnLUTShift, TraceFixedPoint> m_LnLUT;
};
} // BCPNN