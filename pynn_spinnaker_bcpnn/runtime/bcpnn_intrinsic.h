#pragma once

// Common includes
#include "common/fixed_point_number.h"

// BCPNN includes
#include "bcpnn.h"
#include "ln_lut.h"

// Namespaces
using namespace Common::FixedPointNumber;

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
    // Calculate new trace values
    // **TODO** could we store these scaled by Aj?
    const int32_t newZjStar = __smulbb(m_ZjStarDecay, m_Traces[neuron].m_Word) >> TraceFixedPoint;
    const int32_t newPjStar = __smulbt(m_PjStarDecay, m_Traces[neuron].m_Word) >> TraceFixedPoint;
    m_Traces[neuron] = Trace(newZjStar, newPjStar);

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\tZj*:%d, Pj*:%d",
              newZjStar, newPjStar);

    // If bias is enabled, calculate it and return
    if(m_BiasEnabled)
    {
      return CalculateBias(newZjStar, newPjStar);
    }
    // Otherwise, return 0
    else
    {
      return 0;
    }
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
    // If neuron has spiked, add Aj to Zj* and Pj* traces
    if(spiked)
    {
      m_Traces[neuron].m_HalfWords[0] += m_Aj;
      m_Traces[neuron].m_HalfWords[1] += m_Aj;
    }
  }

  bool ReadSDRAMData(uint32_t *region, uint32_t, unsigned int numNeurons)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "BCPNN::BCPNNIntrinsic::ReadSDRAMData");

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

    // Copy plasticity region data from region
    m_Aj = *reinterpret_cast<int32_t*>(region++);
    m_Phi = *reinterpret_cast<int32_t*>(region++);
    m_Epsilon = *reinterpret_cast<int32_t*>(region++);
    m_ZjStarDecay = *reinterpret_cast<int32_t*>(region++);
    m_PjStarDecay = *reinterpret_cast<int32_t*>(region++);
    m_BiasEnabled = *region++;

    LOG_PRINT(LOG_LEVEL_INFO, "\tAj:%d, Phi:%k, Epsilon:%d, Zj* decay:%d, Pj* decay:%d, Bias enabled:%u",
              m_Aj, m_Phi, m_Epsilon, m_ZjStarDecay, m_PjStarDecay, m_BiasEnabled);

    // Copy LUTs from subsequent memory
    m_LnLUT.ReadSDRAMData(region);

    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Private methods
  //-----------------------------------------------------------------------------
  S1615 CalculateBias(int32_t zjStar, int32_t pjStar) const
  {
    // From these calculate Pj
    const int32_t pj = zjStar - pjStar;

    // Calculate log
    const int32_t logPj = m_LnLUT.Get(pj + m_Epsilon);

    // Multiply by phi to scale into S1615
    const S1615 bias = Mul<int32_t, int32_t, TraceFixedPoint>(logPj, m_Phi);

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\tPj:%d, log(Pj):%d, bias:%k",
              pj, logPj, bias);

    return bias;
  }

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Postsynaptic scaling factor
  int32_t m_Aj;

  // Intrinsic bias multiplier
  S1615 m_Phi;

  // Epsilon for eliminating ln(0)
  int32_t m_Epsilon;

  // 1 timestep of decay for Zj* and Pj*
  int32_t m_ZjStarDecay;
  int32_t m_PjStarDecay;

  // Should bias actually be returned
  uint32_t m_BiasEnabled;

  // Per neuron traces
  Trace *m_Traces;

  // Natural log lookup table
  LnLUT<LnLUTShift, TraceFixedPoint> m_LnLUT;
};
} // BCPNN