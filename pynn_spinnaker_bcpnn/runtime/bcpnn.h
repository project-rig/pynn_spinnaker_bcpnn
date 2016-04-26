#pragma once

// Standard includes
#include <cstdint>
#include <cstring>

// Common includes
#include "common/disable_interrupts.h"
#include "common/exp_decay_lut.h"
#include "common/log.h"

// Synapse processor includes
#include "synapse_processor/plasticity/post_events.h"

// BCPNN includes
#include "ln_lut.h"

// Namespaces
using namespace Common;
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// BCPNN::SynapseTypes::BCPNN
//-----------------------------------------------------------------------------
namespace BCPNN
{
template<typename C, unsigned int D, unsigned int I,
  unsigned int StarFixedPoint, unsigned int TraceFixedPoint,
  unsigned int TauZiLUTNumEntries, unsigned int TauZiLUTShift,
  unsigned int TauZjLUTNumEntries, unsigned int TauZjLUTShift,
  unsigned int TauPLUTNumEntries, unsigned int TauPLUTShift,
  unsigned int LnLUTShift,
  unsigned int T>
class BCPNN
{
private:
  //-----------------------------------------------------------------------------
  // Unions
  //-----------------------------------------------------------------------------
  union Pair
  {
    Pair(){}
    Pair(int16_t a, int16_t b) : m_HalfWords{a, b} {}
    Pair(int32_t w) : m_Word(w) {}

    int16_t m_HalfWords[2];
    int32_t m_Word;
  };

  //-----------------------------------------------------------------------------
  // Typedefines
  //-----------------------------------------------------------------------------
  typedef Pair PlasticSynapse;
  typedef Pair Trace;
  typedef Trace PreTrace;
  typedef Trace PostTrace;
  typedef SynapseProcessor::Plasticity::PostEventHistory<PostTrace, T> PostEventHistory;

  //-----------------------------------------------------------------------------
  // Constants
  //-----------------------------------------------------------------------------
  static const unsigned int PreTraceWords = (sizeof(PreTrace) / 4) + (((sizeof(PreTrace) % 4) == 0) ? 0 : 1);
  static const uint32_t DelayMask = ((1 << D) - 1);
  static const uint32_t IndexMask = ((1 << I) - 1);
  static const int32_t StarFixedPointOne = (1 << StarFixedPoint);

public:
  //-----------------------------------------------------------------------------
  // Constants
  //-----------------------------------------------------------------------------
  // One word for a synapse-count, two delay words, a time of last update, 
  // time and trace associated with last presynaptic spike and 512 synapses
  static const unsigned int MaxRowWords = 517 + PreTraceWords;

  //-----------------------------------------------------------------------------
  // Public methods
  //-----------------------------------------------------------------------------
  template<typename F, typename E, typename R>
  bool ProcessRow(uint tick, uint32_t (&dmaBuffer)[MaxRowWords], uint32_t *sdramRowAddress, bool flush,
                  F applyInputFunction, E addDelayRowFunction, R writeBackRowFunction)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "\tProcessing BCPNN row with %u synapses at tick:%u (flush:%u)",
              dmaBuffer[0], tick, flush);

    // If this row has a delay extension, call function to add it
    if(dmaBuffer[1] != 0)
    {
      addDelayRowFunction(dmaBuffer[1] + tick, dmaBuffer[2], flush);
    }

    // Get time of last update from DMA buffer and write back updated time
    const uint32_t lastUpdateTick = dmaBuffer[3];
    const PreTrace lastPreTrace = GetPreTrace(dmaBuffer);
    dmaBuffer[3] = tick;

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\tUpdating pre-synaptic trace at tick:%u",
              tick);

    // Calculate new pre-trace
    PreTrace newPreTrace = UpdateTrace(tick, !flush, lastPreTrace, lastUpdateTick, m_TauZiLUT);
    CSVPrint("%u,%u,%d,%d,,,,,,,,,\n",
              tick, flush, newPreTrace.m_HalfWords[0], newPreTrace.m_HalfWords[1]);

    // Write back updated presynaptic trace to row
    SetPreTrace(dmaBuffer, newPreTrace);

    // Extract first plastic and control words; and loop through synapses
    uint32_t count = dmaBuffer[0];
    PlasticSynapse *plasticWords = GetPlasticWords(dmaBuffer);
    const C *controlWords = GetControlWords(dmaBuffer, count);
    for(; count > 0; count--)
    {
      // Get the next control word from the synaptic_row
      // (should autoincrement pointer in single instruction)
      const uint32_t controlWord = *controlWords++;

      // Extract control word components
      const uint32_t delayDendritic = GetDelay(controlWord);
      const uint32_t delayAxonal = 0;
      const uint32_t postIndex = GetIndex(controlWord);

      // Create update state from the Pij* component of plastic word
      int32_t pIJStar = plasticWords->m_HalfWords[1];

      // Apply axonal delay to last presynaptic spike and update tick
      const uint32_t delayedLastUpdateTick = lastUpdateTick + delayAxonal;

      // Get the post-synaptic window of events to be processed
      // **NOTE** this is the window since the last UPDATE rather than the last presynaptic spike
      const uint32_t windowBeginTick = (delayedLastUpdateTick >= delayDendritic) ?
        (delayedLastUpdateTick - delayDendritic) : 0;
      const uint32_t windowEndTick = tick + delayAxonal - delayDendritic;

      // Get post event history within this window
      auto postWindow = m_PostEventHistory[postIndex].GetWindow(windowBeginTick,
                                                                windowEndTick);

      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tPerforming deferred synapse update for post neuron:%u", postIndex);
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tWindow begin tick:%u, window end tick:%u: Previous time:%u, Num events:%u",
          windowBeginTick, windowEndTick, postWindow.GetPrevTime(), postWindow.GetNumEvents());

      // Process events in post-synaptic window
      uint32_t lastCorrelationTime = delayedLastUpdateTick;
      while (postWindow.GetNumEvents() > 0)
      {
        const uint32_t delayedPostTick = postWindow.GetNextTime() + delayDendritic;

        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tApplying post-synaptic event at delayed tick:%u",
                  delayedPostTick);

        // Update correlation based on post-spike
        pIJStar = UpdateCorrelation(delayedPostTick, true,
                                    lastCorrelationTime, pIJStar,
                                    delayedLastUpdateTick, lastPreTrace,
                                    m_TauZiLUT);

        // Update last correlation time
        lastCorrelationTime = delayedPostTick;

        // Go onto next event
        postWindow.Next(delayedPostTick);
      }

      const uint32_t delayedPreTick = tick + delayAxonal;
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tApplying pre-synaptic event at tick:%u, last post tick:%u",
                delayedPreTick, postWindow.GetPrevTime());

      // Update correlation based on pre-spike/flush
      pIJStar = UpdateCorrelation(delayedPreTick, !flush,
                                  lastCorrelationTime, pIJStar,
                                  postWindow.GetPrevTime(), postWindow.GetPrevTrace(),
                                  m_TauZjLUT);

      // Calculate final state after all updates
      PlasticSynapse finalState = CalculateFinalState(pIJStar,
                                                      delayedPreTick, newPreTrace,
                                                      postWindow.GetPrevTime(), postWindow.GetPrevTrace());

      // If this isn't a flush, add weight to ring-buffer
      /*if(!flush)
      {
        applyInputFunction(delayDendritic + delayAxonal + tick,
          postIndex, finalState.m_HalfWords[0]);
      }*/

      // Write back updated synaptic word to plastic region
      *plasticWords++ = finalState.m_Word;
    }

    // Write back row and all plastic data to SDRAM
    writeBackRowFunction(&sdramRowAddress[3], &dmaBuffer[3],
      1 + PreTraceWords + GetNumPlasticWords(dmaBuffer[0]));
    return true;
  }

  void AddPostSynapticSpike(uint tick, unsigned int neuronID)
  {
    // If neuron ID is valid
    if(neuronID < 256)
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "Updating postsynaptic trace at tick:%u",
                tick);

      // Get neuron's post history
      auto &postHistory = m_PostEventHistory[neuronID];

      // Update last trace entry based on spike at tick
      // and add new trace and time to post history
      PostTrace trace = UpdateTrace(tick, true, postHistory.GetLastTrace(),
                                    postHistory.GetLastTime(), m_TauZjLUT);
      CSVPrintDisableIRQ("%u,,,,%d,%d,,,,,,,,\n",
                         tick, trace.m_HalfWords[0], trace.m_HalfWords[1]);
      postHistory.Add(tick, trace);
    }
  }

  unsigned int GetRowWords(unsigned int rowSynapses) const
  {
    // Four header word and a synapse
    return 4 + PreTraceWords + GetNumPlasticWords(rowSynapses) + GetNumControlWords(rowSynapses);
  }

  bool ReadSDRAMData(uint32_t *region, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "BCPNN::BCPNN::ReadSDRAMData");

    // Copy plasticity region data from region
    m_Ai = *reinterpret_cast<int32_t*>(region++);
    m_Aj = *reinterpret_cast<int32_t*>(region++);
    m_Aij = *reinterpret_cast<int32_t*>(region++);

    m_Epsilon = *reinterpret_cast<int32_t*>(region++);
    m_EpsilonSquared = *reinterpret_cast<int32_t*>(region++);

    m_PHI = *reinterpret_cast<int32_t*>(region++);

    m_MaxWeight = *reinterpret_cast<int32_t*>(region++);

    m_Mode = *region++;

    LOG_PRINT(LOG_LEVEL_INFO, "\tAi:%d, Aj:%d, Aij:%d, epsilon:%d, epsilon squared:%d, phi:%d, max weight:%d, mode:%08x",
              m_Ai, m_Aj, m_Aij, m_Epsilon, m_EpsilonSquared, m_PHI, m_MaxWeight, m_Mode);

    // Copy LUTs from subsequent memory
    m_TauZiLUT.ReadSDRAMData(region);
    m_TauZjLUT.ReadSDRAMData(region);
    m_TauPLUT.ReadSDRAMData(region);
    m_LnLUT.ReadSDRAMData(region);

    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Private methods
  //-----------------------------------------------------------------------------
  int32_t GetP(Trace trace, int32_t a)
  {
    // Extrace components from trace and scale by A
    const int32_t zStar = __smulbb(a, trace.m_Word);
    const int32_t pStar = __smulbt(a, trace.m_Word);

    // Now, if only we could subtract with __smlabt :)
    return (zStar - pStar) >> StarFixedPoint;
  }

  int32_t GetPij(Trace preTrace, Trace postTrace, int32_t pijStar)
  {
    const int32_t correlation = Mul16<9>(preTrace.m_Word, postTrace.m_Word);

    return StarMul32(m_Aij, correlation - pijStar);
  }

  template<typename TauZLUT>
  Trace UpdateTrace(uint32_t tick, bool spike,
                    Trace lastTrace, uint32_t lastTick,
                    const TauZLUT &tauZLut)
  {
    // Get time since last spike
    const uint32_t elapsedTicks = tick - lastTick;

    // Lookup exponential decay over delta-time with both time constants
    const int32_t zStarDecay =  tauZLut.Get(elapsedTicks);
    const int32_t pStarDecay = m_TauPLUT.Get(elapsedTicks);

    // Calculate new trace values
    int32_t newZStar = __smulbb(zStarDecay, lastTrace.m_Word) >> StarFixedPoint;
    int32_t newPStar = __smulbt(pStarDecay, lastTrace.m_Word) >> StarFixedPoint;

    // Add energy caused by new spike to trace
    if(spike)
    {
      newZStar += StarFixedPointOne;
      newPStar += StarFixedPointOne;
    }

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tElapsed ticks:%u, Z*:%d, P*:%d, Z* decay:%d, P* decay:%d, New Z*:%d, New P*:%d\n",
      elapsedTicks, lastTrace.m_HalfWords[0], lastTrace.m_HalfWords[1],
      zStarDecay, pStarDecay, newZStar, newPStar);

    // Combine Z* and P* values into trace and return
    return Trace(newZStar, newPStar);
  }

  template<typename TauZLUT>
  int32_t UpdateCorrelation(uint32_t tick, bool spike,
                            uint32_t lastUpdateTick, int32_t lastPIJStar,
                            uint32_t lastOtherTick, Trace lastOtherTrace,
                            const TauZLUT &otherTauZLut)
  {
    // Get time since last update of Pij*
    const uint32_t elapsedTicks = tick - lastUpdateTick;

    // Decay Pij*
    const int32_t pStarDecay = m_TauPLUT.Get(elapsedTicks);
    int32_t newPIJStar = __smulbb(lastPIJStar, pStarDecay);

    if(spike)
    {
      uint32_t otherTraceElapsedTicks = tick - lastOtherTick;
      int32_t otherTraceZStarDecay = otherTauZLut.Get(otherTraceElapsedTicks);

      LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tOther trace elapsed ticks:%u, other trace Z* decay::%d",
              otherTraceElapsedTicks, otherTraceZStarDecay);

      newPIJStar = __smlabb(lastOtherTrace.m_Word, otherTraceZStarDecay, newPIJStar);
    }
    newPIJStar >>= StarFixedPoint;

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tElapsed ticks:%u, Pij*:%d, Pij* decay:%d, New Pij*:%d",
              elapsedTicks, lastPIJStar, pStarDecay, newPIJStar);

    // Build new trace structure and return
    return newPIJStar;
  }

  PlasticSynapse CalculateFinalState(int32_t pIJStar,
    uint32_t lastPreTick, Trace lastPreTrace,
    uint32_t lastPostTick, Trace lastPostTrace)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tLast pre tick:%u, last post tick:%u",
              lastPreTick, lastPostTick);

    // Last correlation will have occured at last pre time so decay last post trace to this time
    Trace finalPostTrace = UpdateTrace(lastPreTick, false,
                                       lastPostTrace, lastPostTick,
                                       m_TauZjLUT);

    // Convert final Pi*, Pj* and Pij* traces into final Pi, Pj and Pij values
    const int32_t finalPi = GetP(lastPreTrace, m_Ai);
    const int32_t finalPj = GetP(finalPostTrace, m_Aj);
    const int32_t finalPij = GetPij(lastPreTrace, finalPostTrace, pIJStar);

    // Take logs of the correlation trace and the product of the other traces
    const int32_t logPij = m_LnLUT.Get(finalPij + m_EpsilonSquared);
    const int32_t logPiPj = m_LnLUT.Get(TraceMul16(finalPi + m_Epsilon,
                                                   finalPj + m_Epsilon));

    // Calculate bayesian weight (using log identities to remove divide)
    const int32_t weight = logPij - logPiPj;

    // Scale weight into ring-buffer format
    const int32_t weightScaled = TraceMul16(weight, m_MaxWeight);

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tZi*:%d, Zj*:%d, Pi*:%d, Pj*:%d, Pij*:%d, Pi:%d, Pj:%d, Pij:%d, log(Pi * Pj):%d, log(Pij):%d, weight:%d, weight scaled:%d",
      lastPreTrace.m_HalfWords[0], finalPostTrace.m_HalfWords[0], lastPreTrace.m_HalfWords[1], finalPostTrace.m_HalfWords[1], pIJStar,
      finalPi, finalPj, finalPij, logPiPj, logPij, weight, weightScaled);

    CSVPrint("%u,,,,,,%d,%d,%d,%d,%d,,\n",
              lastPreTick, pIJStar, finalPi, finalPj, finalPij, weight);

    // Return final state containing new eligibility value and weight
    return Pair(weightScaled, pIJStar);
  }

  //-----------------------------------------------------------------------------
  // Private static methods
  //-----------------------------------------------------------------------------
  template<typename... Args>
  static void CSVPrint(char *fmt, Args... args )
  {
#ifdef GENERATE_CSV
    io_printf(IO_BUF, fmt, args...);
#endif  // GENERATE_CSV
  }

  template<typename... Args>
  static void CSVPrintDisableIRQ(char *fmt, Args... args )
  {
#ifdef GENERATE_CSV
    DisableIRQ d;
    io_printf(IO_BUF, fmt, args...);
#endif  // GENERATE_CSV
  }

  static int32_t StarMul16(int32_t a, int32_t b)
  {
    return Mul16<StarFixedPoint>(a, b);
  }

  static int32_t StarMul32(int32_t a, int32_t b)
  {
    return Mul<int32_t, int32_t, StarFixedPoint>(a, b);
  }

  static int32_t TraceMul16(int32_t a, int32_t b)
  {
    return Mul16<TraceFixedPoint>(a, b);
  }

  static uint32_t GetIndex(uint32_t word)
  {
    return (word & IndexMask);
  }

  static uint32_t GetDelay(uint32_t word)
  {
    return ((word >> I) & DelayMask);
  }

  static unsigned int GetNumPlasticWords(unsigned int numSynapses)
  {
    const unsigned int plasticBytes = numSynapses * sizeof(PlasticSynapse);
    return (plasticBytes / 4) + (((plasticBytes % 4) == 0) ? 0 : 1);
  }

  static unsigned int GetNumControlWords(unsigned int numSynapses)
  {
    const unsigned int controlBytes = numSynapses * sizeof(C);
    return (controlBytes / 4) + (((controlBytes % 4) == 0) ? 0 : 1);
  }

  static PreTrace GetPreTrace(uint32_t (&dmaBuffer)[MaxRowWords])
  {
    // **NOTE** GCC will optimise this memcpy out it
    // is simply strict-aliasing-safe solution
    PreTrace preTrace;
    memcpy(&preTrace, &dmaBuffer[4], sizeof(PreTrace));
    return preTrace;
  }

  static void SetPreTrace(uint32_t (&dmaBuffer)[MaxRowWords], PreTrace preTrace)
  {
    // **NOTE** GCC will optimise this memcpy out it
    // is simply strict-aliasing-safe solution
    memcpy(&dmaBuffer[4], &preTrace, sizeof(PreTrace));
  }

  static PlasticSynapse *GetPlasticWords(uint32_t (&dmaBuffer)[MaxRowWords])
  {
    return reinterpret_cast<PlasticSynapse*>(&dmaBuffer[4 + PreTraceWords]);
  }

  static const C *GetControlWords(uint32_t (&dmaBuffer)[MaxRowWords], unsigned int numSynapses)
  {
    return reinterpret_cast<C*>(&dmaBuffer[4 + PreTraceWords + GetNumPlasticWords(numSynapses)]);
  }

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Ai;
  int32_t m_Aj;
  int32_t m_Aij;

  // Epsilon values for eliminating ln(0)
  int32_t m_Epsilon;
  int32_t m_EpsilonSquared;

  // Intrinsic bias multiplier
  int32_t m_PHI;

  // Weight multiplier
  int32_t m_MaxWeight;

  // Operating modes for toggling training etc
  uint32_t m_Mode;

  // Event history
  PostEventHistory m_PostEventHistory[256];

  // Exponential lookup tables
  Common::ExpDecayLUT<TauZiLUTNumEntries, TauZiLUTShift> m_TauZiLUT;
  Common::ExpDecayLUT<TauZjLUTNumEntries, TauZjLUTShift> m_TauZjLUT;
  Common::ExpDecayLUT<TauPLUTNumEntries, TauPLUTShift> m_TauPLUT;

  // Natural log lookup table
  LnLUT<LnLUTShift, TraceFixedPoint> m_LnLUT;
};
} // BCPNN