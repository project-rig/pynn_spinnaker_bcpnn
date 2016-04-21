#pragma once

// Standard includes
#include <cstdint>
#include <cstring>

// Common includes
#include "../../common/exp_decay_lut.h"
#include "../../common/log.h"

// Synapse processor includes
#include "../plasticity/post_events.h"

// BCPNN includes
#include "ln_lut.h"

//-----------------------------------------------------------------------------
// BCPNN::SynapseTypes::BCPNN
//-----------------------------------------------------------------------------
namespace BCPNN
{
template<typename C, unsigned int D, unsigned int I, unsigned int T>
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
  typedef Plasticity::PostEventHistory<PostTrace, T> PostEventHistory;

  //-----------------------------------------------------------------------------
  // Constants
  //-----------------------------------------------------------------------------
  static const unsigned int PreTraceWords = (sizeof(PreTrace) / 4) + (((sizeof(PreTrace) % 4) == 0) ? 0 : 1);
  static const uint32_t DelayMask = ((1 << D) - 1);
  static const uint32_t IndexMask = ((1 << I) - 1);

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
    dmaBuffer[3] = tick;

    // Get time of last presynaptic spike and associated trace entry from DMA buffer
    const uint32_t lastPreTick = dmaBuffer[4];
    const PreTrace lastPreTrace = GetPreTrace(dmaBuffer);

    // If this is an actual spike (rather than a flush event)
    PreTrace newPreTrace;
    if(!flush)
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tAdding pre-synaptic event to trace at tick:%u",
                tick);
      // Calculate new pre-trace
      newPreTrace = UpdateTrace(tick, lastPreTrace, lastPreTick, m_TauZiLUT);
      
      // Write back updated last presynaptic spike time and trace to row
      dmaBuffer[4] = tick;
      SetPreTrace(dmaBuffer, newPreTrace);
    }

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
      const uint32_t delayedLastPreTick = lastPreTick + delayAxonal;
      uint32_t delayedLastUpdateTick = lastUpdateTick + delayAxonal;

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
      while (postWindow.GetNumEvents() > 0)
      {
        const uint32_t delayedPostTick = postWindow.GetNextTime() + delayDendritic;

        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tApplying post-synaptic event at delayed tick:%u",
                  delayedPostTick);

        // Apply post-synaptic spike to Pij*
        /*pIJStar = ApplySpike(delayedPostTick, postWindow.GetNextTrace(),
                             delayedLastPreTick, lastPreTrace,
                             postWindow.GetPrevTime(), postWindow.GetPrevTrace());
        */
        pIJStar = ApplySpike(delayedPostTick,
                             delayedLastUpdateTick, pIJStar,
                             delayedLastPreTick, lastPreTrace,
                             m_TauZiLUT);

        // Update
        delayedLastUpdateTick = delayedPostTick;

        // Go onto next event
        postWindow.Next(delayedPostTick);
      }

      // If this isn't a flush
      if(!flush)
      {
        const uint32_t delayedPreTick = tick + delayAxonal;
        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tApplying pre-synaptic event at tick:%u, last post tick:%u",
                  delayedPreTick, postWindow.GetPrevTime());

        // Apply pre-synaptic spike to Pij*
        pIJStar = ApplySpike(delayedPreTick,
                             delayedLastUpdateTick, pIJStar,
                             postWindow.GetPrevTime(), postWindow.GetPrevTrace(),
                             m_TauZjLUT);
      }

      // Calculate final state after all updates
      auto finalState = updateState.CalculateFinalState(m_WeightDependence);

      // If this isn't a flush, add weight to ring-buffer
      if(!flush)
      {
        applyInputFunction(delayDendritic + delayAxonal + tick,
          postIndex, finalState.GetWeight());
      }

      // Write back updated synaptic word to plastic region
      *plasticWords++ = finalState.GetPlasticSynapse();
    }

    // Write back row and all plastic data to SDRAM
    writeBackRowFunction(&sdramRowAddress[3], &dmaBuffer[3],
      2 + PreTraceWords + GetNumPlasticWords(dmaBuffer[0]));
    return true;
  }

  void AddPostSynapticSpike(uint tick, unsigned int neuronID)
  {
    // If neuron ID is valid
    if(neuronID < 512)
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "Adding post-synaptic event to trace at tick:%u",
                tick);

      // Get neuron's post history
      auto &postHistory = m_PostEventHistory[neuronID];

      // Update last trace entry based on spike at tick
      // and add new trace and time to post history
      PostTrace trace = UpdateTrace(tick, postHistory.GetLastTrace(),
                                    postHistory.GetLastTime(), m_TauZjLUT);
      postHistory.Add(tick, trace);
    }
  }

  unsigned int GetRowWords(unsigned int rowSynapses) const
  {
    // Three header word and a synapse
    return 5 + PreTraceWords + GetNumPlasticWords(rowSynapses) + GetNumControlWords(rowSynapses);
  }

  bool ReadSDRAMData(uint32_t *region, uint32_t flags)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "SynapseTypes::STDP::ReadSDRAMData");

    // Read timing dependence data
    if(!m_TimingDependence.ReadSDRAMData(region, flags))
    {
      return false;
    }

    // Read weight dependence data
    if(!m_WeightDependence.ReadSDRAMData(region, flags))
    {
      return false;
    }

    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Private methods
  //-----------------------------------------------------------------------------
  int32_t GetP(Trace trace, int32_t a)
  {
    // Extrace components from trace and scale by A
    int32_t zStar = __smulbb(a, trace.m_Word);
    int32_t pStar = __smulbt(a, trace.m_Word);

    // Now, if only we could subtract with __smlabt :)
    return (zStar - pStar) >> 9;
  }

  int32_t GetPij(Trace preTrace, Trace postTrace, int32_t pijStar)
  {
    int32_t correlation = Mul16<9>(preTrace.m_Word, postTrace.m_Word);

    return STAR_FIXED_MUL_32X32(m_Aij, correlation - pijStar);
  }

  template<typename TauZLUT>
  Trace UpdateTrace(uint32_t tick, Trace lastTrace, uint32_t lastTick, const TauZLUT &tauZLut)
  {
    // Get time since last spike
    uint32_t elapsedTicks = tick - lastTick;

    // Lookup exponential decay over delta-time with both time constants
    int32_t zStarDecay =  tauZLut.Get(elapsedTicks);
    int32_t pStarDecay = m_TauPLUT(elapsedTicks);

    // Calculate new trace values
    int32_t newZStar = __smulbb(zStarDecay, lastTrace.m_Word) >> 9;
    int32_t newPStar = __smulbt(pStarDecay, lastTrace.m_Word) >> 9;

    // Add energy caused by new spike to trace
    newZStar += S229One;
    newPStar += S229One;

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\tElapsed ticks:%u, Z*:%d, P*:%d, Z* decay:%d, P* decay:%d, New Z*:%d, New P*:%d\n",
      elapsedTicks, lastTrace.m_HalfWords[0], lastTrace.m_HalfWords[1],
      zStarDecay, pStarDecay, newZStar, newPStar);

    // Combine Z* and P* values into trace and return
    return Trace(newZStar, newPStar);
  }

  template<typename TauZLUT>
  int32_t ApplySpike(uint32_t tick,
                     uint32_t lastUpdateTick, int32_t lastPIJStar
                     uint32_t lastOtherTick, Trace lastOtherTrace, /*, bool flush, */
                     const TauZLUT &otherTauZLut)
  {
    // Get time since last update of Pij*
    uint32_t elapsedTicks = tick - lastUpdateTick;

    // Decay Pij*
    int32_t pStarDecay = m_TauPLUT.Get(elapsedTicks);
    int32_t newPIJStar = __smulbb(lastPIJStar, pStarDecay);

    //if(!flush)
    //{
      uint32_t otherTraceElapsedTicks = tick - lastOtherTick;
      int32_t otherTraceZStarDecay = otherTauZLut.Get(otherTraceElapsedTicks);

      newPIJStar = __smlabb(lastOtherTrace, otherTraceZStarDecay, newPIJStar);
    //}
    newPIJStar >>= STAR_FIXED_POINT;

    log_debug("\t\t\tApplying deferred spike - New Pij*:%d", newPIJStar);

    // Build new trace structure and return
    return newPIJStar;
  }

  PlasticSynapse CalculateFinalState(int32_t pIJStar, uint32_t lastUpdateTick,
    uint32_t lastPreTick, Trace lastPreTrace,
    uint32_t lastPostTick, Trace lastPostTrace)
  {
    log_debug("\tlast_correlation_time:%u, last_pre_time:%u, last_post_time:%u", last_correlation_time, last_pre_time, last_post_time);

    // Last correlation will have occured at last pre time so decay last post trace to this time
    correlation_trace_t finalPostTrace = bcpnn_update_trace(last_correlation_time,
      last_post_time, last_post_trace,
      ZJ_LUT_INPUT_SHIFT, ZJ_LUT_SIZE, bcpnn_zj_lut);

    // Convert final ei*, ej* and eij* traces into final ei, ej and eij values
    int32_t finalPi = GetP(lastPreTrace, m_Ai);
    int32_t finalPj = GetP(lastPostTrace, m_Aj);
    int32_t finalPij = GetPij(lastPreTrace, finalPostTrace, pIJStar);

    // Take logs of the correlation trace and the product of the other traces
    int32_t logPij = m_LnLUT.Get(finalPij + m_EpsilonSquared);
    int32_t logPiPj = bcpnn_ln(Mul16<13>(finalPi + m_Epsilon,
                                       finalPj + m_Epsilon));

    // Calculate bayesian weight (using log identities to remove divide)
    int32_t weight = logPij - logPiPj;

    // Scale weight into ring-buffer format
    int32_t weightScaled = Mul16<13>(weight, m_MaxWeight);

    log_debug("\t\tzi_star:%d, zj_star:%d, ei_star:%d, ej_star:%d, eij_star:%d, ei:%d, ej:%d, eij:%d, log(ei * ej):%d, log(eij):%d, weight:%d, weight_scaled:%d",
      bcpnn_trace_get_z_star(last_pre_trace), bcpnn_trace_get_z_star(final_post_trace), bcpnn_trace_get_z_star(last_pre_trace), bcpnn_trace_get_z_star(final_post_trace), new_state.eij_star,
      final_ei, final_ej, final_eij, log_ei_ej, log_eij, weight, weight_scaled);

  #ifdef GENERATE_CSV
    io_printf(IO_BUF, "%u,,,,,,%d,%d,%d,%d,%d,,\n", last_correlation_time, new_state.eij_star, final_ei, final_ej, final_eij, weight);
  #endif  // GENERATE_CSV

    // Return final state containing new eligibility value and weight
    return Pair(weightScaled, pIJStar);
  }
  //-----------------------------------------------------------------------------
  // Private static methods
  //-----------------------------------------------------------------------------
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
    memcpy(&preTrace, &dmaBuffer[5], sizeof(PreTrace));
    return preTrace;
  }

  static void SetPreTrace(uint32_t (&dmaBuffer)[MaxRowWords], PreTrace preTrace)
  {
    // **NOTE** GCC will optimise this memcpy out it
    // is simply strict-aliasing-safe solution
    memcpy(&dmaBuffer[5], &preTrace, sizeof(PreTrace));
  }

  static PlasticSynapse *GetPlasticWords(uint32_t (&dmaBuffer)[MaxRowWords])
  {
    return reinterpret_cast<PlasticSynapse*>(&dmaBuffer[5 + PreTraceWords]);
  }

  static const C *GetControlWords(uint32_t (&dmaBuffer)[MaxRowWords], unsigned int numSynapses)
  {
    return reinterpret_cast<C*>(&dmaBuffer[5 + PreTraceWords + GetNumPlasticWords(numSynapses)]);
  }

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Ai;
  int32_t m_Aj;
  int32_t m_Aij;

  int32_t m_Epsilon;
  int32_t m_EpsilonSquared;

  int32_t m_PHI;

  uint32_t m_Mode;

  int32_t m_MaxWeight;

  PostEventHistory m_PostEventHistory[512];

  Common::ExpDecayLUT<TauPlusLUTNumEntries, TauPlusLUTShift> m_TauZiLUT;
  Common::ExpDecayLUT<TauPlusLUTNumEntries, TauPlusLUTShift> m_TauZjLUT;
  Common::ExpDecayLUT<TauPlusLUTNumEntries, TauPlusLUTShift> m_TauPLUT;

  LnLUT<TauPlusLUTNumEntries, TauPlusLUTShift, 13> m_LnLUT;
};
} // SynapseTypes
} // SynapseProcessor