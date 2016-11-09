#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/spinnaker.h"

//-----------------------------------------------------------------------------
// BCPNN::LnLUT
//-----------------------------------------------------------------------------
namespace BCPNN
{
template<unsigned int Shift, unsigned int FixedPoint>
class LnLUT
{
public:
  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  void ReadSDRAMData(uint32_t *&inputPointer)
  {
     // Pad to number of words
    const unsigned int numWords = (NumEntries / 2) + (((NumEntries & 1) != 0) ? 1 : 0);

    // Copy entries to LUT
    spin1_memcpy(m_LUT, inputPointer, sizeof(int16_t) * NumEntries);

    // Advance word-aligned input pointer
    inputPointer += numWords;
  }

  int Get(int x) const
  {
    // Use CLZ to get integer log2 of x
    int integerLog2 = 31 - __builtin_clz(x);

    // Use this to extract fractional part (should be in range of fixed-point (1.0, 2.0)
    int fractionalPart = (x << FixedPoint) >> integerLog2;

    // Convert this to LUT index and thus get log2 of fractional part
    int fractionalPartLookupIndex = (fractionalPart - FixedPointOne) >> Shift;
    int fractionalPartLn = m_LUT[fractionalPartLookupIndex];

    // Scale the integer log2 to fixed point and multiply to FixedPoint integer natural log
    int integerPartLn = __smulbb(integerLog2 - FixedPoint, Log2ToNaturalLogConvert);

    // Add the two logs together and return
    return fractionalPartLn + integerPartLn;
  }

private:
  //-----------------------------------------------------------------------------
  // Constants
  //-----------------------------------------------------------------------------
  // What is 1.0 in desired fixed point format
  static const int FixedPointOne = (1 << FixedPoint);

  // LUT always represents 1.0 - 2.0 so this is reflected in number of entries
  static const unsigned int NumEntries = FixedPointOne >> Shift;

  // **NOTE** could micro-optimise and use a number that can be represented as an ARM literal
  static const int Log2ToNaturalLogConvert = 5678;// (int)roundf((float)TRACE_FIXED_POINT_ONE / log2f(expf(1.0f)));

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int16_t m_LUT[NumEntries];
};
} // Common