#ifndef _BASIC_H
#define _BASIC_H

#include <limits>
#include <vtkm/Types.h>

#define VECTORIZE(Func, func) \
    template <typename T>   \
    VTKM_EXEC_CONT_EXPORT T Func(const T &x) {  \
     return func(x);    \
    }   \
    template <typename T, vtkm::IdComponent N>  \
    VTKM_EXEC_CONT_EXPORT vtkm::Vec<T,N> Func(const vtkm::Vec<T,N> &x) {    \
     vtkm::Vec<T,N> temp;   \
     for(vtkm::IdComponent i =0; i < N; ++i)    \
     { temp[i]=func(x[i]); }    \
     return temp;   \
    }


// ----------------------------------------------------------------------------

namespace vtkm {
namespace math {

VECTORIZE(Sqrt, sqrt)
VECTORIZE(Abs, abs)

/// Returns the difference between 1 and the least value greater than 1
/// that is representable.
///
template<typename T>
VTKM_EXEC_CONT_EXPORT T Epsilon()
{
  return std::numeric_limits<T>::epsilon();
}


}} //vtkm::math
#endif
