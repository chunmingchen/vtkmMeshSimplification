//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __vtkm_math_VectorAnalysis_h
#define __vtkm_math_VectorAnalysis_h

// This header file defines math functions that deal with linear albegra funcitons

#include <vtkm/Types.h>
#include "math/Basic.h"

namespace vtkm {
namespace math {


// ----------------------------------------------------------------------------
/// \brief Returns the linear interpolation of two scalar or vector values based on weight
///
/// lerp interpolates return the linerar interpolation of x and y based on w.  x
/// and y are Scalars or vectors of same lenght. w can either be a scalar or a
/// vector of the same lenght as x and y. If w is outside [0,1] then lerp
/// exterpolates. If w=0 => a is returned if w=1 => b is returned.
///
template <typename T>
VTKM_EXEC_CONT_EXPORT T Lerp(const T &x, const T &y, const vtkm::FloatDefault w) {
 return x + w * (y-x);
}


// ----------------------------------------------------------------------------
template <typename T, vtkm::IdComponent N>
VTKM_EXEC_CONT_EXPORT
vtkm::FloatDefault Norm2(const vtkm::Vec<T, N> &x ) {
    vtkm::FloatDefault t = 0;
    for (vtkm::IdComponent i = 0; i<N; i++)
        t += x[i]* (vtkm::FloatDefault)x[i];
    return Sqrt( t );
}

// ----------------------------------------------------------------------------
/// \brief Changes a vector to be normal.
///
/// The given vector is scaled to be unit length.
///
template <typename T, vtkm::IdComponent N>
VTKM_EXEC_CONT_EXPORT
void Normalize(vtkm::Vec<T, N> &x) {
    return x * ((vtkm::FloatDefault)1. /Norm2(x));
}

// ----------------------------------------------------------------------------
/// \brief Find the cross product of two vectors.
///
template <typename T>
VTKM_EXEC_CONT_EXPORT
vtkm::Vec<T, 3> Cross(const vtkm::Vec<T, 3> &x, const vtkm::Vec<T, 3> &y)
{
  return vtkm::make_Vec<T>(x[1]*y[2] - x[2]*y[1],
                           x[2]*y[0] - x[0]*y[2],
                           x[0]*y[1] - x[1]*y[0]);
}

//-----------------------------------------------------------------------------
/// \brief Find the normal of a triangle.
///
/// Given three coordinates in space, which, unless degenerate, uniquely define
/// a triangle and the plane the triangle is on, returns a vector perpendicular
/// to that triangle/plane.
///
template <typename T>
VTKM_EXEC_CONT_EXPORT
vtkm::Vec<T, 3> TriangleNormal(const vtkm::Vec<T, 3> &a,
                                            const vtkm::Vec<T, 3> &b,
                                            const vtkm::Vec<T, 3> &c)
{
  return vtkm::math::Cross(b-a, c-a);
}


}} // namespace vtkm::math

#endif //__vtkm_math_VectorAnalysis_h
