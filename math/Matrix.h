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
#ifndef __vtkm_math_Matrix_h
#define __vtkm_math_Matrix_h

#include <vtkm/Types.h>
#include <vtkm/TypeTraits.h>
#include <vtkm/VecTraits.h>
#include "math/Basic.h"

//#include <vtkm/math/Precision.h>
//#include <vtkm/math/Sign.h>

namespace vtkm {
namespace math {

// Making non-square matricies may be overkill.

// If matricies are really useful, they may be promoted to vtkm/Types.h (and
// the vtkm namespace).

/// Basic Matrix type.
///
template<typename T, int NumRow, int NumCol>
class Matrix {
public:
  typedef T ComponentType;
  static const int NUM_ROWS = NumRow;
  static const int NUM_COLUMNS = NumCol;

  VTKM_EXEC_CONT_EXPORT Matrix() { }
  VTKM_EXEC_CONT_EXPORT explicit Matrix(const ComponentType &value)
    : Components(vtkm::Vec<ComponentType, NUM_COLUMNS>(value)) { }

  /// Brackets are used to reference a matrix like a 2D array (i.e.
  /// matrix[row][column]).
  VTKM_EXEC_CONT_EXPORT
  const vtkm::Vec<ComponentType, NUM_COLUMNS> &operator[](int rowIndex) const {
    return this->Components[rowIndex];
  }
  /// Brackets are used to referens a matrix like a 2D array i.e.
  /// matrix[row][column].
  VTKM_EXEC_CONT_EXPORT
  vtkm::Vec<ComponentType, NUM_COLUMNS> &operator[](int rowIndex) {
    return this->Components[rowIndex];
  }

  /// Parentheses are used to reference a matrix using mathematical tuple
  /// notation i.e. matrix(row,column).
  VTKM_EXEC_CONT_EXPORT
  const ComponentType &operator()(int rowIndex, int colIndex) const {
    return (*this)[rowIndex][colIndex];
  }
  /// Parentheses are used to reference a matrix using mathematical tuple
  /// notation i.e. matrix(row,column).
  VTKM_EXEC_CONT_EXPORT
  ComponentType &operator()(int rowIndex, int colIndex) {
    return (*this)[rowIndex][colIndex];
  }

private:
  vtkm::Vec<vtkm::Vec<ComponentType, NUM_COLUMNS>, NUM_ROWS> Components;
};

/// A common square matrix.
///
template <typename T>
class Matrix2x2 : public Matrix<T, 2, 2> {
  typedef Matrix<T, 2, 2> Superclass;
public:
  VTKM_EXEC_CONT_EXPORT Matrix2x2() {  }
  VTKM_EXEC_CONT_EXPORT explicit Matrix2x2(const T &value)
    : Superclass(value) {  }
  VTKM_EXEC_CONT_EXPORT Matrix2x2(const Matrix<T, 2, 2> &values)
    : Superclass(values) {  }
};

/// A common square matrix.
///
template <typename T>
class Matrix3x3 : public Matrix<T, 3, 3> {
  typedef Matrix<T, 3, 3> Superclass;
public:
  VTKM_EXEC_CONT_EXPORT Matrix3x3() {  }
  VTKM_EXEC_CONT_EXPORT explicit Matrix3x3(const T &value)
    : Superclass(value) {  }
  VTKM_EXEC_CONT_EXPORT Matrix3x3(const Matrix<T, 3, 3> &values)
    : Superclass(values) {  }
};

/// A common square matrix.
///
template <typename T>
class Matrix4x4 : public Matrix<T, 4, 4> {
  typedef Matrix<T, 4, 4> Superclass;
public:
  VTKM_EXEC_CONT_EXPORT Matrix4x4() {  }
  VTKM_EXEC_CONT_EXPORT explicit Matrix4x4(const T &value)
    : Superclass(value) {  }
  VTKM_EXEC_CONT_EXPORT Matrix4x4(const Matrix<T, 4, 4> &values)
    : Superclass(values) {  }
};

/// Returns a tuple containing the given row (indexed from 0) of the given
/// matrix.
///
template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT const vtkm::Vec<T, NumCol> &MatrixRow(
    const vtkm::math::Matrix<T,NumRow,NumCol> &matrix, int rowIndex)
{
  return matrix[rowIndex];
}

/// Returns a tuple containing the given column (indexed from 0) of the given
/// matrix.  Might not be as efficient as the Row function.
///
template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT vtkm::Vec<T, NumRow> MatrixColumn(
    const vtkm::math::Matrix<T,NumRow,NumCol> &matrix, int columnIndex)
{
  vtkm::Vec<T, NumRow> columnValues;
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    columnValues[rowIndex] = matrix(rowIndex, columnIndex);
    }
  return columnValues;
}

/// Convenience function for setting a row of a matrix.
///
template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT
void MatrixSetRow(vtkm::math::Matrix<T,NumRow,NumCol> &matrix,
                  int rowIndex,
                  vtkm::Vec<T,NumCol> rowValues)
{
  matrix[rowIndex] = rowValues;
}

/// Convenience function for setting a column of a matrix.
///
template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT
void MatrixSetColumn(vtkm::math::Matrix<T,NumRow,NumCol> &matrix,
                     int columnIndex,
                     vtkm::Vec<T,NumRow> columnValues)
{
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    matrix(rowIndex, columnIndex) = columnValues[rowIndex];
    }
}

/// Standard matrix multiplication.
///
template<typename T, int NumRow, int NumCol, int NumInternal>
VTKM_EXEC_CONT_EXPORT
vtkm::math::Matrix<T,NumRow,NumCol> MatrixMultiply(
    const vtkm::math::Matrix<T,NumRow,NumInternal> &leftFactor,
    const vtkm::math::Matrix<T,NumInternal,NumCol> &rightFactor)
{
  vtkm::math::Matrix<T,NumRow,NumCol> result;
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    for (int colIndex = 0; colIndex < NumCol; colIndex++)
      {
      T sum = leftFactor(rowIndex, 0) * rightFactor(0, colIndex);
      for (int internalIndex = 1; internalIndex < NumInternal; internalIndex++)
        {
        sum += leftFactor(rowIndex, internalIndex)
            * rightFactor(internalIndex, colIndex);
        }
      result(rowIndex, colIndex) = sum;
      }
    }
  return result;
}

/// Standard matrix-vector multiplication.
///
template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT
vtkm::Vec<T,NumRow> MatrixMultiply(
    const vtkm::math::Matrix<T,NumRow,NumCol> &leftFactor,
    const vtkm::Vec<T,NumCol> &rightFactor)
{
  vtkm::Vec<T,NumRow> product;
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    product[rowIndex] =
        vtkm::dot(vtkm::math::MatrixRow(leftFactor,rowIndex), rightFactor);
    }
  return product;
}

/// Standard vector-matrix multiplication
///
template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT
vtkm::Vec<T,NumCol> MatrixMultiply(
    const vtkm::Vec<T,NumRow> &leftFactor,
    const vtkm::math::Matrix<T,NumRow,NumCol> &rightFactor)
{
  vtkm::Vec<T,NumCol> product;
  for (int colIndex = 0; colIndex < NumCol; colIndex++)
    {
    product[colIndex] =
        vtkm::dot(leftFactor,
                 vtkm::math::MatrixColumn(rightFactor, colIndex));
    }
  return product;
}

/// Returns the identity matrix.
///
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
vtkm::math::Matrix<T,Size,Size> MatrixIdentity()
{
  vtkm::math::Matrix<T,Size,Size> result(0);
  for (int index = 0; index < Size; index++)
    {
    result(index,index) = 1.0;
    }
  return result;
}

/// Fills the given matrix with the identity matrix.
///
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
void MatrixIdentity(vtkm::math::Matrix<T,Size,Size> &matrix)
{
  matrix = vtkm::math::MatrixIdentity<T,Size>();
}

/// Returns the transpose of the given matrix.
///
template<typename T, int NumRows, int NumCols>
VTKM_EXEC_CONT_EXPORT
vtkm::math::Matrix<T,NumCols,NumRows> MatrixTranspose(
    const vtkm::math::Matrix<T,NumRows,NumCols> &matrix)
{
  vtkm::math::Matrix<T,NumCols,NumRows> result;
  for (int index = 0; index < NumRows; index++)
    {
    vtkm::math::MatrixSetColumn(result, index, matrix[index]);
    }
  return result;
}


namespace detail {

// Used with MatrixLUPFactor.
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
void MatrixLUPFactorFindPivot(vtkm::math::Matrix<T,Size,Size> &A,
                              vtkm::Vec<int,Size> &permutation,
                              int topCornerIndex,
                              T &inversionParity,
                              bool &valid)
{
  int maxRowIndex = topCornerIndex;
  T maxValue = vtkm::math::Abs(A(maxRowIndex, topCornerIndex));
  for (int rowIndex = topCornerIndex + 1; rowIndex < Size; rowIndex++)
    {
    T compareValue =
        vtkm::math::Abs(A(rowIndex, topCornerIndex));
    if (maxValue < compareValue)
      {
      maxValue = compareValue;
      maxRowIndex = rowIndex;
      }
    }

  if (maxValue < vtkm::math::Epsilon<T>()) { valid = false; }

  if (maxRowIndex != topCornerIndex)
    {
    // Swap rows in matrix.
    vtkm::Vec<T,Size> maxRow =
        vtkm::math::MatrixRow(A, maxRowIndex);
    vtkm::math::MatrixSetRow(A,
                            maxRowIndex,
                            vtkm::math::MatrixRow(A,topCornerIndex));
    vtkm::math::MatrixSetRow(A, topCornerIndex, maxRow);

    // Record change in permutation matrix.
    int maxOriginalRowIndex = permutation[maxRowIndex];
    permutation[maxRowIndex] = permutation[topCornerIndex];
    permutation[topCornerIndex] = maxOriginalRowIndex;

    // Keep track of inversion parity.
    inversionParity = -inversionParity;
    }
}

// Used with MatrixLUPFactor
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
void MatrixLUPFactorFindUpperTriangleElements(
    vtkm::math::Matrix<T,Size,Size> &A,
    int topCornerIndex)
{
  // Compute values for upper triangle on row topCornerIndex
  for (int colIndex = topCornerIndex+1; colIndex < Size; colIndex++)
    {
    A(topCornerIndex,colIndex) /= A(topCornerIndex,topCornerIndex);
    }

  // Update the rest of the matrix for calculations on subsequent rows
  for (int rowIndex = topCornerIndex+1; rowIndex < Size; rowIndex++)
    {
    for (int colIndex = topCornerIndex+1; colIndex < Size; colIndex++)
      {
      A(rowIndex,colIndex) -=
          A(rowIndex,topCornerIndex)*A(topCornerIndex,colIndex);
      }
    }
}

/// Performs an LUP-factorization on the given matrix using Crout's method. The
/// LU-factorization takes a matrix A and decomposes it into a lower triangular
/// matrix L and upper triangular matrix U such that A = LU. The
/// LUP-factorization also allows permutation of A, which makes the
/// decomposition always posible so long as A is not singular. In addition to
/// matrices L and U, LUP also finds permutation matrix P containing all zeros
/// except one 1 per row and column such that PA = LU.
///
/// The result is done in place such that the lower triangular matrix, L, is
/// stored in the lower-left triangle of A including the diagonal. The upper
/// triangular matrix, U, is stored in the upper-right triangle of L not
/// including the diagonal. The diagonal of U in Crout's method is all 1's (and
/// therefore not explicitly stored).
///
/// The permutation matrix P is represented by the permutation vector. If
/// permutation[i] = j then row j in the original matrix A has been moved to
/// row i in the resulting matrices. The permutation matrix P can be
/// represented by a matrix with p_i,j = 1 if permutation[i] = j and 0
/// otherwise. If using LUP-factorization to compute a determinant, you also
/// need to know the parity (whether there is an odd or even amount) of
/// inversions. An inversion is an instance of a smaller number appearing after
/// a larger number in permutation. Although you can compute the inversion
/// parity after the fact, this function keeps track of it with much less
/// compute resources. The parameter inversionParity is set to 1.0 for even
/// parity and -1.0 for odd parity.
///
/// Not all matrices (specifically singular matrices) have an
/// LUP-factorization. If the LUP-factorization succeeds, valid is set to true.
/// Otherwise, valid is set to false and the result is indeterminant.
///
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
void MatrixLUPFactor(vtkm::math::Matrix<T,Size,Size> &A,
                     vtkm::Vec<int,Size> &permutation,
                     T &inversionParity,
                     bool &valid)
{
  // Initialize permutation.
  for (int index = 0; index < Size; index++) { permutation[index] = index; }
  inversionParity = 1;
  valid = true;

  for (int rowIndex = 0; rowIndex < Size; rowIndex++)
    {
    MatrixLUPFactorFindPivot(A, permutation, rowIndex, inversionParity, valid);
    MatrixLUPFactorFindUpperTriangleElements(A, rowIndex);
    }
}

/// Use a previous factorization done with MatrixLUPFactor to solve the
/// system Ax = b.  Instead of A, this method takes in the LU and P
/// matrices calculated by MatrixLUPFactor from A.
///
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
void MatrixLUPSolve(const vtkm::math::Matrix<T,Size,Size> &LU,
                    const vtkm::Vec<int,Size> &permutation,
                    const vtkm::Vec<T,Size> &b,
                    vtkm::Vec<T,Size> &x)
{
  // The LUP-factorization gives us PA = LU or equivalently A = inv(P)LU.
  // Substituting into Ax = b gives us inv(P)LUx = b or LUx = Pb.
  // Now consider the intermediate vector y = Ux.
  // Substituting in the previous two equations yields Ly = Pb.
  // Solving Ly = Pb is easy because L is triangular and P is just a
  // permutation.
  vtkm::Vec<T,Size> y;
  for (int rowIndex = 0; rowIndex < Size; rowIndex++)
    {
    y[rowIndex] = b[permutation[rowIndex]];
    // Recall that L is stored in the lower triangle of LU including diagonal.
    for (int colIndex = 0; colIndex < rowIndex; colIndex++)
      {
      y[rowIndex] -= LU(rowIndex,colIndex)*y[colIndex];
      }
    y[rowIndex] /= LU(rowIndex,rowIndex);
    }

  // Now that we have y, we can easily solve Ux = y for x.
  for (int rowIndex = Size-1; rowIndex >= 0; rowIndex--)
    {
    x[rowIndex] = y[rowIndex];
    // Recall that U is stored in the upper triangle of LU with the diagonal
    // implicitly all 1's.
    for (int colIndex = rowIndex+1; colIndex < Size; colIndex++)
      {
      x[rowIndex] -= LU(rowIndex,colIndex)*x[colIndex];
      }
    }
}

} // namespace detail

/// Solve the linear system Ax = b for x. If a single solution is found, valid
/// is set to true, false otherwise.
///
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
vtkm::Vec<T,Size> SolveLinearSystem(
    const vtkm::math::Matrix<T,Size,Size> &A,
    const vtkm::Vec<T,Size> &b,
    bool &valid)
{
  // First, we will make an LUP-factorization to help us.
  vtkm::math::Matrix<T,Size,Size> LU = A;
  vtkm::Vec<int,Size> permutation;
  T inversionParity;  // Unused.
  vtkm::math::detail::MatrixLUPFactor(LU,
                                           permutation,
                                           inversionParity,
                                           valid);

  // Next, use the decomposition to solve the system.
  vtkm::Vec<T,Size> x;
  vtkm::math::detail::MatrixLUPSolve(LU, permutation, b, x);
  return x;
}

/// Find and return the inverse of the given matrix. If the matrix is singular,
/// the inverse will not be correct and valid will be set to false.
///
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
vtkm::math::Matrix<T,Size,Size> MatrixInverse(
    const vtkm::math::Matrix<T,Size,Size> &A,
    bool &valid)
{
  // First, we will make an LUP-factorization to help us.
  vtkm::math::Matrix<T,Size,Size> LU = A;
  vtkm::Vec<int,Size> permutation;
  T inversionParity;  // Unused
  vtkm::math::detail::MatrixLUPFactor(LU,
                                     permutation,
                                     inversionParity,
                                     valid);

  // We will use the decomposition to solve AX = I for X where X is
  // clearly the inverse of A.  Our solve method only works for vectors,
  // so we solve for one column of invA at a time.
  vtkm::math::Matrix<T,Size,Size> invA;
  vtkm::Vec<T,Size> ICol(0);
  vtkm::Vec<T,Size> invACol;
  for (int colIndex = 0; colIndex < Size; colIndex++)
    {
    ICol[colIndex] = 1;
    vtkm::math::detail::MatrixLUPSolve(LU, permutation, ICol, invACol);
    ICol[colIndex] = 0;
    vtkm::math::MatrixSetColumn(invA, colIndex, invACol);
    }
  return invA;
}

/// Compute the determinant of a matrix.
///
template<typename T, int Size>
VTKM_EXEC_CONT_EXPORT
T MatrixDeterminant(
    const vtkm::math::Matrix<T,Size,Size> &A)
{
  // First, we will make an LUP-factorization to help us.
  vtkm::math::Matrix<T,Size,Size> LU = A;
  vtkm::Vec<int,Size> permutation;
  T inversionParity;
  bool valid;
  vtkm::math::detail::MatrixLUPFactor(LU,
                                     permutation,
                                     inversionParity,
                                     valid);

  // If the matrix is singular, the factorization is invalid, but in that
  // case we know that the determinant is 0.
  if (!valid) { return 0; }

  // The determinant is equal to the product of the diagonal of the L matrix,
  // possibly negated depending on the parity of the inversion. The
  // inversionParity variable is set to 1.0 and -1.0 for even and odd parity,
  // respectively. This sign determines whether the product should be negated.
  T product = inversionParity;
  for (int index = 0; index < Size; index++)
    {
    product *= LU(index,index);
    }
  return product;
}

// Specializations for common small determinants.

template<typename T>
VTKM_EXEC_CONT_EXPORT
T MatrixDeterminant(
    const vtkm::math::Matrix<T,1,1> &A)
{
  return A(0,0);
}

template<typename T>
VTKM_EXEC_CONT_EXPORT
T MatrixDeterminant(
    const vtkm::math::Matrix<T,2,2> &A)
{
  return A(0,0)*A(1,1) - A(1,0)*A(0,1);
}

template<typename T>
VTKM_EXEC_CONT_EXPORT
T MatrixDeterminant(
    const vtkm::math::Matrix<T,3,3> &A)
{
  return A(0,0) * A(1,1) * A(2,2) + A(1,0) * A(2,1) * A(0,2) +
         A(2,0) * A(0,1) * A(1,2) - A(0,0) * A(2,1) * A(1,2) -
         A(1,0) * A(0,1) * A(2,2) - A(2,0) * A(1,1) * A(0,2);
}

}
} // namespace vtkm::math

// Implementations of traits for matrices.
#if 0
namespace vtkm {

/// Tag used to identify 2 dimensional types (matrices). A TypeTraits class
/// will typedef this class to DimensionalityTag.
///
struct TypeTraitsMatrixTag {};

template<typename T, int NumRow, int NumCol>
struct TypeTraits<vtkm::math::Matrix<T, NumRow, NumCol> > {
  typedef typename TypeTraits<T>::NumericTag NumericTag;
  typedef TypeTraitsMatrixTag DimensionalityTag;
};

template<typename T>
struct TypeTraits<vtkm::math::Matrix2x2> {
  typedef TypeTraits<T>::NumericTag NumericTag;
  typedef TypeTraitsMatrixTag DimensionalityTag;
};
template<typename T>
struct TypeTraits<vtkm::math::Matrix3x3> {
  typedef TypeTraits<T>::NumericTag NumericTag;
  typedef TypeTraitsMatrixTag DimensionalityTag;
};
template<typename T>
struct TypeTraits<vtkm::math::Matrix4x4> {
  typedef TypeTraits<T>::NumericTag NumericTag;
  typedef TypeTraitsMatrixTag DimensionalityTag;
};

/// A matrix has vector traits to implement component-wise operations.
///
template<typename T, int NumRow, int NumCol>
struct VectorTraits<vtkm::math::Matrix<T, NumRow, NumCol> > {
private:
  typedef vtkm::math::Matrix<T, NumRow, NumCol> MatrixType;
public:
  typedef T ComponentType;
  static const int NUM_COMPONENTS = NumRow*NumCol;
  typedef vtkm::VectorTraitsTagMultipleComponents HasMultipleComponents;

  VTKM_EXEC_CONT_EXPORT static const ComponentType &GetComponent(
      const MatrixType &matrix, int component) {
    int colIndex = component % NumCol;
    int rowIndex = component / NumCol;
    return matrix(rowIndex,colIndex);
  }
  VTKM_EXEC_CONT_EXPORT static ComponentType &GetComponent(
      MatrixType &matrix, int component) {
    int colIndex = component % NumCol;
    int rowIndex = component / NumCol;
    return matrix(rowIndex,colIndex);
  }
  VTKM_EXEC_CONT_EXPORT static void SetComponent(MatrixType &matrix,
                                           int component,
                                           T value)
  {
    GetComponent(matrix, component) = value;
  }
};

#if 0
template<typename T>
struct VectorTraits<vtkm::math::Matrix2x2>
    : public VectorTraits<vtkm::math::Matrix<T,2,2> > {
  typedef T ComponentType;
  typedef vtkm::VectorTraitsTagMultipleComponents HasMultipleComponents;
};
template<typename T>
struct VectorTraits<vtkm::math::Matrix3x3>
    : public VectorTraits<vtkm::math::Matrix<T,3,3> > {
  typedef T ComponentType;
  typedef vtkm::VectorTraitsTagMultipleComponents HasMultipleComponents;
};
template<typename T>
struct VectorTraits<vtkm::math::Matrix4x4>
    : public VectorTraits<vtkm::math::Matrix<T,4,4> > {
  typedef T ComponentType;
  typedef vtkm::VectorTraitsTagMultipleComponents HasMultipleComponents;
};
#endif

} // namespace vtkm

#endif

// Basic comparison operators.

template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT bool operator==(
    const vtkm::math::Matrix<T,NumRow,NumCol> &a,
    const vtkm::math::Matrix<T,NumRow,NumCol> &b)
{
  for (int colIndex = 0; colIndex < NumCol; colIndex++)
    {
    for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
      {
      if (a(rowIndex, colIndex) != b(rowIndex, colIndex)) return false;
      }
    }
  return true;
}
template<typename T, int NumRow, int NumCol>
VTKM_EXEC_CONT_EXPORT bool operator!=(
    const vtkm::math::Matrix<T,NumRow,NumCol> &a,
    const vtkm::math::Matrix<T,NumRow,NumCol> &b)
{
  return !(a == b);
}


// Slightly modified version of  Stan Melax's code for 3x3 matrix diagonalization (Thanks Stan!)
// source: http://www.melax.com/diag.html?attredirects=0
// See: http://stackoverflow.com/questions/4372224/fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition
template <typename Real>
VTKM_EXEC_CONT_EXPORT
void Diagonalize(const Real A[3][3], Real Q[3][3], Real D[3][3])
{
    // A must be a symmetric matrix.
    // returns Q and D such that
    // Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
    const int maxsteps=24;  // certainly wont need that many.
    int k0, k1, k2;
    Real o[3], m[3];
    Real q [4] = {0.0,0.0,0.0,1.0};
    Real jr[4];
    Real sqw, sqx, sqy, sqz;
    Real tmp1, tmp2, mq;
    Real AQ[3][3];
    Real thet, sgn, t, c;
    for(int i=0;i < maxsteps;++i)
    {
        // quat to matrix
        sqx      = q[0]*q[0];
        sqy      = q[1]*q[1];
        sqz      = q[2]*q[2];
        sqw      = q[3]*q[3];
        Q[0][0]  = ( sqx - sqy - sqz + sqw);
        Q[1][1]  = (-sqx + sqy - sqz + sqw);
        Q[2][2]  = (-sqx - sqy + sqz + sqw);
        tmp1     = q[0]*q[1];
        tmp2     = q[2]*q[3];
        Q[1][0]  = 2.0 * (tmp1 + tmp2);
        Q[0][1]  = 2.0 * (tmp1 - tmp2);
        tmp1     = q[0]*q[2];
        tmp2     = q[1]*q[3];
        Q[2][0]  = 2.0 * (tmp1 - tmp2);
        Q[0][2]  = 2.0 * (tmp1 + tmp2);
        tmp1     = q[1]*q[2];
        tmp2     = q[0]*q[3];
        Q[2][1]  = 2.0 * (tmp1 + tmp2);
        Q[1][2]  = 2.0 * (tmp1 - tmp2);

        // AQ = A * Q
        AQ[0][0] = Q[0][0]*A[0][0]+Q[1][0]*A[0][1]+Q[2][0]*A[0][2];
        AQ[0][1] = Q[0][1]*A[0][0]+Q[1][1]*A[0][1]+Q[2][1]*A[0][2];
        AQ[0][2] = Q[0][2]*A[0][0]+Q[1][2]*A[0][1]+Q[2][2]*A[0][2];
        AQ[1][0] = Q[0][0]*A[0][1]+Q[1][0]*A[1][1]+Q[2][0]*A[1][2];
        AQ[1][1] = Q[0][1]*A[0][1]+Q[1][1]*A[1][1]+Q[2][1]*A[1][2];
        AQ[1][2] = Q[0][2]*A[0][1]+Q[1][2]*A[1][1]+Q[2][2]*A[1][2];
        AQ[2][0] = Q[0][0]*A[0][2]+Q[1][0]*A[1][2]+Q[2][0]*A[2][2];
        AQ[2][1] = Q[0][1]*A[0][2]+Q[1][1]*A[1][2]+Q[2][1]*A[2][2];
        AQ[2][2] = Q[0][2]*A[0][2]+Q[1][2]*A[1][2]+Q[2][2]*A[2][2];
        // D = Qt * AQ
        D[0][0] = AQ[0][0]*Q[0][0]+AQ[1][0]*Q[1][0]+AQ[2][0]*Q[2][0];
        D[0][1] = AQ[0][0]*Q[0][1]+AQ[1][0]*Q[1][1]+AQ[2][0]*Q[2][1];
        D[0][2] = AQ[0][0]*Q[0][2]+AQ[1][0]*Q[1][2]+AQ[2][0]*Q[2][2];
        D[1][0] = AQ[0][1]*Q[0][0]+AQ[1][1]*Q[1][0]+AQ[2][1]*Q[2][0];
        D[1][1] = AQ[0][1]*Q[0][1]+AQ[1][1]*Q[1][1]+AQ[2][1]*Q[2][1];
        D[1][2] = AQ[0][1]*Q[0][2]+AQ[1][1]*Q[1][2]+AQ[2][1]*Q[2][2];
        D[2][0] = AQ[0][2]*Q[0][0]+AQ[1][2]*Q[1][0]+AQ[2][2]*Q[2][0];
        D[2][1] = AQ[0][2]*Q[0][1]+AQ[1][2]*Q[1][1]+AQ[2][2]*Q[2][1];
        D[2][2] = AQ[0][2]*Q[0][2]+AQ[1][2]*Q[1][2]+AQ[2][2]*Q[2][2];
        o[0]    = D[1][2];
        o[1]    = D[0][2];
        o[2]    = D[0][1];
        m[0]    = fabs(o[0]);
        m[1]    = fabs(o[1]);
        m[2]    = fabs(o[2]);

        k0      = (m[0] > m[1] && m[0] > m[2])?0: (m[1] > m[2])? 1 : 2; // index of largest element of offdiag
        k1      = (k0+1)%3;
        k2      = (k0+2)%3;
        if (o[k0]==0.0)
        {
            break;  // diagonal already
        }
        thet    = (D[k2][k2]-D[k1][k1])/(2.0*o[k0]);
        sgn     = (thet > 0.0)?1.0:-1.0;
        thet   *= sgn; // make it positive
        t       = sgn /(thet +((thet < 1.E6)?sqrt(thet*thet+1.0):thet)) ; // sign(T)/(|T|+sqrt(T^2+1))
        c       = 1.0/sqrt(t*t+1.0); //  c= 1/(t^2+1) , t=s/c
        if(c==1.0)
        {
            break;  // no room for improvement - reached machine precision.
        }
        jr[0 ]  = jr[1] = jr[2] = jr[3] = 0.0;
        jr[k0]  = sgn*sqrt((1.0-c)/2.0);  // using 1/2 angle identity sin(a/2) = sqrt((1-cos(a))/2)
        jr[k0] *= -1.0; // since our quat-to-matrix convention was for v*M instead of M*v
        jr[3 ]  = sqrt(1.0f - jr[k0] * jr[k0]);
        if(jr[3]==1.0)
        {
            break; // reached limits of floating point precision
        }
        q[0]    = (q[3]*jr[0] + q[0]*jr[3] + q[1]*jr[2] - q[2]*jr[1]);
        q[1]    = (q[3]*jr[1] - q[0]*jr[2] + q[1]*jr[3] + q[2]*jr[0]);
        q[2]    = (q[3]*jr[2] + q[0]*jr[1] - q[1]*jr[0] + q[2]*jr[3]);
        q[3]    = (q[3]*jr[3] - q[0]*jr[0] - q[1]*jr[1] - q[2]*jr[2]);
        mq      = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q[0]   /= mq;
        q[1]   /= mq;
        q[2]   /= mq;
        q[3]   /= mq;
    }
}



#endif //__VTKM_math_Matrix_h
