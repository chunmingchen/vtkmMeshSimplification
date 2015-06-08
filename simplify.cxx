#include <vtkCellArray.h>
#include <vtkType.h>
#include <vtkPolyData.h>

#include <vtkTriangle.h>

#include <vtkm/cont/DeviceAdapter.h>

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/ArrayHandlePermutation.h>

#include <vtkm/cont/DynamicArrayHandle.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/Pair.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/cont/ArrayHandleConstant.h>
#include <vtkm/cont/ArrayHandleCompositeVector.h>

#include "simplify.h"
#include "math/VectorAnalysis.h"
#include "math/Matrix.h"
#include "algorithm.h"

#if (VTKM_DEVICE_ADAPTER==VTKM_DEVICE_ADAPTER_SERIAL)
#pragma message ("Using Serial")
#elif (VTKM_DEVICE_ADAPTER==VTKM_DEVICE_ADAPTER_TBB)
#pragma message ("Using TBB")
#endif
typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

using namespace std;


typedef vtkm::Vec<vtkm::Float32,3> Vector3;
typedef vtkm::Vec<vtkm::Float32,4> Vector4;
typedef vtkm::Vec<vtkm::Float32,9> Vector9;
typedef vtkm::math::Matrix3x3<vtkm::Float32> Matrix3x3;
typedef Vector3 PointType;
struct Triangle{
    vtkIdType cellType;
    vtkIdType pointId[3];
};

// VTK
#define vsp_new(type, name) vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

template <typename T, vtkm::Id N>
VTKM_EXEC_EXPORT void print(const vtkm::Vec<T, N> &vec)
{
    cout << '[';
    for (int i=0; i<N; i++)
        cout << vec[i] << ',';
    cout << ']';
}

struct GridInfo
{
    int dim[3];
    Vector3 origin;
    float grid_width;
    float inv_grid_width; // = 1/grid_width
};


/// determine grid resolution for clustering
vtkm::Id get_cluster_id( const double p_[3], GridInfo &grid )
{
    Vector3 p;
    p[0] = p_[0]; p[1] = p_[1]; p[2] = p_[2];
    Vector3 p_rel = (p - grid.origin) * grid.inv_grid_width;
    vtkm::Id x = min((int)p_rel[0], grid.dim[0]-1);
    vtkm::Id y = min((int)p_rel[1], grid.dim[1]-1);
    vtkm::Id z = min((int)p_rel[2], grid.dim[2]-1);
    return x + grid.dim[0] * (y + grid.dim[1] * z);  // get a unique hash value
}

#if 0 // Quadric Clustering
/// Lindstrom
/// For each triangle, compute pairs <clusterId, QuadMetric>
class ComputeQuadricsWorklet : public vtkm::worklet::WorkletMapField
{
private:
    typedef typename vtkm::cont::ArrayHandle<PointType> PointArrayHandle;
    typedef typename vtkm::cont::ArrayHandle<Triangle> TriangleArrayHandle;
    typedef typename PointArrayHandle::ExecutionTypes<DeviceAdapter>::PortalConst PointPortalType;
    typedef typename TriangleArrayHandle::ExecutionTypes<DeviceAdapter>::PortalConst TrianglePortalType;
    const PointPortalType pointPortal;
    const TrianglePortalType trianglePortal;
    const VTKM_EXEC_CONSTANT_EXPORT GridInfo grid;
public:
    typedef void ControlSignature(FieldIn<> , FieldOut<>, FieldOut<> );
    typedef void ExecutionSignature(_1, _2, _3);
    typedef _1 InputDomain;
    typedef _2 OutputDomain;
    typedef _3 OutputDomain2;

    VTKM_CONT_EXPORT
    ComputeQuadricsWorklet(
            const PointArrayHandle &pointArray,
            const TriangleArrayHandle &triangleArray,
            const GridInfo &grid_
            )
        : pointPortal( pointArray.PrepareForInput(DeviceAdapter() ) ),
          trianglePortal( triangleArray.PrepareForInput(DeviceAdapter() )),
          grid(grid_)
    { }

    VTKM_EXEC_EXPORT
    void get_cell_points(vtkm::Id cellId, PointType tri_points[3]) const
    {
        Triangle tri = trianglePortal.Get( cellId );
        assert(tri.cellType == 3);
        tri_points[0] = pointPortal.Get( tri.pointId[0] );
        tri_points[1] = pointPortal.Get( tri.pointId[1] );
        tri_points[2] = pointPortal.Get( tri.pointId[2] );

    }

    /// quadric weighted by the triangle size
    VTKM_EXEC_EXPORT
    void make_quadric9(const Vector3 tris[3], Vector9 &quad) const
    {
        Vector3 normal = vtkm::math::TriangleNormal(tris[0], tris[1], tris[2]);
        Vector4 plane = vtkm::make_Vec( normal[0], normal[1], normal[2],
                                        -vtkm::dot(normal, tris[0]));
        quad[0] = plane[0]*plane[0];
        quad[1] = plane[0]*plane[1];
        quad[2] = plane[0]*plane[2];
        quad[3] = plane[0]*plane[3];
        quad[4] = plane[1]*plane[1];
        quad[5] = plane[1]*plane[2];
        quad[6] = plane[1]*plane[3];
        quad[7] = plane[2]*plane[2];
        quad[8] = plane[2]*plane[3];
    }

    /// determine grid resolution for clustering
    VTKM_EXEC_EXPORT
    vtkm::Id get_cluster_id( const Vector3 &p) const
    {
        Vector3 p_rel = (p - grid.origin) * grid.inv_grid_width;
        vtkm::Id x = min((int)p_rel[0], grid.dim[0]-1);
        vtkm::Id y = min((int)p_rel[1], grid.dim[1]-1);
        vtkm::Id z = min((int)p_rel[2], grid.dim[2]-1);
        return x + grid.dim[0] * (y + grid.dim[1] * z);  // get a unique hash value
    }

    VTKM_EXEC_EXPORT
    void rotate(vtkm::Id &i1, vtkm::Id &i2, vtkm::Id &i3) const
    {
        int temp=i1; i1 = i2; i2 = i3; i3 = temp;
    }

    //
    VTKM_EXEC_EXPORT
    void sort_ids(vtkm::Id &i1, vtkm::Id &i2, vtkm::Id &i3) const
    {
        while (i1>i2 || i1>i3)
            rotate(i1, i2, i3);
    }

    VTKM_EXEC_EXPORT
    void operator()(const vtkm::Id &counter,
                    vtkm::Vec<vtkm::Id,3> &cidAry, vtkm::Vec<Vector9,3> &quadricAry) const
    {
        //cout << points[0] << "," << points[1] << "," << points[2] << endl;
        //cout << cellId << endl;

        PointType tri_points[3];
        get_cell_points(counter, tri_points);

        Vector9 quadric9;
        make_quadric9( tri_points, quadric9 );

        for (int i=0; i<3; i++)
        {
            cidAry[i] = get_cluster_id(tri_points[i]);
            quadricAry[i] = quadric9;
        }

        // prepare for pass 3
        //sort_ids(cidAry[0], cidAry[1], cidAry[2]);
    }
};

/// Lindstrom
class ComputeRepresentativeWorklet : public vtkm::worklet::WorkletMapField
{
    const VTKM_EXEC_CONSTANT_EXPORT GridInfo grid;
public:
    typedef void ControlSignature(FieldIn<>, FieldIn<>, FieldOut<>);
    typedef void ExecutionSignature(_1, _2, _3);

    VTKM_CONT_EXPORT
    ComputeRepresentativeWorklet( const GridInfo &grid_ )
        : grid(grid_)
    {  }

    VTKM_EXEC_EXPORT
    void unhash( const vtkm::Id &cid,
                 int &x, int &y, int &z ) const
    {
        x = cid % grid.dim[0];
        int tmp = cid / grid.dim[0];
        y = tmp % grid.dim[1];
        z = tmp / grid.dim[1];
    }

    VTKM_EXEC_EXPORT
    void get_grid_center ( const vtkm::Id &cid, Vector3 &point ) const
    {
        int gridx, gridy, gridz;
        unhash(cid, gridx, gridy, gridz);
        point[0] = grid.origin[0] + (gridx+.5) * grid.grid_width;
        point[1] = grid.origin[1] + (gridy+.5) * grid.grid_width;
        point[2] = grid.origin[2] + (gridz+.5) * grid.grid_width;
    }

    /// pull the solution back into the cell if falling outside of cell
    VTKM_EXEC_EXPORT
    void pull_back(Vector3 &p, Vector3 &center) const
    {
        float dist = vtkm::math::Norm2(p-center);
        if ( dist > grid.grid_width*1.732 )
        {
            //cout << "Pulling back" << endl;
            p = center + (p - center) * (grid.grid_width*1.732/dist);
        }

    }

    VTKM_EXEC_EXPORT
    void operator()(const vtkm::Id &cid, const Vector9 &quadric,
                    Vector3 &result_pos) const
    {
        get_grid_center( cid, result_pos );

        // debug
        //cout << "cid=" << cid << ": " ;
        //print (quadric);
        //cout << endl;

        /*
        for (int i=0; i<3; i++)
            cout << result_pos[i]*grid.inv_grid_width << ",";
            */

        /// Diagonalize symetric matrix to get A = Q*D*QT
        ///  Note: vtk has   vtkMath::SingularValueDecomposition3x3(A, U, w, VT);
        Matrix3x3 A, Q, D;
        Vector3 b;
        {
            A(0,0)          = quadric[0];
            A(0,1) = A(1,0) = quadric[1];
            A(0,2) = A(2,0) = quadric[2];
            b[0]            =-quadric[3];
            A(1,1)          = quadric[4];
            A(1,2) = A(2,1) = quadric[5];
            b[1]            =-quadric[6];
            A(2,2)          = quadric[7];
            b[2]            =-quadric[8];
        }


        Diagonalize( (vtkm::Float32 (*)[3])&A[0], (vtkm::Float32 (*)[3])&Q[0], (vtkm::Float32 (*)[3])&D[0] );

        // debug
        //cout << "Diagonalize:" << endl;
        //print ( *(Vector9 *) &A[0]); cout << endl;
        //print ( *(Vector9 *) &Q[0]); cout << endl;
        //print ( *(Vector9 *) &D[0]); cout << endl;

        vtkm::Float32 dmax = max(D(0,0), max(D(1,1), D(2,2)));
        if (dmax == 0 || dmax != dmax) { // check 0 or nan
            cout << "Cannot diagonalize.  Use center point." << endl;
            // do nothing
        } else
        {
            Matrix3x3 invD(0);
            #define SVTHRESHOLD (1e-3)
            invD(0,0) = D(0,0) > SVTHRESHOLD*grid.grid_width*dmax ? 1./D(0,0) : 0;
            invD(1,1) = D(1,1) > SVTHRESHOLD*grid.grid_width*dmax ? 1./D(1,1) : 0;
            invD(2,2) = D(2,2) > SVTHRESHOLD*grid.grid_width*dmax ? 1./D(2,2) : 0;

            D = vtkm::math::MatrixMultiply( vtkm::math::MatrixMultiply( Q, invD ),
                                            vtkm::math::MatrixTranspose( Q ) );

            Vector3 center = result_pos;
            result_pos = result_pos + vtkm::math::MatrixMultiply(D ,
                                 (b - vtkm::math::MatrixMultiply( A, result_pos)));

            pull_back(result_pos, center);
        }
        /*
        cout << "->";
        for (int i=0; i<3; i++)
            cout << result_pos[i]*grid.inv_grid_width << ",";
        cout << endl;
        */

    }
};
#endif

// input: points  output: cid of the points
class MapPointsWorklet : public vtkm::worklet::WorkletMapField {
private:
    const VTKM_EXEC_CONSTANT_EXPORT GridInfo grid;
public:
    typedef void ControlSignature(FieldIn<> , FieldOut<>);
    typedef void ExecutionSignature(_1, _2);

    VTKM_CONT_EXPORT
    MapPointsWorklet(const GridInfo &grid_)
        : grid(grid_)
    { }

    /// determine grid resolution for clustering
    VTKM_EXEC_EXPORT
    vtkm::Id get_cluster_id( const Vector3 &p) const
    {
        Vector3 p_rel = (p - grid.origin) * grid.inv_grid_width;
        vtkm::Id x = min((int)p_rel[0], grid.dim[0]-1);
        vtkm::Id y = min((int)p_rel[1], grid.dim[1]-1);
        vtkm::Id z = min((int)p_rel[2], grid.dim[2]-1);
        return x + grid.dim[0] * (y + grid.dim[1] * z);  // get a unique hash value
    }

    VTKM_EXEC_EXPORT
    void operator()(const PointType &point, vtkm::Id &cid) const
    {
        cid = get_cluster_id(point);
        if (cid < 0)
        {
            cout << "!!!" ;
            cid = get_cluster_id(point);

        }
        VTKM_ASSERT_CONT(cid>=0);  // the id could become overloaded if too many cells
    }
};


class MapCellsWorklet: public vtkm::worklet::WorkletMapField {
    typedef typename vtkm::cont::ArrayHandle<vtkm::Id> IdArrayHandle;
    typedef typename IdArrayHandle::ExecutionTypes<DeviceAdapter>::PortalConst IdPortalType;
    IdPortalType pointCidPortal;
public:
    typedef void ControlSignature(FieldIn<> , FieldOut<>);
    typedef void ExecutionSignature(_1, _2);

    VTKM_CONT_EXPORT
    MapCellsWorklet(const IdArrayHandle &pointCidArray)
        : pointCidPortal( pointCidArray.PrepareForInput(DeviceAdapter()) )
    { }

    VTKM_EXEC_EXPORT
    void operator()(const Triangle &tri, vtkm::Id3 &cid3) const
    {
        assert(tri.cellType == 3);
        cid3[0] = pointCidPortal.Get( tri.pointId[0] );
        cid3[1] = pointCidPortal.Get( tri.pointId[1] );
        cid3[2] = pointCidPortal.Get( tri.pointId[2] );
    }
};

/// pass 3
class IndexingWorklet : public vtkm::worklet::WorkletMapField
{
public:
    typedef typename vtkm::cont::ArrayHandle<vtkm::Id> IdArrayHandle;
private:
    typedef typename IdArrayHandle::ExecutionTypes<DeviceAdapter>::Portal IdPortalType;
    IdArrayHandle cidIndexArray;
    IdPortalType cidIndexRaw;
public:
    typedef void ControlSignature(FieldIn<>, FieldIn<>);
    typedef void ExecutionSignature(_1, _2);

    VTKM_CONT_EXPORT
    IndexingWorklet( size_t n )
    {
        cidIndexRaw = cidIndexArray.PrepareForOutput(n, DeviceAdapter() );
    }

    VTKM_EXEC_EXPORT
    void operator()(const vtkm::Id &counter, const vtkm::Id &cid) const
    {
        cidIndexRaw.Set(cid, counter);
        //printf("cid[%d] = %d\n", cid, counter);
    }

    VTKM_CONT_EXPORT
    IdArrayHandle &getOutput()
    {
        return cidIndexArray;
    }
};


/// pass 4
class Cid2PointIdWorklet : public vtkm::worklet::WorkletMapField
{
public:
    typedef typename vtkm::cont::ArrayHandle<vtkm::Id> IdArrayHandle;
private:
    typedef typename IdArrayHandle::ExecutionTypes<DeviceAdapter>::PortalConst IdPortalType;
    const IdPortalType cidIndexRaw;
public:
    typedef void ControlSignature(FieldIn<>, FieldOut<>);
    typedef void ExecutionSignature(_1, _2);

    VTKM_CONT_EXPORT
    Cid2PointIdWorklet( IdArrayHandle &cidIndexArray )
        : cidIndexRaw ( cidIndexArray.PrepareForInput(DeviceAdapter()) )
    {
    }

    VTKM_EXEC_EXPORT
    void operator()(const vtkm::Id3 &cid3, vtkm::Id3 &pointId3) const
    {
        if (cid3[0]==cid3[1] || cid3[0]==cid3[2] || cid3[1]==cid3[2])
        {
            pointId3[0] = pointId3[1] = pointId3[2] = -1;
        } else {
            pointId3[0] = cidIndexRaw.Get( cid3[0] );
            pointId3[1] = cidIndexRaw.Get( cid3[1] );
            pointId3[2] = cidIndexRaw.Get( cid3[2] );
        }
    }

};

class Id3Less{
public:
    bool operator() (const vtkm::Id3 & a, const vtkm::Id3 & b) const
    {
#if 0
        cout << "Comparing: ";
        print(a);
        print(b);
        cout << (a[0] < b[0] ||
                (a[0]==b[0] && a[1] < b[1]) ||
                (a[0]==b[0] && a[1]==b[1] && a[2] < b[2])) << endl;
#endif
        return (a[0] < b[0] ||
            (a[0]==b[0] && a[1] < b[1]) ||
            (a[0]==b[0] && a[1]==b[1] && a[2] < b[2]));
    }
};

template<typename T, int N>
vtkm::cont::ArrayHandle<T> copyFromVec( vtkm::cont::ArrayHandle< vtkm::Vec<T, N> > const& other)
{
    const T *vmem = reinterpret_cast< const T *>(& *other.GetPortalConstControl().GetRawIterator());
    vtkm::cont::ArrayHandle<T> mem = vtkm::cont::make_ArrayHandle(vmem, other.GetNumberOfValues()*N);
    vtkm::cont::ArrayHandle<T> result;
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy(mem,result);
    return result;
}


///////////////////////////////////////////////////
/// \brief simplify: Mesh simplification extending Lindstrom 2000
/// \param data
/// \param output_data
/// \param grid_width
///
void simplify(vtkSmartPointer<vtkPolyData> data, vtkSmartPointer<vtkPolyData> &output_data, float grid_width)
{
    /// inputs:
    PointType *rawPoints ;
    if (data->GetPoints()->GetDataType() == VTK_FLOAT) {
        rawPoints = reinterpret_cast<PointType *>( data->GetPoints()->GetVoidPointer(0) );
    } else {
        // convert to float
        rawPoints = new PointType[data->GetNumberOfPoints()];
        for (int i=0; i<data->GetNumberOfPoints(); i++)
        {
            double * p = data->GetPoints()->GetPoint(i);
            rawPoints[i][0] = p[0];
            rawPoints[i][1] = p[1];
            rawPoints[i][2] = p[2];
        }
    }
    Triangle *rawTriangles = reinterpret_cast<Triangle *>( data->GetPolys()->GetPointer() );

    vtkm::cont::ArrayHandle<PointType> pointArray = vtkm::cont::make_ArrayHandle( rawPoints, data->GetNumberOfPoints() );
    vtkm::cont::ArrayHandle<Triangle> triangleArray = vtkm::cont::make_ArrayHandle( rawTriangles, data->GetNumberOfCells() );
    vtkm::cont::ArrayHandleCounting<vtkm::Id> counterArray(0, data->GetNumberOfCells());



    /// determine grid resolution for clustering
    GridInfo gridInfo;
    {
        gridInfo.grid_width = grid_width;
        double inv_grid_width = gridInfo.inv_grid_width = 1. / grid_width;

        double *bounds = data->GetBounds();
        gridInfo.dim[0] = ceil((bounds[1]-bounds[0])*inv_grid_width);
        gridInfo.dim[1] = ceil((bounds[3]-bounds[2])*inv_grid_width);
        gridInfo.dim[2] = ceil((bounds[5]-bounds[4])*inv_grid_width);
        gridInfo.origin[0] = (bounds[1]+bounds[0])*0.5 - grid_width*(gridInfo.dim[0])*.5;
        gridInfo.origin[1] = (bounds[3]+bounds[2])*0.5 - grid_width*(gridInfo.dim[1])*.5;
        gridInfo.origin[2] = (bounds[5]+bounds[4])*0.5 - grid_width*(gridInfo.dim[2])*.5;
    }

    //construct the scheduler that will execute all the worklets
    vtkm::cont::Timer<> timer;

    //////////////////////////////////////////////
    /// start algorithm

    /// pass 1 : convert points and cells
    ///
    /// map points
    vtkm::cont::ArrayHandle<vtkm::Id> pointCidArray;

    vtkm::worklet::DispatcherMapField<MapPointsWorklet>(MapPointsWorklet(gridInfo))
                                                      .Invoke(pointArray, pointCidArray );
    //cout << "Time (s): " << timer.GetElapsedTime() << endl;
    /// pass 2 : compute average point position
    ///
    vtkm::cont::ArrayHandle<vtkm::Id> pointCidArrayReduced;
    vtkm::cont::ArrayHandle<Vector3> repPointArray;  // representative point

    AverageByKey( pointCidArray, pointArray, pointCidArrayReduced, repPointArray );
    //cout << "Time (s): " << timer.GetElapsedTime() << endl;


    /// Pass 3 : Decimated mesh generation
    /// For each original triangle, only output vertices from three different clusters


    /// map cells
    vtkm::cont::ArrayHandle<vtkm::Id3> cid3Array;

    vtkm::worklet::DispatcherMapField<MapCellsWorklet>(MapCellsWorklet(pointCidArray))
            .Invoke(triangleArray, cid3Array );

#if 0
    for (int l=0; l<cidArray.GetNumberOfValues(); l++)
    {
        cout << cidArray.GetPortalConstControl().Get(l)[0]  << ",";
        cout << cidArray.GetPortalConstControl().Get(l)[1]  << ",";
        cout << cidArray.GetPortalConstControl().Get(l)[2]  << ",";
    }
    cout << endl;
#endif



    /// Pass 3 preparation: Get index of pointCidArrayReduced
    //cout << "Time (s): " << timer.GetElapsedTime() << endl;
    vtkm::cont::ArrayHandleCounting<vtkm::Id> counterArray3(0, pointCidArrayReduced.GetNumberOfValues());
    IndexingWorklet worklet3 ( gridInfo.dim[0]*gridInfo.dim[1]*gridInfo.dim[2] );

    vtkm::worklet::DispatcherMapField<IndexingWorklet> ( worklet3 )
                                                    .Invoke(counterArray3, pointCidArrayReduced);


    ///
    /// Pass 3 map: update id's in cid3Array
    ///
    vtkm::cont::ArrayHandle<vtkm::Id3> pointId3Array;

    vtkm::worklet::DispatcherMapField<Cid2PointIdWorklet> ( Cid2PointIdWorklet( worklet3.getOutput() ) )
                                                    .Invoke(cid3Array, pointId3Array);


    ///
    /// Pass 3: Unique
    ///
    vtkm::cont::ArrayHandle<vtkm::Id3 > uniquePointId3Array;

    //cout << "Time (s): " << timer.GetElapsedTime() << endl;
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy(pointId3Array,uniquePointId3Array);

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Sort(uniquePointId3Array, Id3Less());

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Unique(uniquePointId3Array);


    /// end algorithm
    /// ////////////////////////////////////////

    cout << "Time (s): " << timer.GetElapsedTime() << endl;

    /// output
    ///
    {
        int i;
        vsp_new(vtkCellArray, out_cells);  /// the output cell array
        int CELL_START = 1;
        for (i=CELL_START; i<uniquePointId3Array.GetNumberOfValues(); i++)
        {
            vtkm::Id3 ids = uniquePointId3Array.GetPortalConstControl().Get(i);
            //print(ids); cout << endl;

            vsp_new(vtkTriangle, triangle);
            triangle->GetPointIds()->SetId(0, ids[0]);
            triangle->GetPointIds()->SetId(1, ids[1]);
            triangle->GetPointIds()->SetId(2, ids[2]);

            out_cells->InsertNextCell(triangle);
        }
        vsp_new(vtkPoints, out_points);
        out_points->SetNumberOfPoints(repPointArray.GetNumberOfValues() );
        for (i=0; i<repPointArray.GetNumberOfValues(); i++)
        {
            Vector3 p = repPointArray.GetPortalConstControl().Get(i);
            out_points->SetPoint((vtkIdType)i, p[0], p[1], p[2]);
        }
        output_data = vtkPolyData::New();
        output_data->SetPoints(out_points);
        output_data->SetPolys(out_cells);
    }

    cout << "num points: " << output_data->GetNumberOfPoints()  << endl;
    cout << "num cells: " << output_data->GetNumberOfCells()  << endl;

    //    saveAsPly(verticesArray, writeLoc);

}

