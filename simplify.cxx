#include <vtkCellArray.h>
#include <vtkType.h>
#include <vtkPolyData.h>

#include <vtkTriangle.h>


// Uncomment one ( and only one ) of the following to reconfigure the Dax
// code to use a particular device . Comment them all to automatically pick a
// device .
#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_SERIAL
//#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_CUDA
//#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_OPENMP
//#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_TBB

#include <vtkm/cont/DeviceAdapter.h>

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/DynamicArrayHandle.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/Pair.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/DeviceAdapterAlgorithm.h>

#include "simplify.h"
#include "math/VectorAnalysis.h"
#include "math/Matrix.h"

using namespace std;

typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

typedef vtkm::Vec<vtkm::Float32,3> Vector3;
typedef vtkm::Vec<vtkm::Float32,4> Vector4;
typedef vtkm::Vec<vtkm::Float32,9> Vector9;
typedef vtkm::math::Matrix3x3<vtkm::Float32> Matrix3x3;
typedef Vector3 PointType;
typedef vtkIdType PointIdType;

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


/// For each trianglge, compute pairs <clusterId, QuadMetric>
class ComputeQuadricsWorklet : public vtkm::worklet::WorkletMapField
{
    const VTKM_EXEC_CONSTANT_EXPORT PointType * rawPoints;  // maybe it will not work in Cuda?
    const VTKM_EXEC_CONSTANT_EXPORT PointIdType * rawPointIds;
    const VTKM_EXEC_CONSTANT_EXPORT GridInfo grid;
public:
    typedef void ControlSignature(FieldIn<> , FieldOut<>, FieldOut<> );
    typedef void ExecutionSignature(_1, _2, _3);
    typedef _1 InputDomain;
    typedef _2 OutputDomain;
    typedef _3 OutputDomain2;

    VTKM_CONT_EXPORT
    ComputeQuadricsWorklet(
            const PointType * rawPoints_,
            const PointIdType * rawPointIds_,
            const GridInfo &grid_
            )
        : rawPoints(rawPoints_), rawPointIds(rawPointIds_), grid(grid_)
    { }

    VTKM_EXEC_EXPORT
    void get_cell_points(vtkm::Id cellId, PointType &point) const
    {
        point = rawPoints[rawPointIds[cellId]];
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
        vtkm::Id base = counter*4;  // cell ids in VTK format

        assert(rawPointIds[base] == 3);
        get_cell_points(base+1, tri_points[0]);
        get_cell_points(base+2, tri_points[1]);
        get_cell_points(base+3, tri_points[2]);

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

/// For each trianglge, compute pairs <clusterId, QuadMetric>
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
            cout << "Pulling back" << endl;
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

template<typename T, int N>
vtkm::cont::ArrayHandle<T> copyFromVec( vtkm::cont::ArrayHandle< vtkm::Vec<T, N> > const& other)
{
#if 0
    std::size_t index = 0;
    std::vector<T> vmem;
    vmem.resize(other.GetNumberOfValues()*N);
    for (int l=0; l<other.GetNumberOfValues(); l++)
    {
        vtkm::Vec<T, N> value = other.GetPortalConstControl().Get(l);
        for(int j=0; j<N; ++j)
        {
            vmem[index]=value[j];
            ++index;
        }
    }
    vtkm::cont::ArrayHandle<T> mem = vtkm::cont::make_ArrayHandle(vmem);
#else
    const T *vmem = reinterpret_cast< const T *>(& *other.GetPortalConstControl().GetRawIterator());
    vtkm::cont::ArrayHandle<T> mem = vtkm::cont::make_ArrayHandle(vmem, other.GetNumberOfValues()*N);
#endif
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
    PointType *rawPoints = reinterpret_cast<PointType *>( data->GetPoints()->GetVoidPointer(0) );
    PointIdType *rawPointIds = reinterpret_cast<PointIdType *>( data->GetPolys()->GetPointer() );

    vtkm::cont::ArrayHandleCounting<vtkm::Id> counterArray(0, data->GetNumberOfCells());

    /// outputs:
    vtkm::cont::ArrayHandle<PointType> outputPointArray;
    vtkm::cont::ArrayHandle<PointIdType> outputCellArray;

    //construct the scheduler that will execute all the worklets
    vtkm::cont::Timer<> timer;

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

    //////////////////////////////////////////////
    /// start algorithm

    /// pass 1 : Cluster-quadric map generation
    /// For each triangle, compute error quadric and add to each cluster
    ///
    /// pass 1 map
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id, 3> > cidArray; // don't need to initialize?
    vtkm::cont::ArrayHandle<vtkm::Vec<Vector9, 3> > quadricArray; // don't need to initialize?

    ComputeQuadricsWorklet worklet1 ( rawPoints, rawPointIds, gridInfo );
    vtkm::worklet::DispatcherMapField<ComputeQuadricsWorklet> dispatcher(worklet1);

    // invoke
    dispatcher.Invoke(counterArray, cidArray, quadricArray);

#if 0
    for (int l=0; l<cidArray.GetNumberOfValues(); l++)
    {
        cout << cidArray.GetPortalConstControl().Get(l)[0]  << ",";
        cout << cidArray.GetPortalConstControl().Get(l)[1]  << ",";
        cout << cidArray.GetPortalConstControl().Get(l)[2]  << ",";
    }
    cout << endl;
#endif

    /// pass 1 reduce
    vtkm::cont::ArrayHandle<vtkm::Id> cidArrayToReduce = copyFromVec(cidArray);
    vtkm::cont::ArrayHandle<Vector9> quadricArrayToReduce = copyFromVec(quadricArray);

    vtkm::cont::ArrayHandle<vtkm::Id> cidArrayReduced;
    vtkm::cont::ArrayHandle<Vector9> quadricArrayReduced;

    // !!! YOu have to sort first !!!
#if 0
    {
        cout << cidArrayToReduce.GetNumberOfValues() << endl;
        for (int k=0; k<cidArrayToReduce.GetNumberOfValues(); k++)
            cout << cidArrayToReduce.GetPortalConstControl().Get(k) << " ";
        cout << endl;
    }
#endif

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::SortByKey(cidArrayToReduce, quadricArrayToReduce);

#if 0
    {
        cout << cidArrayToReduce.GetNumberOfValues() << endl;
        for (int k=0; k<cidArrayToReduce.GetNumberOfValues(); k++)
            cout << cidArrayToReduce.GetPortalConstControl().Get(k) << " ";
        cout << endl;
    }
#endif

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::ReduceByKey(cidArrayToReduce,    quadricArrayToReduce,
                                                                   cidArrayReduced,     quadricArrayReduced,
                                                                   vtkm::internal::Add());
#if 1
    {
        cout << cidArrayReduced.GetNumberOfValues() << endl;
        for (int k=0; k<cidArrayReduced.GetNumberOfValues(); k++)
            cout << cidArrayReduced.GetPortalConstControl().Get(k) << " ";
        for (int k=0; k<quadricArrayReduced.GetNumberOfValues(); k++) {
            print( quadricArrayReduced.GetPortalConstControl().Get(k) );
            cout << endl;
        }
        cout << endl;
    }
#endif

    //cidArray.ReleaseResources();
    //quadricArray.ReleaseResources();
    //cidArrayToReduce.ReleaseResources();
    //quadricArray.ReleaseResources();


    /// pass 2 : Optimal representative computation
    /// For each cluster, compute the representative vertex
    vtkm::cont::ArrayHandle<Vector3> repPointArray;  // representative point

    ComputeRepresentativeWorklet worklet2 ( gridInfo );
    vtkm::worklet::DispatcherMapField<ComputeRepresentativeWorklet> dispatcher2 (worklet2);

    dispatcher2.Invoke(cidArrayReduced, quadricArrayReduced, repPointArray);

    quadricArrayReduced.ReleaseResources();


    /// Pass 3 : Decimated mesh generation
    /// For each original triangle, only output vertices from three different clusters


    int i;
    vtkm::Id mapping[gridInfo.dim[0]*gridInfo.dim[1]*gridInfo.dim[2]];
    for (i=0; i<cidArrayReduced.GetNumberOfValues(); i++)
    {
        mapping[ cidArrayReduced.GetPortalConstControl().Get(i) ] = i;
    }


    vsp_new(vtkCellArray, out_cells);  /// the output cell array

    vtkIdType npts;
    vtkIdType *pointIds;
    vtkPoints* points = data->GetPoints();

    data->GetPolys()->InitTraversal();
    while( data->GetPolys()->GetNextCell(npts, pointIds) )
    {
        double p[3];
        vtkm::Id cid0, cid1, cid2;

        /// get cluster id for each vertex
        points->GetPoint(pointIds[0], p);
        cid0 = get_cluster_id(p, gridInfo);

        points->GetPoint(pointIds[1], p);
        cid1 = get_cluster_id(p, gridInfo);

        points->GetPoint(pointIds[2], p);
        cid2 = get_cluster_id(p, gridInfo);

        if (cid0 == cid1 || cid0 == cid2 || cid1 == cid2 )
            continue;

        vsp_new(vtkTriangle, new_tri);
        new_tri->GetPointIds()->SetId(0, mapping[cid0]);
        new_tri->GetPointIds()->SetId(1, mapping[cid1]);
        new_tri->GetPointIds()->SetId(2, mapping[cid2]);

        out_cells->InsertNextCell(new_tri);
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


    /// end algorithm
    /// ////////////////////////////////////////

    double time = timer.GetElapsedTime();

    cout << "num points: " << (outputPointArray.GetNumberOfValues()/3)  << endl;
    cout << "num cells: " << (outputCellArray.GetNumberOfValues()/3)  << endl;
    cout << "Time: " << timer.GetElapsedTime() << endl;

    //    saveAsPly(verticesArray, writeLoc);

}
