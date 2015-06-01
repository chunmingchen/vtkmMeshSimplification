#include <vtkCellArray.h>
#include <vtkType.h>
#include <vtkCellArray.h>

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
#include "VectorAnalysis.h"

using namespace std;

typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

typedef vtkm::Vec<vtkm::Float32,3> Vector3;
typedef vtkm::Vec<vtkm::Float32,4> Vector4;
typedef vtkm::Vec<vtkm::Float32,9> Vector9;
typedef Vector3 PointType;
typedef vtkIdType PointIdType;


struct GridInfo
{
    int dim[3];
    Vector3 origin;
    float grid_width;
    float inv_grid_width; // = 1/grid_width
};




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
    void operator()(const vtkm::Id &counter,
                    vtkm::Vec<vtkm::Id,3> &cidAry, vtkm::Vec<Vector9,3> &quadricAry) const
    {
        //cout << points[0] << "," << points[1] << "," << points[2] << endl;
        //cout << cellId << endl;

        PointType tri_points[3];
        vtkm::Id base = counter*4;

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
    }
};

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
    vtkm::worklet::DispatcherMapField dispatcher(worklet1);

    // invoke
    dispatcher.Invoke(counterArray, cidArray, quadricArray);

#if 0
    vtkm::cont::ArrayHandle< vtkm::Vec<vtkm::Id,3> >::PortalConstControl cidArrayPortal = cidArray.GetPortalConstControl();
    vtkm::cont::ArrayHandle< vtkm::Vec<Vector9 ,3> >::PortalConstControl quadricArrayPortal = quadricArray.GetPortalConstControl();
    for (int i=0; i<1000; i++)
    {
        cout << portal.Get(i)[0] << ",";
    }
#endif

    /// pass 1 reduce
    vtkm::cont::ArrayHandle<vtkm::Id> cidArrayToReduce = vtkm::cont::make_ArrayHandle(
                reinterpret_cast< const vtkm::Id *>(&*cidArray.GetPortalConstControl().GetRawIterator()), cidArray.GetNumberOfValues()*3 );
    vtkm::cont::ArrayHandle<Vector9 > quadricArrayToReduce = vtkm::cont::make_ArrayHandle(
                reinterpret_cast< const Vector9 *>(&*quadricArray.GetPortalConstControl().GetRawIterator()), cidArray.GetNumberOfValues()*3 );
    vtkm::cont::ArrayHandle<vtkm::Id> cidArrayReduced ; // don't need to initialize?
    vtkm::cont::ArrayHandle<Vector9> quadricArrayReduced; // don't need to initialize?

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::ReduceByKey(cidArrayToReduce,    quadricArrayToReduce,
                                                                   cidArrayReduced,     quadricArrayReduced,
                                                                   vtkm::internal::Add());
    //cidArray.ReleaseResources();
    //quadricArray.ReleaseResources();
    //cidArrayToReduce.ReleaseResources();
    //quadricArray.ReleaseResources();


    /// pass 2 : Optimal representative computation
    /// For each cluster, compute the representative vertex
    ComputeRepresentativeWorklet worklet2 ( );
    vtkm::worklet::DispatcherMapField dispatcher(worklet2);


    /// end algorithm
    /// ////////////////////////////////////////

    double time = timer.GetElapsedTime();

    cout << "num points: " << (outputPointArray.GetNumberOfValues()/3)  << endl;
    cout << "num cells: " << (outputCellArray.GetNumberOfValues()/3)  << endl;
    cout << "Time: " << timer.GetElapsedTime() << endl;

    //    saveAsPly(verticesArray, writeLoc);
    output_data = data;

}
