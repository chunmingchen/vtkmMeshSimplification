
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

#if 1
// added to vtkm: Pair.h
class ZipAdd{
public:
    template<typename T, typename U>
    vtkm::Pair<T, U> operator()(const vtkm::Pair<T, U>& a, const vtkm::Pair<T, U> &b)const
    {
        return vtkm::Pair<T,U>(a.first+b.first, a.second + b.second);
    }
};
#endif

// TODO: custom Less()
template <class KeyType, class ValueType, class DeviceAdapter = VTKM_DEFAULT_DEVICE_ADAPTER_TAG>
void AverageByKey( const vtkm::cont::ArrayHandle<KeyType> &keyArray,
                   const vtkm::cont::ArrayHandle<ValueType> &valueArray,
                   vtkm::cont::ArrayHandle<KeyType> &outputKeyArray,
                   vtkm::cont::ArrayHandle<ValueType> &outputValueArray)
{
    vtkm::cont::Timer<> timer;

    vtkm::cont::ArrayHandle<ValueType> sumArray;
    vtkm::cont::ArrayHandle<KeyType> keyArraySorted;

#if 0
    vtkm::cont::ArrayHandle<ValueType> valueArraySorted;
    cout << "AVG Time (s): " << timer.GetElapsedTime() << endl;
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy( keyArray, keyArraySorted );
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy( valueArray, valueArraySorted );

    cout << "AVG Time (s): " << timer.GetElapsedTime() << endl;
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::SortByKey( keyArraySorted, valueArraySorted, std::less<KeyType>() ) ;

    cout << "AVG Time (s): " << timer.GetElapsedTime() << endl;

    vtkm::cont::ArrayHandleConstant<vtkm::Id> constOneArray(1, valueArray.GetNumberOfValues());
    vtkm::cont::ArrayHandle<vtkm::Id> countArray;

    auto inputZipHandle = vtkm::cont::make_ArrayHandleZip(valueArraySorted, constOneArray);
    auto outputZipHandle = vtkm::cont::make_ArrayHandleZip(sumArray, countArray);

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::ReduceByKey( keyArraySorted, inputZipHandle,
                                                                    outputKeyArray, outputZipHandle,
                                                                    vtkm::internal::Add()  );
#else
    vtkm::cont::ArrayHandleCounting<vtkm::Id> indexArray(0, keyArray.GetNumberOfValues());
    vtkm::cont::ArrayHandle<vtkm::Id> indexArraySorted;

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy( keyArray, keyArraySorted );
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy( indexArray, indexArraySorted );
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::SortByKey( keyArraySorted, indexArraySorted, std::less<KeyType>() ) ;

    auto valueArraySorted
            = vtkm::cont::make_ArrayHandlePermutation( indexArraySorted, valueArray );

    vtkm::cont::ArrayHandleConstant<vtkm::Id> constOneArray(1, valueArray.GetNumberOfValues());
    vtkm::cont::ArrayHandle<vtkm::Id> countArray;
#if 1 // reduce twice
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::ReduceByKey( keyArraySorted, valueArraySorted,
                                                                    outputKeyArray, sumArray,
                                                                    vtkm::internal::Add()  );
    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::ReduceByKey( keyArraySorted, constOneArray,
                                                                    outputKeyArray, countArray,
                                                                    vtkm::internal::Add()  );
#else // use zip (slower)
    auto inputZipHandle = vtkm::cont::make_ArrayHandleZip(valueArraySorted, constOneArray);
    auto outputZipHandle = vtkm::cont::make_ArrayHandleZip(sumArray, countArray);

    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::ReduceByKey( keyArraySorted, inputZipHandle,
                                                                    outputKeyArray, outputZipHandle,
                                                                    ZipAdd()  );
#endif
#endif
    cout << "AVG Time (s): " << timer.GetElapsedTime() << endl;



//    cout << "AVG Time (s): " << timer.GetElapsedTime() << endl;
//    vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::ReduceByKey( keyArraySorted, constOneArray, outputKeyArray, countArray,
//                                                                    vtkm::internal::Add());

    // Using local structure with templates : Only works after c++11
    struct DivideWorklet: public vtkm::worklet::WorkletMapField{
        typedef void ControlSignature(FieldIn<>, FieldIn<>, FieldOut<>);
        typedef void ExecutionSignature(_1, _2, _3);
        VTKM_EXEC_EXPORT void operator()(const ValueType &v, vtkm::Id &count, ValueType &vout) const
        {  vout = v * (1./count);  }
    };

    vtkm::worklet::DispatcherMapField<DivideWorklet >().Invoke(sumArray, countArray, outputValueArray);

}

