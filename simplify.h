#include "vtkSmartPointer.h"
#include "vtkPolyData.h"

void simplify(vtkSmartPointer<vtkPolyData> data, vtkSmartPointer<vtkPolyData> &output_data, float grid_width);
