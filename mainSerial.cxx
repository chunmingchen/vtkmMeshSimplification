#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_SERIAL

#include <cstdio>
#include <iostream>
#include <vector>
#include <map>


#include "vtkSmartPointer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkObjectFactory.h"
#include "vtkXMLPolyDataReader.h"
#include "vtkAxes.h"
#include "vtkAxesActor.h"
#include "vtkImageData.h"
#include <vtkExtractEdges.h>
#include "vtkVector.h"
#include "vtkTriangle.h"
#include "vtkTriangleFilter.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkQuadricClustering.h"
#include "vtkPLYReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkQuadric.h"

#include "vtkPolyDataMapper.h"
#include "vtkRenderWindow.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkMatrix4x4.h"
#include "vtkMatrix3x3.h"

#include "simplify.h"
#include "cp_time.h"

using namespace std;
//using namespace boost::numeric;

#define vsp_new(type, name) vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

vtkRenderer *ren;
vtkRenderWindow *renWin;

bool bShowGrids = false;
#ifdef PROFILING
bool bShowOriginal = false;
#else
bool bShowOriginal = true;
#endif
bool bShowOutput = false;
bool bUseVTKSimplify = false;

/// for mesh simplification
vtkSmartPointer<vtkPolyData> data;
vtkSmartPointer<vtkPolyData> output_data;

double origin[3];
int xdim, ydim, zdim;
void set_origin_and_dim();
float grid_width = 0.015625;
float one_over_grid_width = 1./grid_width;


/// determine grid resolution for clustering
void set_origin_and_dim()
{
    one_over_grid_width = 1. / grid_width;

    double *bounds = data->GetBounds();
    xdim = ceil((bounds[1]-bounds[0])*one_over_grid_width);
    ydim = ceil((bounds[3]-bounds[2])*one_over_grid_width);
    zdim = ceil((bounds[5]-bounds[4])*one_over_grid_width);
    if (0) {
        origin[0] = bounds[0];
        origin[1] = bounds[2];
        origin[2] = bounds[4];
    }else {
        origin[0] = (bounds[1]+bounds[0])*0.5 - grid_width*(xdim)*.5;
        origin[1] = (bounds[3]+bounds[2])*0.5 - grid_width*(ydim)*.5;
        origin[2] = (bounds[5]+bounds[4])*0.5 - grid_width*(zdim)*.5;
    }
}

/// the vtk version
void vtk_simplify(vtkSmartPointer<vtkPolyData> data, vtkSmartPointer<vtkPolyData> &output_data_)
{
    Timer timer;
    timer.start();

    set_origin_and_dim();   // compute xdim, ydim, zdim

    vsp_new(vtkQuadricClustering, filter);
    filter->SetInputData(data);
    filter->SetNumberOfXDivisions(xdim);
    filter->SetNumberOfYDivisions(ydim);
    filter->SetNumberOfZDivisions(zdim);
    filter->AutoAdjustNumberOfDivisionsOff();
    filter->Update();

    output_data = vtkSmartPointer<vtkPolyData>::New();
    output_data->DeepCopy(filter->GetOutput());

    timer.end();
    cout << "Time (ms): " << timer.getElapsedMS() << endl;

    cout << "Number of output points: " << output_data->GetNumberOfPoints() << endl;
    cout << "Number of output cells: " << output_data->GetNumberOfCells() << endl;

    filter->GetDivisionOrigin(origin);
    double *spacing = filter->GetDivisionSpacing();
    printf ("Spacing: %lf %lf %lf\n", spacing[0], spacing[1], spacing[2]);
}

void draw()
{
    ren->RemoveAllViewProps();

    // draw data
    if (bShowOriginal)
    {
        vsp_new(vtkPolyDataMapper, polymapper);
        polymapper->SetInputData(data);

        vsp_new(vtkActor, polyactor);
        polyactor->SetMapper(polymapper);

        ren->AddActor(polyactor);
    }

    // draw data
    if (bShowOutput)
    {
        vsp_new(vtkPolyDataMapper, polymapper);
        polymapper->SetInputData(output_data);

        vsp_new(vtkActor, polyactor);
        polyactor->SetMapper(polymapper);

        ren->AddActor(polyactor);
    }


    // axes
    vsp_new(vtkAxesActor, axes);
    ren->AddActor(axes);

    if (bShowGrids)
    {
        cout << "grid width=" << grid_width << endl;
        set_origin_and_dim();

        vsp_new(vtkImageData, image);
        image->SetDimensions(xdim+1, ydim+1, zdim+1); // grids are larger than cells by one in each dimension
        image->SetOrigin(origin[0], origin[1], origin[2]);
        image->SetSpacing(grid_width, grid_width, grid_width);

        vsp_new(vtkExtractEdges, edges);
        edges->SetInputData(image);

        vsp_new(vtkPolyDataMapper, polymapper);
        polymapper->SetInputConnection(edges->GetOutputPort());

        vsp_new(vtkActor, polyactor);
        polyactor->SetMapper(polymapper);

        ren->AddActor(polyactor);
    }

    ren->SetBackground(0,0,.5); // Background color
    renWin->Render();
}

// Define interaction style
class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
  public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);

    virtual void OnKeyPress()
    {
      // Get the keypress
      vtkRenderWindowInteractor *rwi = this->Interactor;
      std::string key = rwi->GetKeySym();

      // Output the key that was pressed
      std::cout << "Pressed " << key << std::endl;

      // Handle an arrow key
      if(key == "Up")
        {
        std::cout << "The up arrow was pressed." << std::endl;
        }

      // Handle a "normal" key
      if(key == "s")
        {
          if (bUseVTKSimplify)
              vtk_simplify(data, output_data);
          else
              simplify(data, output_data, grid_width);
          bShowOriginal = false;
          bShowOutput = true;
          draw();
        }
      if (key == "o" )
      {
          bShowOriginal = !bShowOriginal ;
          bShowOutput = !bShowOutput;
          draw();
      }
      if (key=="minus") {
          grid_width *= .5;
          cout << "grid width=" << grid_width << endl;
          if (bShowGrids)
            draw();
      }
      if (key=="plus") {
          grid_width *= 2.;
          cout << "grid width=" << grid_width << endl;
          if (bShowGrids)
            draw();
      }
      if (key=="g") {
          bShowGrids = ! bShowGrids;
          draw();
      }
      if (key=="v") {
          bUseVTKSimplify = ! bUseVTKSimplify;
          cout << "Use vtk = " << bUseVTKSimplify << endl;
      }
      if (key=="z") {
          cout << "Saving to output.vtk ..." << endl;
          vsp_new(vtkPolyDataWriter , writer);
          writer->SetFileName("output.vtk");
          writer->SetInputData(output_data);
          writer->Write();
      }

      // Forward events
      vtkInteractorStyleTrackballCamera::OnKeyPress();
    }

};
vtkStandardNewMacro(KeyPressInteractorStyle);

void load_input(const char *filename)
{
    printf("Loading file: %s\n", filename);

    int len = strlen(filename);
    const char *ext = filename + (len-3);

    vsp_new(vtkPolyData, data_in);
    if (strcasecmp(ext, "ply")==0)
    {
        vsp_new(vtkPLYReader, reader);
        reader->SetFileName(filename);
        reader->Update();
        data_in->DeepCopy(reader->GetOutput());
    } else {
        vsp_new(vtkXMLPolyDataReader,reader);
        reader->SetFileName(filename);
        reader->Update();
        data_in->DeepCopy(reader->GetOutput());
    }

    // triangulate
    vsp_new(vtkTriangleFilter, tri);
    tri->SetInputData(data_in);
    tri->Update();

    data = vtkSmartPointer<vtkPolyData>::New();
    data->DeepCopy(tri->GetOutput());
}


int main( int argc, char **argv )
{
    if (argc>1)
        load_input(argv[1]);
    else
        load_input(DATA_FILE);

    printf("Keys:\n"
           "g: Toggle showing grids\n"
           "+/-: Increase/decrease grid size by 2\n"
           "<<< s: Simplify mesh >>>\n"
           "o: Toggle showing orginal model or simplified model\n"
           "v: Toggle using VTK simplification filter\n"
           "z: Save output data\n"
           );

    // Visualize
    vsp_new(vtkRenderer, ren);
    vsp_new(vtkRenderWindow, renWin);
    ::ren = ren.GetPointer();
    ::renWin = renWin.GetPointer();

    renWin->AddRenderer(ren);
    renWin->SetSize(800,600);

    vsp_new(vtkRenderWindowInteractor, renderWindowInteractor );
    renderWindowInteractor->SetRenderWindow(renWin);

    vsp_new(KeyPressInteractorStyle, style);
    style->SetCurrentRenderer(ren);
    renderWindowInteractor->SetInteractorStyle(style);

    draw();

    ren->ResetCamera();

    renWin->Render();

    renderWindowInteractor->Start();

    return EXIT_SUCCESS;

}
