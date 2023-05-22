#include <CGAL/Simple_cartesian.h>

#include <CGAL/bilateral_smooth_point_set.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/IO/write_points.h>
#include <CGAL/property_map.h>
#include <CGAL/tags.h>

#include <utility> // defines std::pair
#include <iostream>
#include "IO.h"
// Types
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Pt;
typedef K::Vector_3 Vector;

typedef std::array<std::size_t,3> Facet;
// Point with normal vector stored in a std::pair.
typedef std::pair<Pt, Vector> PointVectorPair;

// Concurrency
typedef CGAL::Parallel_if_available_tag Concurrency_tag;



int main(int argc, char*argv[])
{
  const char* input_filename = argv[1];
  const char* output_xyz = argv[2];
  const char* output_off = argv[3];
  const int k = std::stoi(argv[4]);
  const double sharpness_angle = std::stod(argv[5]);
  const int iter_number = std::stoi(argv[6]);

  char result_name[160];
  // Reads a point set file in points[] * with normals *.
  std::vector<PointVectorPair> points;
  std::cout<<output_xyz<<std::endl;
  std::cout<<"iter time = "<<iter_number<<std::endl;
  if(!CGAL::IO::read_points(input_filename, std::back_inserter(points),
                            CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
                                             .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())))
  {
     std::cerr << "Error: cannot read file " << input_filename << std::endl;
     return EXIT_FAILURE;
  }

  // Algorithm parameters
//  int k = 50;                 // size of neighborhood. The bigger the smoother the result will be.
//                               // This value should bigger than 1.
//  double sharpness_angle = 60; // control sharpness of the result.
//                               // The bigger the smoother the result will be
//  int iter_number = 3;         // number of times the projection is applied

  for(int i = 0; i < iter_number; ++i)
  {
    /* double error = */
    CGAL::bilateral_smooth_point_set <Concurrency_tag>(
      points,
      k,
      CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
                       .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())
                       .sharpness_angle(sharpness_angle));
  }
    // Save point set.
    if(!CGAL::IO::write_XYZ(output_xyz, points,
                          CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
                          .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())
                          .stream_precision(17)));    
    // save surface
    Eigen::MatrixXd V;
    readXYZ(output_xyz, V);
    save_surface_advance_front(output_off, V);
    
}

