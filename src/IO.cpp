#include "IO.h"

void saveXYZ(const std::string filename, const Eigen::MatrixXd& denoised_xyz) {
  std::ofstream file(filename);
  if (file)
  {
    file << denoised_xyz << "\n";
    file.close();
    std::cout<<"xyz file saved!!"<<std::endl;
  }
  else {
    std::cerr<<"saving error..."<<std::endl;
  }

}


void saveIndices(const std::string filename, const Eigen::VectorXi& idx) {
  std::ofstream file(filename);
  if (file)
  {
    file << idx << "\n";
    file.close();
    std::cout<<"idx saved!!"<<std::endl;
  }
  else {
    std::cerr<<"saving error..."<<std::endl;
  }

}


void point2vtk(const std::string& filename, const Eigen::MatrixXd& points) {
    int n = points.rows();
    std::ofstream file(filename);
    if (file) {
        file << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";
        file << "POINTS "<<n<<" float\n";
        for (unsigned int i=0; i<n; ++i) {
            file<<points(i, 0)<<" "<<points(i, 1)<<" "<<points(i, 2)<<"\n";
        }
        file<<"CELLS "<<n<<" "<<2*n<<" \n";
        for (unsigned int i=0; i<n; ++i) {
            file<<"1 "<<i<<"\n";
        }
        file<<"CELL_TYPES "<<n<<"\n";
        for (unsigned int i=0; i<n; ++i) {
            file<<"1\n";
        }
        if (points.cols() == 4) {
            // with scalar
            file<<"POINT_DATA "<<n<<"\n";
            file<<"SCALARS v float\nLOOKUP_TABLE v\n";
            for (unsigned int i=0; i<n; ++i) {
                file<<points(i, 3)<<"\n";
            }
        }
        file.close();
        // std::cout<<"saved point vtk file ..."<<std::endl;
    }
}


void line2vtk(const std::string& filename, const Eigen::MatrixXd& start_points, const Eigen::MatrixXd& end_points) {
    assert(start_points.rows() == end_points.rows());
    int n = start_points.rows();
    std::ofstream file(filename);
    if (file) {
        file<<"# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";
        file<<"POINTS "<<2*n<<" float\n";
        for (unsigned int i=0; i<n; ++i) {
            file<<start_points(i, 0)<<" "<<start_points(i, 1)<<" "<<start_points(i, 2)<<"\n";
            file<<end_points(i, 0)<<" "<<end_points(i, 1)<<" "<<end_points(i, 2)<<"\n";
        }
        file<<"CELLS "<<n<<" "<<3*n<<"\n";
        for (unsigned int i=0; i<n; ++i) {
            file<<"2 "<<2*i<<" "<<2*i+1<<"\n";
        }
        file<<"CELL_TYPES "<<n<<"\n";
        for (unsigned int i=0; i<n; ++i) {
            file<<"3\n";
        }

        file.close();
        std::cout<<"saved line vtk file ..."<<std::endl;
    }
}

void write_line_with_scalar(const std::string& filename, const Eigen::MatrixXd& fit_x, const Eigen::SparseMatrix<double>& lij) {
    unsigned int node_num = fit_x.rows();
    unsigned int line_num = lij.nonZeros();
    std::ofstream file(filename);
    if (file) {
        file<<"# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";
        file<<"POINTS "<<node_num<<"float\n";
        for (unsigned int i=0; i<node_num; ++i) {
            file<<fit_x(i, 0)<<" "<<fit_x(i, 1)<<" "<<fit_x(i, 2)<<"\n";
        }
        file<<"CELLS "<<line_num<<" "<<3*line_num<<"\n";
        for (int j = 0; j < lij.outerSize(); ++j) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(lij, j); it; ++it) {
            file<<"2 "<<j<<" "<<it.row()<<"\n";
          }
        }
        file<<"CELL_TYPES "<<line_num<<"\n";
        for(unsigned int i=0; i<line_num; ++i) {
            file<<"3\n";
        }
        file<<"CELL_DATA "<<line_num<<"\n";
        file<<"SCALARS v float\nLOOKUP_TABLE v\n";
        for (int j = 0; j < lij.outerSize(); ++j) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(lij, j); it; ++it) {
            file<<it.value()<<"\n";
          }
        }
    }
}

void save_surface (const std::string& outputname, const Eigen::MatrixXd& denoised_pts) {
  Point_set points;
  // read pts
  for (unsigned int i = 0; i < denoised_pts.rows(); ++i) {
    points.insert(Point(denoised_pts(i, 0), denoised_pts(i, 1), denoised_pts(i, 2)));
  }
  // reconstruction
  Reconstruction reconstruct (points.points().begin(), points.points().end());
  reconstruct.increase_scale<Smoother> (4);
  reconstruct.reconstruct_surface (Mesher (0.5));
  std::ofstream out (outputname);
  out << "OFF" << std::endl << points.size() << " " << reconstruct.number_of_facets() << " 0" << std::endl;

  for (Point_set::iterator it = points.begin(); it != points.end(); ++ it)
    out << points.point(*it) << std::endl;
  for (Reconstruction::Facet_iterator it = reconstruct.facets_begin();
       it != reconstruct.facets_end(); ++ it)
    out << "3 " << (*it)[0] << " " << (*it)[1] << " " << (*it)[2] << std::endl;
  std::cout<<"OFF file saved!"<<std::endl;

}



struct Perimeter {

  double bound;

  Perimeter(double bound)
    : bound(bound)
  {}

  template <typename AdvancingFront, typename Cell_handle>
  double operator() (const AdvancingFront& adv, Cell_handle& c,
                     const int& index) const
  {
    // bound == 0 is better than bound < infinity
    // as it avoids the distance computations
    if(bound == 0){
      return adv.smallest_radius_delaunay_sphere (c, index);
    }

    // If perimeter > bound, return infinity so that facet is not used
    double d  = 0;
    d = sqrt(squared_distance(c->vertex((index+1)%4)->point(),
                              c->vertex((index+2)%4)->point()));
    if(d>bound) return adv.infinity();
    d += sqrt(squared_distance(c->vertex((index+2)%4)->point(),
                               c->vertex((index+3)%4)->point()));
    if(d>bound) return adv.infinity();
    d += sqrt(squared_distance(c->vertex((index+1)%4)->point(),
                               c->vertex((index+3)%4)->point()));
    if(d>bound) return adv.infinity();

    // Otherwise, return usual priority value: smallest radius of
    // delaunay sphere
    return adv.smallest_radius_delaunay_sphere (c, index);
  }
};


void save_surface_advance_front (const std::string& outputname, const Eigen::MatrixXd& denoised_pts) {
  std::vector<Point_3> points;
  std::vector<Facet> facets;
  for (unsigned int i = 0; i < denoised_pts.rows(); ++i) {
    points.push_back(Point_3(denoised_pts(i, 0), denoised_pts(i, 1), denoised_pts(i, 2)));
  }
  // reconstruction
  Perimeter perimeter(0);
    CGAL::advancing_front_surface_reconstruction(points.begin(),
                                               points.end(),
                                               std::back_inserter(facets),
                                               perimeter,
                                               5.0);
    
  std::ofstream out (outputname);
  out << "OFF" << std::endl << points.size() << " " << facets.size() << " 0\n";

  for (int i=0; i<points.size(); ++i)
    out << points[i][0] << " " << points[i][1] << " " << points[i][2] << std::endl;
  for (int i=0; i<facets.size(); ++i)
    out << "3 " << facets[i][0] << " " << facets[i][1] << " " << facets[i][2] << std::endl;
  std::cout<<"OFF file saved!"<<std::endl;

}

