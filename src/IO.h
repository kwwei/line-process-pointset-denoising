#ifndef IO_H
#define IO_H
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/SparseCholesky>
#include <eigen3/Eigen/Eigen>
#include <igl/list_to_matrix.h>
// cgal setup
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Scale_space_surface_reconstruction_3.h>
#include <CGAL/Scale_space_reconstruction_3/Advancing_front_mesher.h>
#include <CGAL/Scale_space_reconstruction_3/Jet_smoother.h>
#include <CGAL/Point_set_3/IO.h>

#include <algorithm>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/tuple.h>
#include <boost/lexical_cast.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel     Kernel;
typedef CGAL::Scale_space_surface_reconstruction_3<Kernel>                    Reconstruction;
typedef CGAL::Scale_space_reconstruction_3::Advancing_front_mesher<Kernel>    Mesher;
typedef CGAL::Scale_space_reconstruction_3::Jet_smoother<Kernel>              Smoother;
typedef Kernel::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;
typedef Reconstruction::Facet_const_iterator                   Facet_iterator;


typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3  Point_3;

typedef std::array<std::size_t,3> Facet;



namespace EigenSup {
template<class Matrix>
inline void write_binary(const std::string& filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if(out.is_open()) {
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index));
        out.write(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index));
        out.write(reinterpret_cast<const char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
        out.close();
    }
    else {
        std::cout << "Can not write to file: " << filename << std::endl;
    }
}

template<class Matrix>
inline void read_binary(const std::string& filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        typename Matrix::Index rows=0, cols=0;
        in.read(reinterpret_cast<char*>(&rows),sizeof(typename Matrix::Index));
        in.read(reinterpret_cast<char*>(&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read(reinterpret_cast<char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
        in.close();
    }
    else {
        std::cout << "Can not open binary matrix file: " << filename << std::endl;
    }
}


template <class SparseMatrix>
inline void write_binary_sparse(const std::string& filename, const SparseMatrix& matrix) {
    assert(matrix.isCompressed() == true);
    std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    if(out.is_open())
    {
        typename SparseMatrix::Index rows, cols, nnzs, outS, innS;
        rows = matrix.rows()     ;
        cols = matrix.cols()     ;
        nnzs = matrix.nonZeros() ;
        outS = matrix.outerSize();
        innS = matrix.innerSize();

        out.write(reinterpret_cast<char*>(&rows), sizeof(typename SparseMatrix::Index));
        out.write(reinterpret_cast<char*>(&cols), sizeof(typename SparseMatrix::Index));
        out.write(reinterpret_cast<char*>(&nnzs), sizeof(typename SparseMatrix::Index));
        out.write(reinterpret_cast<char*>(&outS), sizeof(typename SparseMatrix::Index));
        out.write(reinterpret_cast<char*>(&innS), sizeof(typename SparseMatrix::Index));

        typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
        typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
        out.write(reinterpret_cast<const char*>(matrix.valuePtr()),       sizeScalar * nnzs);
        out.write(reinterpret_cast<const char*>(matrix.outerIndexPtr()),  sizeIndexS  * outS);
        out.write(reinterpret_cast<const char*>(matrix.innerIndexPtr()),  sizeIndexS  * nnzs);

        out.close();
    }
    else {
        std::cout << "Can not write to file: " << filename << std::endl;
    }
}

template <class SparseMatrix>
inline void read_binary_sparse(const std::string& filename, SparseMatrix& matrix) {
    std::ifstream in(filename, std::ios::binary | std::ios::in);
    if(in.is_open()) {
        typename SparseMatrix::Index rows, cols, nnz, inSz, outSz;
        typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
        typename SparseMatrix::Index sizeIndex  = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Index       ));
        typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
        std::cout << sizeScalar << " " << sizeIndex << std::endl;
        in.read(reinterpret_cast<char*>(&rows ), sizeIndex);
        in.read(reinterpret_cast<char*>(&cols ), sizeIndex);
        in.read(reinterpret_cast<char*>(&nnz  ), sizeIndex);
        in.read(reinterpret_cast<char*>(&outSz), sizeIndex);
        in.read(reinterpret_cast<char*>(&inSz ), sizeIndex);

        matrix.resize(rows, cols);
        matrix.makeCompressed();
        matrix.resizeNonZeros(nnz);

        in.read(reinterpret_cast<char*>(matrix.valuePtr())     , sizeScalar * nnz  );
        in.read(reinterpret_cast<char*>(matrix.outerIndexPtr()), sizeIndexS * outSz);
        in.read(reinterpret_cast<char*>(matrix.innerIndexPtr()), sizeIndexS * nnz );

        matrix.finalize();
        in.close();
    } // file is open
    else {
        std::cout << "Can not open binary sparse matrix file: " << filename << std::endl;
    }
}
}

template <typename Scalar>
bool readXYZ(
  const std::string xyz_file_name,
  std::vector<std::vector<Scalar > > & V,
  std::vector<std::vector<Scalar > > & N)
{
  using namespace std;
  FILE * xyz_file = fopen(xyz_file_name.c_str(),"r");
  if(NULL==xyz_file)
  {
    printf("IOError: %s could not be opened...\n",xyz_file_name.c_str());
    return false;
  }
  return readXYZ(xyz_file,V,N);
}

template <typename Scalar>
bool readXYZ(
  FILE * xyz_file,
  std::vector<std::vector<Scalar > > & V,
  std::vector<std::vector<Scalar > > & N)
{
  using namespace std;
  V.clear();
  N.clear();
  // get num of pts
  unsigned int number_of_vertices = 0;
  int ch;
  bool has_normals = false;
  char line[1000];

  while (fgets(line, 1000, xyz_file)!= NULL) {
      number_of_vertices ++;
      double x,y,z,nx,ny,nz;
      if (sscanf(line, "%lg %lg %lg %lg %lg %lg",&x,&y,&z,&nx,&ny,&nz) == 6)
          has_normals = true;
  }


  fseek(xyz_file, 0, SEEK_SET);

  char tic_tac_toe;

  V.resize(number_of_vertices);
  N.resize(number_of_vertices);
  for(int i = 0;i<number_of_vertices;)
  {
      fgets(line, 1000, xyz_file);

    double x,y,z,nx,ny,nz;
    if(sscanf(line, "%lg %lg %lg %lg %lg %lg",&x,&y,&z,&nx,&ny,&nz)>= 3)
    {
      std::vector<Scalar > vertex;
      vertex.resize(3);
      vertex[0] = x;
      vertex[1] = y;
      vertex[2] = z;
      V[i] = vertex;


        std::vector<Scalar > normal;
        normal.resize(3);
        normal[0] = nx;
        normal[1] = ny;
        normal[2] = nz;
        N[i] = normal;

      i++;
    }
    else if(sscanf(line, "%lg %lg %lg %lg %lg %lg",&x,&y,&z,&nx,&ny,&nz)== 3)
    {
        std::vector<Scalar > vertex;
        vertex.resize(3);
        vertex[0] = x;
        vertex[1] = y;
        vertex[2] = z;
        V[i] = vertex;
        i++;
      }
    else if(
        fscanf(xyz_file,"%[#]",&tic_tac_toe)==1)
    {
      char comment[1000];
      fscanf(xyz_file,"%[^\n]",comment);
    }else
    {
      printf("Error: bad line (%d)\n",i);
      printf("value = %i", sscanf(line, "%lg %lg %lg %lg %lg %lg",&x,&y,&z,&nx,&ny,&nz));
      if(feof(xyz_file))
      {
        fclose(xyz_file);
        return false;
      }
    }
  }
  fclose(xyz_file);
  return true;
}

template <typename DerivedV>
bool readXYZ(
  const std::string str,
  Eigen::PlainObjectBase<DerivedV>& V)
{
  std::vector<std::vector<double> > vV;
  std::vector<std::vector<double> > vN;
  bool success = readXYZ(str,vV,vN);
  if(!success)
  {
    // readXYZ(str,vV,vF,vN,vC) should have already printed an error
    // message to stderr
    return false;
  }
  bool V_rect = igl::list_to_matrix(vV,V);
  if(!V_rect)
  {
    // igl::list_to_matrix(vV,V) already printed error message to std err
    return false;
  }
//  bool F_rect = igl::list_to_matrix(vN,N);
//  if(!F_rect)
//  {
//    // igl::list_to_matrix(vF,F) already printed error message to std err
//    return false;
//  }
  return true;
}


template <typename DerivedV>
bool readXYZ(
  const std::string str,
  Eigen::PlainObjectBase<DerivedV>& V,
  Eigen::PlainObjectBase<DerivedV>& N)
{
  std::vector<std::vector<double> > vV;
  std::vector<std::vector<double> > vN;

  bool success = readXYZ(str,vV,vN);
  if(!success)
  {
    // readXYZ(str,vV,vF,vC) should have already printed an error
    // message to stderr
    return false;
  }
  bool V_rect = igl::list_to_matrix(vV,V);

  if(!V_rect)
  {

    // igl::list_to_matrix(vV,V) already printed error message to std err
    return false;
  }
//  bool F_rect = igl::list_to_matrix(vF,F);
//  if(!F_rect)
//  {
//    // igl::list_to_matrix(vF,F) already printed error message to std err
//    return false;
//  }

  if (vN.size())
  {
    bool N_rect = igl::list_to_matrix(vN,N);
    if(!N_rect)
    {
      // igl::list_to_matrix(vN,N) already printed error message to std err
      return false;
    }
  }

  return true;
}


void saveXYZ(const std::string filename, const Eigen::MatrixXd& denoised_xyz);
void saveIndices(const std::string filename, const Eigen::VectorXi& idx);
void point2vtk(const std::string& filename, const Eigen::MatrixXd& points);
void line2vtk(const std::string& filename, const Eigen::MatrixXd& start_points, const Eigen::MatrixXd& end_points);
void write_line_with_scalar(const std::string& filename, const Eigen::MatrixXd& fit_x, const Eigen::SparseMatrix<double>& lij);
void save_surface (const std::string& outputname, const Eigen::MatrixXd& denoised_pts);
void save_surface_advance_front (const std::string& outputname, const Eigen::MatrixXd& denoised_pts);
#endif // IO_H
