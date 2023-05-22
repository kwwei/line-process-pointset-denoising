#ifndef PRECOND_CONJUGATE_GRADIENT_H
#define PRECOND_CONJUGATE_GRADIENT_H

#include <fstream>
#include <Eigen/Sparse>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggr_emin.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/chebyshev.hpp>


typedef Eigen::SparseMatrix<double> MatrixType;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> CSRMatrixType;
typedef Eigen::VectorXd VectorType;

class preconditioner
{
 public:
  virtual ~preconditioner() {}
  virtual int analyse_pattern(const MatrixType &mat) = 0;
  virtual int factorize(const MatrixType &mat, const bool verbose=true) = 0;
  int compute(const MatrixType &mat, const bool verbose=true) {
    int rtn = 0;
    rtn |= analyse_pattern(mat);
    rtn |= factorize(mat, verbose);
    return rtn;
  }
  virtual VectorType solve(const VectorType &rhs) = 0;
};

class amg_precon : public preconditioner
{
 public:
  typedef amgcl::backend::builtin<double> backend;
  typedef backend::matrix spmat;
  typedef amgcl::amg<backend, amgcl::coarsening::ruge_stuben, amgcl::relaxation::spai0> AMG_t;
  //  typedef amgcl::amg<backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0> AMG_t;  
  typedef MatrixType::StorageIndex index_t;

  amg_precon(const size_t nrelax, const size_t maxits);
  amg_precon(const size_t nrelax, const size_t maxits, const size_t coarse_enough);
  int analyse_pattern(const MatrixType &mat) { return 0; }
  int factorize(const MatrixType &mat, const bool verbose=true);
  VectorType solve(const VectorType &rhs);
 private:
  const size_t maxits_;
  
  std::vector<index_t> ptr_, ind_;
  std::vector<double> val_;
  std::vector<double> u_;
  
  std::shared_ptr<AMG_t> slv_;
  AMG_t::params prm_;
};

class precond_cg_solver
{
 public:
  typedef Eigen::Index Index;
  typedef double RealScalar;
  typedef double Scalar;

  // //-> final error and real iteration number
  // RealScalar m_tol_error;
  // Index      m_iters;

  precond_cg_solver() { //}: precond_(std::make_shared<identity_precon>()) {
    tol_ = 1e-12;
  }
  precond_cg_solver(const std::shared_ptr<preconditioner> &precond) : precond_(precond) {
    tol_ = 1e-12;
  }

  int analyse_pattern(const MatrixType &mat) {
    return precond_->analyse_pattern(mat);
  }
  int factorize(const MatrixType &mat, const bool verbose=true) {
    mat_    = &mat;
    maxits_ = 2*mat.cols();
    return precond_->factorize(mat, verbose);
  }
  
  void set_maxits(const Index maxits) {
    maxits_ = maxits;
  }
  void set_tol(const RealScalar tol) {
    tol_ = tol;
  }
  VectorType solve(const VectorType &rhs) const {
    RealScalar m_tol_error;
    Index m_iters;
    
    Eigen::Map<const CSRMatrixType> MAT(mat_->rows(), mat_->cols(), mat_->nonZeros(),
                                        mat_->outerIndexPtr(), mat_->innerIndexPtr(),
                                        mat_->valuePtr());
    
    Index n = mat_->cols();
    VectorType x(n);
    x.setZero();
 
    VectorType residual = rhs-MAT*x; //initial residual 
    RealScalar rhsNorm2 = rhs.squaredNorm();
    
    if(rhsNorm2 == 0) 
    {
      x.setZero();
      m_iters = 0;
      m_tol_error = 0;
      return x;
    }
    const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
    RealScalar threshold = std::max(tol_*tol_*rhsNorm2,considerAsZero);
    RealScalar residualNorm2 = residual.squaredNorm();
    if (residualNorm2 < threshold)
    {
      m_iters = 0;
      m_tol_error = sqrt(residualNorm2 / rhsNorm2);
      return x;
    }

    VectorType p = precond_->solve(residual);   // initial search direction
    
    VectorType tmp(n);
    RealScalar absNew = residual.dot(p);  // the square of the absolute value of r scaled by invM
    Index i = 0;
    
    while(i < maxits_)
    {
      tmp.noalias() = MAT*p;                  // the bottleneck of the algorithm
 
      Scalar alpha = absNew / p.dot(tmp);         // the amount we travel on dir
      x += alpha * p;                             // update solution
      residual -= alpha * tmp;                    // update residual
     
      residualNorm2 = residual.squaredNorm();
      
      if(residualNorm2 < threshold)
        break;
     
      const VectorType &&z = precond_->solve(residual); // approximately solve for "A z = residual"
      if ( std::isnan(z.sum()) ) {
        std::cerr << "# preconditioner produced NaN!" << std::endl;
        break;
      }
 
      RealScalar absOld = absNew;
      absNew = residual.dot(z);                   // update the absolute value of r
      RealScalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                           // update search direction
      i++;
    }
    m_tol_error = sqrt(residualNorm2 / rhsNorm2);
    m_iters = i;

    return x;
  }

 protected:
  MatrixType const *mat_;
  const std::shared_ptr<preconditioner> precond_;

  Index maxits_;
  RealScalar tol_;
};

#endif
