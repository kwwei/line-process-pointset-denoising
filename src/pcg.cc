#include <iostream>
#include <memory>
#include <amgcl/adapter/crs_tuple.hpp>

#include "pcg.h"

using namespace std;
using namespace Eigen;

amg_precon::amg_precon(const size_t nrelax, const size_t maxits)
    : maxits_(maxits) {
  prm_.npre = prm_.npost = nrelax;
}

amg_precon::amg_precon(const size_t nrelax, const size_t maxits, const size_t coarse_enough)
    : maxits_(maxits) {
  prm_.npre = prm_.npost = nrelax;
  prm_.coarse_enough = coarse_enough;
}

int amg_precon::factorize(const MatrixType &mat, const bool verbose) {
  const index_t n = mat.rows(), nnz = mat.nonZeros();

  if ( ptr_.size() != n+1 ) ptr_.resize(n+1);
  if ( ind_.size() != nnz ) ind_.resize(nnz);
  if ( val_.size() != nnz ) val_.resize(nnz);
  if ( u_.size() != n ) u_.resize(n);  

  std::copy(mat.outerIndexPtr(), mat.outerIndexPtr()+ptr_.size(), ptr_.begin());
  std::copy(mat.innerIndexPtr(), mat.innerIndexPtr()+ind_.size(), ind_.begin());
  std::copy(mat.valuePtr(), mat.valuePtr()+val_.size(), val_.begin());
  
  //-> coarsening
  slv_ = std::make_shared<AMG_t>(std::tie(n, ptr_, ind_, val_), prm_);  
  cout << *slv_ << endl;

  return 0;
}

VectorType amg_precon::solve(const VectorType &rhs) {
  std::fill(u_.begin(), u_.end(), 0);
  for (size_t iter = 0; iter < maxits_; ++iter) {
    slv_->cycle(rhs, u_);
  }  
  return Map<VectorXd>(&u_[0], u_.size());
}
