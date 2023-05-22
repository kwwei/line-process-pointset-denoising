#include "utils.h"
#include "IO.h"

void normalize_shape(Eigen::MatrixXd& x, double& bbox_diag_length, Eigen::RowVectorXd& mean_x) {
    bbox_diag_length = (x.colwise().maxCoeff() - x.colwise().minCoeff()).norm();
    x /= bbox_diag_length;
    mean_x = x.colwise().mean();
    x = x.rowwise() - mean_x;
}


void put_back_shape(Eigen::MatrixXd& x, const double bbox_diag_length, const Eigen::RowVectorXd mean_x) {
  x = x.rowwise() + mean_x;
  x *= bbox_diag_length;
}


Eigen::VectorXd vector_divide(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {

  Eigen::VectorXd result(a.rows());
  std::transform(a.data(), a.data() + a.rows(), b.data(), result.data(), std::divides<double>());
  return result;
}


// // minimize J(x) = x.'*Q*x/2-dot(b,x) such that ||x|| = 1
// Eigen::VectorXd trustregprob(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
//   int n = A.rows();
//   Eigen::VectorXd x;
//   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver((A+A)*0.5);
//   Eigen::VectorXd w = eigensolver.eigenvalues();
//   Eigen::MatrixXd v = eigensolver.eigenvectors();
//   if ( b.norm() < 1e-10 ) {
//     x = v.col(0);
//     if (x(3)<0){
//       x = -x;
//     }
//     // std::cout<<"b = 0, v="<<x<<std::endl;
//     return x;
//   }  
//   Eigen::VectorXd beta = v.transpose() * b;
//   std::vector<double> beta_vec(beta.data(), beta.data() + beta.rows()*beta.cols());
//   std::vector<double> beta_nz_vec;
//   std::vector<double> w_nz_vec;
//   beta_nz_vec.clear();
//   w_nz_vec.clear();
//   for (int i = 0; i < beta_vec.size(); ++i) {
//     if (beta_vec[i]!=0) {
//       beta_nz_vec.push_back(beta_vec[i]);
//       w_nz_vec.push_back(w[i]);
//     }
//   }
//   Eigen::VectorXd beta_nz = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(beta_nz_vec.data(), beta_nz_vec.size());
//   Eigen::VectorXd w_nz = Eigen::Map<Eigen::VectorXd , Eigen::Unaligned>(w_nz_vec.data(), w_nz_vec.size());

//   auto f = [beta_nz, w_nz] (double l) {return (1 - (1./vector_divide((beta_nz),(w_nz.array() - l)).norm()));};
//   double fhigh = f(w(0));
//   double ld = w(0);
//   if (fhigh > 0) {
//     double ub = w(0);
//     int i = 0;
//     double lb = ub - pow(10, i);
//     while (f(lb) >= 0) {
//       i += 1;
//       lb = ub - pow(10, i);
//     }
//     auto ldd = boost::math::tools::bisect([beta_nz, w_nz](double l){return (1 - (1./vector_divide((beta_nz),(w_nz.array() - l)).norm()));},
//                                           lb, ub, [](double x,double y){return abs(x-y) < 0.0001;});
//     ld = ldd.first;
//   }
//   Eigen::VectorXd hl = w.array() - ld;
//   x.setZero(beta.rows());

//   for (int i = 0; i < hl.rows(); ++i) {
//     if (hl(i) > 0) {
//       x(i) = (double) beta(i) / (double) hl(i);
//     }
//   }
//   x = v * x;
//   if (x(3)<0){
//     x = -x;
//   }

//   return x;
// }


Eigen::VectorXd trustregprob(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
  int n = A.rows();
  Eigen::VectorXd x;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);
  const Eigen::VectorXd w = eigensolver.eigenvalues();
  const Eigen::MatrixXd v = eigensolver.eigenvectors();
  if ( b.norm() < 1e-10 ) { // zero RHS
    x = v.col(0);
    if (x(3)<0){
      x = -x;
    }
    return x;
  }  
  Eigen::VectorXd beta = v.transpose() * b;

  auto f =
      [&] (double l) {
        Eigen::VectorXd tmp = beta.array()/(w.array() - l);
        return 1 - 1./tmp.norm();
      };
  
  double fhigh = f(w(0));
  double ld = w(0);
  if (fhigh > 0) { // non-degenerate
    const double ub = w(0);
    int i = 0;
    double lb = ub - pow(10, i);
    while (f(lb) >= 0) {
      i += 1;
      lb = ub - pow(10, i);
    }
    auto ldd = boost::math::tools::bisect(f, lb, ub, [](double x,double y){return abs(x-y) < 1e-6;});
    ld = ldd.first;
  }
  Eigen::VectorXd hl = w.array() - ld;

  x.setZero(beta.rows());
  int nn = 0;
  for (int i = 0; i < hl.rows(); ++i) {
    if (hl(i) > 0) {
      x(i) = beta(i) / hl(i);
    } else {
      nn += 1;
    }
  }

  double cc = 1 - x.squaredNorm();
  double remain_v = sqrt(cc/nn);
  for (int i = 0; i < hl.size(); ++i) {
    if ( hl(i) <= 0 ) {
      x(i) = remain_v;
    }
  }

  x = v * x;
  if (x(3)<0){
    x = -x;
  }
  return x;
}

double error_MSE(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const std::string object_name) {
  kd_tree a_tree(3, std::cref(A), 10);
  kd_tree b_tree(3, std::cref(B), 10);
  nanoflann::SearchParams params;
  a_tree.index -> buildIndex();
  b_tree.index -> buildIndex();
  double dist_a = 0.0;
  double dist_b = 0.0;
  Eigen::MatrixXd heatmap(B.rows(), 4);
  std::vector<double> heatvalue(B.rows(), 0);
  for (int i=0; i<A.rows(); ++i) {
    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = A(i, j);
    b_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    //test
    // if (std::find(edge_idx.begin(), edge_idx.end(), ret_indexes[0]) != edge_idx.end()) continue;
    dist_b += (A.row(i) - B.row(ret_indexes[0])).squaredNorm();
    heatvalue[ret_indexes[0]] += (A.row(i) - B.row(ret_indexes[0])).squaredNorm();
  }

  for (int i=0; i<B.rows(); ++i) {
    //test
    // if (std::find(edge_idx.begin(), edge_idx.end(), i) != edge_idx.end()) continue;

    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = B(i, j);
    a_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    dist_a += (B.row(i) - A.row(ret_indexes[0])).squaredNorm();
    heatvalue[i] += (B.row(i) - A.row(ret_indexes[0])).squaredNorm();
  }
  dist_b /= A.rows();
  dist_a /= B.rows();


  for (int i=0; i<B.rows(); ++i) {
    heatmap.row(i) << B(i, 0), B(i, 1), B(i, 2), heatvalue[i];
  }

  point2vtk(("/home/wwei/Documents/INF574/Denoising/result/figure_cad/heatmap/"+object_name+"_mse.vtk").c_str(), heatmap);
  return 0.5 * (dist_a + dist_b);

}


double error_MSE_normal(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& A_normal, const Eigen::MatrixXd& B_normal) {
  kd_tree a_tree(3, std::cref(A), 10);
  kd_tree b_tree(3, std::cref(B), 10);
  nanoflann::SearchParams params;
  a_tree.index -> buildIndex();
  b_tree.index -> buildIndex();
  double dist_a = 0.0;
  double dist_b = 0.0;
  for (int i=0; i<A.rows(); ++i) {
    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = A(i, j);
    b_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    //test
    // if (std::find(edge_idx.begin(), edge_idx.end(), ret_indexes[0]) != edge_idx.end()) continue;
    dist_b += (A_normal.row(i) - B_normal.row(ret_indexes[0])).squaredNorm();

  }

  for (int i=0; i<B.rows(); ++i) {
    //test
    // if (std::find(edge_idx.begin(), edge_idx.end(), i) != edge_idx.end()) continue;

    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = B(i, j);
    a_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    dist_a += (B_normal.row(i) - A_normal.row(ret_indexes[0])).squaredNorm();
  }
  dist_b /= A.rows();
  dist_a /= B.rows();
  return 0.5 * (dist_a + dist_b);
  
}


double error_MCD(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const std::string object_name) {
  kd_tree a_tree(3, std::cref(A), 10);
  kd_tree b_tree(3, std::cref(B), 10);
  nanoflann::SearchParams params;
  a_tree.index -> buildIndex();
  b_tree.index -> buildIndex();
  double dist_a = 0.0;
  double dist_b = 0.0;
  Eigen::MatrixXd heatmap(B.rows(), 4);
  std::vector<double> heatvalue(B.rows(), 0);

  for (int i=0; i<A.rows(); ++i) {
    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = A(i, j);
    b_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    //test
    // if (std::find(edge_idx.begin(), edge_idx.end(), ret_indexes[0]) != edge_idx.end()) continue;
    heatvalue[ret_indexes[0]] += (A.row(i) - B.row(ret_indexes[0])).lpNorm<1>();
    dist_b += (A.row(i) - B.row(ret_indexes[0])).lpNorm<1>();

  }

  for (int i=0; i<B.rows(); ++i) {
    //test
    // if (std::find(edge_idx.begin(), edge_idx.end(), i) != edge_idx.end()) continue;

    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = B(i, j);
    a_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    heatvalue[i] += (B.row(i) - A.row(ret_indexes[0])).lpNorm<1>();
    dist_a += (B.row(i) - A.row(ret_indexes[0])).lpNorm<1>();
  }
  dist_b /= A.rows();
  dist_a /= B.rows();

  for (int i=0; i<B.rows(); ++i) {
    heatmap.row(i) << B(i, 0), B(i, 1), B(i, 2), heatvalue[i];
  }

  point2vtk(("/home/wwei/Documents/INF574/Denoising/result/figure_cad/heatmap/"+object_name+"_mse.vtk").c_str(), heatmap);

  return 0.5 * (dist_a + dist_b);  
}


double error_SNR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
  // A is groundtrth  
  kd_tree a_tree(3, std::cref(A), 10);
  kd_tree b_tree(3, std::cref(B), 10);
  nanoflann::SearchParams params;
  a_tree.index -> buildIndex();
  b_tree.index -> buildIndex();

  double dist_a = 0.0;
  double dist_b = 0.0;
  std::vector<double> heatvalue(B.rows(), 0);
  for (int i=0; i<A.rows(); ++i) {
    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = A(i, j);
    b_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    dist_b += (A.row(i) - B.row(ret_indexes[0])).squaredNorm();
  }

  // this is the denoised points
  double total_noise = 0.0;
  for (int i=0; i<B.rows(); ++i) {
    std::vector<size_t> ret_indexes(1);
    std::vector<double> out_dists_sqr(1);
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
    std::vector<double> query_pt(3);
    for (int j=0; j<3; j++) query_pt[j] = B(i, j);
    a_tree.index->findNeighbors(resultSet, &query_pt[0], params);
    dist_a += (B.row(i) - A.row(ret_indexes[0])).squaredNorm();

    total_noise += B.row(i).squaredNorm();
  }
  dist_b /= A.rows();
  dist_a /= B.rows();

  double MSE = 0.5*(dist_a+dist_b);
  return 10*std::log10(total_noise/B.rows()/MSE);
}



Eigen::MatrixXi refine_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double param) {
  double bbox = (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
  std::vector<int> rows_to_del;
  for (int i = 0; i < F.rows(); ++i) {
    int id_1 = F(i, 0);
    int id_2 = F(i, 1);
    int id_3 = F(i, 2);
    Eigen::VectorXd pos_1 = V.row(id_1);
    Eigen::VectorXd pos_2 = V.row(id_2);
    Eigen::VectorXd pos_3 = V.row(id_3);
    double d_12 = (pos_1-pos_2).norm();
    double d_13 = (pos_1-pos_3).norm();
    double d_23 = (pos_2-pos_3).norm();
    if ((d_12 > param * bbox) || (d_13 > param * bbox) || (d_23 > param * bbox)) {
      rows_to_del.push_back(i);
    }
  }

  Eigen::MatrixXi new_F(F.rows()-rows_to_del.size(), 3);
  int cnt = 0;
  int cnt2 = 0;
  for (int i = 0; i < F.rows(); ++i) {
    if (i == rows_to_del[cnt]) {
      cnt += 1;
      continue;
    }
    new_F.row(cnt2) = F.row(i);
    cnt2 += 1;
  }

  return new_F;
}
