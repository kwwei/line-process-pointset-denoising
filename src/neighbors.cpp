#include "neighbors.h"

Eigen::SparseMatrix<double> find_neighbors_radius(const Eigen::MatrixXd& x, const kd_tree& tree, const double radius, const int lower_bound) {
  int n = x.rows();
  Eigen::SparseMatrix<double> neighbors(n, n);
  {
    std::vector<T> tripletList;
    nanoflann::SearchParams params;
    const double search_radius = static_cast<double>(pow(radius, 2)); // should be r^2
    int sum_ind = 0;
    int num_few_nei = 0;
    for (int i = 0; i < n; ++i) {
      std::vector<std::pair<long, double>> ret_matches;
      std::vector<double> query_pt(3);
      for (int j=0; j<3; j++) query_pt[j] = x(i, j);
      const int nMatches = tree.index->radiusSearch(&query_pt[0], search_radius, ret_matches, params);
      if (nMatches < lower_bound) {
        num_few_nei += 1;
        std::vector<size_t> ret_indexes_k(lower_bound);
        std::vector<double> out_dists_sqr_k(lower_bound);
        nanoflann::KNNResultSet<double> resultSet_k(lower_bound);
        resultSet_k.init(&ret_indexes_k[0],&out_dists_sqr_k[0]);
        tree.index->findNeighbors(resultSet_k, &query_pt[0], params);
        for (int t=0; t<lower_bound; ++t)
          tripletList.push_back(T(ret_indexes_k[t], i, 1.0));
        sum_ind += lower_bound;
        continue;
      }
      sum_ind += nMatches;
      for (int k=0; k<nMatches; ++k)
        tripletList.push_back(T(ret_matches[k].first, i, 1.0));
    }
    neighbors.setFromTriplets(tripletList.begin(), tripletList.end());
    int avr_neighbor_size = sum_ind / n;
  }
  return neighbors;
}

Eigen::SparseMatrix<double> find_neighbors_k(const Eigen::MatrixXd& x, const kd_tree& tree, const size_t k) {
  int n = x.rows();
  Eigen::SparseMatrix<double> neighbors(n, n);
  {
    std::vector<T> tripletList;
    nanoflann::SearchParams params;
    //      const double search_radius = static_cast<double>(pow(radius, 2)); // should be r^2
    int sum_ind = 0;
    for (int i = 0; i < n; ++i) {
      std::vector<size_t> ret_indexes(k);
      std::vector<double> out_dists_sqr(k);
      nanoflann::KNNResultSet<double> resultSet(k);
      resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
      std::vector<double> query_pt(3);
      for (int j=0; j<3; j++) query_pt[j] = x(i, j);
      tree.index->findNeighbors(resultSet, &query_pt[0], params);
      sum_ind += k;
      for (int t=0; t<k; ++t)
        tripletList.push_back(T(ret_indexes[t], i, 1.0));
    }
    neighbors.setFromTriplets(tripletList.begin(), tripletList.end());
  }
  return neighbors;
}

Eigen::SparseMatrix<double> adaptive_find_neighbors_k(const Eigen::MatrixXd& x, const std::vector<int>& edge_v, const kd_tree& tree, const size_t k) {
  int n = x.rows();
  std::set<int> e_v(edge_v.begin(), edge_v.end());
  Eigen::SparseMatrix<double> neighbors(n, n);
  {
    std::vector<T> tripletList;
    nanoflann::SearchParams params;
    //      const double search_radius = static_cast<double>(pow(radius, 2)); // should be r^2
    for (int i = 0; i < n; ++i) {
      int t = 1;
      bool search = true;
      std::vector<size_t> ret_indexes;
      std::vector<double> out_dists_sqr;


      while (search) {
        ret_indexes.resize(t*k);
        out_dists_sqr.resize(t*k);
        nanoflann::KNNResultSet<double> resultSet(t*k);
        resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
        std::vector<double> query_pt(3);
        for (int j=0; j<3; j++) query_pt[j] = x(i, j);
        tree.index->findNeighbors(resultSet, &query_pt[0], params);
        int cnt = 0;
        for (int t=0; t<ret_indexes.size(); ++t) {
          if (e_v.find(ret_indexes[t])==e_v.end())
          {
            cnt += 1;
          }      }
        if (cnt<20) {
          t *= 2;
        }
        else {
          search = false;
        }
      }
      for (int t=0; t<ret_indexes.size(); ++t)
        tripletList.push_back(T(ret_indexes[t], i, 1.0));
    }
    neighbors.setFromTriplets(tripletList.begin(), tripletList.end());
   
  }
  return neighbors;
}

