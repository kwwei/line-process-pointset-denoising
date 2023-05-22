#ifndef NEIGHBORS_H
#define NEIGHBORS_H
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <set>

#include "kmeans.hpp"
#include "dbscan.h"
#include "nanoflann.hpp"
#include <eigen3/Eigen/SparseCholesky>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
typedef nanoflann::KDTreeEigenMatrixAdaptor<mat> kd_tree;
typedef Eigen::Triplet<double> T;

Eigen::SparseMatrix<double> find_neighbors_radius(const Eigen::MatrixXd& x, const kd_tree& tree, const double radius, const int lower_bound);

Eigen::SparseMatrix<double> find_neighbors_k(const Eigen::MatrixXd& x, const kd_tree& tree, const size_t k);

Eigen::SparseMatrix<double> adaptive_find_neighbors_k(const Eigen::MatrixXd& x, const std::vector<int>& edge_v, const kd_tree& tree, const size_t k);
#endif
