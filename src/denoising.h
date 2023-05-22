#ifndef DENOISING_H
#define DENOISING_H
#include <iostream>
#include <math.h>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/SparseCholesky>
#include <ctime>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/grad.h>
#include <igl/doublearea.h>
#include <igl/per_vertex_normals.h>
#include <random>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include "nanoflann.hpp"
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
typedef nanoflann::KDTreeEigenMatrixAdaptor<mat> kd_tree;
typedef Eigen::Triplet<double> T;


void initialize_params(const Eigen::MatrixXd &x, const kd_tree &tree, const int k_neighbor, std::vector<double>& alpha, Eigen::SparseMatrix<double> &m, Eigen::SparseMatrix<double> &l, Eigen::SparseMatrix<double> &r, Eigen::SparseMatrix<double> &s);

Eigen::SparseMatrix<double> assembleA(const unsigned int n, const std::vector<double>& alpha_weight);

Eigen::SparseMatrix<double> assemble_smooth_matrix(const unsigned int n, const Eigen::SparseMatrix<double>&s, const Eigen::SparseMatrix<double>&l, const Eigen::SparseMatrix<double>&r);

void denoising(Eigen::MatrixXd &x, Eigen::MatrixXd &c, std::vector<Eigen::MatrixXd> &inter_res, const kd_tree &tree, const int k_neighbor, const double w_a, const double w_b, const double w_c, const double mu_fit, const double mu_smooth, const double lp_threshold, const int max_smooth_iter, const std::string& output_dir, std::vector<int>& edge_v, const int need_s);

void compute_mesh_dirichlet(const std::string filename);

#endif
