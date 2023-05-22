#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H
#include <eigen3/Eigen/Eigen>
#include "kmeans.hpp"
#include "dbscan.h"
#include "nanoflann.hpp"
#include "neighbors.h"
#include <eigen3/Eigen/SparseCholesky>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
typedef nanoflann::KDTreeEigenMatrixAdaptor<mat> kd_tree;
typedef Eigen::Triplet<double> T;

std::pair<Eigen::VectorXd, Eigen::VectorXd> compute_intersection_line(const Eigen::VectorXd& pl1, const Eigen::VectorXd& pl2);

Eigen::VectorXd project(const Eigen::VectorXd& pt, const Eigen::VectorXd& normal);

Eigen::VectorXd project_onto_line(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::VectorXd& pt);

void convert_db_pts(const std::vector<Eigen::VectorXd>& pts, std::vector<dbPoint>& db_pts);

double compute_Manhattan_dist(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2);

Eigen::MatrixXd post_processing(Eigen::MatrixXd& x, const double bbox_diag_length, const Eigen::RowVector3d mean_x, const kd_tree& tree, const Eigen::MatrixXd& c, const std::vector<int>& edge_v, const int post_process_k, std::string output_dir, const bool need_post_process);


#endif
