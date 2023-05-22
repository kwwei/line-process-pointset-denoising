#ifndef UTILS_H
#define UTILS_H
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <boost/math/tools/roots.hpp>
#include <boost/lexical_cast.hpp>
#include "kmeans.hpp"
#include "nanoflann.hpp"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
typedef nanoflann::KDTreeEigenMatrixAdaptor<mat> kd_tree;
typedef Eigen::Triplet<double> T;

void normalize_shape(Eigen::MatrixXd& x, double& bbox_diag_length, Eigen::RowVectorXd& mean_x);
void put_back_shape(Eigen::MatrixXd& x, const double bbox_diag_length, const Eigen::RowVectorXd mean_x);

Eigen::VectorXd vector_divide(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

Eigen::VectorXd trustregprob(Eigen::MatrixXd& A, Eigen::VectorXd& b);

double error_MSE(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const std::string object_name);
double error_MSE_normal(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& A_normal, const Eigen::MatrixXd& B_normal);
double error_MCD(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const std::string object_name);
double error_SNR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);
Eigen::MatrixXi refine_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double param);

#endif // UTILS_H
