#pragma once

#include <Eigen/Core>

void reconstruct(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const Eigen::MatrixXd &P, const Eigen::Vector3i &res,
                 const Eigen::VectorXd& mass, const Eigen::VectorXd & density, double h, double isovalue);