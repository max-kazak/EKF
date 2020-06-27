#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);

  if(estimations.size() == 0) {
    std::cout << "ERROR: CalculateRMSE: estimations is empty" << std::endl;
    return rmse;
  }
  if(ground_truth.size() == 0) {
    std::cout << "ERROR: CalculateRMSE: ground_truth is empty" << std::endl;
    return rmse;
  }
  if(estimations.size() != ground_truth.size()) {
    std::cout << "ERROR: CalculateRMSE: estimations ground_truth are not the same size" << std::endl;
    return rmse;
  }

  for(int i=0; i<estimations.size(); i++){
    VectorXd diff = estimations[i] - ground_truth[i];
    diff = diff.array() * diff.array();
    rmse += diff;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);

  if(x_state.size() != 4){
    std::cout << "ERROR: CalculateJacobian: state vector must be of size 4." << std::endl;
    return Hj;
  }

  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  if (fabs(px) < 0.0001 and fabs(py) < 0.0001){
      px = 0.0001;
      py = 0.0001;
    }

  double c1 = px*px + py*py;

  if(fabs(c1) < 0.0000001) {
      std::cout << "WARNING: CalculateJacobian: prevent division by zero." << std::endl;
      c1 = 0.0000001;
  }

  double c2 = sqrt(c1);
  double c3 = c1 * c2;

  //compute the Jacobian matrix
  Hj << (px/c2),              (py/c2),                0,    0,
        -(py/c1),             (px/c1),                0,     0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
