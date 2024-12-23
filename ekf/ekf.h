#pragma once

#include <Eigen/Dense>
#include <functional>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
Base EKF
system dynamics are modeled with non-additive noise
measurements are modeled with additive noise
*/
class BaseEKF
{
public:

    BaseEKF();
    virtual ~BaseEKF();

    void initialize(const VectorXd& x0, const MatrixXd& P0, const MatrixXd& D, const double t0);

    void setProcessNoiseCovariance(const MatrixXd& Q);

    void setMeasurementNoiseCovariance(const MatrixXd& R);

    void timeUpdate();

    void measUpdate(const VectorXd& z);

private:

    // state vector
    VectorXd x_;
    // state covariance matrix
    MatrixXd P_;
    // process noise covariance matrix
    MatrixXd Q_;
    // measurement noise covariance matrix
    MatrixXd R_;
    // dynamics Jacobian matrix
    MatrixXd A_;
    // measurement Jacobian matrix
    MatrixXd H_;
    // linearized state transition matrix
    MatrixXd STM_;
    // linearized process noise matrix
    MatrixXd Gamma_;

    /* 
    Pure virtual function that computes the non-linear dynamics function of the system
    as dx/dt = f(x, u, t) where x is the state, u is control input, t is the time
    */
    virtual VectorXd f_dyn(const VectorXd& x, const VectorXd& u, const double t) = 0;

    /*
    Pure virtual function that computes the Jacobian of the system dynamics
    as A = df/dx
    */
    virtual MatrixXd dynJacobian(const VectorXd& x, const VectorXd& u) = 0;

    /*
    Pure virtual function that computes the non-linear measurement function
    as z = h(x) where x is the state
    */
    virtual VectorXd h_meas(const VectorXd& x) = 0;

    /*
    Pure virtual function that computes the Jacobian of the measurement function
    as H = dh/dx
    */
    virtual MatrixXd measJacobian(const VectorXd& x) = 0;

    /*
    Pure virtual function that implements an integrator that solves a differential
    equation of the form dX/dt = f(X, u, t) as X = integral of f(X, u , t') dt' from t to t + dt
    where X is some vector quantity that needs to be solved for, u is the control input, t is the time
    */
    virtual VectorXd integrator(
        std::function<VectorXd(const VectorXd&, const VectorXd&,  double)> f, 
        const VectorXd& X, 
        const double t, 
        const double dt) = 0;

    /*
    Function the propagates a generic state vector from t to t + dt using the integrator function
    */
    VectorXd propagate(const VectorXd& X, const double t, const double dt);
};