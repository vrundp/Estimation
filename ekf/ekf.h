// (c) 2024 Vrund Patel
#pragma once

#include <Eigen/Dense>
#include <functional>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
Extended Kalman Filter
dynamics model: dx/dt = f(t, x, u) + D * v(t)
note: process noise in the dynamics function of the filter
      is assumed to be zero mean; the model uncertainty is 
      accounted for in the covariance based on D and Q
measurement model: z = h(t, x) + w(t)
*/
class BaseEKF
{
public:

    BaseEKF();
    virtual ~BaseEKF();

    /*
    Initialize EKF with state and covariance along with process
    noise mapping matrix, timestamp corresponding to initial state,
    and filter update interval
    */
    void initialize(
        const VectorXd& x0, 
        const MatrixXd& P0, 
        const MatrixXd& D,
        const double t0, 
        const double dt);

    /*
    Set a control input for the dynamics
    note: derived class will need to use member variable in
          dynamics model implementation if control input needed
    */
    void setControlInput(const VectorXd& u);

    /*
    Set the process noise covariance matrix
    note: usually a diagonal matrix
    */
    void setProcessNoiseCovariance(const MatrixXd& Q);

    /*
    Set the measurement noise covariance matrix
    note: usually a diagonal matrix
    */
    void setMeasurementNoiseCovariance(const MatrixXd& R);

    /*
    Compute the a priori state and covariance
    */
    void timeUpdate();

    /*
    Compute the a posteriori state and covariance by processing 
    raw measurement (observation)
    */
    void measUpdate(const VectorXd& z);

protected:

    // dimension of state
    unsigned int nx_;
    // dimension of control input
    unsigned int nu_;
    // dimension of process noise
    unsigned int nv_;
    // dimension of measurement
    unsigned int nz_;

    // time
    double t_;
    // filter update interval
    double dt_;
    // state vector
    VectorXd x_;
    // state covariance matrix
    MatrixXd P_;
    // control input
    VectorXd u_;
    // process noise covariance matrix
    MatrixXd Q_;
    // measurement noise covariance matrix
    MatrixXd R_;
    // dynamics Jacobian matrix
    MatrixXd A_;
    // measurement Jacobian matrix
    MatrixXd H_;
    // process noise mapping matrix
    MatrixXd D_;
    // linearized state transition matrix
    MatrixXd STM_;
    // linearized process noise matrix
    MatrixXd L_;
    // pre-fit measurement residual (innovation)
    VectorXd delz_;
    // innovation covariance matrix
    MatrixXd S_;
    // Kalman gain matrix
    MatrixXd K_;

private:

    /*
    Dynamics function that computes the time derivative of state vector, STM, and L
    */
    VectorXd f_full(const double t, const VectorXd& X);
    
    /* 
    Pure virtual function that computes the non-linear dynamics function of the system
    as dx/dt = f(x, u, t) where x is the state, u is control input, t is the time
    */
    virtual VectorXd f_dyn(const double t, const VectorXd& x) = 0;

    /*
    Pure virtual function that computes the Jacobian of the system dynamics
    as A = df/dx evaluated at the a posteriori state (before time update)
    */
    virtual MatrixXd dynJacobian(const double t, const VectorXd& x) = 0;

    /*
    Pure virtual function that computes the non-linear measurement function
    as z = h(x, t) where x is the state and t is the time
    */
    virtual VectorXd h_meas(const double t, const VectorXd& x) = 0;

    /*
    Pure virtual function that computes the Jacobian of the measurement function
    as H = dh/dx evaluated at the a priori state (after time update, before measurement update)
    */
    virtual MatrixXd measJacobian(const double t, const VectorXd& x) = 0;

    /*
    Virtual function that implements an integrator that solves a differential
    equation of the form dX/dt = f(t, X) as X = integral of f(t', X) dt' from t to t + dt
    where X is some vector quantity that needs to be solved for and t is the time
    */
    virtual std::pair<double, VectorXd> integrator(
        std::function<VectorXd(const double, const VectorXd&)> f, 
        const VectorXd& X, 
        const double t, 
        const double dt);
};