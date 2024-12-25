// (c) 2024 Vrund Patel
#include "ekf.h"

BaseEKF::BaseEKF()
{

}

BaseEKF::~BaseEKF()
{

}

void BaseEKF::initialize(const VectorXd& x0, const MatrixXd& P0, const MatrixXd& D, const double t0, const double dt)
{
    x_ = x0;
    P_ = P0;
    D_ = D;
    t_ = t0;
    dt_ = dt;

    nx_ = x0.size();
    nv_ = D.row(0).size();
}

void BaseEKF::setControlInput(const VectorXd& u)
{
    u_ = u;

    nu_ = u.size();
}

void BaseEKF::setProcessNoiseCovariance(const MatrixXd& Q)
{
    Q_ = Q;
}

void BaseEKF::setMeasurementNoiseCovariance(const MatrixXd& R)
{
    R_ = R;

    nz_ = R.cols();
}

void BaseEKF::timeUpdate()
{
    // set the "initial conditions" for the integrator
    VectorXd x0 = x_;
    MatrixXd STM0 = MatrixXd::Identity(nx_, nx_);
    MatrixXd L0 = MatrixXd::Zero(nx_, nv_);

    // stack the "initial conditions" into a single vector
    VectorXd X0(nx_ * (1 + nx_ + nv_));
    X0 << x0, Eigen::Map<VectorXd>(STM0.data(), nx_ * nx_), Eigen::Map<VectorXd>(L0.data(), nx_ * nv_);

    // propagate the state, STM, and L forward
    std::pair<double, VectorXd> sol = integrator(f_full(t_, X0), X0, t_, dt_);

    // extract time and update
    t_ = sol.first;
    // extract state, STM, and L stacked vector
    VectorXd X = sol.second;

    // extract STM
    MatrixXd STM = Eigen::Map<MatrixXd>(X.segment(nx_, nx_ * nx_).data(), nx_, nx_);
    // extract L
    MatrixXd L = Eigen::Map<MatrixXd>(X.segment(nx_ * nx_, nx_ * nv_).data(), nx_, nv_);

    // update state (now a priori state)
    x_ = X.segment(0, nx_);
    // update state covariance (now a priori covariance)
    P_ = STM * P_ * STM.transpose() + L * Q_ * L.transpose();
}

void BaseEKF::measUpdate(const VectorXd& z)
{

    // computed measurement using the measurement model with a priori state
    VectorXd zbar = h_meas(t_, x_);

    // compute measurement Jacobian evaluated at a priori state
    H_ = measJacobian(t_, x_);

    // compute pre-fit measurement residual (observed - computed)
    delz_ = z - zbar;

    // compute innovation covariance matrix
    S_ = H_ * P_ * H_.transpose() + R_;

    // compute Kalman gain matrix
    K_ = P_ * H_.transpose() * S_.inverse();

    // update state (now a posteriori state)
    x_ = x_ + K_ * delz_;
    // update state covariance (now a posteriori covariance)
    MatrixXd I = MatrixXd::Identity(nx_, nx_);
    P_ = (I - K_ * H_) * P_ * (I - K_ * H_).transpose() + K_ * R_ * K_.transpose();
}

std::pair<double, VectorXd> BaseEKF::integrator(
    std::function<VectorXd(const double, const VectorXd&)> f, 
    const VectorXd& X, 
    const double t, 
    const double dt)
{
    VectorXd k1 = f(t, X);
    VectorXd k2 = f(t + dt / 2, X + k1 * dt / 2);
    VectorXd k3 = f(t + dt / 2, X + k2 * dt / 2);
    VectorXd k4 = f(t + dt, X + k3 * dt);

    VectorXd X_sol =  X + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt;
    double t_sol = t + dt;

    return std::make_pair(t_sol, X_sol);
}

VectorXd BaseEKF::f_full(const double t, const VectorXd& X)
{
    // extract state vector
    VectorXd x = X.segment(0, nx_);

    // compute dynamics Jacobian at current state vector
    A_ = dynJacobian(t, x);

    // extract STM
    MatrixXd STM = Eigen::Map<const MatrixXd>(X.segment(nx_, nx_ * nx_).data(), nx_, nx_);

    // extract L
    MatrixXd L = Eigen::Map<const MatrixXd>(X.segment(nx_ * nx_, nx_ * nv_).data(), nx_, nv_);

    // compute STM time derivative
    MatrixXd STM_dot = A_ * STM;
    // compute process noise matrix time derivative
    MatrixXd L_dot = A_ * L + D_;
    // compute state time derivative
    VectorXd x_dot = f_dyn(t, x);

    // stack derivatives
    VectorXd X_dot(nx_ * (1 + nx_ + nv_));
    X_dot << x_dot, Eigen::Map<VectorXd>(STM_dot.data(), nx_ * nx_), Eigen::Map<VectorXd>(L_dot.data(), nx_ * nv_);

    return X_dot;
}
