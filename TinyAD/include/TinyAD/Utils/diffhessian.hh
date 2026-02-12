/*
 * This file released under the MIT license.
 * Author: Jennifer Zhang
 */
// differentiable_clamping.hpp
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <functional>

namespace TinyAD {
template<typename Scalar>
class DifferentiableHessianClamping {
public:
    // 构造函数，设置参数
    DifferentiableHessianClamping(
        Scalar epsilon = 1e-6, Scalar delta = 1e-3, Scalar beta = 10.0) 
        : epsilon_(epsilon), delta_(delta), beta_(beta) {}

    // 对Hessian矩阵进行可微分clamping
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> 
    clamp(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H)
    {
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H) {
        
        // 确保矩阵是对称的（Hessian矩阵应该对称）
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H_sym = 
            0.5 * (H + H.transpose());
        
        // 进行可微分的特征值分解
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q;
        Eigen::Vector<Scalar, Eigen::Dynamic> lambda;
        differentiable_eigendecomposition(H_sym, Q, lambda);
        
        // 对特征值进行可微分正则化
        Eigen::Vector<Scalar, Eigen::Dynamic> lambda_reg(lambda.size());
        for (int i = 0; i < lambda.size(); ++i) {
            lambda_reg(i) = regulate_eigenvalue(lambda(i));
        }
        
        // 重构矩阵
        return Q * lambda_reg.asDiagonal() * Q.transpose();
    }
    
    
    
    // 对对称矩阵进行可微分正则化
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
    regularize_symmetric(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M){
        
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> 
            solver(M);
        
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen decomposition failed");
        }
        
        Eigen::Vector<Scalar, Eigen::Dynamic> eigenvalues = solver.eigenvalues();
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigenvectors = 
            solver.eigenvectors();
        
        // 对特征值进行可微分正则化
        for (int i = 0; i < eigenvalues.size(); ++i) {
            eigenvalues(i) = regulate_eigenvalue(eigenvalues(i));
        }
        
        return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    }
    
    // 获取特征值正则化函数（用于自动微分）
    std::function<Scalar(Scalar)> get_eigenvalue_regulator() const {
        return [this](Scalar lambda) -> Scalar {
            return regulate_eigenvalue(lambda);
        };
    }

    
    // 可微分的特征值分解（使用Jacobi方法）
    void DifferentiableHessianClamping<Scalar>::differentiable_eigendecomposition(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& eigenvectors,
        Eigen::Vector<Scalar, Eigen::Dynamic>& eigenvalues,
        int max_iterations) const {
        
        int n = M.rows();
        eigenvectors = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A = M;
        
        // Jacobi迭代（可微分）
        for (int iter = 0; iter < max_iterations; ++iter) {
            Scalar max_offdiag = 0;
            int p = 0, q = 1;
            
            // 找到最大非对角线元素
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (std::abs(A(i, j)) > max_offdiag) {
                        max_offdiag = std::abs(A(i, j));
                        p = i;
                        q = j;
                    }
                }
            }
            
            if (max_offdiag < 1e-12) break;
            
            // 计算旋转角度（可微分）
            Scalar tau = (A(q, q) - A(p, p)) / (Scalar(2) * A(p, q));
            Scalar t = (tau >= 0 ? Scalar(1) : Scalar(-1)) / 
                    (std::abs(tau) + std::sqrt(Scalar(1) + tau * tau));
            Scalar c = Scalar(1) / std::sqrt(Scalar(1) + t * t);
            Scalar s = t * c;
            
            // 应用Jacobi旋转
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J = 
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
            J(p, p) = c; J(q, q) = c;
            J(p, q) = s; J(q, p) = -s;
            
            A = J.transpose() * A * J;
            eigenvectors = eigenvectors * J;
        }
        
        eigenvalues = A.diagonal();
    }
    
    // 设置参数
    void set_epsilon(Scalar eps) { epsilon_ = eps; }
    void set_delta(Scalar delta) { delta_ = delta; }
    void set_beta(Scalar beta) { beta_ = beta; }
    
private:
    Scalar epsilon_;  // 小特征值阈值
    Scalar delta_;    // 过渡区间宽度
    Scalar beta_;     // 光滑参数
    
    // 可微分的SVD（使用Jacobi方法）
    void differentiable_svd(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M,
                          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U,
                          Eigen::Vector<Scalar, Eigen::Dynamic>& S,
                          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V,
                          int max_iterations = 20) const;
    
    // 可微分的特征值分解（对称矩阵）
    void differentiable_eigendecomposition(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& eigenvectors,
        Eigen::Vector<Scalar, Eigen::Dynamic>& eigenvalues,
        int max_iterations = 30) const;

    // 可微分的特征值正则化
    Scalar DifferentiableHessianClamping<Scalar>::regulate_eigenvalue(Scalar lambda) const {
        // 处理负值：使用可微分的绝对值近似
        Scalar sign = std::tanh(beta_ * lambda);  // 近似符号函数
        Scalar abs_lambda = soft_abs(lambda);
        
        // 处理小特征值：光滑的clamping
        if (abs_lambda < epsilon_) {
            // 在小值区域，光滑地投影到epsilon
            Scalar t = (abs_lambda - epsilon_) / delta_;
            Scalar sigmoid = Scalar(1) / (Scalar(1) + std::exp(-beta_ * t));
            return sign * (epsilon_ * sigmoid + abs_lambda * (Scalar(1) - sigmoid));
        } 
        else if (abs_lambda < epsilon_ + delta_) {
            // 在过渡区域
            Scalar t = (abs_lambda - epsilon_) / delta_;
            Scalar sigmoid = Scalar(1) / (Scalar(1) + std::exp(-beta_ * t));
            Scalar clamped = epsilon_ + (abs_lambda - epsilon_) * sigmoid;
            return sign * clamped;
        }
        else {
            // 大特征值，保持不变
            return lambda;
        }
    }

    
    // 光滑的clamping函数
    Scalar smooth_clamp(Scalar x, Scalar min_val, Scalar max_val) const{
    
        // 使用log-sum-exp实现光滑clamping
        Scalar alpha = beta_;  // 光滑参数
        
        // 下界clamping
        Scalar lower = min_val + 
                    std::log(Scalar(1) + std::exp(alpha * (x - min_val))) / alpha;
        
        // 上界clamping（如果需要）
        // Scalar upper = max_val - 
        //                std::log(Scalar(1) + std::exp(alpha * (max_val - x))) / alpha;
        
        return lower;
    }
    
    // 软绝对值（可微近似）
    Scalar soft_abs(Scalar x) const{
        // 使用平滑的绝对值近似
        // sqrt(x^2 + epsilon) 或 x * tanh(beta * x)
        Scalar alpha = Scalar(10);  // 陡峭度参数
        return std::sqrt(x * x + Scalar(1e-8));
        // 或者：return x * std::tanh(alpha * x);
    }
};


// 显式实例化常用类型
template class DifferentiableHessianClamping<float>;
template class DifferentiableHessianClamping<double>;

}; // namespace TinyAD