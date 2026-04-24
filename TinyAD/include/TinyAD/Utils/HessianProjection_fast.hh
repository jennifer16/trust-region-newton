/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Eigenvalues>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>
#include <TinyAD/Utils/Helpers.hh>
//#include <TinyAD/Utils/DiffHessian2.hh>
#include <TinyAD/Utils/DiffHessianProjection.hh>
namespace TinyAD
{

Eigen::VectorXd computeGradientProjection(Eigen::MatrixXd eigenvectors, Eigen::VectorXd& g)
{
    return Eigen::VectorXd::Ones(eigenvectors.rows());
}

/**
 * @brief 高精度模式（牛顿迭代）
 * * return [m_eps, |lambda|]
 */
double optimizeEigenvaluePrecise(
    double lambda,
    double a2,
    double wk,
    double beta,
    double m_eps)
{
    m_eps = TinyAD::EPS_1E_8;
    if (std::abs(lambda) <= m_eps) {
        return m_eps;
    }
    
    // 初始猜测
    double x;
    if (lambda > m_eps) {
        x = lambda;  // 正特征值从原值开始
    } else {
        x = -lambda; // 负特征值从绝对值开始
    }
    
    // 牛顿迭代（3次足够）
    const int max_iter = 3;
    for (int iter = 0; iter < max_iter; ++iter) {
        double f, df, ddf;
        
        if (lambda > m_eps) {
            // 正特征值情形
            double x2 = x * x;
            double x3 = x2 * x;
            double x4 = x3 * x;
            
            f = -a2 / x + 0.5 * a2 * lambda / x2 + beta * wk * (x - lambda) * (x - lambda);
            df = a2 / x2 - a2 * lambda / x3 + 2.0 * beta * wk * (x - lambda);
            ddf = -2.0 * a2 / x3 + 3.0 * a2 * lambda / x4 + 2.0 * beta * wk;
        } else {
            // 负特征值情形
            double b = -lambda;
            double x2 = x * x;
            double x3 = x2 * x;
            double x4 = x3 * x;
            
            f = -a2 / x - 0.5 * a2 * b / x2 + beta * wk * (x + b) * (x + b);
            df = a2 / x2 + a2 * b / x3 + 2.0 * beta * wk * (x + b);
            ddf = -2.0 * a2 / x3 - 3.0 * a2 * b / x4 + 2.0 * beta * wk;
        }
        
        // 牛顿步
        double step = df / ddf;
        x = x - step;
        
        // 确保在可行域内
        if (lambda > m_eps) {
            x = std::max(m_eps, std::min(x, lambda));
        } else {
            x = std::max(m_eps, std::min(x, -lambda));
        }
        
        // 提前收敛
        if (std::abs(step) < 1e-6) break;
    }
    
    return x;
}

/**
 * @brief 简化版优化（无梯度信息）
 */
double optimizeEigenvalueSimple(
    double lambda,
    double beta,
    double m_eps,
    int fast_mode)
{
    m_eps = TinyAD::EPS_1E_8;
    if (fast_mode) 
    {
        if (std::abs(lambda) <= m_eps) 
        {
            return m_eps;
        }
        else if (lambda > m_eps) {
            double alpha = 1.0 - 0.2 * beta / 4.0;  // 简化
            alpha = std::max(0.8, std::min(1.0, alpha));
            return alpha * lambda + (1.0 - alpha) * m_eps;   
        } 
        else 
        {
            double b = -lambda;
            double alpha = std::min(1.0, beta / 4.0);
            return alpha * b + (1.0 - alpha) * m_eps;
        }
    } 
    else 
    {
        // 无梯度信息时，用均匀权重
        return optimizeEigenvaluePrecise(lambda, 1.0, 1.0, beta, m_eps);
    }
}


/**
 * @brief 快速近似模式（解析公式）
 */
double optimizeEigenvalueFast(
    double lambda,
    double a2, //暂时无用
    double wk,
    double beta,
    double m_eps)
{
    m_eps = TinyAD::EPS_1E_8;
    //wk = 1;
    if (std::abs(lambda) <= m_eps) {
        return m_eps;
    }
    else if (lambda > m_eps) {
        // 正特征值：轻微收缩
        // 基础收缩系数
        double alpha = 1.0 - 0.05 * beta * (1.0 + wk); // 收缩程度受β和权重影响
        alpha = std::max(0.8, std::min(1.0, alpha));
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,beta,wk:"<<alpha<<", "<<beta<<", "<< wk);  
        #endif
        return alpha * lambda + (1.0 - alpha) * m_eps;

    } else {
        // 负特征值：放大到绝对值
        double b = -lambda;
        // 基础放大系数,α = min(1, β/4) 为基础，再用权重微调
        double alpha_base = std::min(1.0, beta / 4.0);
        // 用权重微调（贡献大的特征值放大更多）
        double alpha = alpha_base * (0.8 + 0.2 * wk);
        alpha = std::max(0.0, std::min(1.0, alpha));
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("fu特征值增强系数alpha_n,beta,wk:"<<alpha<<", "<<beta<<", "<< wk);  
        #endif
        return alpha * b + (1.0 - alpha) * m_eps;
    }
}


/**
 * @brief 优化单个特征值（带梯度信息）
 */
double optimizeEigenvalue( 
    double lambda, double a2, double wk, double beta, 
    int fast_mode, double m_eps)
{
    
    m_eps = TinyAD::EPS_1E_8;
    switch(fast_mode) 
    {   case 1:
        {
            return optimizeEigenvalueFast(lambda, a2, wk, beta, m_eps);
        }
        default:
        {
            return optimizeEigenvaluePrecise(lambda, a2, wk, beta,m_eps);
        }        
    } 
    
    // if (std::abs(lambda) <= m_eps) {
    //     // 小特征值
    //     new_lambda = m_eps;

    //     modified = true;
    //     // #if DEBUG_OUTPUT
    //     //     TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);
    //     // #endif   
    // }
    // else if (lambda > m_eps) { 
    //     // 正特征值轻微收缩，最大收缩20%
    //     new_lambda = simplifyPositive(lambda, beta,m_eps, w);

    //     modified = true;
    //     #if DEBUG_OUTPUT
    //         TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,lambda,kappa:"<<alpha_p<<  ", " << lambda<<  ", " << kappa); 
    //         // TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
    //     #endif
        
    // }
    // else {
    //     // 只处理负特征值（λ < -ε）
    //     new_lambda = simplifyNegative(lambda, beta,m_eps, w);
    
    //     modified = true;
    //     #if DEBUG_OUTPUT
    //         TINYAD_DEBUG_OUT("负特征值增强系数alpha_n,lambda,kappa,beta:"<<alpha_n<<  ", " << lambda <<  ", " << kappa<<  ", " << beta_n); 
    //         //TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
    //     #endif
    // }
    //return modified;
}

// β 随曲率线性变化：从2到4
double  getBeta(double kappa, double beta0 = 2.0, double gamma = 2.0)
{
    //double beta = 2.0 + 2.0 * kappa; //[2,4]
    double beta = beta0 * (1.0 + gamma * kappa); //[2,6]
    return beta;
}

bool correctLambda(double& new_lambda, double lambda, double m_eps, double kappa)
{
    //double new_lambda = lambda;
    bool modified = false;
    m_eps = TinyAD::EPS_1E_8;
    if (std::abs(lambda) <= m_eps) {
        // 小特征值
        new_lambda = m_eps;
        modified = true;
        
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);
        // #endif
        
    }
    else if (lambda > m_eps) { //正特征值不做修改
        // 正特征值轻微收缩，最大收缩20%
        double alpha_p = 1.0 - 0.3 * kappa;  // 最大收缩20%
        alpha_p = std::max(0.7, std::min(1.0, alpha_p));
        
        // 应用混合
        new_lambda = alpha_p * lambda + (1.0 - alpha_p) * m_eps;
        modified = true;
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,lambda,kappa:"<<alpha_p<<  ", " << lambda<<  ", " << kappa); 
            // TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
        #endif
        
    }
    else {
        // 只处理负特征值（λ < -ε）,曲率越大，beta越大，alpha_n越大，负特征值增强越大
        double b = -lambda;  // 绝对值个体
        // β_n 随曲率线性变化：从2到5
        double beta_n = 2.0 + 2.0 * kappa;
        
        // 当 b >> eps 时，α ≈ β/2 - 1(从0到1.5)
        // 使用精确公式保证连续性
        double numerator = beta_n * b * b - 2.0 * (b * b - m_eps * m_eps);
        double denominator = 2.0 * (b - m_eps) * (b - m_eps);
        double alpha_n = numerator / denominator;
        
        alpha_n = std::max(0.0, std::min(1.0, alpha_n));
        
        // 应用混合策略
        new_lambda = alpha_n * b + (1 - alpha_n) * m_eps;
    
        modified = true;
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("负特征值增强系数alpha_n,lambda,kappa,beta:"<<alpha_n<<  ", " << lambda <<  ", " << kappa<<  ", " << beta_n); 
            //TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
        #endif
    }
    return modified;
}

bool correctLambdaBlend(double& new_lambda, double lambda, double m_eps, double beta)
{
    //double new_lambda = lambda;
    bool modified = false;
    m_eps = TinyAD::EPS_1E_8;
    if (std::abs(lambda) <= m_eps) {
        // 小特征值
        new_lambda = m_eps;
        modified = true;
        
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);
        // #endif
        
    }
    else if (lambda > m_eps) { //正特征值不做修改
        // 正特征值轻微收缩，最大收缩20%
        double alpha_p = 1.0 - 0.1 * beta;  // 最大收缩20%
        //alpha_p = std::max(0.7, std::min(1.0, alpha_p));
        
        // 应用混合
        new_lambda = alpha_p * lambda + (1.0 - alpha_p) * m_eps;
        modified = true;
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,lambda,new_lambda:"<<alpha_p<<  ", " << lambda<<  ", " << new_lambda); 
            // TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
        #endif
        
    }
    else {
        // 只处理负特征值（λ < -ε）,曲率越大，beta越大，alpha_n越大，负特征值增强越大
        double b = -lambda;  // 绝对值
        
        // 当 b >> eps 时，α ≈ β/2 - 1(从0到1.5)
        // 使用精确公式保证连续性
        // double numerator = beta_n * b * b - 2.0 * (b * b - m_eps * m_eps);
        // double denominator = 2.0 * (b - m_eps) * (b - m_eps);
        // double alpha_n = numerator / denominator;
        double alpha_n = std::min(1.0, beta/4.0);
        //double alpha_n = std::min(1.2, beta/2.0 - 1.0);
        

        //alpha_n = std::max(0.0, std::min(1.0, alpha_n));
        
        // 应用混合策略
        new_lambda = alpha_n * b + (1 - alpha_n) * m_eps;
    
        modified = true;
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("负特征值增强系数alpha_n,lambda,new_lambda,beta:"<<alpha_n<<  ", " << lambda <<  ", " << new_lambda<<  ", " << beta); 
            //TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
        #endif
    }
    return modified;
}

constexpr double default_hessian_projection_eps = 1e-9;

/**
 * Check if matrix is diagonally dominant and has positive diagonal entries.
 * This is a sufficient condition for positive-definiteness
 * and can be used as an early out to avoid eigen decomposition.
 */
template <int k, typename PassiveT>
bool positive_diagonally_dominant(
        Eigen::Matrix<PassiveT, k, k>& _H,
        const PassiveT& _eps)
{
    for (Eigen::Index i = 0; i < _H.rows(); ++i)
    {
        PassiveT off_diag_abs_sum = 0.0;
        for(Eigen::Index j = 0; j < _H.cols(); ++j)
        {
            if (i != j)
                off_diag_abs_sum += std::abs(_H(i, j));
        }

        if (_H(i, i) < off_diag_abs_sum + _eps)
            return false;
    }

    return true;
}

/**
 * Project symmetric matrix to positive-definite matrix
 * via eigen decomposition.
 */
template <int k, typename PassiveT>
void project_positive_definite(
        Eigen::Matrix<PassiveT, k, k>& _H,
        const PassiveT& _eigenvalue_eps)
{
    if constexpr (k == 0)
    {
        return;
    }
    else
    {
        using MatT = Eigen::Matrix<PassiveT, k, k>;

        // Early out if sufficient condition is fulfilled
        if (positive_diagonally_dominant<k, PassiveT>(_H, _eigenvalue_eps))
            return;

        // Compute eigen-decomposition (of symmetric matrix)
        Eigen::SelfAdjointEigenSolver<MatT> eig(_H);
        MatT D = eig.eigenvalues().asDiagonal();  

        // Clamp all eigenvalues to eps
        bool all_positive = true;
        for (Eigen::Index i = 0; i < _H.rows(); ++i)
        {
            if (_eigenvalue_eps < 0) {
                // project to absolute value
                if (D(i, i) < 0)
                {
                    D(i, i) = -D(i, i);
                    all_positive = false;
                }
            }
            else {
                // project to epsilon
                if (D(i, i) < _eigenvalue_eps)
                {
                    D(i, i) = _eigenvalue_eps;
                    all_positive = false;
                }
            }
        }

        // Do nothing if all eigenvalues were already at least eps
        if (all_positive)
            return;

        // Re-assemble matrix using clamped eigenvalues
        _H = eig.eigenvectors() * D * eig.eigenvectors().transpose();
        TINYAD_ASSERT_FINITE_MAT(_H);
    }
}

// 范围 [0, 1]
double computeKappa(int kappaMethod, const Eigen::VectorXd& eigenvalues, double m_eps)
{
    double kappa = 0; // 范围 [0, 1]
    m_eps = TinyAD::EPS_1E_8;
    switch(kappaMethod) {
        case 1:
        {
            //==== 计算 kappa(只有有正有负的时候正确) ====
            // 1. 计算特征值的最大值和最小值
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();
            // 2. 计算曲率度量 κ_i (方案A: 基于特征值各向异性)
            // κ = (λ_max - λ_min) / (|λ_max| + |λ_min| + ε)
            if (TinyAD::isPostive(lambda_max * lambda_min)) //同号，这是不正常的
            {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("ERROR:lambda_max * lambda_min >0,lambda_max,lambda_min :"<<lambda_max<<","<<lambda_min ); 
                #endif
                
                // 这个比值度量了Hessian的各向异性程度
                double nominator = lambda_max - lambda_min;
                double denominator =  std::abs(lambda_max)+ std::abs(lambda_min) + m_eps;
                
                kappa = nominator / denominator;
                // 确保κ在合理范围内 [0, 1]
                kappa = std::min(1.0, kappa);
            }
            else //不同号
            {
                // 这个比值度量了Hessian的各向异性程度
                double nominator = lambda_max - lambda_min;
                double denominator = nominator + std::abs(lambda_max+lambda_min) + m_eps;
                
                kappa = nominator / denominator;
                // 确保κ在合理范围内 [0, 1]
                kappa = std::min(1.0, kappa);
            }
        
            break;
        }
        case 2:
        {
            //==== 计算 kappa ====
            // 计算负特征值能量占比
            double sum_abs_all = 0.0;
            double sum_abs_neg = 0.0;
            for (int i = 0; i < eigenvalues.size(); ++i) {
                double val = eigenvalues(i);
                double abs_val = std::abs(val);
                sum_abs_all += abs_val;
                if (TinyAD::isNonPostive(val)) {
                    sum_abs_neg += abs_val;
                }
            }
            double neg_ratio = sum_abs_neg / (sum_abs_all + m_eps);

            double alpha = 4.0;  // 敏感度参数
            kappa = 1.0 - std::exp(-alpha * neg_ratio);
            
            // // // 可选：体积比增强
            // // double J_factor = std::abs(std::log(J)) / 5.0;
            // // J_factor = std::min(1.0, J_factor);
            
            // // 取最大，确保非凸时曲率大
            // return std::max(kappa, J_factor);

            break;
        }
        case 3:
        {
            //==== 计算 kappa(只有有正有负的时候正确) ====
            // 1. 计算特征值的最大值和最小值
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();
            
            // 判断2：相对于第一拉梅系数（捕捉体积变形）
            if (lambda_max < TinyAD::g_LAMBDA * 1e-6)
            {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("lambda_max < TinyAD::g_LAMBDA * 1e-6 " ); 
                #endif
                return 1.0;
            }
            // 判断1：相对于剪切模量
            if (lambda_max < TinyAD::g_MU * 1e-4)
            {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("lambda_max < TinyAD::g_MU * 1e-4 " ); 
                #endif
                return 1.0;
            }

            // // 判断3：相对阈值
            // double min_abs_value = eigenvalues.cwiseAbs().minCoeff();
            // if (min_abs_value < 1e-8)
            // {
            //     min_abs_value = 1e-8;
            // }
            // if (min_abs_value < lambda_max * 1e-4) {
                
            //     #if DEBUG_OUTPUT
            //         TINYAD_DEBUG_OUT("min_abs_value < lambda_max * 1e-4: "<< min_abs_value); 
            //     #endif
            //     return 1.0;
            // }

            // // 判断4：最小正特征值与最大负特征值对比
            // if (lambda_min < 0 && min_abs_value < -lambda_min * 1e-4) {
            //     #if DEBUG_OUTPUT
            //         TINYAD_DEBUG_OUT("min_abs_value < -lambda_min * 1e-4 " ); 
            //     #endif
            //     return 1.0;
            // }

            // 2. 计算曲率度量 κ_i (方案A: 基于特征值各向异性)
            // κ = (λ_max - λ_min) / (|λ_max| + |λ_min| + ε)
            // 这个比值度量了Hessian的各向异性程度
            double nominator = lambda_max - lambda_min;
            double denominator = nominator + std::abs(lambda_max+lambda_min) + m_eps;
            
            kappa = nominator / denominator;
            // 确保κ在合理范围内 [0, 1]
            kappa = std::min(1.0, kappa);

            break;
        }
    }

    return kappa;
}

/**
 * Project symmetric matrix to positive-definite matrix
 * via eigen decomposition. 
 * added by zj (Differentiable version)
 */
template <int k, typename PassiveT>
void project_positive_definite_diff(
        Eigen::Matrix<PassiveT, k, k>& _H,
        const PassiveT& _eigenvalue_eps,
        HessianProjectionMode _mode = HessianProjectionMode::AUTO)
{
    // #if DEBUG_OUTPUT
    //     TINYAD_DEBUG_OUT("Projection mode in project_positive_definite_diff: " << static_cast<int>(_mode)); 
    // #endif
    if constexpr (k == 0)
    {
        return;
    }
    else
    {
        using MatT = Eigen::Matrix<PassiveT, k, k>;

        // Early out if sufficient condition is fulfilled
        if (positive_diagonally_dominant<k, PassiveT>(_H, _eigenvalue_eps))
            return;

        // ===== 1. diff project =====
        // Compute eigen-decomposition (of symmetric matrix)
        Eigen::SelfAdjointEigenSolver<MatT> eig(_H);
        auto eigenvalues = eig.eigenvalues();
        auto eigenvectors = eig.eigenvectors();
        
        // 创建正则化器
        EigenvalueRegularizer<PassiveT> regularizer(_eigenvalue_eps,_mode);
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("_eigenvalue_eps in project_positive_definite_diff:"<<static_cast<double>(_eigenvalue_eps));   
        // #endif
        bool modified = false;
        if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING
        ||_mode == HessianProjectionMode::CLAMP_ABS_BLENDING2)
        {
            // 将β代入变分框架，计算混合系数α
            double m_eps = 0;  // 小特征值阈值
            m_eps = TinyAD::EPS_1E_8;

            // 2. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            // #if DEBUG_OUTPUT
            //     TINYAD_DEBUG_OUT("Type: "<< typeid(eigenvalues).name()); 
            // #endif
            
            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 2;
            double kappa = computeKappa(kappaMethod, eigenvalues, m_eps);
            // // fast by zj
            double beta0 = 2.0 *_eigenvalue_eps; 
            double gamma = 2.0; 
            // // adaptive by zj
            // double beta0 = 2.0 *_eigenvalue_eps; 
            // double gamma = 1.0;
            double beta = getBeta(kappa, beta0, gamma);// [0-6], beta =0, clamp,其他情况为blending
            
            /* 
            * correct lambda
            */
            int correctLambdaMethod = 2; // 1: 非统一框架；2，统一框架; 3:统一框架+正负blending
            int fast_mode = 1; //1.fast;其他 newton
            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT("kappa, beta0, gamma, beta:"<<kappa<<", "<<beta0<<", "<<gamma<<", "<<beta); 
                TINYAD_DEBUG_OUT("kappaMethod, correctLambdaMethod, fast_mode:"<<kappaMethod<<", "<<correctLambdaMethod<<", "<<fast_mode); 
            #endif
            switch(correctLambdaMethod) {
                case 1:
                {
                    for (size_t i = 0; i < k; ++i) {
                        double lambda = eigenvalues(i);
                        double new_lambda = lambda;
                        if (correctLambda(new_lambda, lambda, m_eps, kappa))
                        {
                            modified = true;
                        }
                        
                        eigenvalues(i) = new_lambda;    
                    }
                    break;
                }
                case 2:
                {
                    // // 4. 计算正则化参数 β
                    // double beta0 = 2.0; // 
                    // double gamma = 2.0; // 
                    // double beta = beta0 * (1.0 + gamma * kappa);

                    // 5. 计算梯度投影 a_k^2
                    Eigen::VectorXd g = Eigen::VectorXd::Ones(k);
                    Eigen::VectorXd a2 = computeGradientProjection(eigenvectors, g);
                    double sum_a2 = a2.sum();
                    if (sum_a2 < 1e-16) 
                    {
                        sum_a2 = 1.0;  // 避免除零
                    }

                    // 6. 优化每个特征值
                    for (size_t i = 0; i < k; ++i) {
                        double lambda = eigenvalues(i);
                        double new_lambda = lambda;
                        double wi = a2(i) / sum_a2;
                        //correctLambda(double& new_lambda, double lambda, double m_eps, double beta, double w = 1.0)
                        new_lambda = optimizeEigenvalue(
                            eigenvalues(i), a2(i), wi, beta, fast_mode,m_eps);
                        
                        eigenvalues(i) = new_lambda;    
                    }
                    modified = true;
                    break;
                }
                case 3:
                {
                    for (size_t i = 0; i < k; ++i) {
                        double lambda = eigenvalues(i);
                        double new_lambda = lambda;
                        if (correctLambdaBlend(new_lambda, lambda, m_eps, beta))
                        {
                            modified = true;
                        }
                        
                        eigenvalues(i) = new_lambda;    
                    }

                    break;
                }
            }
            
            
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else
        { 
            for (Eigen::Index i = 0; i < k; ++i) {
                PassiveT old_val = eigenvalues(i);
                
                if (regularizer.needs_regularization(old_val)) {

                    eigenvalues(i) = regularizer.regularize(old_val);
                    modified = true;

                    // #if DEBUG_OUTPUT
                    //     //TINYAD_DEBUG_OUT("Projection mode in project_positive_definite_diff: " << static_cast<int>(_mode));
                    //     TINYAD_DEBUG_OUT("ev("<<i<<"):" << old_val << " -> " << eigenvalues(i)); 
                    //     //TINYAD_DEBUG_OUT("*evec("<<i<<")=" << eigenvectors.col(i).transpose()); 
                    // #endif
                }
            }
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }

        #if 0
        auto shift_value = 0.0;
        //if (_eigenvalue_eps > 1.0)// shift
        if (_mode == HessianProjectionMode::ABS_SHIFT_CLAMP_NONDIFF)
        {
            auto alpha_shear = 0.0;
            // 找到特征值的"断点"
            std::vector<double> vals(eigenvalues.data(), 
                                        eigenvalues.data() + eigenvalues.size());
            
            // 计算剪切模态的能量占比
            double sum_shear = 0.0;
            double sum_total = 0.0;
            double lambda_min = vals[0];
            double lambda_max = vals[0];
            for (size_t i = 0; i < vals.size(); ++i) {
                if (std::abs(vals[i]) < 1e-4){
                    sum_shear += std::abs(vals[i]);
                }
                
                sum_total += std::abs(vals[i]); 
                if (vals[i] < lambda_min) 
                {
                    lambda_min = vals[i];
                }
                
                if (vals[i] > lambda_max) 
                {
                    lambda_max = vals[i];
                }
            }

            alpha_shear = sum_shear / sum_total;
            
            
            if (lambda_min < 0)
            {
                shift_value += (std::abs(lambda_min) + 1e-8);
                lambda_min += shift_value;
                lambda_max += shift_value;
            }
            double alpha_opt = std::sqrt(std::abs(lambda_min) * std::abs(lambda_max));
        
            // 根据剪切程度调整
            if (alpha_shear > 0.95) {
                // 高度剪切：需要保护小特征值
                shift_value +=  alpha_opt * 0.1;
            } else if (alpha_shear > 0.8) { // shear dominated
                shift_value += alpha_opt * 0.5;
            } else if (alpha_shear > 0.3) { // mixed
                shift_value += alpha_opt;
            } else {
                // 体积主导：可以大一些加速收敛
                shift_value += alpha_opt * 2.0;
            }
            #if SHEAR_DEBUG_OUTPUT
                TINYAD_DEBUG_OUT("alpha_shear,alpha_opt,min,max,shift: "<<alpha_shear <<","<<alpha_opt<<","<<lambda_min <<","<<lambda_max <<","<<shift_value <<","); 
            #endif

            // for (Eigen::Index i = 0; i < k; ++i) {
            //     PassiveT old_val = eigenvalues(i);
            //     eigenvalues(i) = old_val + shift_value;
                    
            //     #if SHEAR_DEBUG_OUTPUT
            //         //TINYAD_DEBUG_OUT("Projection mode in project_positive_definite_diff: " << static_cast<int>(_mode));
            //         TINYAD_DEBUG_OUT("ev("<<i<<"):" << old_val << " -> " << eigenvalues(i)); 
            //         //TINYAD_DEBUG_OUT("*evec("<<i<<")=" << eigenvectors.col(i).transpose()); 
            //     #endif   
            // }
            //modified = true;
        }
        #endif
        //else
        #if 0
        if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING)
        {
            // #if DEBUG_OUTPUT
            //     TINYAD_DEBUG_OUT("blending: " );
            //     //TINYAD_DEBUG_OUT("ev("<<i<<"):" << old_val << " -> " << eigenvalues(i));  
            // #endif

            double m_eps = 1e-8;
            // TinyAD::CurvatureBetaCalculator betaCalc(1.0, 1.5, eps);
            // auto [kappa, beta] = betaCalc.compute(eigenvalues);
            // 或者使用体积比增强版本
            // double J = computeDetF(F);  // 计算det(F)
            // auto [kappa, beta] = betaCalc.computeWithJ(eigenvalues, J);
            
            // 将β代入变分框架，计算混合系数α
            double kappa = 1.0;
            double beta = 1.0;

            double m_beta0 = 1.5;
            double m_gamma = 1.5;

            // 1. 计算特征值的最大值和最小值
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();
            
            // 2. 计算曲率度量 κ_i (方案A: 基于特征值各向异性)
            // κ = (λ_max - λ_min) / (λ_max + λ_min + ε)
            // 这个比值度量了Hessian的各向异性程度
            double denominator = std::abs(lambda_max) + std::abs(lambda_min) + m_eps;
            
            kappa = (std::abs(lambda_max) - std::abs(lambda_min)) / denominator;
            kappa = std::abs(kappa);  // 取绝对值，保证非负
            
            // 确保κ在合理范围内 [0, 1]
            kappa = std::max(0.0, std::min(1.0, kappa));
            
            // 3. 计算自适应正则化参数 β_i
            // β = β₀ * (1 + γ * κ)
            // 曲率越大 → β越大 → 更激进的滤波（放大负特征值）
            beta = m_beta0 * (1.0 + m_gamma * kappa);
            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT("kappa, beta,lambda_min, lambda_max:"<<kappa<<"," << beta<<","<< lambda_min<<"," << lambda_max); 
            #endif

            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                double new_lambda = lambda;

                if (std::abs(lambda) <= m_eps) {
                    // 小特征值
                    new_lambda = m_eps;

                    modified = true;
                    #if DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);
                    #endif
                }
                else if (lambda > m_eps) { //正特征值不做修改
                    // // 正特征值，曲率越大，beta越大，alpha_p越小,正特征值越被弱化
                    double numerator = beta * lambda * lambda;
                    double denominator = 2.0 * (lambda - m_eps) * (lambda - m_eps);
                    double alpha_p = 1.0 - numerator / denominator;
                    
                    // 钳位到 [0,1]
                    alpha_p = std::max(0.0, std::min(1.0, alpha_p));
                    
                    // 应用混合
                    new_lambda = alpha_p * lambda + (1.0 - alpha_p) * m_eps;
                    #if DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("正特征值增强系数alpha_p:"<<alpha_p); 
                        TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
                    #endif
                }
                else {
                    // 只处理负特征值（λ < -ε）,曲率越大，beta越大，alpha_n越大，负特征值增强越大
                    double b = -lambda;  // 绝对值
                    
                    // 混合系数的解析解（由变分问题导出）
                    // α* = clamp((β·b² - 2(b² - ε²)) / (2(b - ε)²), 0, 1)
                    //double eps = 1e-8;
                    double numerator = beta * b * b - 2.0 * (b * b - m_eps * m_eps);
                    double denominator = 2.0 * (b - m_eps) * (b - m_eps);
                    double alpha_n = numerator / denominator;
                    
                    // 钳位到[0,1]
                    alpha_n = std::max(0.0, std::min(1.0, alpha_n));
                    
                    // 应用混合策略
                    new_lambda = alpha_n * b + (1.0 - alpha_n) * m_eps;
                
                    modified = true;
                    #if DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("负特征值增强系数alpha_n:"<<alpha_n); 
                        TINYAD_DEBUG_OUT("blending ev("<<i<<"):" << lambda << " -> " << new_lambda);  
                    #endif
                }
                eigenvalues(i) = new_lambda;
                
            }
            
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        #endif
        
        
        
        #if 0
        if (modified) {
            MatT D = eig.eigenvalues().asDiagonal();  

            // Clamp all eigenvalues to eps
            bool all_positive = true;
            for (Eigen::Index i = 0; i < _H.rows(); ++i)
            {
                if (_eigenvalue_eps < 0) {
                    // project to absolute value
                    if (D(i, i) < 0)
                    {
                        D(i, i) = -D(i, i);
                        all_positive = false;
                    }
                }
                else {
                    // project to epsilon
                    if (D(i, i) < _eigenvalue_eps)
                    {
                        D(i, i) = _eigenvalue_eps;
                        all_positive = false;
                    }
                }
            }
             
            //check eigenvalues and D.diagonal() have the same values after regularization
            TINYAD_DEBUG_OUT("Checking regularized eigenvalues...");
            TINYAD_ASSERT_EQ(eigenvalues.size(), D.diagonal().size());
            for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
                TINYAD_ASSERT_EPS(eigenvalues(i), D(i,i), 1e-6);
            }
            
            //check eigenvectors are the same as beforeregularization
            TINYAD_DEBUG_OUT("Checking regularized eigenvectors...");
            TINYAD_ASSERT_EPS_MAT(eig.eigenvectors(), eigenvectors, 1e-6);   
        }
        #endif 

        if (modified) {
            _H = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
            TINYAD_ASSERT_FINITE_MAT(_H);
        }

        //#endif
       
    }
}

}




