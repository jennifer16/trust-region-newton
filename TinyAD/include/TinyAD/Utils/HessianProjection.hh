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

    class EigenvalueBlendingRegularizer {
    public:
        EigenvalueBlendingRegularizer(double rho = TinyAD::EPS_1E_8 , double eps = TinyAD::EPS_1E_8, 
            int kappa_mode = 0, int beta_mode = 0,
            int  pos_mode = 0, int neg_mode = 0,
            bool lower_bound = false) 
            : m_rho(rho), m_eps(eps),  
            m_pos_mode(pos_mode), m_neg_mode(neg_mode), 
            m_beta_mode(beta_mode), m_kappa_mode(kappa_mode),
            m_use_lower_bound(lower_bound){
            // 自动设置平滑参数
                m_beta_max = 4.0; // 限制beta最大值，避免过激
                m_gamma = 2.0; 

                m_lambda_min = 0.0;
                
                m_beta = 0.0;
                m_kappa = 0.0;
             
        }

        void print()
        {
            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT("m_pos_mode, m_neg_mode:"<<m_pos_mode<<", "<<m_neg_mode); 
                TINYAD_DEBUG_OUT("m_beta_mode, m_kappa_mode:"<<m_beta_mode<<", "<<m_kappa_mode); 
                TINYAD_DEBUG_OUT("m_use_lower_bound:"<<m_use_lower_bound); 
                TINYAD_DEBUG_OUT("m_kappa, m_beta:"<<m_kappa<<", "<<m_beta); 
                TINYAD_DEBUG_OUT("m_beta_max, m_gamma:"<<m_beta_max<<", "<<m_gamma); 
                TINYAD_DEBUG_OUT("m_rho, m_eps:"<<m_rho<<", "<<m_eps); 
            #endif
        }

        void setBeta_mode(int mode) {
            m_beta_mode = mode;
        }
        void set_pos_mode(int mode) {
            m_pos_mode = mode;
        }
        void set_neg_mode(int mode) {
            m_neg_mode = mode;
        }

        void set_kappa_mode(int mode) {
            m_kappa_mode = mode;
        }



        /**
         * @brief 快速近似模式（解析公式）
         */
        double blending(double lambda,double wk)
        {
            //m_eps = TinyAD::EPS_1E_8;
            //wk = 1;
            if (std::abs(lambda) <= m_eps) {
                return m_eps;
            }
            else if (lambda > m_eps) {
                return blendingPositiveEigenvalue(lambda, wk);

            } else {
                return blendingNegativeEigenvalue(lambda, wk);
            }
        }

        /**
         * @brief 处理正特征值的混合策略
          1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w)，其中w是权重，β是正则化强度。这样可以根据β和权重调整收缩程度。
         */
        double blendingPositiveEigenvalue(double lambda, double wk)
        {
            double new_lambda = lambda;
            double alpha = 1.0;
            switch (m_pos_mode)
            {
                case 1:
                {
                    // 正特征值：轻微收缩
                    // 基础收缩系数
                    alpha = 1.0 - 0.05 * m_beta * (1.0 + wk); // 收缩程度受β和权重影响
                    alpha = std::max(0.8, std::min(1.0, alpha));
                    // #if DEBUG_OUTPUT
                    //     TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,beta,wk:"<<alpha<<", "<<beta<<", "<< wk);  
                    // #endif
                    // if (m_neg_pos_ratio < 1.0 && m_kappa < 0.3) // 正特征值主导，不需要缩步长
                    // {
                    //     alpha = 1.0;
                    // }
                    new_lambda =  alpha * lambda + (1.0 - alpha) * m_eps;

                    #if DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("正alpha, lambda, new_lambda:"<<alpha<<", "<<lambda<<", "<<new_lambda); 
                    #endif
                
                    break;  
                }
                case 2:
                {
                    // 正特征值：轻微收缩
                    // 基础收缩系数
                    alpha = 1.0 - 0.1 * m_beta * (1.0 + wk); // 收缩程度受β和权重影响
                    alpha = std::max(0.6, std::min(1.0, alpha));
                    // #if DEBUG_OUTPUT
                    //     TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,beta,wk:"<<alpha<<", "<<beta<<", "<< wk);  
                    // #endif
                    // if (m_neg_pos_ratio < 1.0 && m_kappa < 0.3) // 正特征值主导，不需要缩步长
                    // {
                    //     alpha = 1.0;
                    // }
                    new_lambda =  alpha * lambda + (1.0 - alpha) * m_eps;

                    #if DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("正alpha, lambda, new_lambda:"<<alpha<<", "<<lambda<<", "<<new_lambda); 
                    #endif
                
                    break;  
                }
                case 3:
                {
                    // 正特征值：轻微收缩
                    // 基础收缩系数
                    alpha = 1.0 - 0.12 * m_beta * (1.0 + wk); // 收缩程度受β和权重影响
                    alpha = std::max(0.5, std::min(1.0, alpha));
                    // #if DEBUG_OUTPUT
                    //     TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,beta,wk:"<<alpha<<", "<<beta<<", "<< wk);  
                    // #endif
                    // if (m_neg_pos_ratio < 1.0 && m_kappa < 0.3) // 正特征值主导，不需要缩步长
                    // {
                    //     alpha = 1.0;
                    // }
                    new_lambda =  alpha * lambda + (1.0 - alpha) * m_eps;

                    #if DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("正alpha, lambda, new_lambda:"<<alpha<<", "<<lambda<<", "<<new_lambda); 
                    #endif
                
                    break;  
                }
                default:
                {
                    // 不收缩
                    new_lambda = lambda;
                    break;
                }

            } 

            

            return new_lambda;
            
        }

        /**
         * @brief 处理负特征值的混合策略
         * 1. α_n = min(1, β/4)
         * 2. α_n = 1.0/(1+ m_beta / 4.0)
         * 基于解析公式的混合策略：α_n = clamp((β·b² - 2(b² - ε²)) / (2(b - ε)²), 0, 1)，其中b = -λ是负特征值的绝对值。这个公式来源于变分问题的解析解，能够根据β和特征
         * 值的大小自动调整混合系数，实现更智能的负特征值增强。曲率越大，β越大，alpha_n越大，负特征值增强越大。
         * 2. 基于权重调节的混合策略：在上述解析公式的基础上，加入权重调节，使得贡献较大的特征值（权重较大）得到更强的增强，而贡献较小的特征值得到较弱的增强。这可以通过将α_n乘以一个基于权重的调节因子来实现，例如α_n = α_n_base * (0.8 + 0.2 * w)，其中α_n_base是解析公式计算得到的混合系数，w是特征值的权重。这样可以使得增强效果更加细腻和智能。
         * 其他：不修正
         */
        double blendingNegativeEigenvalue(double lambda,double wk)
        {
            double new_lambda = lambda;
            double alpha = 0.0;
            switch (m_neg_mode)
            {
                
                case 1:
                { 
                    // 负特征值：放大到绝对值
                    double b = -lambda;
                    // 基础放大系数,α = min(1, β/4) 为基础，再用权重微调
                    double alpha_base = std::min(1.0, m_beta / 4.0);
                    // 用权重微调（贡献大的特征值放大更多）
                    alpha = alpha_base * (0.8 + 0.2 * wk);
                    alpha = std::max(0.0, std::min(1.0, alpha));
                    
                    new_lambda = alpha * b + (1.0 - alpha) * m_eps;
                    break;
                }
                case 2: //新公式
                {
                    // 负特征值：放大到绝对值
                    double b = -lambda;
                    // 基础放大系数,α = min(1, β/4) 为基础，再用权重微调
                    double alpha_base = std::min(1.0, 1.0/(1+ m_beta / 4.0));  
                    // 用权重微调（贡献大的特征值放大更多）
                    alpha = alpha_base * (0.8 + 0.2 * wk);
                    alpha = std::max(0.0, std::min(1.0, alpha));
                    
                    new_lambda =  alpha * b + (1.0 - alpha) * m_eps;
                    break;
                }
                case 3: //新公式
                {
                    // 负特征值：放大到绝对值
                    double b = -lambda;
                    // 基础放大系数,α = min(1, β/4) 为基础，再用权重微调
                    double alpha_base = std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));  
                    // 用权重微调（贡献大的特征值放大更多）
                    alpha = alpha_base * (0.8 + 0.2 * wk);
                    alpha = std::max(0.0, std::min(1.0, alpha));
                    
                    new_lambda =  alpha * b + (1.0 - alpha) * m_eps;
                    break;
                }
                case 4:
                { 
                    alpha = 1.0;
                    new_lambda = -lambda;
                    break;
                }
                case 0:
                { 
                    new_lambda = m_eps;
                    break;
                }
                
                default:
                {
                    // 不修改
                    new_lambda = lambda;
                    break;
                }
                
            }

            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT("负alpha, lambda, new_lambda:"<<alpha<<", "<<lambda<<", "<<new_lambda); 
            #endif

            return new_lambda;
        } 

        // double getBeta()
        // {
        //     return m_beta;
        // }

        double setBeta_max(double beta_max)
        {
            m_beta_max = beta_max;
            return m_beta;
        }

        double setGamma(double gamma)
        {
            m_gamma = gamma;
            return m_gamma;
        }
        
        double computeBeta()
        {
            switch (m_beta_mode)
            {
                case 1:
                {
                    m_beta = m_beta_max * m_kappa; 
                    break;
                }           
                case 2:
                {
                    // 基于曲率的自适应β
                    m_beta =  m_beta_max * (1.0 + m_gamma * m_kappa);
                    break;
                }
                default:
                {
                    m_beta = 0.0; // 不增强
                    break;
                }
            }

            return m_beta;
        }

        double computeLambdaMin(const Eigen::VectorXd& eigenvalues)
        {
            m_lambda_min = std::max(m_eps, eigenvalues.cwiseAbs().minCoeff());
            return m_lambda_min;
        }

        double getKappa()
        {
            return m_kappa;
        }

        double computeKappa(const Eigen::VectorXd& eigenvalues)
        {
            
            if (m_use_lower_bound)
            {
                m_lambda_min = computeLambdaMin(eigenvalues);
            }

            // double _eigenvalue_eps = m_eps;

            switch(m_kappa_mode) {
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
                        
                        m_kappa = nominator / denominator;
                        // 确保κ在合理范围内 [0, 1]
                        m_kappa = std::min(1.0, m_kappa);
                    }
                    else //不同号
                    {
                        // 这个比值度量了Hessian的各向异性程度
                        double nominator = lambda_max - lambda_min;
                        double denominator = nominator + std::abs(lambda_max+lambda_min) + m_eps;
                        
                        m_kappa = nominator / denominator;
                        // 确保κ在合理范围内 [0, 1]
                        m_kappa = std::min(1.0, m_kappa);
                    }
                
                    break;
                }
                case 2:
                {
                    //==== 计算 kappa ====
                    
                    // 计算负特征值能量占比
                    double sum_abs_all = 0.0;
                    double sum_abs_neg = 0.0;
                    double lambda_max = eigenvalues.maxCoeff();
                    double lambda_min = eigenvalues.minCoeff();

                    // TINYAD_DEBUG_OUT("lambda_max,lambda_min: "<< lambda_max << "," << lambda_min);

                    if (lambda_min > 0) {
                        TINYAD_DEBUG_OUT("边界修正触发:全正, kappa =0,lambda_min: "<< lambda_min); 
                        break;
                    }

                    for (int i = 0; i < eigenvalues.size(); ++i) {
                        double val = eigenvalues(i);
                        double abs_val = std::abs(val);
                        
                        sum_abs_all += abs_val;
                        if (TinyAD::isNegative(val)) { //非正特征值能量占比,倾向于kappa为0
                            sum_abs_neg += abs_val;
                        }
                    }
                    double neg_ratio = sum_abs_neg / (sum_abs_all + TinyAD::EPS_1E_8);

                    double alpha = 4.0;  // 敏感度参数
                    m_kappa = 1.0 - std::exp(-alpha * neg_ratio);

                    m_neg_pos_ratio = std::abs(lambda_min / lambda_max); //>1，负特征值主导

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

                    // 2. 计算曲率度量 κ_i (方案A: 基于特征值各向异性)
                    // κ = (λ_max - λ_min) / (|λ_max| + |λ_min| + ε)
                    // 这个比值度量了Hessian的各向异性程度
                    double nominator = lambda_max - lambda_min;
                    double denominator = nominator + std::abs(lambda_max+lambda_min) + m_eps;
                    
                    m_kappa = nominator / denominator;
                    // 确保κ在合理范围内 [0, 1]
                    m_kappa = std::min(1.0, m_kappa);

                    break;
                }
                case 4:
                {
                    //==== 计算 kappa ====
                    // 计算负特征值能量占比
                    //计算特征值跨度
                    double lambda_max = eigenvalues.maxCoeff();
                    double lambda_min = eigenvalues.minCoeff();
                    if (lambda_min > 0) {
                        #if DEBUG_OUTPUT_SHEAR
                            TINYAD_DEBUG_OUT("边界修正触发:全正, kappa =0,lambda_min > 0, lambda_min: "<< lambda_min); 
                        #endif
                        return 0.0;
                    }

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
                    double kappa1 = 1.0 - std::exp(-alpha * neg_ratio);

                    
                    double c = (lambda_max - lambda_min) / (lambda_max+lambda_min + m_eps); // 特征值跨度归一化
                    double kappa2 = 1.0 - std::exp(-2.0 * c); // 基于特征值跨度的增强

                    m_kappa = kappa1 * 0.6 + kappa2 * 0.4; // 综合两种度量


                    // 边界条件修正
                    // 纯剪切/温和拉伸：α ≈ 0.2-0.4（保守）
                    // 大拉伸/大压缩：α ≈ 0.4-0.6（中等）
                    // 负特征值出现：α ≈ 0.6-0.9（激进）
                    // 单元翻转：α = 1.0（最激进）
                    // 边界修正
                    if (lambda_min < -1e-4 && lambda_max < 0) // 若 λ₃ < -1e-4 且 λ₁ < 0，则 kappa = max(kappa, 0.9),全负，强制激进
                    {
                        #if SHEAR_DEBUG_OUTPUT
                            TINYAD_DEBUG_OUT("边界修正触发:全负，强制激进,kappa->1.0 lambda_min, lambda_max: "<< lambda_min << ", " << lambda_max); 
                        #endif
                        m_kappa = std::max(m_kappa, 0.8); // 强制增强
                    }

                    // 若 λ₁ > 0 且 λ₃ < -λ₁，则 α = max(α, 0.8)         # 负特征值绝对值大于正特征值
                    if (lambda_max > 0 && lambda_min < -lambda_max) 
                    {
                        m_kappa = std::max(m_kappa, 0.8); // 强制增强
                        #if SHEAR_DEBUG_OUTPUT
                            TINYAD_DEBUG_OUT("边界修正触发: 负特征值绝对值大于正特征值,k>0.8,lambda_max,lambda_min,kappa " << lambda_max << ", " << lambda_min << ", " << m_kappa ); 
                        #endif
                    }

                    //若 λ₁ - λ₃ < 0.1·|λ₁|，则 α = max(α, 0.3)         # 各向同性，保守
                    if (lambda_max - lambda_min < 0.1 * std::abs(lambda_max)) 
                    {
                        m_kappa = std::max(m_kappa, 0.3); // 保守增强
                        #if SHEAR_DEBUG_OUTPUT
                            TINYAD_DEBUG_OUT("边界修正触发: 各向同性,保守,k<0.3 lambda_max,lambda_min,kappa " << lambda_max << ", " << lambda_min << ", " << m_kappa ); 
                        #endif
                    }

                    // 若 λ₃ > 0 且 λ₁/λ₃ > 100，则 α = min(α, 0.6)      # 条件数极大但全正，保守
                    if (lambda_min > 0 && lambda_max / lambda_min > 100) 
                    {
                        
                        m_kappa = std::min(m_kappa, 0.6); // 保守增强
                        #if SHEAR_DEBUG_OUTPUT
                            TINYAD_DEBUG_OUT("边界修正触发: 条件数极大但全正，保守,k>0.6,lambda_min, lambda_max, kappa " << lambda_min << ", " << lambda_max << ", " << m_kappa ); 
                        #endif
                    }

                    if (m_rho < 0.25
                        && m_kappa < 0.4) // 剪切伴随局部屈曲，强制增强
                    {
                        if (m_lambda_min < 0 && 
                            neg_ratio < 0.3) // 负特征值能量占比较小，但出现了负特征值，可能是剪切伴随局部屈曲，强制增强
                        {
                            m_kappa = std::max(m_kappa, 0.8);
                            #if SHEAR_DEBUG_OUTPUT
                                TINYAD_DEBUG_OUT("边界修正触发:剪切伴随局部屈曲，强制增强:_eigenvalue_eps, kappa "<< m_rho<<","<<m_kappa); 
                            #endif
                        }
                        
                        
                    }

                    m_kappa = std::max(0.0, std::min(1.0, m_kappa));
                    
                    break;
                }
            }

            return m_kappa;
        }

    protected:
        double m_eps = TinyAD::EPS_1E_8;
        double m_rho = TinyAD::EPS_1E_8; // rho ratio
        int m_pos_mode = 0;
        int m_neg_mode = 0;
        
        double m_lambda_min = 0;
        bool m_use_lower_bound = false;
        
        int m_kappa_mode = 0;
        
        int m_beta_mode = 0;
        double m_beta_max = 4.0; // beta的最大值，避免过激
        double m_gamma = 2.0; // beta对kappa的敏感度
        double m_beta_min = 0.0; // beta的最小值，避免过弱

        double m_beta = 0.0; // 当前计算得到的beta值
        double m_kappa = 0;
        double m_neg_pos_ratio = 0.0;

    };

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
    //m_eps = TinyAD::EPS_1E_8;
    //wk = 1;
    if (std::abs(lambda) <= m_eps) {
        return m_eps;
    }
    else if (lambda > m_eps) {
        // 正特征值：轻微收缩
        // 基础收缩系数
        double alpha = 1.0 - 0.05 * beta * (1.0 + wk); // 收缩程度受β和权重影响
        alpha = std::max(0.8, std::min(1.0, alpha));
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("正特征值增强系数alpha_p,beta,wk:"<<alpha<<", "<<beta<<", "<< wk);  
        // #endif
        return alpha * lambda + (1.0 - alpha) * m_eps;

    } else {
        // 负特征值：放大到绝对值
        double b = -lambda;
        // 基础放大系数,α = min(1, β/4) 为基础，再用权重微调
        double alpha_base = std::min(1.0, beta / 4.0);
        // 用权重微调（贡献大的特征值放大更多）
        double alpha = alpha_base * (0.8 + 0.2 * wk);
        alpha = std::max(0.0, std::min(1.0, alpha));
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("fu特征值增强系数alpha_n,beta,wk:"<<alpha<<", "<<beta<<", "<< wk);  
        // #endif
        return alpha * b + (1.0 - alpha) * m_eps;
    }
}





/**
 * @brief 快速近似模式（解析公式）
 */
double optimizeEigenvalueFastSmooth(
    double lambda,
    double a2, //暂时无用
    double wk,
    double beta,
    double m_eps)
{
    m_eps = TinyAD::EPS_1E_8;
    //wk = 1;
    if (std::abs(lambda) <= m_eps) {
        double delta = m_eps - std::abs(lambda);
        double t = m_eps * (1.0 - 0.1 * delta * delta); // 轻微平滑过渡
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("光滑钳制:"<<lambda<<"-> "<<t);  
        #endif

        return t;
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

double optimizeEigenvalueByKappa( 
    double old_lambda, double kappa, double m_eps)
{
    
    m_eps = TinyAD::EPS_1E_8;

    double lambda = (1-kappa) * old_lambda + kappa * std::abs(old_lambda); // λ' = (1-κ)λ + κ|λ|

    if (std::abs(lambda) < m_eps) { 
        // 小特征值
        lambda = m_eps; 
    }
    return lambda;
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
        case 2:
        {
            return optimizeEigenvalueFastSmooth(lambda, a2, wk, beta, m_eps);
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
    double beta = beta0 * (1.0 + gamma * kappa); //beta:[2,7.2];beta0:[0,2.4]
    
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
 * Check if matrix is diagonally dominant and has positive diagonal entries.
 * This is a sufficient condition for positive-definiteness
 * and can be used as an early out to avoid eigen decomposition.
 */
template <typename PassiveT>
bool positive_diagonally_dominant(
        Eigen::MatrixX<PassiveT>& _H,
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
    double _eigenvalue_eps = m_eps;
    //m_eps = TinyAD::EPS_1E_8;
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
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();

            // TINYAD_DEBUG_OUT("lambda_max,lambda_min: "<< lambda_max << "," << lambda_min);

            if (lambda_min > 0) {
                TINYAD_DEBUG_OUT("边界修正触发:全正, kappa =0,lambda_min: "<< lambda_min); 
                break;
            }

            for (int i = 0; i < eigenvalues.size(); ++i) {
                double val = eigenvalues(i);
                double abs_val = std::abs(val);
                
                sum_abs_all += abs_val;
                if (TinyAD::isNegative(val)) { //非正特征值能量占比,倾向于kappa为0
                    sum_abs_neg += abs_val;
                }
            }
            double neg_ratio = sum_abs_neg / (sum_abs_all + TinyAD::EPS_1E_8);

            double alpha = 4.0;  // 敏感度参数
            kappa = 1.0 - std::exp(-alpha * neg_ratio);
            
            // // // 可选：体积比增强
            // // double J_factor = std::abs(std::log(J)) / 5.0;
            // // J_factor = std::min(1.0, J_factor);
            
            // // 取最大，确保非凸时曲率大
            // return std::max(kappa, J_factor);

            
            // if ( lambda_max > (-lambda_min )  //正特征值较大，且条件数极大
            //     && lambda_max / std::max(m_eps, -lambda_min) > 100) 
            // {
            //     kappa = std::min(kappa, 0.6); // 保守增强
            //     #if SHEAR_DEBUG_OUTPUT
            //         TINYAD_DEBUG_OUT("边界修正触发: 条件数极大,k>0.6,lambda_min, lambda_max, kappa " << lambda_min << ", " << lambda_max << ", " << kappa ); 
            //     #endif
            // }

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
        case 4:
        {
            //==== 计算 kappa ====
            // 计算负特征值能量占比
            //计算特征值跨度
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();
            if (lambda_min > 0) {
                #if DEBUG_OUTPUT_SHEAR
                    TINYAD_DEBUG_OUT("边界修正触发:全正, kappa =0,lambda_min > 0, lambda_min: "<< lambda_min); 
                #endif
                return 0.0;
            }

            // if (lambda_max < 0) {
            //     #if DEBUG_OUTPUT_SHEAR
            //         TINYAD_DEBUG_OUT("lambda_max < 0, lambda_max: "<< lambda_max); 
            //     #endif
            //     return 1.0;
            // }

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
            double kappa1 = 1.0 - std::exp(-alpha * neg_ratio);

            
            double c = (lambda_max - lambda_min) / (lambda_max+lambda_min + m_eps); // 特征值跨度归一化
            double kappa2 = 1.0 - std::exp(-2.0 * c); // 基于特征值跨度的增强

            kappa = kappa1 * 0.6 + kappa2 * 0.4; // 综合两种度量


            // 边界条件修正
            // 纯剪切/温和拉伸：α ≈ 0.2-0.4（保守）
            // 大拉伸/大压缩：α ≈ 0.4-0.6（中等）
            // 负特征值出现：α ≈ 0.6-0.9（激进）
            // 单元翻转：α = 1.0（最激进）
            // 边界修正
            if (lambda_min < -1e-4 && lambda_max < 0) // 若 λ₃ < -1e-4 且 λ₁ < 0，则 kappa = max(kappa, 0.9),全负，强制激进
            {
                #if SHEAR_DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("边界修正触发:全负，强制激进,kappa->1.0 lambda_min, lambda_max: "<< lambda_min << ", " << lambda_max); 
                #endif
                kappa = std::max(kappa, 0.8); // 强制增强
            }

            // 若 λ₁ > 0 且 λ₃ < -λ₁，则 α = max(α, 0.8)         # 负特征值绝对值大于正特征值
            if (lambda_max > 0 && lambda_min < -lambda_max) 
            {
                kappa = std::max(kappa, 0.8); // 强制增强
                #if SHEAR_DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("边界修正触发: 负特征值绝对值大于正特征值,k>0.8,lambda_max,lambda_min,kappa " << lambda_max << ", " << lambda_min << ", " << kappa ); 
                #endif
            }

            //若 λ₁ - λ₃ < 0.1·|λ₁|，则 α = max(α, 0.3)         # 各向同性，保守
            if (lambda_max - lambda_min < 0.1 * std::abs(lambda_max)) 
            {
                kappa = std::max(kappa, 0.3); // 保守增强
                #if SHEAR_DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("边界修正触发: 各向同性,保守,k<0.3 lambda_max,lambda_min,kappa " << lambda_max << ", " << lambda_min << ", " << kappa ); 
                #endif
            }

            // 若 λ₃ > 0 且 λ₁/λ₃ > 100，则 α = min(α, 0.6)      # 条件数极大但全正，保守
            if (lambda_min > 0 && lambda_max / lambda_min > 100) 
            {
                
                kappa = std::min(kappa, 0.6); // 保守增强
                #if SHEAR_DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("边界修正触发: 条件数极大但全正，保守,k>0.6,lambda_min, lambda_max, kappa " << lambda_min << ", " << lambda_max << ", " << kappa ); 
                #endif
            }

            if (_eigenvalue_eps < 0.25
                && kappa < 0.4) // 剪切伴随局部屈曲，强制增强
            {
                if (lambda_min < 0 && 
                    neg_ratio < 0.3) // 负特征值能量占比较小，但出现了负特征值，可能是剪切伴随局部屈曲，强制增强
                {
                    kappa = std::max(kappa, 0.8);
                    #if SHEAR_DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("边界修正触发:剪切伴随局部屈曲，强制增强:_eigenvalue_eps, kappa "<< _eigenvalue_eps<<","<<kappa); 
                    #endif
                }
                
                
            }

            kappa = std::max(0.0, std::min(1.0, kappa));
            
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
        
        double m_eps = TinyAD::EPS_1E_8;  // 小特征值阈值
        bool use_shift = false; // 是否阻尼牛顿法

        #if DEBUG_OUTPUT
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();
            
            TINYAD_DEBUG_OUT("lambda_max,lambda_min:"<<lambda_max<<","<<lambda_min);   
        #endif

        // if (TinyAD::isNegative(_eigenvalue_eps)
        //     && _mode != HessianProjectionMode::CLAMP_ABS_BLENDING) // 负数表示投影到绝对值
        // {
        //     m_eps = 0; // 不设置下限，直接投影到绝对值
        // }

        // 创建正则化器
        EigenvalueRegularizer<PassiveT> regularizer(_eigenvalue_eps,_mode);
        //EigenvalueRegularizer<PassiveT> regularizer(m_eps,_mode);
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("_eigenvalue_eps in project_positive_definite_diff:"<<static_cast<double>(_eigenvalue_eps));   
        // #endif
        bool modified = false;
        if (_mode == HessianProjectionMode::SMOOTH_TR) 
        {
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            int method = 3;

            switch (method) {
                case 1:
                {
                    int kappaMethod = 2; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
                    // 2. 计算几何曲率 κ（基于特征值分布）
                    double kappa = computeKappa(kappaMethod, eigenvalues, m_eps);
                    // 3. 优化每个特征值
                    for (size_t i = 0; i < k; ++i) {
                        eigenvalues(i) = optimizeEigenvalueByKappa(eigenvalues(i), kappa, m_eps);
                    }
                    modified = true;
                    break;
                }
                case 2:
                {
                    double thresh_upper = 1e-6; // 
                    double thresh_lower = -1e-3; // 
                    if (_eigenvalue_eps >= 0) //激进的单元策略
                    {
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < thresh_upper && eigenvalues(i) > thresh_lower) {
                                eigenvalues(i) = thresh_upper; // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue in (thresh_lower, thresh_upper) " ); 
                                #endif
                            }
                            else if (eigenvalues(i) <= thresh_lower) {
                                eigenvalues(i) = std::abs(eigenvalues(i)); // 直接投影到绝对值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue <= thresh_lower " ); 
                                #endif
                            }
                        }
                    }
                    else // abs策略
                    {   
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < 0) {
                                eigenvalues(i) = - eigenvalues(i); // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue < 0  and eps < 0" ); 
                                #endif
                            }
                            
                        }

                    }
                    //modified = true;
                    break;
                    
                }
                case 3:
                {
                    double thresh_upper = 1e-6; // 
                    double thresh_lower = -1e-3; // 
                    if (_eigenvalue_eps >= 0) //激进的单元策略
                    {
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < _eigenvalue_eps) {
                                eigenvalues(i) = _eigenvalue_eps; // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue < 0 and eps >= 0" ); 
                                #endif
                            }
                            
                        }
                    }
                    else // abs策略
                    {   
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < thresh_upper && eigenvalues(i) > thresh_lower) {
                                eigenvalues(i) = thresh_upper; // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue in (thresh_lower, thresh_upper) " ); 
                                #endif
                            }
                            else if (eigenvalues(i) <= thresh_lower) {
                                eigenvalues(i) = -eigenvalues(i); // 直接投影到绝对值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue <= thresh_lower " ); 
                                #endif
                            }
                        }

                    }
                    //modified = true;
                    break;
                    
                }
            }
            
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_SHEAR)
        {
            // 将β代入变分框架，计算混合系数α
            
            // 2. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT_BLEND_SHEAR
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 4; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
            double kappa = computeKappa(kappaMethod, eigenvalues, _eigenvalue_eps);
            // // fast by zj
            double beta0 = 2.0 ; // modified by zj 0402
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
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING3
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING4
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING5) 
        {
            // 将β代入变分框架，计算混合系数α
            // m_eps = 0;
            // 2. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    // TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 

            int kappa_mode = 2; //fixed by zj, 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
            //待测试参数1
            int beta_mode = 1; //{0,1,2} 1: beta_max * kappa, beta随kappa线性变化 ; 2:m_beta =  m_beta_max * (1.0 + m_gamma * m_kappa) ;  others, 0.0(支持clamp)
            //[2,2][0,2]
            int pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); 3:alpha = 1.0 - 0.12 * m_beta * (1.0 + wk);; others. 不收缩
            int neg_blending_mode = 2; // {0,1, 2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
            bool use_lower_bound = false; //{false, true} 是否使用下界，false表示完全blending
            //待测试参数2
            double beta_max = 4.0; // beta的上限，防止过度增强，可调整,可调整的基准值，beta = beta0 * (1.0 + gamma * kappa)
            double gamma = 2.0; // beta对kappa的敏感度，gamma越大，beta随kappa的变化越剧烈
            // m_eps = 0;

            switch(_mode){
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: // not good
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 2; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: // not good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 2; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: // not good
                // {
                //     pos_blending_mode = 1; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 2; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                case HessianProjectionMode::CLAMP_ABS_BLENDING3: // best, stretch/compress/bend/shear, 怎么解决twist 不要缩放pos
                {
                    pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                    neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                    use_lower_bound = true; // not necceary
                    beta_max *= _eigenvalue_eps;
                    break;
                }
                
                case HessianProjectionMode::CLAMP_ABS_BLENDING4: // good, stretch/compress/bend/shear
                {
                    pos_blending_mode = 3; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                    neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                    use_lower_bound = true;
                    beta_max *= _eigenvalue_eps;
                    break;
                }
                case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, stretch/compress/shear
                {
                    pos_blending_mode = 1; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                    neg_blending_mode = 3; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                    // use_lower_bound = true;  //not clear
                    // beta_max *= _eigenvalue_eps; // not clear
                    break;
                }

                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // not good
                // {
                //     pos_blending_mode = 3; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: // good
                // {
                //     pos_blending_mode = 3; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;  //not clear
                //     beta_max *= _eigenvalue_eps; // not clear
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: // not good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                
                
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: //not good， good for twist?
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: //not good
                // {
                //     pos_blending_mode = 0; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // not good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // to check  *,3; *,4
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: //good, for compress&stretch&shear
                // {
                //     pos_blending_mode = 1; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: //good
                // {
                //     pos_blending_mode = 1; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: //good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, modify neg only
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, modify neg only
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, 从abs到clamp
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // 和clamp切换效果不好
                default:
                {
                    break;
                }
            }

            
            if (beta_mode < 1 || beta_mode > 2)
            {
                if (_eigenvalue_eps == 0) // clamp mode
                {
                    m_eps = 0;
                    pos_blending_mode = 0; // 保持
                    neg_blending_mode = 0; //clamp
                }
                //否则, a_p = 1
            }

            // if (_eigenvalue_eps == 0) // clamp mode
            // {
            //     // m_eps = 0;
            //     pos_blending_mode = 0; // 保持
            //     neg_blending_mode = 0; //clamp
            // }

            EigenvalueBlendingRegularizer regularizer_blending(_eigenvalue_eps, m_eps, 
                kappa_mode, beta_mode, pos_blending_mode, neg_blending_mode, 
                use_lower_bound);
            regularizer_blending.setBeta_max(beta_max * _eigenvalue_eps);
            regularizer_blending.setGamma(gamma);
            
            
            // 2. 计算非凸强度
            double kappa = regularizer_blending.computeKappa(eigenvalues);

            // 3. 计算正则化参数 β
            double beta = regularizer_blending.computeBeta();
            regularizer_blending.print();

            // 3. 优化每个特征值
            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                double new_lambda = lambda;
                double wi = 1.0/12;
                new_lambda = regularizer_blending.blending(lambda, wi);
                
                eigenvalues(i) = new_lambda;    
            }
            
            modified = true;

            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING2)
        {
            // 将β代入变分框架，计算混合系数α
            // m_eps = 0;
            // 2. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    // TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 
            
            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 2;
            double kappa = computeKappa(kappaMethod, eigenvalues, m_eps);
            // // fast by zj
            double beta0 = 2.0 *_eigenvalue_eps; 
            double gamma = 2.0; 
            double beta_max = 4.0; // 限制beta最大值，避免过激
            // // adaptive by zj
            // double beta0 = 2.0 *_eigenvalue_eps; 
            // double gamma = 1.0;
            double beta = 0;// [0-6], beta =0, clamp,其他情况为blending
            
            if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING2)  //blending3，动态beta_max
            {
                // beta = beta_max * kappa * _eigenvalue_eps; // 限制beta最大为4.0，避免过激
                beta = beta_max * kappa; // 限制beta最大为4.0，避免过激
            }
            else
            {
                double kappa_threshhold = 0.1;
                // if (_eigenvalue_eps < 1.0) //模型稳定,扩大优化范围;否则仅关注形变大的单元
                // {
                //     // kappa_threshhold = kappa_threshhold * kappa_threshhold;
                //     kappa_threshhold = std::pow(kappa_threshhold, std::log(_eigenvalue_eps) / std::log(0.9));
                // }
                
                // if (_eigenvalue_eps > 1.0){
                //     kappa_threshhold = kappa_threshhold + 0.02 * (std::log(_eigenvalue_eps) / std::log(1.02));
                // }
                if (kappa < kappa_threshhold ) // 形变小,clamp
                {
                    beta0 = 0;
                }
                
                beta = getBeta(kappa, beta0, gamma); // 限制beta最大为2.0，避免过激
            }

            /* 
            * correct lambda
            */
            int correctLambdaMethod = 2; // 1: 非统一框架；2，统一框架; 3:统一框架+正负blending
            int fast_mode = 1; //1.fast;其他 newton
            if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH)
            {
                fast_mode = 2; 
            }
            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT("kappa, beta:"<<kappa<<", "<<beta); 
                TINYAD_DEBUG_OUT("beta_max, beta0, gamma:"<<beta_max<<", "<<beta0<<", "<<gamma); 
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

                    //m_eps = 0;
                    m_eps = std::max(m_eps, eigenvalues.cwiseAbs().minCoeff());

                    #if DEBUG_OUTPUT
                        double alpha_n_base = std::min(1.0, beta / 4.0);
                        // 用权重微调（贡献大的特征值放大更多）
                        double alpha_n = alpha_n_base * (0.8 + 0.2 * 1.0/12);
                        double alpha_p = 1.0 - 0.05 * beta * (1.0 + 1.0/12); // 收缩程度受β和权重影响
                        alpha_p = std::max(0.8, std::min(1.0, alpha_p));
                        TINYAD_DEBUG_OUT("alpha_p, alpha_n:"<<alpha_p<<", "<<alpha_n);
                        TINYAD_DEBUG_OUT("m_eps:"<<m_eps); 
                    #endif

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
            // if (use_shift)//先shift
            // {
            //     double min_abs = eigenvalues.cwiseAbs().minCoeff();

            //     if (TinyAD::isNegative(min_abs, TinyAD::EPS_1E_8)) //小于最小值
            //     {
            //        double shift_value = min_abs + TinyAD::ZERO; // 根据β调整shift 

            //         for (size_t i = 0; i < k; ++i) { //先shift,再clamp，abs,blending
            //             eigenvalues(i) += shift_value;    
            //         }
            //         modified = true;
            //     }

            // }

            // if (_mode == HessianProjectionMode::CLAMP_ABS_NONDIFF)
            // {
            //     if (_eigenvalue_eps > (-1e-8)) // 只有0,-1两种取值
            //     {
            //         TINYAD_DEBUG_OUT("!!!!skip regularization."); 
            //         return;
            //     }
                
            // }

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
            //TINYAD_ASSERT_FINITE_MAT(_H);
            // TINYAD_DEBUG_OUT("regularized element hessian +1");
        }

        //#endif
       
    }
}


/**
 * Project symmetric matrix to positive-definite matrix
 * via eigen decomposition. 
 * added by zj (Differentiable version)
 */
template < typename PassiveT>
void project_positive_definite_diff(
        Eigen::MatrixX<PassiveT>& _H,
        const PassiveT& _eigenvalue_eps,
        HessianProjectionMode _mode = HessianProjectionMode::AUTO)
{
    // #if DEBUG_OUTPUT
    //     TINYAD_DEBUG_OUT("Projection mode in project_positive_definite_diff: " << static_cast<int>(_mode)); 
    // #endif
    const int k = _H.rows();
    if (k == 0)
    {
        return;
    }
    else
    {
        // using MatT = Eigen::MatrixX<PassiveT>(k, k);
        using MatT = Eigen::MatrixX<PassiveT>;

        // Early out if sufficient condition is fulfilled
        if (positive_diagonally_dominant<PassiveT>(_H, _eigenvalue_eps))
            return;

        // ===== 1. diff project =====
        // Compute eigen-decomposition (of symmetric matrix)
        Eigen::SelfAdjointEigenSolver<MatT> eig(_H);
        auto eigenvalues = eig.eigenvalues();
        auto eigenvectors = eig.eigenvectors();
        
        double m_eps = TinyAD::EPS_1E_8;  // 小特征值阈值
        bool use_shift = false; // 是否阻尼牛顿法

        #if DEBUG_OUTPUT
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();
            
            TINYAD_DEBUG_OUT("lambda_max,lambda_min:"<<lambda_max<<","<<lambda_min);   
        #endif

        // if (TinyAD::isNegative(_eigenvalue_eps)
        //     && _mode != HessianProjectionMode::CLAMP_ABS_BLENDING) // 负数表示投影到绝对值
        // {
        //     m_eps = 0; // 不设置下限，直接投影到绝对值
        // }

        // 创建正则化器
        EigenvalueRegularizer<PassiveT> regularizer(_eigenvalue_eps,_mode);
        //EigenvalueRegularizer<PassiveT> regularizer(m_eps,_mode);
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("_eigenvalue_eps in project_positive_definite_diff:"<<static_cast<double>(_eigenvalue_eps));   
        // #endif
        bool modified = false;
        if (_mode == HessianProjectionMode::SMOOTH_TR) 
        {
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            int method = 3;

            switch (method) {
                case 1:
                {
                    int kappaMethod = 2; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
                    // 2. 计算几何曲率 κ（基于特征值分布）
                    double kappa = computeKappa(kappaMethod, eigenvalues, m_eps);
                    // 3. 优化每个特征值
                    for (size_t i = 0; i < k; ++i) {
                        eigenvalues(i) = optimizeEigenvalueByKappa(eigenvalues(i), kappa, m_eps);
                    }
                    modified = true;
                    break;
                }
                case 2:
                {
                    double thresh_upper = 1e-6; // 
                    double thresh_lower = -1e-3; // 
                    if (_eigenvalue_eps >= 0) //激进的单元策略
                    {
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < thresh_upper && eigenvalues(i) > thresh_lower) {
                                eigenvalues(i) = thresh_upper; // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue in (thresh_lower, thresh_upper) " ); 
                                #endif
                            }
                            else if (eigenvalues(i) <= thresh_lower) {
                                eigenvalues(i) = std::abs(eigenvalues(i)); // 直接投影到绝对值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue <= thresh_lower " ); 
                                #endif
                            }
                        }
                    }
                    else // abs策略
                    {   
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < 0) {
                                eigenvalues(i) = - eigenvalues(i); // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue < 0  and eps < 0" ); 
                                #endif
                            }
                            
                        }

                    }
                    //modified = true;
                    break;
                    
                }
                case 3:
                {
                    double thresh_upper = 1e-6; // 
                    double thresh_lower = -1e-3; // 
                    if (_eigenvalue_eps >= 0) //激进的单元策略
                    {
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < _eigenvalue_eps) {
                                eigenvalues(i) = _eigenvalue_eps; // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue < 0 and eps >= 0" ); 
                                #endif
                            }
                            
                        }
                    }
                    else // abs策略
                    {   
                        for (size_t i = 0; i < k; ++i) {
                            if (eigenvalues(i) < thresh_upper && eigenvalues(i) > thresh_lower) {
                                eigenvalues(i) = thresh_upper; // 直接提升到正数阈值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue in (thresh_lower, thresh_upper) " ); 
                                #endif
                            }
                            else if (eigenvalues(i) <= thresh_lower) {
                                eigenvalues(i) = -eigenvalues(i); // 直接投影到绝对值
                                modified = true;
                                #if DEBUG_OUTPUT_BLEND_TR
                                    TINYAD_DEBUG_OUT(" eigenvalue <= thresh_lower " ); 
                                #endif
                            }
                        }

                    }
                    //modified = true;
                    break;
                    
                }
            }
            
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_SHEAR)
        {
            // 将β代入变分框架，计算混合系数α
            
            // 2. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT_BLEND_SHEAR
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 4; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
            double kappa = computeKappa(kappaMethod, eigenvalues, _eigenvalue_eps);
            // // fast by zj
            double beta0 = 2.0 ; // modified by zj 0402
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
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING3
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING4
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING5) 
        {
            // 将β代入变分框架，计算混合系数α
            // m_eps = 0;
            // 2. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    // TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 

            int kappa_mode = 2; //fixed by zj, 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
            //待测试参数1
            int beta_mode = 1; //{0,1,2} 1: beta_max * kappa, beta随kappa线性变化 ; 2:m_beta =  m_beta_max * (1.0 + m_gamma * m_kappa) ;  others, 0.0(支持clamp)
            //[2,2][0,2]
            int pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); 3:alpha = 1.0 - 0.12 * m_beta * (1.0 + wk);; others. 不收缩
            int neg_blending_mode = 2; // {0,1, 2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
            bool use_lower_bound = false; //{false, true} 是否使用下界，false表示完全blending
            //待测试参数2
            double beta_max = 4.0; // beta的上限，防止过度增强，可调整,可调整的基准值，beta = beta0 * (1.0 + gamma * kappa)
            double gamma = 2.0; // beta对kappa的敏感度，gamma越大，beta随kappa的变化越剧烈
            // m_eps = 0;

            switch(_mode){
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: // not good
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 2; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: // not good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 2; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: // not good
                // {
                //     pos_blending_mode = 1; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 2; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                case HessianProjectionMode::CLAMP_ABS_BLENDING3: // best, stretch/compress/bend/shear, 怎么解决twist 不要缩放pos
                {
                    pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                    neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                    use_lower_bound = true; // not necceary
                    beta_max *= _eigenvalue_eps;
                    break;
                }
                
                case HessianProjectionMode::CLAMP_ABS_BLENDING4: // good, stretch/compress/bend/shear
                {
                    pos_blending_mode = 3; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                    neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                    use_lower_bound = true;
                    beta_max *= _eigenvalue_eps;
                    break;
                }
                case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, stretch/compress/shear
                {
                    pos_blending_mode = 1; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                    neg_blending_mode = 3; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                    // use_lower_bound = true;  //not clear
                    // beta_max *= _eigenvalue_eps; // not clear
                    break;
                }

                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // not good
                // {
                //     pos_blending_mode = 3; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: // good
                // {
                //     pos_blending_mode = 3; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;  //not clear
                //     beta_max *= _eigenvalue_eps; // not clear
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: // not good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                
                
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: //not good， good for twist?
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: //not good
                // {
                //     pos_blending_mode = 0; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // not good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     beta_max *= _eigenvalue_eps;
                //     break;
                // }
                // to check  *,3; *,4
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: //good, for compress&stretch&shear
                // {
                //     pos_blending_mode = 1; // {0,1,2} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING3: //good
                // {
                //     pos_blending_mode = 1; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING4: //good
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, modify neg only
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 3; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, modify neg only
                // {
                //     pos_blending_mode = 0; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // case HessianProjectionMode::CLAMP_ABS_BLENDING5: // good, 从abs到clamp
                // {
                //     pos_blending_mode = 2; // {0,1,2,3} 0:不收缩,clamp,1. 轻微收缩：α_p = 1 - 0.05 * β * (1 + w); 2: α_p = 1 - 0.1 * β * (1 + w); others. 不收缩
                //     neg_blending_mode = 1; // {0,1,2,3,4} 0:clamp;4:abs;1. α_n = min(1, β/4); 2. α_n = 1.0/(1+ β / 4.0); 3:std::min(1.0, 1.0-1.0/(1+ m_beta / 4.0));others. 不修改
                //     // use_lower_bound = true;
                //     break;
                // }
                // 和clamp切换效果不好
                default:
                {
                    break;
                }
            }

            
            if (beta_mode < 1 || beta_mode > 2)
            {
                if (_eigenvalue_eps == 0) // clamp mode
                {
                    m_eps = 0;
                    pos_blending_mode = 0; // 保持
                    neg_blending_mode = 0; //clamp
                }
                //否则, a_p = 1
            }

            // if (_eigenvalue_eps == 0) // clamp mode
            // {
            //     // m_eps = 0;
            //     pos_blending_mode = 0; // 保持
            //     neg_blending_mode = 0; //clamp
            // }

            EigenvalueBlendingRegularizer regularizer_blending(_eigenvalue_eps, m_eps, 
                kappa_mode, beta_mode, pos_blending_mode, neg_blending_mode, 
                use_lower_bound);
            regularizer_blending.setBeta_max(beta_max * _eigenvalue_eps);
            regularizer_blending.setGamma(gamma);
            
            
            // 2. 计算非凸强度
            double kappa = regularizer_blending.computeKappa(eigenvalues);

            // 3. 计算正则化参数 β
            double beta = regularizer_blending.computeBeta();
            regularizer_blending.print();

            // 3. 优化每个特征值
            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                double new_lambda = lambda;
                double wi = 1.0/12;
                new_lambda = regularizer_blending.blending(lambda, wi);
                
                eigenvalues(i) = new_lambda;    
            }
            
            modified = true;

            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH
        || _mode == HessianProjectionMode::CLAMP_ABS_BLENDING2)
        {
            // 将β代入变分框架，计算混合系数α
            // m_eps = 0;
            // 2. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    // TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 
            
            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 2;
            double kappa = computeKappa(kappaMethod, eigenvalues, m_eps);
            // // fast by zj
            double beta0 = 2.0 *_eigenvalue_eps; 
            double gamma = 2.0; 
            double beta_max = 4.0; // 限制beta最大值，避免过激
            // // adaptive by zj
            // double beta0 = 2.0 *_eigenvalue_eps; 
            // double gamma = 1.0;
            double beta = 0;// [0-6], beta =0, clamp,其他情况为blending
            
            if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING2)  //blending3，动态beta_max
            {
                // beta = beta_max * kappa * _eigenvalue_eps; // 限制beta最大为4.0，避免过激
                beta = beta_max * kappa; // 限制beta最大为4.0，避免过激
            }
            else
            {
                double kappa_threshhold = 0.1;
                // if (_eigenvalue_eps < 1.0) //模型稳定,扩大优化范围;否则仅关注形变大的单元
                // {
                //     // kappa_threshhold = kappa_threshhold * kappa_threshhold;
                //     kappa_threshhold = std::pow(kappa_threshhold, std::log(_eigenvalue_eps) / std::log(0.9));
                // }
                
                // if (_eigenvalue_eps > 1.0){
                //     kappa_threshhold = kappa_threshhold + 0.02 * (std::log(_eigenvalue_eps) / std::log(1.02));
                // }
                if (kappa < kappa_threshhold ) // 形变小,clamp
                {
                    beta0 = 0;
                }
                
                beta = getBeta(kappa, beta0, gamma); // 限制beta最大为2.0，避免过激
            }

            /* 
            * correct lambda
            */
            int correctLambdaMethod = 2; // 1: 非统一框架；2，统一框架; 3:统一框架+正负blending
            int fast_mode = 1; //1.fast;其他 newton
            if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH)
            {
                fast_mode = 2; 
            }
            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT("kappa, beta:"<<kappa<<", "<<beta); 
                TINYAD_DEBUG_OUT("beta_max, beta0, gamma:"<<beta_max<<", "<<beta0<<", "<<gamma); 
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

                    //m_eps = 0;
                    m_eps = std::max(m_eps, eigenvalues.cwiseAbs().minCoeff());

                    #if DEBUG_OUTPUT
                        double alpha_n_base = std::min(1.0, beta / 4.0);
                        // 用权重微调（贡献大的特征值放大更多）
                        double alpha_n = alpha_n_base * (0.8 + 0.2 * 1.0/12);
                        double alpha_p = 1.0 - 0.05 * beta * (1.0 + 1.0/12); // 收缩程度受β和权重影响
                        alpha_p = std::max(0.8, std::min(1.0, alpha_p));
                        TINYAD_DEBUG_OUT("alpha_p, alpha_n:"<<alpha_p<<", "<<alpha_n);
                        TINYAD_DEBUG_OUT("m_eps:"<<m_eps); 
                    #endif

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
            // if (use_shift)//先shift
            // {
            //     double min_abs = eigenvalues.cwiseAbs().minCoeff();

            //     if (TinyAD::isNegative(min_abs, TinyAD::EPS_1E_8)) //小于最小值
            //     {
            //        double shift_value = min_abs + TinyAD::ZERO; // 根据β调整shift 

            //         for (size_t i = 0; i < k; ++i) { //先shift,再clamp，abs,blending
            //             eigenvalues(i) += shift_value;    
            //         }
            //         modified = true;
            //     }

            // }

            // if (_mode == HessianProjectionMode::CLAMP_ABS_NONDIFF)
            // {
            //     if (_eigenvalue_eps > (-1e-8)) // 只有0,-1两种取值
            //     {
            //         TINYAD_DEBUG_OUT("!!!!skip regularization."); 
            //         return;
            //     }
                
            // }

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
            //TINYAD_ASSERT_FINITE_MAT(_H);
            // TINYAD_DEBUG_OUT("regularized element hessian +1");
        }

        //#endif
       
    }
}



}




