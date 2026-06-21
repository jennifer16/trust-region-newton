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
#include "../../../include/ElementDeformationState.h"
// #include "../../../include/ElementDeformationState.h"  // 或项目 include 路径

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
// 添加辅助函数（文件作用域）
inline double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}
// 范围 [0, 1]
double computeKappa(int kappaMethod, const Eigen::VectorXd& eigenvalues)
{
    double kappa = 0; // 范围 [0, 1]
    // double _eigenvalue_eps = m_eps;
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
                double denominator =  std::abs(lambda_max)+ std::abs(lambda_min) + TinyAD::EPS;
                
                kappa = nominator / denominator;
                // 确保κ在合理范围内 [0, 1]
                kappa = std::min(1.0, kappa);
            }
            else //不同号
            {
                // 这个比值度量了Hessian的各向异性程度
                double nominator = lambda_max - lambda_min;
                double denominator = nominator + std::abs(lambda_max+lambda_min) + TinyAD::EPS;
                
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
            double neg_ratio = sum_abs_neg / (sum_abs_all + TinyAD::EPS);

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
            double denominator = nominator + std::abs(lambda_max+lambda_min) + TinyAD::EPS;
            
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
            double neg_ratio = sum_abs_neg / (sum_abs_all + TinyAD::EPS);

            double alpha = 4.0;  // 敏感度参数
            double kappa1 = 1.0 - std::exp(-alpha * neg_ratio);

            
            double c = (lambda_max - lambda_min) / (lambda_max+lambda_min + TinyAD::EPS); // 特征值跨度归一化
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

            if (neg_ratio < 0.25
                && kappa < 0.4) // 剪切伴随局部屈曲，强制增强
            {
                if (lambda_min < 0 && 
                    neg_ratio < 0.3) // 负特征值能量占比较小，但出现了负特征值，可能是剪切伴随局部屈曲，强制增强
                {
                    kappa = std::max(kappa, 0.8);
                    #if SHEAR_DEBUG_OUTPUT
                        TINYAD_DEBUG_OUT("边界修正触发:剪切伴随局部屈曲，强制增强:neg_ratio, kappa "<< neg_ratio<<","<<kappa); 
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
 * @brief 计算基于J的增强系数，范围 [0, 1]
 * 当J<J_threshold时，alpha_J接近0; 
 * 当J>J_threshold时，alpha_J接近1;
 * 不要修正到0，1
 */
template <typename PassiveT>
double computeAlpha_J(PassiveT& J)
{
    double alpha_J = 1.0;
    int method = g_j_mode;
    switch(method) 
    {
        case 1:
        {
            double J_c  = 0.9;
            double J_s  = 2.0;
            double k1   = 15.0;
            double k2   = 3.0;
            double beta = 20.0;
            
            // sigmoid: 1 / (1 + exp(-x))，截断防溢出, -inf-> 0, +inf -> 1
            auto sig = [](double x) -> double {
                if (x >  100.0) return 1.0;
                if (x < -100.0) return 0.0;
                return 1.0 / (1.0 + std::exp(-x));
            };

            // ── J<0 安全因子 ──────────────────────────────────────────
            // 当 J<0（单元翻转）时 safe≈0，强制 R→0 → alpha*=1（abs）
            // 否则若不加此项，J<0 时 R1=sig(k1*(J_c-J))→1 会错误给出 clamp
            double safe = sig(50.0 * J);

            // ── 左右区域混合权重 ───────────────────────────────────────
            // w≈0：J<1（压缩侧），使用 R1
            // w≈1：J>1（拉伸侧），使用 R2
            double w = sig(beta * (J - 1.0));

            // ── 压缩侧可靠性 R1 ───────────────────────────────────────
            // J < J_c(0.9): R1→1（高可靠，可激进）
            // J > J_c(0.9): R1→0（低可靠，需保守）
            double R1 = sig(k1 * (J_c - J));

            // ── 拉伸侧可靠性 R2 ───────────────────────────────────────
            // J < J_s(2.0): R2→1（高可靠，可激进）
            // J > J_s(2.0): R2→0（低可靠，需保守）
            double R2 = sig(k2 * (J_s - J));

            // ── 合并：加权混合 + J<0安全 ──────────────────────────────
            double R = safe * ((1.0 - w) * R1 + w * R2);

            alpha_J = std::max(R, TinyAD::EPS);   // 防除零
            alpha_J = std::pow(1.0/alpha_J, 1.0/3.0); // 平滑过渡
            alpha_J = clamp01(alpha_J);
            break;
        }
        case 2:
        {
            // double J_threshold_compress = 0.9; // 阈值，根据实际情况调整
            double J_threshold = 1.5; // 阈值，根据实际情况调整
            if (J < 0) {
                alpha_J = std::min(1.0, std::max(0.0, std::abs(1.0 + J))); // J < 0 时，|J|越小越可信，刚开始翻转;
            }
            else if (J < J_threshold) {
                alpha_J = 0; // J < J_threshold 时，alpha_J=0;
            }
            else
            {
                // J >= J_threshold: 线性增长到1
                double J_max = J_threshold + 1.0;  // 假设最大J为阈值+1
                alpha_J = std::min(1.0, std::max(0.0, (J - J_threshold) / (J_max - J_threshold)));
            }
            break;
        }
        case 3:
        {
            double J_threshold = 0.9; // 阈值，根据实际情况调整
            alpha_J = J - J_threshold;
            alpha_J = alpha_J * alpha_J;
            alpha_J = std::max(0.0, std::min(1.0, alpha_J));
            break;
        }
        case 4:
        {
            alpha_J = std::abs(J) ;
            break;
        }
        default:
        {
            double J_threshold = 0.9; // 阈值，根据实际情况调整
            if (J < 0) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT("Warning: J < 0, J: " << J); 
                #endif
                if (std::abs(1.0 - J_threshold) < 1e-6) {
                    alpha_J = 0.0;  // 避免除零
                } 
                else
                {
                    alpha_J = std::min(1.0, std::max(0.0, std::abs(J) / (1.0 - J_threshold))); // J < 0 时，alpha_J 线性下降，J=-1时alpha_J=0);
                }
                
            }
            else if (J < J_threshold) // J在[0, J_threshold)范围内不可信
            {
                alpha_J = 0; 
            }
            else
            {
                // J >= J_threshold: 线性增长
                double J_max = J_threshold + 1.0;
                alpha_J = std::min(1.0, std::max(0.0, (J - J_threshold) / (J_max - J_threshold)));
            }
            break;
        }
    }
    
    

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("alpha_J, alpha_J(clamped), J:"<<alpha_J<<","<<clamp01(alpha_J)<<","<<J);   
    #endif

    alpha_J = clamp01(alpha_J);
    return alpha_J;
}

/**
 * 
 */
void updateGamma(double &gamma, double _J)
{
    int method = g_update_gamma_mode;
    _J = std::max(0.0, std::min(1.0, std::abs(_J)));
    switch(method)
    {
        case 1:// 覆盖模式
        {
            gamma = std::pow(std::abs(_J), 1.0/3.0) / TinyAD::g_MU;
            break;
        }
        case 2: // 累积模式（需要确保 gamma 已初始化）
        {
            double coeff = std::pow(std::abs(_J), 1.0/3.0) / TinyAD::g_MU;
            gamma *= coeff;
            break;
        }
        case 3:// 覆盖模式
        {
            gamma =  TinyAD::g_MU/(std::pow(std::abs(_J), 1.0/3.0)+ TinyAD::EPS_1E_8);
            break;
        }
        default: // 保持 gamma 不变
        {
            break;
        }
    }
}

/**
 * @brief 计算基于特征值的hessian信息指标，衡量负特征值的占比，范围0-1，越大说明负特征值占比越大，可能需要更激进的增强
 */

double computeAlpha_hessian( const Eigen::VectorXd& eigenvalues)
{
    double alpha_hessian = 0.0;
    double sum_lambda = 0;
    double sum_neg_lambda = 0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        double lambda_i = eigenvalues(i);
        double abs_lambda_i = std::abs(lambda_i);

        sum_lambda += abs_lambda_i; // ✅ 累加绝对值
        if (lambda_i < 0) {
            sum_neg_lambda += abs_lambda_i; // ✅ 累加负特征值的绝对值
        }
    }
    double r = sum_neg_lambda / (sum_lambda + TinyAD::EPS); // 负特征值占比

    // 使用指数映射，可选择调整敏感度系数
    const double sensitivity = 4.0;  // 可调参数
    alpha_hessian = 1.0 - std::exp(-sensitivity * r); // 通过指数函数映射到0-1，增强敏感度,当 r 接近 1 时，exp(-4) ≈ 0.0183
    // alpha_hessian = 1 - std::exp(-4.0 * r); 

    alpha_hessian = std::max(0.0, std::min(1.0, alpha_hessian)); // 确保在0-1范围内
    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("alpha_hessian:"<<alpha_hessian<<", r:"<<r);  
    #endif

    return alpha_hessian;
}
/**
 * @brief 优化单个负特征值（带梯度信息）
 * case 1: 收缩程度受grad影响, 0-1.0
 * case 2: 收缩程度受hessian影响, 0-1.0
 * case 3: hessian信息和梯度信息共同决定收缩程度，hessian主导，grad微调，0.8-1.0
 * case 4: hessian信息和梯度信息共同决定收缩程度，grad主导，hessian微调，0.8-1.0
 * case 5: 不收缩1.0,abs(lambda)
 * case 0: 强收缩0.0，clamp(lambda),直接投影到m_eps
 */
double optimizeVpnEigenvalue_neg( 
    double lambda, double alpha_grad, double alpha_hessian, double alpha_J,
    int mode, const double _eigenvalue_eps)
{
    double alpha_base = 1.0 ;
    double alpha_lambda = 1.0;
    double alpha_eps = 0.0;
    
    double b = -lambda;

    switch(mode) 
    {   
        case 1: // 收缩程度受grad影响, 0-1.0 ，理论公式
        {
            alpha_base = std::max(0.0, std::min(1.0, alpha_grad));
            alpha_lambda = alpha_base;
            // alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));

            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 2: // 收缩程度受hessian影响, 0-1.0
        {
            alpha_base = std::max(0.0, std::min(1.0, alpha_hessian));
            alpha_lambda = alpha_base;
            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));

            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 3: // hessian信息和梯度信息共同决定收缩程度，hessian主导，grad微调，0.8-1.0
        {
            

            alpha_base = std::max(0.0, std::min(1.0, alpha_hessian));
            
            alpha_grad = std::max(0.0, std::min(1.0, alpha_grad));
            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_grad);

            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 4: // hessian信息和梯度信息共同决定收缩程度，grad主导，hessian微调，0.8-1.0
        {
            
            alpha_base = std::max(0.0, std::min(1.0, alpha_grad));

            alpha_hessian = std::max(0.0, std::min(1.0, alpha_hessian));
            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_hessian);
            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 5: // abs
        {
            alpha_lambda = 1.0; 
            alpha_eps = 0.0;
            break;
        }
        case 6: // hessian信息和梯度信息共同决定收缩程度，grad主导，J，0.8-1.0
        {
            
            alpha_base = std::max(0.0, std::min(1.0, alpha_grad));

            alpha_hessian = std::max(0.0, std::min(1.0, alpha_hessian));
            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_hessian);
            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));

            alpha_J = std::max(0.0, std::min(1.0, alpha_J));
            alpha_lambda = std::max(alpha_J,  alpha_lambda); //翻转时，J主导，grad微调
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 7: // grad,J决定收缩程度，J主导，grad微调,翻转时，J主导，grad微调
        {
            
            alpha_base = std::max(0.0, std::min(1.0, alpha_grad));

            alpha_J = std::max(0.0, std::min(1.0, alpha_J));
            alpha_lambda = alpha_base;
            alpha_lambda = std::max(alpha_J,  alpha_lambda); //

            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 8: // J，0.0-1.0
        {
            
            alpha_base = std::max(0.0, std::min(1.0, alpha_J));
            alpha_lambda = std::max(0.0, std::min(1.0, alpha_base));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 9: // grad+J，0.0-1.0, 理论公式
        {
            
            alpha_lambda = alpha_grad * alpha_J;
            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 10: // J base+_grad，0.0-1.0
        {
            
            alpha_base = std::max(0.0, std::min(1.0, alpha_J));
            
            alpha_grad = std::max(0.0, std::min(1.0, alpha_grad));
            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_grad);

            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 11: // grad base+J，0.0-1.0
        {
            
            alpha_base = std::max(0.0, std::min(1.0, alpha_grad));

            alpha_J = std::max(0.0, std::min(1.0, alpha_J));
            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_J);

            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        default: // clamp
        {
            alpha_lambda = 0.0; 
            alpha_eps = 1.0;
            break;
        }
              
    }
    
    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("负alpha_lambda:"<<alpha_lambda<<", 负alpha_base:"<<alpha_base); 
        TINYAD_DEBUG_OUT("负alpha_hessian:"<<alpha_hessian<<", 负alpha_grad:"<<alpha_grad<<", 负alpha_J:"<<alpha_J); 
    #endif
    return alpha_lambda * b + alpha_eps * _eigenvalue_eps;
}

/**
 * @brief 优化单个正特征值（带梯度信息）
 * case 1: 固定0.8,有eps
 * case 2: 固定0.8,无eps
 * case 3: 收缩程度受grad影响, 0.8-1.0
 * case 4: hessian信息和梯度信息共同决定收缩程度，hessian主导，grad微调，0.8-1.0
 * case 5: hessian信息和梯度信息共同决定收缩程度，grad主导，hessian微调，0.8-1.0
 * case 0: 不收缩1.0,clamp(lambda)
 */
double optimizeVpnEigenvalue_pos( 
    double lambda, double alpha_grad, double alpha_hessian, double alpha_J,
    int mode, const double _eigenvalue_eps)
{
    double alpha_lambda = 1.0;
    double alpha_eps = 0.0;
    double alpha_base = 1.0 ;

    switch(mode) 
    {   case 1: //固定0.8,有eps
        {
            // if (alpha_grad >= 1.0)
            // {
            //     alpha_lambda = 0.8;
            // }
            alpha_lambda = 0.8;
              
            alpha_eps = 1.0 - alpha_lambda;
            break;  
        }
        case 2: //固定0.8,无eps
        {
            alpha_lambda = 0.8;  
            alpha_eps = 0.0;
            break;
        }
        case 3: // 收缩程度受grad影响, 0.8-1.0
        {
             // 收缩程度受grad影响
            alpha_base = 1.0 ;
            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_grad);

            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));

            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 4: // hessian信息和梯度信息共同决定收缩程度，hessian主导，grad微调，0.8-1.0
        {
            
            // alpha_base = std::min(1.0, alpha_hessian);
            alpha_base = std::clamp(alpha_hessian, 0.0,1.0);

            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_grad);
            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        case 5: // hessian信息和梯度信息共同决定收缩程度，grad主导，hessian微调，0.8-1.0
        {
            
            alpha_base = std::clamp(alpha_grad, 0.0,1.0);

            alpha_lambda = alpha_base * (0.8 + 0.2 * alpha_hessian);
            alpha_lambda = std::max(0.0, std::min(1.0, alpha_lambda));
            
            alpha_eps = 1.0 - alpha_lambda;
            break;
        }
        default:
        {
            alpha_lambda = 1.0; // 收缩程度受β和权重影响
            alpha_eps = 0.0;
            break;
        }
              
    } 

    #if DEBUG_OUTPUT
        // TINYAD_DEBUG_OUT("alpha_lambda:"<<alpha_lambda<<", alpha_base:"<<alpha_base); 
        // TINYAD_DEBUG_OUT("alpha_hessian:"<<alpha_hessian<<", alpha_grad:"<<alpha_grad<<", alpha_J:"<<alpha_J); 
    #endif

    return alpha_lambda * lambda + alpha_eps * _eigenvalue_eps;

    
}

/**
 * @brief 优化单个特征值（带梯度信息）
 * **
 * 优化单个正特征值（带梯度信息）
 * case 1: 固定0.8,有eps
 * case 2: 固定0.8,无eps
 * case 3: 收缩程度受grad影响, 0.8-1.0
 * case 4: hessian信息和梯度信息共同决定收缩程度,hessian主导,grad微调,0.8-1.0
 * case 5: hessian信息和梯度信息共同决定收缩程度,grad主导,hessian微调,0.8-1.0
 * case 0: 不收缩1.0,clamp(lambda)
 * **
 * 优化单个负特征值（带梯度信息）
 * case 1: 收缩程度受grad影响, 0-1.0
 * case 2: 收缩程度受hessian影响, 0-1.0
 * case 3: hessian信息和梯度信息共同决定收缩程度,hessian主导,grad微调,0.8-1.0
 * case 4: hessian信息和梯度信息共同决定收缩程度,grad主导,hessian微调,0.8-1.0
 * case 5: 不收缩1.0,abs(lambda)
 * case 0: 强收缩0.0,clamp(lambda),直接投影到m_eps
 *
 */
 double optimizeVpnEigenvalue( 
    double lambda, double alpha_grad, double alpha_hessian, double alpha_J,
    const double _eigenvalue_eps = TinyAD::EPS_1E_8)
{
    
    // m_eps = TinyAD::EPS_1E_8;
    int pos_mode = 1; 
    int neg_mode = 1; 

    pos_mode = g_pos_mode;
    neg_mode = g_neg_mode;

    if (std::abs(lambda) <= _eigenvalue_eps) {
        return _eigenvalue_eps;
    }
    else if (lambda > _eigenvalue_eps) {
        return optimizeVpnEigenvalue_pos(lambda, alpha_grad, alpha_hessian, alpha_J, pos_mode, _eigenvalue_eps);
    } else {
        return optimizeVpnEigenvalue_neg(lambda, alpha_grad, alpha_hessian, alpha_J, neg_mode, _eigenvalue_eps);
    }
    // return 0.0;
    
}

// 输入: Q = eigenvectors (12x12), g_e = 单元梯度向量 (12维)
// 输出: a2 = 每个特征向量方向的 g_i
Eigen::VectorXd computeGradientProjection_vpn(
    const Eigen::MatrixXd& eigenvectors, 
    const Eigen::VectorXd& g)
{
    // g_rot = Q^T * g_e  →  梯度在特征向量上的投影
    Eigen::VectorXd g_rot = eigenvectors.transpose() * g;

    return g_rot;
}

void computeAlpha_grad_eta( double& eta_pos, double& eta_neg, double Kappa = 1, 
    double pos_lambda_ratio = 0.0, double neg_lambda_ratio = 0.0, 
    double pos_grad_ratio = 0.0, double neg_grad_ratio = 0.0)
{

    int method = g_eta_mode;
    Kappa = clamp01(Kappa);
    switch(method)
    {
        case 1: // Kappa 缩放模式
        {
            eta_pos = 0.7 * Kappa;
            eta_neg = 0.2 * Kappa;
            break;
        }
        case 2: // 固定值模式（忽略 Kappa）
        {
            eta_pos = 0.7;
            eta_neg = 0.2;
            break;
        }
        case 3: // lambda_ratio（忽略 Kappa）
        {
            eta_pos = pos_lambda_ratio;
            eta_neg = neg_lambda_ratio;
            break;
        }
        case 4: // lambda_grad_ratio（忽略 Kappa）
        {
            eta_pos = pos_grad_ratio;
            eta_neg = neg_grad_ratio;
            break;
        }
        case 5: // 固定值模式（忽略 Kappa）
        {
            eta_pos = 1.0;
            eta_neg = 1.0;
            break;
        }
        case 6: // lambda_ratio（忽略 Kappa）
        {
            eta_pos = pos_lambda_ratio+neg_lambda_ratio;
            eta_neg = eta_pos;
            break;
        }
        case 7: // lambda_grad_ratio（忽略 Kappa）
        {
            eta_pos = pos_grad_ratio + neg_grad_ratio;
            eta_neg = eta_pos;
            break;
        }
        case 8: // kappa
        {
            eta_pos = Kappa;
            eta_neg = Kappa;
            break;
        }
        case 9: // 固定值模式（忽略 Kappa）
        {
            eta_pos = 0.5;
            eta_neg = 0.5;
            break;
        }
        case 10: // 固定值模式（忽略 Kappa）
        {
            eta_pos = 0.1;
            eta_neg = 0.1;
            break;
        }
        default: // 保守的对称模式
        {
            eta_pos = 0.5;
            eta_neg = 0.5;
            break;
        }
    }
    if (eta_pos >1.0 || eta_neg > 1.0) {
        TINYAD_ERROR("eta_pos,eta_neg :"<<eta_pos<<","<<eta_neg ); 
    }
    
    eta_pos = std::max(0.0, std::min(1.0, eta_pos));
    eta_neg = std::max(0.0, std::min(1.0, eta_neg));
}

double computeAlpha_grad_kappa(const Eigen::VectorXd& eigenvalues, const double alpha_J = 1.0)
{
    int method = g_kappa_mode; //tbd
    double Kappa = 1.0;
    switch(method)
    {
        case 1:
        {
            Kappa = computeKappa(2, eigenvalues);
            break;
        }
        case 2:
        {
            Kappa = alpha_J;
            break;
        }
        case 3:
        {
            Kappa = computeKappa(2, eigenvalues);
            Kappa = std::max(Kappa, alpha_J);
            break;
        }
        default:
        {
            break; // designed
        }
    }
    
    Kappa = std::max(0.0, std::min(1.0, Kappa));
    return Kappa;
}


// 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
double computeAlpha_grad_safe(double P, double B, double beta)
{
    double alpha = 0.0;
    double C = P / ( 2 * beta * B * B + TinyAD::EPS);
    alpha = std::pow(C, 1.0 / 3.0);
    // alpha = std::min(1.0, std::max(0.0, alpha));
    return alpha;
}

// (P+(P*P+2* eta *energy_e *p)^1/2)/(2* eta *energy_e)
double computeAlpha_grad_safe_min(double P, double energy_e, double eta)
{
    
    double alpha_min = 0.0;

    double eta_energy = 2 * eta * energy_e;
    alpha_min = (P + std::sqrt(P * (P +  eta_energy))) / ( eta_energy + TinyAD::EPS);
    // alpha_min = std::min(1.0, std::max(0.0, alpha_min));

    #if DEBUG_OUTPUT
        TINYAD_INFO("P:"<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
    #endif
    return alpha_min;
}

// (P+(P*P+2* eta *energy_e *p)^1/2)/(2* eta *energy_e)
double computeAlpha_grad_min(double P, double energy_e, double eta)
{
    return computeAlpha_grad_safe_min(P, energy_e, eta);
}



// std::pow(C/2.0, 1/3) > std::pow(C/2.0, 1/2)
// 基于解析公式的混合策略：α = (S_g / (4γS_λ))^(1/2 or 1/3)，其中γ是一个超参数，需标定。
double computeAlpha_grad_unsafe(double P, double B, double beta, double coeff = 1.0/3.0)
{
    double alpha = 0.0;
    double C = P / ( 4 * beta * B * B + TinyAD::EPS);
    alpha = std::pow(C, coeff); //different from safe version
    // alpha = std::min(1.0, std::max(0.0, alpha));

    #if DEBUG_OUTPUT
        TINYAD_INFO("P:"<<P <<",B:"<<B<<",beta:"<<beta<<",coeff:"<<coeff<<")"); 
    #endif
    return alpha;
}

// P/(eta_neg * energy)
double computeAlpha_grad_unsafe_min(double P, double energy_e, double eta)
{
    double alpha_min = 0.0;

    alpha_min = P / (eta * energy_e + TinyAD::EPS);
    // alpha_min = std::min(1.0, alpha_min);
    // alpha_min = std::min(1.0, std::max(0.0, alpha_min));

    return alpha_min;
}

#include <cmath>
#include <algorithm>
#include <stdexcept>

/**
 * 无参数自适应能量约束的特征值修正系数计算
 * 
 * @param P        负特征值聚合参数: sum( (p_k^2) / b_k )
 * @param C        正特征值聚合参数: sum( (p_k^2) / lambda_k )
 * @param E_elem   单元当前弹性势能 (必须 > 0)
 * @return         返回 pair{alpha_n, alpha_p}
 *                 alpha_n ∈ (0,1] 负特征值混合系数
 *                 alpha_p ∈ [0.5,1] 正特征值收缩系数
 */
bool computeAlpha_energy_thresh5(const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    const double E_elem, double& alpha_p, double& alpha_n, const double _eigenvalue_eps)
{
    
    alpha_n = 1.0;
    double alpha_p_min = 0.65; //0.7
    alpha_p = alpha_p_min;
    // 边界与非法输入处理
    if (E_elem <= 0.0) {
        // 单元能量为零或负，无法形成有效约束，采用最保守策略
        // alpha_p = 0.5;
        // alpha_n = 1.0;
        TINYAD_WARNING("No energy"); 
        return false;
    }

    // compute P, C
    double P = 0.0, C = 0.0;
    double P_and_C = 0.0;
    double S_lambda_neg = 0.0, S_lambda_pos = 0.0,S_lambda_total = 0.0;

    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i] * proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] < -_eigenvalue_eps) {
            P += g_i_sq / (abs_lambda_i+TinyAD::EPS);  
            S_lambda_neg += abs_lambda_i;
        }
        if (eigenvalues[i] > _eigenvalue_eps) {
            C += g_i_sq / (abs_lambda_i +TinyAD::EPS);  
            S_lambda_pos += abs_lambda_i;
        }  
    }
    P_and_C = P + C;
    S_lambda_total = S_lambda_neg + S_lambda_pos;

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" P "<<P<<", C "<<C <<", E_elem "<<E_elem); 
        TINYAD_DEBUG_OUT(" S_lambda_pos "<<S_lambda_pos<<", S_lambda_neg "<<S_lambda_neg); 
    #endif

    if (P <= TinyAD::EPS )
    {
        alpha_n = 1.0;
        return true;
    }
    
    double eta_pos = 1.0;
    double eta_neg = 1.0;
    double pos_lambda_ratio = S_lambda_pos/(S_lambda_total+TinyAD::EPS);
    double neg_lambda_ratio = S_lambda_neg/(S_lambda_total+TinyAD::EPS);
    double pos_grad_ratio = std::sqrt(C/(P_and_C+TinyAD::EPS));
    double neg_grad_ratio = std::sqrt(P/(P_and_C+TinyAD::EPS));
    double kappa = 1.0;
    computeAlpha_grad_eta(eta_pos, eta_neg, kappa, pos_lambda_ratio, neg_lambda_ratio,
         pos_grad_ratio, neg_grad_ratio);
    // eta_pos = pos_grad_ratio;
    // eta_neg = neg_grad_ratio;
    
    eta_pos = std::clamp(eta_pos, 0.0, 1.0);
    eta_neg = std::clamp(eta_neg, 0.0, 1.0);

    // neg eigenvalue
    const double f1 = 1.5 * P;   // 最小下降量
    double eta_E = eta_neg * E_elem;
    if (f1 > eta_E) // 最小下降量
    {
        alpha_n = 1.0;
    }
    else // 更大下降量
    {
        double sqrt_disc = std::sqrt(P * (P + 2.0 * eta_E ));
        alpha_n = (P + sqrt_disc) / (2.0 * eta_E +TinyAD::EPS);   
    }

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" f1(min neg) "<<f1 ); 
        TINYAD_DEBUG_OUT(" eta_pos "<<eta_pos<<", eta_neg "<< eta_neg ); 
        TINYAD_DEBUG_OUT(" alpha_p "<<alpha_p<<", alpha_n "<< alpha_n ); 
    #endif 

    
    if (alpha_n > 1.0)
    {
        TINYAD_WARNING("alpha_n("<<alpha_n<<") > "<<1.0);
    }
    
    alpha_p = std::clamp(alpha_p, alpha_p_min, 1.0);
    alpha_n = std::clamp(alpha_n, 0.0, 1.0);

    return true;
}

/**
 * 无参数自适应能量约束的特征值修正系数计算
 * 
 * @param P        负特征值聚合参数: sum( (p_k^2) / b_k )
 * @param C        正特征值聚合参数: sum( (p_k^2) / lambda_k )
 * @param E_elem   单元当前弹性势能 (必须 > 0)
 * @return         返回 pair{alpha_n, alpha_p}
 *                 alpha_n ∈ (0,1] 负特征值混合系数
 *                 alpha_p ∈ [0.5,1] 正特征值收缩系数
 */
bool computeAlpha_energy_thresh4(const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    const double E_elem, double& alpha_p, double& alpha_n, const double _eigenvalue_eps)
{
    
    alpha_n = 1.0;
    double alpha_p_min = 0.5;
    alpha_p = 1.0;
    // 边界与非法输入处理
    if (E_elem <= 0.0) {
        // 单元能量为零或负，无法形成有效约束，采用最保守策略
        // alpha_p = 0.5;
        // alpha_n = 1.0;
        TINYAD_WARNING("No energy"); 
        return false;
    }

    // compute P, C
    double P = 0.0, C = 0.0;
    double P_and_C = 0.0;
    double S_lambda_neg = 0.0, S_lambda_pos = 0.0,S_lambda_total = 0.0;

    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i] * proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] < -_eigenvalue_eps) {
            P += g_i_sq / (abs_lambda_i+TinyAD::EPS);  
            S_lambda_neg += abs_lambda_i;
        }
        if (eigenvalues[i] > _eigenvalue_eps) {
            C += g_i_sq / (abs_lambda_i +TinyAD::EPS);  
            S_lambda_pos += abs_lambda_i;
        }  
    }
    P_and_C = P + C;
    S_lambda_total = S_lambda_neg + S_lambda_pos;

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" P "<<P<<", C "<<C <<", E_elem "<<E_elem); 
        TINYAD_DEBUG_OUT(" S_lambda_pos "<<S_lambda_pos<<", S_lambda_neg "<<S_lambda_neg); 
    #endif

    if (P <= TinyAD::EPS )
    {
        alpha_n = 1.0;
        return true;
    }
    if (C <= TinyAD::EPS)
    {
        alpha_p = 1.0;
        return true;
    }

    double eta_pos = 1.0;
    double eta_neg = 1.0;
    double pos_lambda_ratio = S_lambda_pos/(S_lambda_total+TinyAD::EPS);
    double neg_lambda_ratio = S_lambda_neg/(S_lambda_total+TinyAD::EPS);
    double pos_grad_ratio = std::sqrt(C/(P_and_C+TinyAD::EPS));
    double neg_grad_ratio = std::sqrt(P/(P_and_C+TinyAD::EPS));
    double kappa = 1.0;
    computeAlpha_grad_eta(eta_pos, eta_neg, kappa, pos_lambda_ratio, neg_lambda_ratio,
         pos_grad_ratio, neg_grad_ratio);
    // eta_pos = pos_grad_ratio;
    // eta_neg = neg_grad_ratio;
    
    eta_pos = std::clamp(eta_pos, 0.0, 1.0);
    eta_neg = std::clamp(eta_neg, 0.0, 1.0);

    // pos eigenvalue
    const double Q0 = 0.5 * C;
    double eta_E = eta_pos * E_elem;
    if (Q0 <= eta_E) // 最大下降量 [0.5,1]
    {
        alpha_p = 1.0;
    }
    else // 更小下降量 [0.5,1]
    {
        if (eta_E < C * TinyAD::EPS_1E_12 )
        {
            alpha_p = alpha_p_min;
        }
        else
        {
            double sqrt_disc = std::sqrt(C * (C - 2.0 * eta_E ));
            alpha_p = (C - sqrt_disc) / (2.0 * eta_E +TinyAD::EPS); 
        }
          
    }

    // neg eigenvalue
    const double f1 = 1.5 * P;   // 最小下降量
    eta_E = eta_neg * E_elem;
    if (f1 > eta_E) // 最小下降量
    {
        alpha_n = 1.0;
    }
    else // 更大下降量
    {
        double sqrt_disc = std::sqrt(P * (P + 2.0 * eta_E ));
        alpha_n = (P + sqrt_disc) / (2.0 * eta_E +TinyAD::EPS);   
    }

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" f1(min neg) "<<f1<<", Q0(max pos) "<<Q0 ); 
        TINYAD_DEBUG_OUT(" eta_pos "<<eta_pos<<", eta_neg "<< eta_neg ); 
        TINYAD_DEBUG_OUT(" alpha_p "<<alpha_p<<", alpha_n "<< alpha_n ); 
    #endif 

    if (alpha_p < alpha_p_min)
    {
        TINYAD_WARNING("alpha_p("<<alpha_p<<") < "<<alpha_p_min);
    }
    if (alpha_n > 1.0)
    {
        TINYAD_WARNING("alpha_n("<<alpha_n<<") > "<<1.0);
    }
    
    alpha_p = std::clamp(alpha_p, alpha_p_min, 1.0);
    alpha_n = std::clamp(alpha_n, 0.0, 1.0);

    return true;
}

/**
 * 无参数自适应能量约束的特征值修正系数计算
 * 
 * @param P        负特征值聚合参数: sum( (p_k^2) / b_k )
 * @param C        正特征值聚合参数: sum( (p_k^2) / lambda_k )
 * @param E_elem   单元当前弹性势能 (必须 > 0)
 * @return         返回 pair{alpha_n, alpha_p}
 *                 alpha_n ∈ (0,1] 负特征值混合系数
 *                 alpha_p ∈ [0.5,1] 正特征值收缩系数
 */
bool computeAlpha_energy_thresh(const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    const double E_elem, double& alpha_p, double& alpha_n)
{
    alpha_p = 1.0;
    alpha_n = 1.0;
    double alpha_p_min = 0.8;
    // 边界与非法输入处理
    if (E_elem <= 0.0) {
        // 单元能量为零或负，无法形成有效约束，采用最保守策略
        // alpha_p = 0.5;
        // alpha_n = 1.0;
        TINYAD_WARNING("No energy"); 
        return false;
    }

    // compute P, C
    double P=0.0, C=0.0;
    bool hasNegLambda = false;
    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i] * proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] < 0.0) {
            P += g_i_sq / (abs_lambda_i+TinyAD::EPS);
            hasNegLambda = true;   
        }
        if (eigenvalues[i] > 0.0) {
            C += g_i_sq / (abs_lambda_i +TinyAD::EPS);  
        }  
    }
    if (hasNegLambda == false) //no neg lambda
    {
        alpha_p = 1.0;
        alpha_n = 1.0;
        TINYAD_WARNING("no negative lamda"); 
        return false;
    }

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" P "<<P<<", C "<<C <<", E_elem "<<E_elem); 
    #endif
    // if (P < 0.0 || C < 0.0) {//obsolete
    //     throw std::invalid_argument("P and C must be non-negative");
    // }

    if (P <= TinyAD::EPS || C <= TinyAD::EPS) //和梯度正交，由梯度主导
    {
        alpha_p = 1.0;
        alpha_n = 1.0;
        TINYAD_WARNING("P ("<<P<<") or C("<<C<<") is close to EPS"); 
        return false;
    }

    // 正特征值在 alpha_p = 1 时的固有能量下降
    const double Q0 = 0.5 * C;

    // 扣除正特征值占用后留给负特征值调控的能量预算
    const double R = E_elem - Q0;
    // 负特征值在 alpha_n = 1 时的下降量 (最保守)
    const double f1 = 1.5 * P;   // P*(1/1 + 1/(2*1^2)) = 1.5P
    // // 情形1：正特征值自身已耗尽或超出总能量 → 最保守策略，说明二阶近似不准确
    // if (R <= TinyAD::EPS) {
    //     // alpha_p = 0.5;
    //     alpha_p = 1.0;
    //     alpha_n = 0.0;
    //     return true;
    // }
    // else 

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" f1(min neg) "<<f1<<", Q0(max pos) "<<Q0 <<", R "<<R); 
    #endif 
    if (R < f1) // 情形2：预算不足 —— 即使负特征值保守仍超出预算
    {
        // 强制负特征值最保守
        // const double alpha_n = 1.0;
        alpha_n = 1.0;
        // 正特征值最多可消耗的剩余能量
        const double S = E_elem - f1;

        if (S <= 0.0) {
            // 即使正特征值贡献为零仍超限 → 取最小可能值
            alpha_p = alpha_p_min;
        } else {
            // 解二次方程： 2*S*alpha_p^2 - 2*C*alpha_p + C = 0
            // 判别式
            const double disc = C * C - 2.0 * S * C;
            if (disc < 0.0) {
                // 数值上不应出现，但以防万一取安全值
                alpha_p = alpha_p_min;
            } else {
                // 取 [0.5, 1] 内的根 (另一根可能大于1或小于0.5)
                const double sqrt_disc = std::sqrt(disc);
                const double alpha_p1 = (C - sqrt_disc) / (2.0 * S+TinyAD::EPS);
                const double alpha_p2 = (C + sqrt_disc) / (2.0 * S+TinyAD::EPS);
                double max_alpha_p = std::max(alpha_p1,alpha_p2);
                double min_alpha_p = std::min(alpha_p1,alpha_p2);

                if (alpha_p1 >= alpha_p_min && alpha_p1 <= 1.0) {
                    alpha_p = alpha_p1;
                } else if (alpha_p2 >= alpha_p_min && alpha_p2 <= 1.0) {
                    alpha_p = alpha_p2;
                } else {
                    // 理论解不在范围内时取边界
                    alpha_p = alpha_p_min;
                }
            }
        }
        
    }
    else // 情形3：预算充足 —— 可通过调整负特征值使总下降等于 E_elem
    {
        // 解二次方程： 2*R*alpha_n^2 - 2*P*alpha_n - P = 0
        const double disc = P * P + 2.0 * R * P;  // P^2 + 2RP
        const double sqrt_disc = std::sqrt(disc);
        alpha_n = (P + sqrt_disc) / (2.0 * R+TinyAD::EPS);

        // 截断到合理范围 (理论上应 ≤ 1，数值误差下可能略大于1)
        alpha_n = std::clamp(alpha_n, 0.0, 1.0);

        // 正特征值保持原样    
    }

    return true;
}

/**
 * 无参数自适应能量约束的特征值修正系数计算
 * 
 * @param P        负特征值聚合参数: sum( (p_k^2) / b_k )
 * @param C        正特征值聚合参数: sum( (p_k^2) / lambda_k )
 * @param E_elem   单元当前弹性势能 (必须 > 0)
 * @return         返回 pair{alpha_n, alpha_p}
 *                 alpha_n ∈ (0,1] 负特征值混合系数
 *                 alpha_p ∈ [0.5,1] 正特征值收缩系数
 */
bool computeAlpha_energy_thresh2(const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    const double E_elem, double& alpha_p, double& alpha_n)
{
    alpha_p = 1.0;
    alpha_n = 1.0;
    double alpha_p_max = 2;
    // 1.边界与非法输入处理
    if (E_elem <= 0.0) {
        // 单元能量为零或负，无法形成有效约束，采用最保守策略
        // alpha_p = 0.5;
        // alpha_n = 1.0;
        TINYAD_WARNING("No energy"); 
        return false;
    }

    // 2.compute P, C
    double P=0.0, C=0.0;
    bool hasNegLambda = false;
    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i] * proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] < 0.0) {
            P += g_i_sq / (abs_lambda_i+TinyAD::EPS);
            hasNegLambda = true;   
        }
        if (eigenvalues[i] > 0.0) {
            C += g_i_sq / (abs_lambda_i+TinyAD::EPS );  
        }  
    }
    if (hasNegLambda == false) //no neg lambda
    {
        alpha_p = 1.0;
        alpha_n = 1.0;
        TINYAD_WARNING("no negative lamda"); 
        return false;
    }

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" P "<<P<<", C "<<C <<", E_elem "<<E_elem); 
    #endif
 
    // 正特征值在 alpha_p = 1 时的固有能量下降
    const double Q0 = 0.5 * C;

    // 扣除正特征值占用后留给负特征值调控的能量预算
    const double R = E_elem - Q0;
    // 负特征值在 alpha_n = 1 时的下降量 (最保守)
    const double f1 = 1.5 * P;   // P*(1/1 + 1/(2*1^2)) = 1.5P

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" f1(min neg) "<<f1<<", Q0(max pos) "<<Q0 <<", R "<<R); 
    #endif 
    
    
    // 特殊情形：无负特征值
    if (P < TinyAD::EPS) {
        // 只有正特征值贡献
        alpha_n = 0.0;
        if (Q0 <= E_elem) {
            // 基准点可行，最小修正
            return false;
        } else {
            // 需降低正特征值贡献，通过增大 αp
            // 解 Q(αp) = E_elem
            // 2*E_elem*αp² - 2*C*αp + C = 0
            double disc = C * C - 2.0 * E_elem * C;
            if (disc < 0.0) {
                // 无实数解，取极限
                alpha_p = alpha_p_max;
                return false;
            }
            double sqrt_disc = std::sqrt(disc);
            alpha_p = (C + sqrt_disc) / (2.0 * E_elem+TinyAD::EPS);
            if (alpha_p < 1.0) alpha_p = 1.0; // 数值裁剪
            if (alpha_p > alpha_p_max) alpha_p = alpha_p_max;
            return false;
        }
    }
    
    // 特殊情形：无正特征值
    if (C < TinyAD::EPS) {
        // 只有负特征值贡献
    
        if (f1 <= E_elem) {
            // 可减小 αn 以降低矩阵修正量，同时释放能量预算
            double R = E_elem;   // Q0 = 0
            alpha_n = (P + std::sqrt(P * P + 2.0 * R * P)) / (2.0 * R+TinyAD::EPS);
            if (alpha_n > 1.0) alpha_n = 1.0;
            if (alpha_n < TinyAD::EPS) alpha_n = TinyAD::EPS; // 避免完全钳位
            return false;
        } else {
            // 预算不足，无法通过正特征值补偿，只能接受最保守
            return false; // αp 无影响
        }
    }
    
    // ---------- 一般情况 (P>0, C>0) ----------
    const double deltaE0 = f1 + Q0;          // 基准总下降
    
    if (deltaE0 <= E_elem) {
        // 情况 A：基准点可行，减小 αn 来增加负特征值修正量，直至能量约束边界
        double R = E_elem - Q0;   // 留给负特征值的预算
        // 解 P*(1/α + 1/(2α²)) = R  ⇒ 二次方程 2Rα² - 2Pα - P = 0
        double disc = P * P + 2.0 * R * P;   // P² + 2RP
        double sqrt_disc = std::sqrt(disc);
        alpha_n = (P + sqrt_disc) / (2.0 * R+TinyAD::EPS);
        
        // 截断
        if (alpha_n > 1.0) alpha_n = 1.0;
        if (alpha_n < TinyAD::EPS) alpha_n = TinyAD::EPS;
        return true;
        
    } else {
        // 情况 B：基准点违反约束，固定 αn=1，通过增大 αp 降低正特征值贡献
        alpha_n = 1.0;
        double S = E_elem - f1;   // 正特征值允许的下降量
        
        if (S <= 0.0) {
            // 即使正特征值贡献降为 0 也无法满足约束
            alpha_p = alpha_p_max;
            return false;
        }
        
        // 解 Q(αp) = S  ⇒ C*(1/αp - 1/(2αp²)) = S
        // 转化为 2S αp² - 2C αp + C = 0
        double disc = C * C - 2.0 * S * C;
        if (disc < 0.0) {
            // 无解（理论上 S < 0.5C 保证有解，但数值误差可能导致）
            alpha_p = alpha_p_max;
            return false;
        }
        double sqrt_disc = std::sqrt(disc);
        // 取较大的根 (αp > 1)
        alpha_p = (C + sqrt_disc) / (2.0 * S+TinyAD::EPS);
        
        // 数值安全裁剪
        if (alpha_p < 1.0) alpha_p = 1.0;
        if (alpha_p > alpha_p_max) alpha_p = alpha_p_max;
        return true;
    }


    return true;
}

/**
 * 无参数自适应能量约束的特征值修正系数计算
 * 
 * @param P        负特征值聚合参数: sum( (p_k^2) / b_k )
 * @param C        正特征值聚合参数: sum( (p_k^2) / lambda_k )
 * @param E_elem   单元当前弹性势能 (必须 > 0)
 * @return         返回 pair{alpha_n, alpha_p}
 *                 alpha_n ∈ (0,1] 负特征值混合系数
 *                 alpha_p ∈ [0.5,1] 正特征值收缩系数
 */
bool computeAlpha_energy_thresh3(const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    const double E_elem, double& alpha_p, double& alpha_n)
{
    alpha_p = 1.0;
    alpha_n = 1.0;
    // double alpha_p_max = 2;
    double alpha_p_min = 0.5;
    // 1.边界与非法输入处理
    if (E_elem <= 0.0) {
        // 单元能量为零或负，无法形成有效约束，采用最保守策略
        // alpha_p = 0.5;
        // alpha_n = 1.0;
        TINYAD_WARNING("No energy"); 
        return false;
    }

    // 2.compute P, C
    double P=0.0, C=0.0;
    bool hasNegLambda = false;
    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i] * proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] < 0.0) {
            P += g_i_sq / (abs_lambda_i+TinyAD::EPS);
            hasNegLambda = true;   
        }
        if (eigenvalues[i] > 0.0) {
            C += g_i_sq / (abs_lambda_i+TinyAD::EPS);  
        }  
    }
    if (hasNegLambda == false) //no neg lambda
    {
        alpha_p = 1.0;
        alpha_n = 1.0;
        TINYAD_WARNING("no negative lamda"); 
        return false;
    }

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" P "<<P<<", C "<<C <<", E_elem "<<E_elem); 
    #endif
 
    // 正特征值在 alpha_p = 1 时的固有能量下降
    const double Q0 = 0.5 * C;

    // 扣除正特征值占用后留给负特征值调控的能量预算
    const double R = E_elem - Q0;
    // 负特征值在 alpha_n = 1 时的下降量 (最保守)
    const double f1 = 1.5 * P;   // P*(1/1 + 1/(2*1^2)) = 1.5P

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT(" f1(min neg) "<<f1<<", Q0(max pos) "<<Q0 <<", R "<<R); 
    #endif 
    
    
    // 特殊情形：无负特征值
    if (P < TinyAD::EPS) {
        // 只有正特征值贡献
        alpha_n = 0.0;
        if (Q0 <= E_elem) {
            // 基准点可行，最小修正
            return false;
        } else {
            // 需降低正特征值贡献，通过减小 αp
            // 解 Q(αp) = E_elem
            // 2*E_elem*αp² - 2*C*αp + C = 0
            double disc = C * C - 2.0 * E_elem * C;
            if (disc < 0.0) {
                // 无实数解，取极限
                alpha_p = alpha_p_min;
                return false;
            }
            double sqrt_disc = std::sqrt(disc);
            double alpha_p = (C - sqrt_disc) / (2.0 * E_elem+TinyAD::EPS);
            alpha_p = std::clamp(alpha_p, alpha_p_min, 1.0);
            return false;
        }
    }
    
    // 特殊情形：无正特征值
    if (C < TinyAD::EPS) {
        // 只有负特征值贡献
    
        if (f1 <= E_elem) {
            // 可减小 αn 以降低矩阵修正量，同时释放能量预算
            double R = E_elem;   // Q0 = 0
            alpha_n = (P + std::sqrt(P * P + 2.0 * R * P)) / (2.0 * R+TinyAD::EPS);
            // if (alpha_n > 1.0) alpha_n = 1.0;
            // if (alpha_n < TinyAD::EPS) alpha_n = TinyAD::EPS; // 避免完全钳位
            alpha_n = std::clamp(alpha_n, 0.0, 1.0);
            return false;
        } else {
            // 预算不足，无法通过正特征值补偿，只能接受最保守
            return false; // αp 无影响
        }
    }
    
    // ---------- 一般情况 (P>0, C>0) ----------
    const double deltaE0 = f1 + Q0;          // 基准总下降
    
    if (deltaE0 <= E_elem) {
        // 情况 A：基准点可行，减小 αn 来增加负特征值修正量，直至能量约束边界
        double R = E_elem - Q0;   // 留给负特征值的预算
        // 解 P*(1/α + 1/(2α²)) = R  ⇒ 二次方程 2Rα² - 2Pα - P = 0
        double disc = P * P + 2.0 * R * P;   // P² + 2RP
        double sqrt_disc = std::sqrt(disc);
        alpha_n = (P + sqrt_disc) / (2.0 * R+TinyAD::EPS);
        
        // 截断
        alpha_n = std::clamp(alpha_n, 0.0, 1.0);
        return true;
        
    } else {
        // 情况 B：基准点违反约束，固定 αn=1，通过增大 αp 降低正特征值贡献
        alpha_n = 1.0;
        double S = E_elem - f1;   // 正特征值允许的下降量
        
        if (S <= 0.0) {
            // 即使正特征值贡献降为 0 也无法满足约束
            // alpha_p = alpha_p_max;
            alpha_p = alpha_p_min;
            return false;
        }
        
        // 解 Q(αp) = S  ⇒ C*(1/αp - 1/(2αp²)) = S
        // 转化为 2S αp² - 2C αp + C = 0
        double disc = C * C - 2.0 * S * C;
        if (disc < 0.0) {
            // 无解（理论上 S < 0.5C 保证有解，但数值误差可能导致）
            // alpha_p = alpha_p_max;
            alpha_p = alpha_p_min;
            return false;
        }
        double sqrt_disc = std::sqrt(disc);
        // 取较大的根 (αp > 1)
        alpha_p = (C - sqrt_disc) / (2.0 * S+TinyAD::EPS);
        
        // 数值安全裁剪
        // if (alpha_p < 1.0) alpha_p = 1.0;
        // if (alpha_p > alpha_p_max) alpha_p = alpha_p_max;
        alpha_p = std::clamp(alpha_p, alpha_p_min, 1.0);
        return true;
    }


    return true;
}

void computeAlpha_grad( double& alpha_grad_pos, double& alpha_grad_neg,
    const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    double gamma,
    const double alpha_J = 1.0,
    const double _f = 0.0,
    const double volume = 1.0
    )
{
    int method = g_grad_mode;

    // 2. 计算s_g和s_lambda
    // S_g = Σ (g_i² / |λ_i|)   (i ∈ I_neg)
    // S_λ = Σ |λ_i|²
    double S_g_pos = 0.0, S_lambda_pos = 0.0;
    double S_g_neg = 0.0, S_lambda_neg = 0.0;
    double S_g_total = 0.0, S_lambda_total = 0.0;
    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i]*proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] < 0) {

            if (proj_g[i] > 0) //同向
            {
                S_lambda_pos += abs_lambda_i * abs_lambda_i; // |λ_i|²

                // double g_i_sq = proj_g[i]*proj_g[i];                // g_i² (来自 computeGradientProjection)
                S_g_pos += g_i_sq / (abs_lambda_i + TinyAD::EPS);
            }
            else //异向
            {
                S_lambda_neg += abs_lambda_i * abs_lambda_i; // |λ_i|²

                // double g_i_sq = proj_g[i]*proj_g[i];                // g_i² (来自 computeGradientProjection)
                S_g_neg += g_i_sq / (abs_lambda_i + TinyAD::EPS);
            } 
            // S_g_total += g_i_sq / (abs_lambda_i + m_eps);   
        }
        S_g_total += g_i_sq / (abs_lambda_i + TinyAD::EPS);
        S_lambda_total += abs_lambda_i * abs_lambda_i; // |λ_i|²
    }

    double Kappa = computeAlpha_grad_kappa(eigenvalues, alpha_J); 
    //dobule gamma = g_para_gamma;
    
    double alpha_grad_pos_min = 0;
    double alpha_grad_neg_min = 0;
    double energy_e = _f;
    double eta_pos = 1.0;
    double eta_neg = 1.0;
    double pos_lambda_ratio = S_lambda_pos / (S_lambda_total + TinyAD::EPS);
    double neg_lambda_ratio = S_lambda_neg / (S_lambda_total + TinyAD::EPS);
    double pos_grad_ratio = S_g_pos / (S_g_total + TinyAD::EPS);
    double neg_grad_ratio = S_g_neg / (S_g_total + TinyAD::EPS);

    // energy_e = get_energy();
    switch(method)
    {
        case 1: // safe direction 1/3次方的方案 + unsafe direction, clamp到0.0的方案
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            alpha_grad_neg = 0;
            break;
        }
        case 2: // safe direction 1/3次方的方案  + unsafe direction with lower threshold(min)
        {
            // get energy
            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);

            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            // alpha_grad_pos = computeAlpha_grad_safe(S_g_pos,  S_lambda_pos , gamma, m_eps);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+ TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 3: // safe direction 1/3次方的方案 with lower threshold + unsafe direction with lower threshold
        {
            // get energy
            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            //lower threshold
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            #if DEBUG_OUTPUT
                if (alpha_grad_pos < alpha_grad_pos_min)
                {
                    TINYAD_WARNING("!!!safe vector alpha_pos("<<alpha_grad_pos <<")< alpha_min("<<alpha_grad_pos_min<<")"); 
                }
            #endif
            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 4: // 不区分safe&unsafe
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            alpha_grad_neg = alpha_grad_pos;
            break;
            
        }
        case 5: //same to 2, exact element energy, safe direction 1/3次方的方案  + unsafe direction with lower threshold(min)
        {
            // get energy
            // energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 6: // same to 3, exact element energy,safe direction 1/3次方的方案 with lower threshold + unsafe direction with lower threshold
        {
            // get energy
            // energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            //lower threshold
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            #if DEBUG_OUTPUT
                if (alpha_grad_pos < alpha_grad_pos_min)
                {
                    TINYAD_WARNING("!!!safe vector alpha_pos("<<alpha_grad_pos <<")< alpha_min("<<alpha_grad_pos_min<<")"); 
                }
            #endif

            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 7: // same to 2, avg energy, safe direction 1/3次方的方案  + unsafe direction with lower threshold(min)
        {
            // get energy
            energy_e = TinyAD::g_avg_vol_energy * std::abs(volume); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 8: // same to 3, avg_energy,safe direction 1/3次方的方案 with lower threshold + unsafe direction with lower threshold
        {
            // get energy
            energy_e = TinyAD::g_avg_vol_energy * std::abs(volume);

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            //lower threshold
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            #if DEBUG_OUTPUT
                if (alpha_grad_pos < alpha_grad_pos_min)
                {
                    TINYAD_WARNING("!!!safe vector alpha_pos("<<alpha_grad_pos <<")< alpha_min("<<alpha_grad_pos_min<<")"); 
                }
            #endif

            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }   
        case 9: // same to 4, 不区分safe&unsafe,  lower threshold based on total energy
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            alpha_grad_neg = alpha_grad_pos;

            // get energy
            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            double alpha_grad_min = S_g_total_neg / (eta_pos * energy_e + TinyAD::EPS);

            alpha_grad_pos = std::min(std::max(alpha_grad_pos, alpha_grad_min), 1.0);
            alpha_grad_neg = alpha_grad_pos;

            break;
            
        }  
        case 10: // same to 9,different element energy
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            alpha_grad_neg = alpha_grad_pos;

            // get energy
            energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            double alpha_grad_min = S_g_total_neg / (0.5 * energy_e + TinyAD::EPS);

            alpha_grad_pos = std::min(1.0, alpha_grad_min);
            alpha_grad_neg = alpha_grad_pos;
            break;
            
        } 
        case 11: // no safe&unsafe,  alpha_min, energy = max E
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 12: // no safe&unsafe,  alpha_min, energy = element_e
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        } 
        case 13: // no safe&unsafe,  alpha_min,energy-avg
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 14: // no safe&unsafe,  alpha_min, energy = max E
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 15: // no safe&unsafe,  alpha_min,energy-avg
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            energy_e = g_avg_energy; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        } 
        case 16: // no safe&unsafe,  [alpha_min, 1], energy = element_e
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio = ( gamma * S_lambda_total_neg) / (2.0 * S_g_total_neg + TinyAD::EPS);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_min); // [alpha_min, 1]
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 17: // no safe&unsafe,  [alpha_min, 1], energy = element_e
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio = ( gamma * S_lambda_total_neg) / (2.0 * S_g_total_neg + TinyAD::EPS);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_min); // [alpha_min, 1]
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 21: // safe direction 1/3次方的方案 + unsafe direction 1/2次方的方案
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            break;
        }
        case 22: // safe direction 1/3次方的方案 + unsafe direction 1/2次方的方案(min)
        {
            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = S_g_total;

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            
            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));

            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::max(alpha_grad_neg, alpha_grad_neg_min);

            break;
        }
        case 23: // safe direction 1/3次方的方案(min) + unsafe direction 1/2次方的方案(min)
        {
            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = S_g_total;

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::max(alpha_grad_neg, alpha_grad_neg_min);

            break;
        }
        case 24: // safe direction 1/3次方的方案(min) + unsafe direction 1/2次方的方案(max)
        {
            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = S_g_total;

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            alpha_grad_pos = std::min(alpha_grad_pos, alpha_grad_pos_min);


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::max(alpha_grad_neg, alpha_grad_neg_min);

            break;
        }
        case 31: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_LOG
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 32: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 33: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = g_avg_energy; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 34: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = g_avg_energy; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            double kappa_thresh = 0.1;
            if (Kappa < kappa_thresh)
            {
                TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
                alpha_min = 0;
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 35: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = (g_avg_energy +energy_e)/2.0;; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            // double kappa_thresh = 0.1;
            // if (Kappa < kappa_thresh)
            // {
            //     TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
            //     alpha_min = 0;
            // }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 36: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = std::min(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            // double kappa_thresh = 0.1;
            // if (Kappa < kappa_thresh)
            // {
            //     TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
            //     alpha_min = 0;
            // }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 37: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = (g_avg_energy +energy_e)/2.0;; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            double kappa_thresh = 0.1;
            if (Kappa < kappa_thresh)
            {
                TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
                alpha_min = 0;
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 38: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = std::min(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            double kappa_thresh = 0.1;
            if (Kappa < kappa_thresh)
            {
                TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
                alpha_min = 0;
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 39: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            else if ((neg_grad_ratio + pos_grad_ratio) < TinyAD::EPS 
            && (neg_lambda_ratio + pos_lambda_ratio) > 0.5 )//lamda比值和grad比值不一致,说明特征值方向和梯度方向正交
            {
                alpha_min = 1.0;
                TINYAD_WARNING("grad_ratio: "<<(neg_grad_ratio + pos_grad_ratio) <<",lambda_ratio:"<<neg_lambda_ratio + pos_lambda_ratio); 
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 40: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            if (alpha_min > 1.0)
            {
                // TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            else if ((alpha_min) < TinyAD::EPS 
            && (1.0 - std::abs(alpha_J))<  TinyAD::EPS)//lamda比值和grad比值不一致,说明特征值方向和梯度方向正交
            {
                alpha_min = 1.0;
                TINYAD_WARNING("grad_ratio: "<<(neg_grad_ratio + pos_grad_ratio) <<",lambda_ratio:"<<neg_lambda_ratio + pos_lambda_ratio); 
                TINYAD_WARNING("alpha_min: "<<alpha_min <<",alpha_J:"<<alpha_J); 
            
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        default: // safe direction 1/3次方的方案 + unsafe direction 1/2次方的方案
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            break;
        }
    }
    

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("proj_g:"<<proj_g.transpose() <<", gamma:"<<gamma); 
        TINYAD_DEBUG_OUT("S_g_pos(bigger):"<<S_g_pos<<", S_lambda_pos(smaller):"<<S_lambda_pos);
        TINYAD_DEBUG_OUT("S_g_neg(bigger):"<<S_g_neg<<", S_lambda_neg(smaller):"<<S_lambda_neg); 
        TINYAD_DEBUG_OUT("energy_e:"<<energy_e<<", eta_pos:"<<eta_pos<<", eta_neg:"<<eta_neg);
        TINYAD_DEBUG_OUT("alpha_grad_pos_min:"<<alpha_grad_pos_min<<", alpha_grad_neg_min:"<<alpha_grad_neg_min);
        TINYAD_DEBUG_OUT("alpha_grad_pos:"<<alpha_grad_pos<<", alpha_grad_neg:"<<alpha_grad_neg); 
    #endif

    alpha_grad_pos = std::min(1.0, std::max(0.0, alpha_grad_pos));
    alpha_grad_neg = std::min(1.0, std::max(0.0, alpha_grad_neg));

    // return alpha;
}

void computeAlpha_grad2( double& alpha_grad_pos, double& alpha_grad_neg,
    const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    double gamma,
    const double alpha_J = 1.0,
    const double _f = 0.0,
    const double volume = 1.0,
    const double _eigenvalue_eps = TinyAD::EPS
    )
{
    int method = g_grad_mode;

    // 2. 计算s_g和s_lambda
    // S_g = Σ (g_i² / |λ_i|)   (i ∈ I_neg)
    // S_λ = Σ |λ_i|²
    double S_g_pos = 0.0, S_lambda_pos = 0.0;
    double S_g_neg = 0.0, S_lambda_neg = 0.0;
    double S_g_total = 0.0, S_lambda_total = 0.0;
    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i]*proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] < -_eigenvalue_eps) {

            if (proj_g[i] > TinyAD::EPS) //同向
            {
                S_lambda_pos += abs_lambda_i * abs_lambda_i; // |λ_i|²

                // double g_i_sq = proj_g[i]*proj_g[i];                // g_i² (来自 computeGradientProjection)
                S_g_pos += g_i_sq / (abs_lambda_i );
            }
            else if (proj_g[i] < -TinyAD::EPS) //异向
            {
                S_lambda_neg += abs_lambda_i * abs_lambda_i; // |λ_i|²

                // double g_i_sq = proj_g[i]*proj_g[i];                // g_i² (来自 computeGradientProjection)
                S_g_neg += g_i_sq / (abs_lambda_i );
            } 
            // S_g_total += g_i_sq / (abs_lambda_i + m_eps);   
        }
        if (abs_lambda_i > _eigenvalue_eps
            && std::abs(proj_g[i]) > TinyAD::EPS)
        {
            S_g_total += g_i_sq / (abs_lambda_i );
        }
        if (abs_lambda_i > _eigenvalue_eps)
        {
            S_lambda_total += abs_lambda_i * abs_lambda_i; // |λ_i|²
        }
        
    }

    double Kappa = computeAlpha_grad_kappa(eigenvalues, alpha_J); 
    //dobule gamma = g_para_gamma;
    
    double alpha_grad_pos_min = 0;
    double alpha_grad_neg_min = 0;
    double energy_e = _f;
    double eta_pos = 1.0;
    double eta_neg = 1.0;
    double pos_lambda_ratio = S_lambda_pos / (S_lambda_total + TinyAD::EPS);
    double neg_lambda_ratio = S_lambda_neg / (S_lambda_total + TinyAD::EPS);
    double pos_grad_ratio = S_g_pos / (S_g_total + TinyAD::EPS);
    double neg_grad_ratio = S_g_neg / (S_g_total + TinyAD::EPS);

    // energy_e = get_energy();
    switch(method)
    {
        case 1: // safe direction 1/3次方的方案 + unsafe direction, clamp到0.0的方案
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            alpha_grad_neg = 0;
            break;
        }
        case 2: // safe direction 1/3次方的方案  + unsafe direction with lower threshold(min)
        {
            // get energy
            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);

            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            // alpha_grad_pos = computeAlpha_grad_safe(S_g_pos,  S_lambda_pos , gamma, m_eps);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+ TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 3: // safe direction 1/3次方的方案 with lower threshold + unsafe direction with lower threshold
        {
            // get energy
            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            //lower threshold
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            #if DEBUG_OUTPUT
                if (alpha_grad_pos < alpha_grad_pos_min)
                {
                    TINYAD_WARNING("!!!safe vector alpha_pos("<<alpha_grad_pos <<")< alpha_min("<<alpha_grad_pos_min<<")"); 
                }
            #endif
            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 4: // 不区分safe&unsafe
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            alpha_grad_neg = alpha_grad_pos;
            break;
            
        }
        case 5: //same to 2, exact element energy, safe direction 1/3次方的方案  + unsafe direction with lower threshold(min)
        {
            // get energy
            // energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 6: // same to 3, exact element energy,safe direction 1/3次方的方案 with lower threshold + unsafe direction with lower threshold
        {
            // get energy
            // energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            //lower threshold
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            #if DEBUG_OUTPUT
                if (alpha_grad_pos < alpha_grad_pos_min)
                {
                    TINYAD_WARNING("!!!safe vector alpha_pos("<<alpha_grad_pos <<")< alpha_min("<<alpha_grad_pos_min<<")"); 
                }
            #endif

            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 7: // same to 2, avg energy, safe direction 1/3次方的方案  + unsafe direction with lower threshold(min)
        {
            // get energy
            energy_e = TinyAD::g_avg_vol_energy * std::abs(volume); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }
        case 8: // same to 3, avg_energy,safe direction 1/3次方的方案 with lower threshold + unsafe direction with lower threshold
        {
            // get energy
            energy_e = TinyAD::g_avg_vol_energy * std::abs(volume);

            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);


            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            //lower threshold
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            #if DEBUG_OUTPUT
                if (alpha_grad_pos < alpha_grad_pos_min)
                {
                    TINYAD_WARNING("!!!safe vector alpha_pos("<<alpha_grad_pos <<")< alpha_min("<<alpha_grad_pos_min<<")"); 
                }
            #endif

            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);

            // 4. 计算alpha_grad_neg
            //lower threshold
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::min(1.0, alpha_grad_neg_min);
            
            break;
        }   
        case 9: // same to 4, 不区分safe&unsafe,  lower threshold based on total energy
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            alpha_grad_neg = alpha_grad_pos;

            // get energy
            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            double alpha_grad_min = S_g_total_neg / (eta_pos * energy_e + TinyAD::EPS);

            alpha_grad_pos = std::min(std::max(alpha_grad_pos, alpha_grad_min), 1.0);
            alpha_grad_neg = alpha_grad_pos;

            break;
            
        }  
        case 10: // same to 9,different element energy
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            alpha_grad_neg = alpha_grad_pos;

            // get energy
            energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            double alpha_grad_min = S_g_total_neg / (0.5 * energy_e + TinyAD::EPS);

            alpha_grad_pos = std::min(1.0, alpha_grad_min);
            alpha_grad_neg = alpha_grad_pos;
            break;
            
        } 
        case 11: // no safe&unsafe,  alpha_min, energy = max E
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 12: // no safe&unsafe,  alpha_min, energy = element_e
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        } 
        case 13: // no safe&unsafe,  alpha_min,energy-avg
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            energy_e = S_g_total; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 14: // no safe&unsafe,  alpha_min, energy = max E
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 15: // no safe&unsafe,  alpha_min,energy-avg
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            // double ratio_pos = S_g_total_neg / (2.0 * gamma * S_lambda_total_neg + m_eps);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            // alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);

            energy_e = g_avg_energy; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = alpha_grad_min;
            alpha_grad_neg = alpha_grad_min;
            break;
            
        } 
        case 16: // no safe&unsafe,  [alpha_min, 1], energy = element_e
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio = ( gamma * S_lambda_total_neg) / (2.0 * S_g_total_neg + TinyAD::EPS);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(S_g_total, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_min); // [alpha_min, 1]
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 17: // no safe&unsafe,  [alpha_min, 1], energy = element_e
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
            double S_lambda_total_neg = S_lambda_pos + S_lambda_neg; //与上面的total不同，不包含正特征值
            double ratio = ( gamma * S_lambda_total_neg) / (2.0 * S_g_total_neg + TinyAD::EPS);
            // // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio, 1.0 / 3.0);
            // // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            // alpha_grad_neg = alpha_grad_pos;

            // get energy
            double P = S_g_total_neg;
            computeAlpha_grad_eta(eta_pos, eta_neg , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            // eta_pos = 1;
            double eta_energy = 2 * eta_pos * energy_e;
            double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + TinyAD::EPS);

            alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_min); // [alpha_min, 1]
            alpha_grad_neg = alpha_grad_min;
            break;
            
        }
        case 21: // safe direction 1/3次方的方案 + unsafe direction 1/2次方的方案
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            break;
        }
        case 22: // safe direction 1/3次方的方案 + unsafe direction 1/2次方的方案(min)
        {
            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = S_g_total;

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));

            
            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));

            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::max(alpha_grad_neg, alpha_grad_neg_min);

            break;
        }
        case 23: // safe direction 1/3次方的方案(min) + unsafe direction 1/2次方的方案(min)
        {
            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = S_g_total;

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            alpha_grad_pos = std::max(alpha_grad_pos, alpha_grad_pos_min);


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::max(alpha_grad_neg, alpha_grad_neg_min);

            break;
        }
        case 24: // safe direction 1/3次方的方案(min) + unsafe direction 1/2次方的方案(max)
        {
            computeAlpha_grad_eta(eta_pos, eta_neg, Kappa,
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = S_g_total;

            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));
            double eta_energy = eta_pos * energy_e;
            alpha_grad_pos_min = (S_g_pos + std::sqrt(S_g_pos * (S_g_pos + 2 * eta_energy))) / (2 * eta_energy+TinyAD::EPS);

            alpha_grad_pos = std::min(alpha_grad_pos, alpha_grad_pos_min);


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            alpha_grad_neg_min = S_g_neg / (eta_neg * energy_e+TinyAD::EPS);

            alpha_grad_neg = std::max(alpha_grad_neg, alpha_grad_neg_min);

            break;
        }
        case 31: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            // if (alpha_min > 1.0)
            // {
            //     TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            // }
            #endif
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 32: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 33: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = g_avg_energy; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 34: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = g_avg_energy; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif
            double kappa_thresh = 0.1;
            if (Kappa < kappa_thresh)
            {
                #if DEBUG_OUTPUT
                    TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
                #endif 
                alpha_min = 0;
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 35: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = (g_avg_energy +energy_e)/2.0;; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            // double kappa_thresh = 0.1;
            // if (Kappa < kappa_thresh)
            // {
            //     TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
            //     alpha_min = 0;
            // }
            #endif
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 36: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = std::min(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            // double kappa_thresh = 0.1;
            // if (Kappa < kappa_thresh)
            // {
            //     TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
            //     alpha_min = 0;
            // }
            #endif
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 37: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = (g_avg_energy +energy_e)/2.0;; // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif
            double kappa_thresh = 0.1;
            if (Kappa < kappa_thresh)
            {
                #if DEBUG_OUTPUT
                TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
                #endif
                alpha_min = 0;
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 38: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            energy_e = std::min(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif
            double kappa_thresh = 0.1;
            if (Kappa < kappa_thresh)
            {
                #if DEBUG_OUTPUT
                TINYAD_WARNING("Kappa:("<<Kappa <<")<"<<kappa_thresh);
                #endif
                alpha_min = 0;
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 39: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif
            if ((alpha_min < 1.0)
            &&    (neg_grad_ratio + pos_grad_ratio < TinyAD::EPS )
            && (neg_lambda_ratio + pos_lambda_ratio > 0.5 ))//lamda比值和grad比值不一致,说明特征值方向和梯度方向正交
            {
                alpha_min = 1.0;
                #if DEBUG_OUTPUT
                TINYAD_WARNING("grad_ratio: "<<(neg_grad_ratio + pos_grad_ratio) <<",lambda_ratio:"<<neg_lambda_ratio + pos_lambda_ratio); 
                #endif
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        case 40: // no safe&unsafe,  alpha_min, energy = local E
        {
            // 3. 计算alpha_grad_pos
            
            double S_g_total_neg = S_g_pos + S_g_neg; //与上面的total不同，不包含正特征值
           
            // get energy
            double P = S_g_total_neg;
            double eta = 0.0;
            computeAlpha_grad_eta(eta, eta , Kappa, 
                pos_lambda_ratio, neg_lambda_ratio, pos_grad_ratio, neg_grad_ratio);
            // energy_e = std::max(g_avg_energy, energy_e); // tbd zj: 也可以考虑加权，或者只考虑S_g_pos
            
            double alpha_min = computeAlpha_grad_min(P, eta, energy_e);
            // double eta_energy = 2 * eta_pos * energy_e;
            // double alpha_grad_min = (P + std::sqrt(P * (P + eta_energy))) / (eta_energy + m_eps);

            // alpha_grad_min = std::min(1.0, std::max(0.0, alpha_grad_min));
            
            #if DEBUG_OUTPUT
            if (alpha_min > 1.0)
            {
                // TINYAD_WARNING("P: "<<P <<",energy_e:"<<energy_e<<",eta:"<<eta<<",alpha_min:"<<alpha_min<<")"); 
            }
            #endif

            //非凸性强, alpha不能太小
            if ((alpha_min) < TinyAD::EPS 
            && (Kappa > 0.7) 
            && (1.0 - std::abs(alpha_J))<  TinyAD::EPS)//lamda比值和grad比值不一致,说明特征值方向和梯度方向正交
            {
                alpha_min = 1.0;
                #if DEBUG_OUTPUT
                TINYAD_WARNING("grad_ratio: "<<(neg_grad_ratio + pos_grad_ratio) <<",lambda_ratio:"<<neg_lambda_ratio + pos_lambda_ratio); 
                TINYAD_WARNING("alpha_min: "<<alpha_min <<",alpha_J:"<<alpha_J); 
                #endif 
            }
            alpha_grad_pos = alpha_min;
            alpha_grad_neg = alpha_min;
            break;
            
        }
        default: // safe direction 1/3次方的方案 + unsafe direction 1/2次方的方案
        {
            // 3. 计算alpha_grad_pos
            // double gamma = g_para_gamma ;  // 超参数 γ，需标定,或者自适应 global tbd zj
            double ratio_pos = S_g_pos / (2.0 * gamma * S_lambda_pos + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_pos = std::pow(ratio_pos, 1.0 / 3.0);
            // alpha_grad_pos = std::max(0.0, std::min(1.0, alpha_grad_pos));


            // 4. 计算alpha_grad_neg
            double ratio_neg = S_g_neg / (4.0 * gamma * S_lambda_neg + TinyAD::EPS);
            // 基于解析公式的混合策略：α = (S_g / (2γS_λ))^(1/3)，其中γ是一个超参数，需标定。
            // 这个公式来源于变分问题的解析解，能够根据S_g和S_λ的关系自动调整混合系数，实现更智能的特征值增强。
            // S_g越大（负特征值贡献越大,能量下降越大），S_λ越小（负特征值越小，修正代价较小），则α越大，增强效果越强。
            alpha_grad_neg = std::pow(ratio_neg, 1.0 / 2.0);
            // alpha_grad_neg = std::max(0.0, std::min(1.0, alpha_grad_neg));
            break;
        }
    }
    

    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("proj_g:"<<proj_g.transpose() <<", gamma:"<<gamma); 
        TINYAD_DEBUG_OUT("S_g_pos(bigger):"<<S_g_pos<<", S_lambda_pos(smaller):"<<S_lambda_pos);
        TINYAD_DEBUG_OUT("S_g_neg(bigger):"<<S_g_neg<<", S_lambda_neg(smaller):"<<S_lambda_neg); 
        TINYAD_DEBUG_OUT("energy_e:"<<energy_e<<", eta_pos:"<<eta_pos<<", eta_neg:"<<eta_neg);
        TINYAD_DEBUG_OUT("alpha_grad_pos_min:"<<alpha_grad_pos_min<<", alpha_grad_neg_min:"<<alpha_grad_neg_min);
        TINYAD_DEBUG_OUT("alpha_grad_pos:"<<alpha_grad_pos<<", alpha_grad_neg:"<<alpha_grad_neg); 
    #endif

    alpha_grad_pos = std::min(1.0, std::max(0.0, alpha_grad_pos));
    alpha_grad_neg = std::min(1.0, std::max(0.0, alpha_grad_neg));

    // return alpha;
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
    double old_lambda, double kappa, double _eigenvalue_eps)
{
    
    // m_eps = TinyAD::EPS_1E_8;

    double lambda = (1-kappa) * old_lambda + kappa * std::abs(old_lambda); // λ' = (1-κ)λ + κ|λ|

    if (std::abs(lambda) < _eigenvalue_eps) { 
        // 小特征值
        lambda = _eigenvalue_eps; 
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
        const PassiveT& _eigenvalue_eps) //for vpn
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


void computeAlpha_Q( 
    const int k,
    const Eigen::VectorXd& proj_g, 
    const Eigen::VectorXd& eigenvalues,
    double& w1, double& w2
    // const double EPS = 1e-12
    )
{
    // double EPS =TinyAD::EPS;
    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("computeAlpha_Q: "<<"proj_g:"<<proj_g.transpose() <<", eigenvalues:"<<eigenvalues.transpose()); 
        // TINYAD_DEBUG_OUT("EPS: "<<TinyAD::EPS); 
    #endif
    double q1=0,q2=0;

    // ---------- q1,q2 ----------
    for (size_t i = 0; i < k; ++i) {
        double g_i_sq = proj_g[i]*proj_g[i];
        double abs_lambda_i = std::abs(eigenvalues[i]);  // |λ_i|

        if (eigenvalues[i] > 0) {
            // double g_i_sq = proj_g[i]*proj_g[i];                // g_i² (来自 computeGradientProjection)
            q1 += g_i_sq / (abs_lambda_i + TinyAD::EPS);    
        }
        else
        {
            q2 += g_i_sq / (abs_lambda_i + TinyAD::EPS);    
        }

    }
    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("q1: "<<"q1:"<<q1<<", q2:"<<q2); 
    #endif
    
    // ---------- 常数定义 (论文经验值) ----------
    const double W1_FIXED = 0.8;   // 论文图8 验证的最优值
    const double K_LOWER  = 1.5;   // 论文图9 验证的稳定下界
    // const double EPS      = 1e-12; // 防止除零 check

    // ---------- 分支判断 (Algorithm 1 Line 1) ----------
    if (q2 >= q1) {
        // 情况 1: 负特征值影响不大，直接退化为绝对值投影 (Abs)
        w1 = 1.0;
        w2 = 1.0;
        return;
    }

    // ---------- 情况 2: q2 < q1，需要混合 (Algorithm 1 Line 2-9) ----------
    // 步骤 2a: 固定 w1 = 0.8 (Line 2)
    w1 = W1_FIXED;

    // 步骤 2b: 计算基础比率 ratio = sqrt(q2 / q1) (Line 3)
    double ratio = std::sqrt(q2 / std::max(q1, TinyAD::EPS));

    // 步骤 2c: 构造关于 k 的二次方程 d(w1, k) = A*k^2 + B*k + C = 0
    // 依据论文公式 (13): 
    //   A = (3/2) * q2
    //   B = -sqrt(q1*q2) / w1
    //   C = (1/(2*w1^2) - 1/w1 + 1/2)*q1 - q1/(2*w1^2) = (1/2 - 1/w1) * q1
    double A = (q1/32.0) + 1.5 * q2;
    double B = - std::sqrt(q1 * q2) / w1;  // 注意: B <= 0
    double C = - q1 /(2* w1* w1) ;     // 因为 w1=0.8，C 恒为负

    // 判别式 Delta = B^2 - 4*A*C
    // 由于 A > 0, C < 0, Delta 恒大于 0，无需担心负数开根
    double delta = B * B - 4.0 * A * C;
    if (delta < 0) delta = 0.0; // 防御性编程

    // 取正根 (负根无物理意义，因为 k>1)
    double k_root = (-B + std::sqrt(delta)) / (2.0 * A);

    // 步骤 2d: 计算 k 的候选值 (Line 5-6)
    //   k1 = (1 + k_root) / 2      (取自二次曲线中点的启发式)
    //   k2 = 1 / (ratio * w1)      (约束 w2 <= 1 的上界)
    double k1 = (1.0 + k_root) / 2.0;
    double k2 = 1.0 / (ratio * w1 + TinyAD::EPS);

    // 步骤 2e: 取 min 并施加 k_lower 下界 (Line 7-8)
    double k3 = std::min(std::min(k1, k2),K_LOWER);
    // para_k = std::max(para_k, K_LOWER);

    // 步骤 2f: 计算 w2 (Line 9)
    w2 = k3 * ratio * w1;

    // 步骤 2g: 截断 w2 到 [0, 1] 区间 (论文中隐式要求)
    w2 = std::clamp(w2, 0.0, 1.0);

    // 注: 在实际实现中, 若 q1 或 q2 极小, 可选择性输出调试信息
    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("w1: "<<"w1:"<<w1<<", w2:"<<w2); 
    #endif
}

/**
 * Project symmetric matrix to positive-definite matrix
 * via eigen decomposition. 
 * added by zj (Differentiable version)
 */
template <int k, typename PassiveT>
void project_positive_definite_diff(
        Eigen::Matrix<PassiveT, k, k>& _H,
        const Eigen::VectorX<PassiveT>& _grad,        // ← 新增
        const PassiveT& _eigenvalue_eps,
        HessianProjectionMode _mode = HessianProjectionMode::AUTO,
        const PassiveT& _J = PassiveT(1.0)      ) // for vpn      
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

        // 创建正则化器
        EigenvalueRegularizer<PassiveT> regularizer(_eigenvalue_eps,_mode);

        bool modified = false;
        
        if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_Q) 
        {
            // m_eps = 1e-10;
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 

           
            // 2. 计算梯度投影
            Eigen::VectorXd proj_g = computeGradientProjection_vpn(eigenvectors, _grad);

            // 3. w1,w2
            // int computeAlphaMethod = 1; // 1: 基于梯度投影；2:基于特征值分布；3:基于负特征值能量占比；4:基于多重判定 
            
            double alpha_pos = 0.0, alpha_neg = 0.0;
            computeAlpha_Q(k, proj_g, eigenvalues, alpha_pos, alpha_neg); //element level
            
            // 3. 优化每个特征值
            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                double new_lambda = lambda;
                if (lambda <= _eigenvalue_eps) {
                    modified = true;
                }
                if (abs(lambda) <= _eigenvalue_eps) {
                    new_lambda = _eigenvalue_eps;
                }
                else{
                    new_lambda = (lambda > 0) ? alpha_pos * lambda : alpha_neg * (-lambda); //与梯度同向和异向的特征值使用不同的梯度增强系数
                }
                // double new_lambda = (lambda > 0) ? lambda*alpha_pos : lambda * alpha_neg; //与梯度同向和异向的特征值使用不同的梯度增强系数

                eigenvalues(i) = new_lambda;    
            }

            // modified = true;
            
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_J)
        {

            m_eps = 1e-10;
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > m_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > m_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 

           
            // 2. 计算梯度投影
            Eigen::VectorXd proj_g = computeGradientProjection_vpn(eigenvectors, _grad);

            // 3. alpha
            // int computeAlphaMethod = 1; // 1: 基于梯度投影；2:基于特征值分布；3:基于负特征值能量占比；4:基于多重判定 
            double alpha_J = computeAlpha_J(_J);    //element level
            double gamma = g_para_gamma;
            updateGamma(gamma, _J);
            double alpha_grad_pos = 0.0, alpha_grad_neg = 0.0;
            computeAlpha_grad(alpha_grad_pos, alpha_grad_neg,k, proj_g, eigenvalues, gamma, alpha_J, _J); //element level
            double alpha_hessian = computeAlpha_hessian(eigenvalues); //element level
            
            
            // 3. 优化每个特征值
            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                double new_lambda = lambda;
                double alpha_grad = (proj_g[i] > 0) ? alpha_grad_pos : alpha_grad_neg; //与梯度同向和异向的特征值使用不同的梯度增强系数

                new_lambda = optimizeVpnEigenvalue(eigenvalues[i], alpha_grad, alpha_hessian, alpha_J, _eigenvalue_eps);

                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" new_lambda"<<new_lambda<<", lambda"<<lambda<<",_eigenvalue_eps"<<_eigenvalue_eps); 
                    TINYAD_DEBUG_OUT(" alpha_grad"<<alpha_grad<<", alpha_hessian"<<alpha_hessian<<",alpha_J"<<alpha_J); 
                #endif
                
                eigenvalues(i) = new_lambda;    
            }

            modified = true;
            
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::SMOOTH_TR) 
        {
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            int method = 3;

            switch (method) {
                case 1:
                {
                    int kappaMethod = 2; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
                    // 2. 计算几何曲率 κ（基于特征值分布）
                    double kappa = computeKappa(kappaMethod, eigenvalues);
                    // 3. 优化每个特征值
                    for (size_t i = 0; i < k; ++i) {
                        eigenvalues(i) = optimizeEigenvalueByKappa(eigenvalues(i), kappa, _eigenvalue_eps);
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
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT_BLEND_SHEAR
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 4; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
            double kappa = computeKappa(kappaMethod, eigenvalues);
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
            double kappa = computeKappa(kappaMethod, eigenvalues);
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
            for (Eigen::Index i = 0; i < k; ++i) {
                PassiveT old_val = eigenvalues(i);
                
                if (regularizer.needs_regularization(old_val)) {

                    eigenvalues(i) = regularizer.regularize(old_val);
                    modified = true;

                }
            }
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }

        if (modified) {
            _H = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
            g_reg_element_num++;
        }
   
    }
}


/**
 * Project symmetric matrix to positive-definite matrix
 * via eigen decomposition. 
 * added by zj (Differentiable version)
 */
template <int k, typename PassiveT>
void project_positive_definite_diff_energy(
        Eigen::Matrix<PassiveT, k, k>& _H,
        const Eigen::VectorX<PassiveT>& _grad,        // ← 新增
        const PassiveT& _eigenvalue_eps, //小特征值
        HessianProjectionMode _mode = HessianProjectionMode::AUTO,
        const PassiveT& _J = PassiveT(1.0) ,const PassiveT _f= PassiveT(0.0)    ) // for vpn      
{
    #if DEBUG_OUTPUT
        TINYAD_DEBUG_OUT("Projection mode in project_positive_definite_diff_energy: " << static_cast<int>(_mode)); 
    #endif
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
        
        double m_eps = TinyAD::EPS_1E_8;  // zero
        // double EPS = TinyAD::EPS_1E_8;
        bool use_shift = false; // 是否阻尼牛顿法

        #if DEBUG_OUTPUT
            double lambda_max = eigenvalues.maxCoeff();
            double lambda_min = eigenvalues.minCoeff();
            
            TINYAD_DEBUG_OUT("lambda_max,lambda_min:"<<lambda_max<<","<<lambda_min);   
        #endif

        // 创建正则化器
        EigenvalueRegularizer<PassiveT> regularizer(_eigenvalue_eps,_mode);

        bool modified = false;
        
        if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_Q) 
        {
            // m_eps = 1e-10;
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 
            modified = true;
           
            // 2. 计算梯度投影
            Eigen::VectorXd proj_g = computeGradientProjection_vpn(eigenvectors, _grad);

            // 3. w1,w2
            // int computeAlphaMethod = 1; // 1: 基于梯度投影；2:基于特征值分布；3:基于负特征值能量占比；4:基于多重判定 
            
            double alpha_pos = 0.0, alpha_neg = 0.0;
            computeAlpha_Q(k, proj_g, eigenvalues, alpha_pos, alpha_neg); //element level
            
            // 3. 优化每个特征值
            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                // if (lambda <= _eigenvalue_eps) {
                //     modified = true;
                // }

                double new_lambda = lambda;
                double abs_lambda = std::abs(lambda);
                if (abs_lambda <= _eigenvalue_eps) {
                    new_lambda = _eigenvalue_eps;
                }
                else{
                    new_lambda = (lambda > 0) ? alpha_pos * abs_lambda : alpha_neg * abs_lambda; //与梯度同向和异向的特征值使用不同的梯度增强系数
                }
                
                eigenvalues(i) = new_lambda;   
                
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" new_lambda "<<new_lambda<<", lambda "<<lambda<<",_eigenvalue_eps "<<_eigenvalue_eps); 
                    TINYAD_DEBUG_OUT(" alpha_pos "<<alpha_pos<<", alpha_neg "<<alpha_neg ); 
                #endif
            }
            
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_J_Energy)
        {

            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 

            // 2. 计算梯度投影
            Eigen::VectorXd proj_g = computeGradientProjection_vpn(eigenvectors, _grad);
            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT(" proj_g "<<proj_g.transpose()); 
            #endif
            // 3. alpha
            double alpha_p = 1.0, alpha_n = 1.0;
            // bool tbModify = computeAlpha_energy_thresh(k,
            //     proj_g, eigenvalues, _f, alpha_p, alpha_n);
            // computeAlpha_energy_thresh(k,proj_g, eigenvalues, _f, alpha_p, alpha_n);
            computeAlpha_energy_thresh5(k,proj_g, eigenvalues, _f, alpha_p, alpha_n, _eigenvalue_eps);
            #if DEBUG_OUTPUT
                TINYAD_DEBUG_OUT(" alpha_p: "<<alpha_p<<", alpha_n: "<<alpha_n); 
            #endif 

            // 3. 优化每个特征值
            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                double abs_lambda = std::abs(lambda);
                double new_lambda = lambda;
                if (abs_lambda <= _eigenvalue_eps) {
                    new_lambda = _eigenvalue_eps;
                }
                else if (lambda < 0) 
                {
                    new_lambda = abs_lambda * alpha_n;
                }
                else{
                    new_lambda = abs_lambda * alpha_p;
                }

                eigenvalues(i) = new_lambda; 
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" new_lambda "<<new_lambda<<", lambda "<<lambda<<",_eigenvalue_eps "<<_eigenvalue_eps); 
                #endif  

            }
            modified = true;
            
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_BLENDING_J)
        {

            // m_eps = 1e-10;
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 

           
            // 2. 计算梯度投影
            Eigen::VectorXd proj_g = computeGradientProjection_vpn(eigenvectors, _grad);

            // 3. alpha
            // int computeAlphaMethod = 1; // 1: 基于梯度投影；2:基于特征值分布；3:基于负特征值能量占比；4:基于多重判定 
            double alpha_J = computeAlpha_J(_J);    //element level
            double gamma = g_para_gamma;
            updateGamma(gamma, _J);
            double alpha_grad_pos = 0.0, alpha_grad_neg = 0.0;
            computeAlpha_grad(alpha_grad_pos, alpha_grad_neg,k, proj_g, eigenvalues, gamma, alpha_J, _f); //element level
            
            // computeAlpha_grad2(alpha_grad_pos, alpha_grad_neg,k, proj_g, eigenvalues, gamma, alpha_J, _f, _eigenvalue_eps); //element level
            
            
            double alpha_hessian = computeAlpha_hessian(eigenvalues); //element level
            
            
            // 3. 优化每个特征值
            for (size_t i = 0; i < k; ++i) {
                double lambda = eigenvalues(i);
                double new_lambda = lambda;
                double alpha_grad = (proj_g[i] > 0) ? alpha_grad_pos : alpha_grad_neg; //与梯度同向和异向的特征值使用不同的梯度增强系数

                new_lambda = optimizeVpnEigenvalue(eigenvalues[i], alpha_grad, alpha_hessian, alpha_J, _eigenvalue_eps);
                
                eigenvalues(i) = new_lambda; 
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" new_lambda "<<new_lambda<<", lambda "<<lambda<<",_eigenvalue_eps "<<_eigenvalue_eps); 
                    TINYAD_DEBUG_OUT(" alpha_grad "<<alpha_grad<<", alpha_hessian "<<alpha_hessian<<",alpha_J "<<alpha_J); 
                #endif   
            }

            modified = true;
            
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::SMOOTH_TR) 
        {
            // 1. 检查是否需要滤波
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            int method = 3;

            switch (method) {
                case 1:
                {
                    int kappaMethod = 2; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
                    // 2. 计算几何曲率 κ（基于特征值分布）
                    double kappa = computeKappa(kappaMethod, eigenvalues);
                    // 3. 优化每个特征值
                    for (size_t i = 0; i < k; ++i) {
                        eigenvalues(i) = optimizeEigenvalueByKappa(eigenvalues(i), kappa, _eigenvalue_eps);
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
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT_BLEND_SHEAR
                    TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            }

            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 4; // 1: 基于特征值分布；2:基于负特征值能量占比；3:基于多重判定
            double kappa = computeKappa(kappaMethod, eigenvalues);
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
                        if (correctLambda(new_lambda, lambda, _eigenvalue_eps, kappa))
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
                            eigenvalues(i), a2(i), wi, beta, fast_mode,_eigenvalue_eps);
                        
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
                        if (correctLambdaBlend(new_lambda, lambda, _eigenvalue_eps, beta))
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
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    // TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
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
            if (eigenvalues.minCoeff() > _eigenvalue_eps) {
                #if DEBUG_OUTPUT
                    // TINYAD_DEBUG_OUT(" eigenvalues.minCoeff() > _eigenvalue_eps return"); 
                #endif
                return;  // 已正定，直接返回
            } 
            
            // 3. 计算几何曲率 κ（基于变形梯度）
            //double kappa;
            int kappaMethod = 2;
            double kappa = computeKappa(kappaMethod, eigenvalues);
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
                        if (correctLambda(new_lambda, lambda, _eigenvalue_eps, kappa))
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
        else if (_mode == HessianProjectionMode::CLAMP_NONDIFF)
        { 
            for (Eigen::Index i = 0; i < k; ++i) {
                PassiveT old_val = eigenvalues(i);
                
                if (std::abs(old_val) < _eigenvalue_eps) {

                    eigenvalues(i) = _eigenvalue_eps;
                    modified = true;

                }
            }
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::ABS_NONDIFF)
        { 
            for (Eigen::Index i = 0; i < k; ++i) {
                PassiveT old_val = eigenvalues(i);
                PassiveT abs_val = std::abs(old_val) ;
                
                if (old_val < 0) {

                    eigenvalues(i) = abs_val;
                    modified = true;
                }
            }
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }
        else if (_mode == HessianProjectionMode::CLAMP_ABS_NONDIFF)
        { 
            if (_eigenvalue_eps < 0) // -1,abs, _eigenvalue_eps变成了标志
            {
                for (Eigen::Index i = 0; i < k; ++i) {
                    PassiveT old_val = eigenvalues(i);
                    PassiveT abs_val = std::abs(old_val) ;
                    
                    if (old_val < 0) {

                        eigenvalues(i) = abs_val;
                        modified = true;
                    }
                }
            }
            else{ // _eigenvalue_eps = 0
                for (Eigen::Index i = 0; i < k; ++i) {
                    PassiveT old_val = eigenvalues(i);
                    
                    if (old_val < 0) {
                        eigenvalues(i) = 0;
                        modified = true;
                    }
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

                }
            }
            if (!modified) {
                return; // 如果没有任何特征值需要修改，直接返回
            }
        }

        if (modified) {
            _H = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
            g_reg_element_num++;
        }
   
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
        const Eigen::VectorX<PassiveT>& _grad,        // ← 新增
        const PassiveT& _eigenvalue_eps,
        HessianProjectionMode _mode = HessianProjectionMode::AUTO,
        const PassiveT& _J  =  PassiveT(1.0)      ) // for vpn
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
            g_reg_element_num++;
        }

        //#endif
       
    }
}



}




