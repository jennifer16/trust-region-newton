#pragma once

#ifndef CUBIC_REGULAR_H
#define CUBIC_REGULAR_H

#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Sparse>
#include "mesh_ops.h"

//越大越稳定
enum class SIGMA_STRATEGIES {
    FIXED = 0,      
    GRADIENT = 1,      
    HESSIAN = 2,    
    ENERGY = 3,
    DEFORM = 4,
    ADAPTIVE = 5,
};


class Cubic_Regular {
public:

    Cubic_Regular(SIGMA_STRATEGIES strategy = SIGMA_STRATEGIES::FIXED, double sigma = 1.0, double rho = 1.0, double upper_threshhold = 0.75, 
        double lower_threshhold = 0.25, double step = 2, int level = 0, double lambda = 1e-4)
    {
        this->cubic_regular_strategy = strategy;
        
        this->rho = rho;
        this->sigma = sigma;
        
        this->cubic_regular_ratio_upper_threshhold = upper_threshhold;
        this->cubic_regular_ratio_lower_threshhold = lower_threshhold;
        this->sigma_step = step;
        this->level = level; //保守
        this->lambda = lambda;
    }


    double compute_ratio_rho(double e1, double e0, 
                                  const Eigen::VectorXd &d, 
                                  const Eigen::VectorXd &g,
                                  const Eigen::SparseMatrix<double> &H, 
                                  const double sigma = 0){
        assert(d.size() == g.size());
        assert(d.size() == H.rows());
        assert(d.size() == H.cols());
        double d_norm = d.norm();
        this->rho = (e0 - e1) / (0.0 - (d.dot(g) + 0.5 * d.transpose() * H * d + std::pow(d_norm, 3) * sigma / 3.0));
        return this->rho;
    }
                                
    

    double update_sigma(const double para1 = 0, const double para2 = 0)
    {
        switch (cubic_regular_strategy)
        {
            case SIGMA_STRATEGIES::FIXED:
                return sigma_fixed();
            case SIGMA_STRATEGIES::ADAPTIVE:
                return sigma_adaptive();
            case SIGMA_STRATEGIES::DEFORM:
                return sigma_deform(para1);
            case SIGMA_STRATEGIES::GRADIENT:
                return sigma_gradient(para1);
            case SIGMA_STRATEGIES::ENERGY:
                return sigma_energy(para1, para2);
            case SIGMA_STRATEGIES::HESSIAN:
                return sigma_hessian(para1);
            default:
                return sigma_adaptive();
        }
    }

    double update_lambda(const Eigen::VectorXd &d)
    {
        double norm_d = d.norm();
        this->lambda = this->sigma * norm_d;
        return this->lambda;
    }

    double get_lambda()
    {
        return this->lambda;
    }

    double get_rho()
    {
        return this->rho;
    }
    void set_rho(double rho)
    {
        this->rho = rho;
    }

    double get_sigma()
    {
        return this->sigma;
    }
    void set_sigma(double sigma)
    {
        this->sigma = sigma;
    }

    double get_sigma_step()
    {
        return this->sigma_step;
    }
    void set_sigma_step(double step)
    {
        this->sigma_step = step;
    }


    int get_level()
    {
        return this->level;
    }
    void set_level(int level)
    {
        this->level = level;
    }

    SIGMA_STRATEGIES get_strategy(void)
    {
        return this->cubic_regular_strategy;
    }
    void set_strategy(SIGMA_STRATEGIES strategy)
    {
        this->cubic_regular_strategy = strategy;
    }

    double get_rho_upper_threshhold()
    {
        return this->cubic_regular_ratio_upper_threshhold;
    }
    void set_rho_upper_threshhold(double threshhold)
    {
        this->cubic_regular_ratio_upper_threshhold = threshhold;
    }

    double get_rho_lower_threshhold()
    {
        return this->cubic_regular_ratio_lower_threshhold;
    }
    void set_rho_lower_threshhold(double threshhold)
    {
        this->cubic_regular_ratio_lower_threshhold = threshhold;
    }

private: 
        
    double sigma_adaptive(const double old_sigma)  
    {   
        if (this->rho > cubic_regular_ratio_upper_threshhold)
        {
            return (old_sigma / sigma_step);
        }
        else if (this->rho  < cubic_regular_ratio_lower_threshhold){
            return (old_sigma * sigma_step);
        }
        return old_sigma;
    }

    double sigma_adaptive()  
    {   
        if (this->rho > cubic_regular_ratio_upper_threshhold)
        {
            this->sigma /= this->sigma_step;
            //return (this->sigma / sigma_step);
        }
        else if (this->rho  < cubic_regular_ratio_lower_threshhold){
            this->sigma *= this->sigma_step;
            //return (this->sigma * sigma_step);
        }
        return this->sigma;
    }

    double sigma_fixed()
    {
        return sigma_multiLevelStart();
    }

    // 基于物理量纲的启发式（适用于FSFEM）
    double sigma_gradient(double typical_gradient) {
        // // 估计典型的能量量级
        // double typical_energy = estimateTypicalEnergy(mesh);
        // double typical_gradient = typical_energy / mesh.characteristic_length;
        
        // sigma应该与梯度量级相关
        return 0.1 * typical_gradient;
    }

    double sigma_hessian(double gradient_norm) {
        // 问题规模越大，初始sigma可以稍大
        return 1.0 * gradient_norm;
    }

    // max deform ratio
    double sigma_deform(double deform_ratio)
    {
        if (deform_ratio> 0.5) //大变形
        {
            return 100;
        }else if (deform_ratio > 0.3) //中等变形
        {
            return 10;
        }
        else{
            return 1.0;
        }
        
    }

        


     // 基于问题规模的启发式
    double sigma_problem_size(double n_variables) {
        // 问题规模越大，初始sigma可以稍大
        return 1.0 * std::sqrt(static_cast<double>(n_variables)) / 100.0;
    }

     // 基于问题规模的启发式
    double sigma_energy(double energy, double size) {
        // 问题规模越大，初始sigma可以稍大
        assert(size != 0);
        return energy / size;
    }
    
    // 多级启动策略
    double sigma_multiLevelStart() {
        static const std::vector<double> sigma_levels = {
            10.0,   // 默认
            1.0,    // level 1: 标准启动
            0.1,    // level 2: 激进启动（快速但可能不稳定）
            10.0,   // level 3: 保守启动（慢但稳定）
            0.01,   // level 4: 非常激进（适用于良态问题）
            100.0   // level 5: 非常保守（适用于病态问题）
        };
        
        return sigma_levels[std::min(this->level, (int)sigma_levels.size() - 1)];
    }

    SIGMA_STRATEGIES cubic_regular_strategy; 
    double cubic_regular_ratio_upper_threshhold = 0.75;
    double cubic_regular_ratio_lower_threshhold = 0.25;
    double sigma_step = 1; // >= 1
    double rho;
    int level = 0;
    double sigma = 1.0;
    double lambda = 0.0;

    
};

// class AdaptiveSolverSelector {
// public:
//     struct SolverConfig {
//         std::string solver_type;
//         double sigma_init;           // 三次正则化参数
//         int max_iterations;
//         bool use_line_search;
//         double tolerance;
//     };
    
//     static SolverConfig selectSolver(const DeformationStatus& status) {
//         SolverConfig config;
        
//         if (status.is_small) {
//             // 小变形：标准牛顿法
//             config.solver_type = "Newton";
//             config.sigma_init = 0.0;      // 无正则化
//             config.max_iterations = 20;
//             config.use_line_search = false;
//             config.tolerance = 1e-8;
//         }
//         else if (status.max_strain < 0.2 && status.max_rotation < 20) {
//             // 中等变形：带线搜索的牛顿法
//             config.solver_type = "NewtonLS";
//             config.sigma_init = 0.0;
//             config.max_iterations = 30;
//             config.use_line_search = true;
//             config.tolerance = 1e-7;
//         }
//         else {
//             // 大变形：三次正则化牛顿法
//             config.solver_type = "CubicNewton";
            
//             // 根据变形程度选择初始sigma
//             if (status.max_strain > 0.5 || status.max_rotation > 45) {
//                 config.sigma_init = 100.0;   // 极强正则化
//             } else if (status.max_strain > 0.3) {
//                 config.sigma_init = 10.0;     // 强正则化
//             } else {
//                 config.sigma_init = 1.0;      // 中等正则化
//             }
            
//             config.max_iterations = 50;
//             config.use_line_search = true;
//             config.tolerance = 1e-6;
//         }
        
//         return config;
//     }
// };

// class FSFEMDeformationCheck {
// private:
//     const FSFEM_5NodeData& mesh_data;
    
// public:
//     // 在每个面上判断变形大小
//     struct FaceDeformationInfo {
//         int face_idx;
//         double avg_strain_norm;
//         double max_rotation;
//         double volume_change;
//         bool is_large_deformation;
//     };
    
//     std::vector<FaceDeformationInfo> checkAllFaces(
//         const Eigen::MatrixXd& V_cur,
//         const Eigen::MatrixXd& V_ref) {
        
//         std::vector<FaceDeformationInfo> results;
        
//         for (int f = 0; f < mesh_data.face_vertices.rows(); f++) {
//             FaceDeformationInfo info;
//             info.face_idx = f;
            
//             // 计算该面的平均形变梯度
//             Eigen::Matrix3d F_avg = computeAverageF(f, V_cur, V_ref);
            
//             // 1. 检查应变大小
//             Eigen::Matrix3d E = 0.5 * (F_avg.transpose() * F_avg - Eigen::Matrix3d::Identity());
//             info.avg_strain_norm = E.norm();
            
//             // 2. 检查旋转
//             info.max_rotation = extractRotationAngle(F_avg);
            
//             // 3. 检查体积变化
//             double J = F_avg.determinant();
//             info.volume_change = std::abs(J - 1.0);
            
//             // 4. 综合判断
//             info.is_large_deformation = 
//                 info.avg_strain_norm > 0.1 ||    // 应变 > 10%
//                 info.max_rotation > 10.0 ||       // 旋转 > 10度
//                 std::abs(J - 1.0) > 0.2;           // 体积变化 > 20%
            
//             results.push_back(info);
//         }
        
//         return results;
//     }
    
//     // 全局变形评估（用于选择求解策略）
//     struct GlobalDeformationStatus {
//         bool is_large_deformation;
//         double max_strain;
//         double max_rotation;
//         double max_volume_change;
//         int num_large_faces;
//         std::string recommended_solver;
//     };
    
//     GlobalDeformationStatus assessGlobalDeformation(
//         const Eigen::MatrixXd& V_cur,
//         const Eigen::MatrixXd& V_ref) {
        
//         auto face_results = checkAllFaces(V_cur, V_ref);
        
//         GlobalDeformationStatus status;
//         status.max_strain = 0;
//         status.max_rotation = 0;
//         status.max_volume_change = 0;
//         status.num_large_faces = 0;
        
//         for (const auto& face : face_results) {
//             status.max_strain = std::max(status.max_strain, face.avg_strain_norm);
//             status.max_rotation = std::max(status.max_rotation, face.max_rotation);
//             status.max_volume_change = std::max(status.max_volume_change, face.volume_change);
//             if (face.is_large_deformation) status.num_large_faces++;
//         }
        
//         // 判断全局变形状态
//         double large_face_ratio = static_cast<double>(status.num_large_faces) / 
//                                   face_results.size();
        
//         status.is_large_deformation = 
//             status.max_strain > 0.15 ||           // 最大应变 > 15%
//             status.max_rotation > 15.0 ||          // 最大旋转 > 15度
//             large_face_ratio > 0.3;                 // 超过30%的面是大变形
        
//         // 推荐求解器
//         if (status.is_large_deformation) {
//             status.recommended_solver = "三次正则化牛顿法";
//         } else if (status.max_strain > 0.05) {
//             status.recommended_solver = "修正牛顿法 (带线搜索)";
//         } else {
//             status.recommended_solver = "标准牛顿法";
//         }
        
//         return status;
//     }
    
// private:
//     Eigen::Matrix3d computeAverageF(int face_idx,
//                                      const Eigen::MatrixXd& V_cur,
//                                      const Eigen::MatrixXd& V_ref) {
//         // 使用边界积分计算平均F
//         // 实现细节...
//         return Eigen::Matrix3d::Identity();
//     }
    
//     double extractRotationAngle(const Eigen::Matrix3d& F) {
//         Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
//         Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
//         double trace = R.trace();
//         double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
//         return angle * 180.0 / M_PI;
//     }
// };

// struct DeformationStatus {
//         bool is_small;
//         double max_strain;
//         double max_rotation;
//         std::string recommendation;
//     };

// class DeformationCriterion {
// public:
    
    
//     // 基于位移梯度的范数
//     static DeformationStatus checkByGradientNorm(const Eigen::Matrix3d& grad_u) {
//         DeformationStatus status;
        
//         // 计算位移梯度的各种范数
//         double grad_norm = grad_u.norm();  // Frobenius范数
//         double max_component = grad_u.cwiseAbs().maxCoeff();
        
//         // 常见判据
//         status.max_strain = 0.5 * (grad_u + grad_u.transpose()).norm();
//         status.max_rotation = 0.5 * (grad_u - grad_u.transpose()).norm();
        
//         // 判断标准
//         if (grad_norm < 0.01) {  // 位移梯度 < 1%
//             status.is_small = true;
//             status.recommendation = "小变形，可用线性理论";
//         }
//         else if (grad_norm < 0.1) {  // 位移梯度 < 10%
//             status.is_small = true;
//             status.recommendation = "中等变形，需注意但线性近似可能可行";
//         }
//         else if (grad_norm < 0.3) {  // 位移梯度 < 30%
//             status.is_small = false;
//             status.recommendation = "大变形，必须用有限变形理论";
//         }
//         else {
//             status.is_small = false;
//             status.recommendation = "极大变形，需特殊处理";
//         }
        
//         return status;
//     }
    
//     // 基于格林应变
//     static DeformationStatus checkByGreenStrain(const Eigen::Matrix3d& F) {
//         DeformationStatus status;
        
//         Eigen::Matrix3d E = 0.5 * (F.transpose() * F - Eigen::Matrix3d::Identity());
//         double E_norm = E.norm();
        
//         // 与无穷小应变的比较
//         Eigen::Matrix3d grad_u = F - Eigen::Matrix3d::Identity();
//         Eigen::Matrix3d eps = 0.5 * (grad_u + grad_u.transpose());
//         double eps_norm = eps.norm();
        
//         double strain_ratio = E_norm / (eps_norm + 1e-12);
        
//         status.max_strain = E_norm;
        
//         if (strain_ratio < 1.1) {  // 相差不到10%
//             status.is_small = true;
//             status.recommendation = "格林应变 ≈ 无穷小应变，小变形";
//         }
//         else if (strain_ratio < 1.5) {
//             status.is_small = false;
//             status.recommendation = "中等非线性，考虑几何非线性";
//         }
//         else {
//             status.is_small = false;
//             status.recommendation = "强几何非线性，必须用有限变形";
//         }
        
//         return status;
//     }
// };

// class DeformationMetrics {
// public:
//     // 1. 格林应变张量 (Green-Lagrange Strain)
//     // E = 1/2 (F^T F - I)
//     static Eigen::Matrix3d greenStrain(const Eigen::Matrix3d& F) {
//         return 0.5 * (F.transpose() * F - Eigen::Matrix3d::Identity());
//     }
    
//     // 2. 无穷小应变张量 (小变形假设)
//     // ε = 1/2 (∇u + ∇u^T)
//     static Eigen::Matrix3d infinitesimalStrain(const Eigen::Matrix3d& grad_u) {
//         return 0.5 * (grad_u + grad_u.transpose());
//     }
    
//     // 3. 变形梯度张量
//     // F = I + ∇u
//     static Eigen::Matrix3d deformationGradient(const Eigen::Matrix3d& grad_u) {
//         return Eigen::Matrix3d::Identity() + grad_u;
//     }
// };


#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <iostream>
#include <vector>
#include <string>

class SymmetricIndefiniteAnalyzer {
private:
    const Eigen::SparseMatrix<double>& H;
    double tolerance;
    
public:
    struct AnalysisResult {
        // 分解状态
        bool decomposition_success;
        
        // 惯性指数
        int n_positive;  // 正特征值个数
        int n_negative;  // 负特征值个数
        int n_zero;      // 零特征值个数
        
        // 数值信息
        double min_pivot;
        double max_pivot;
        double min_abs_pivot;
        double condition_number_est;
        
        // 正定性判断
        bool is_positive_definite;
        bool is_negative_definite;
        bool is_indefinite;
        bool is_singular;
        
        // 建议
        double suggested_shift;
        std::string recommendation;
        std::vector<std::string> warnings;
    };
    
    SymmetricIndefiniteAnalyzer(const Eigen::SparseMatrix<double>& mat,
                                double tol = 1e-12)
        : H(mat), tolerance(tol) {}
    
    AnalysisResult analyze() {
        AnalysisResult result;
        
        // 1. 尝试 LDLT 分解
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        
        result.decomposition_success = (solver.info() == Eigen::Success);
        
        if (result.decomposition_success) {
            // 2. 分析 D 矩阵获取惯性指数
            analyzeInertia(solver, result);
            
            // 3. 判断正定性
            determineDefiniteness(result);
            
            // 4. 计算建议移位
            result.suggested_shift = computeSuggestedShift(result);
            
            // 5. 生成建议
            generateRecommendation(result);
        } else {
            result.warnings.push_back("LDLT 分解失败，矩阵可能高度病态");
            result.suggested_shift = 1e-6;  // 默认小量正则化
            result.recommendation = "分解失败，建议使用小量正则化";
        }
        
        return result;
    }
    
    // 快速判断（只需知道是否正定和建议移位）
    std::pair<bool, double> quickCheck() {
        auto result = analyze();
        return {result.is_positive_definite, result.suggested_shift};
    }
    
    // 获取建议的移位参数（用于三次正则化）
    double getSuggestedShift(double current_sigma = 1.0) {
        auto result = analyze();
        
        if (result.is_positive_definite) {
            // 如果正定，可以保持当前 sigma 或稍微减小
            return std::min(current_sigma, 1.0);
        } else {
            // 如果不正定，需要足够的移位
            return std::max(current_sigma, result.suggested_shift);
        }
    }
    
private:
    void analyzeInertia(const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                       AnalysisResult& result) {
        
        auto D = solver.vectorD();
        result.n_positive = 0;
        result.n_negative = 0;
        result.n_zero = 0;
        result.min_pivot = std::numeric_limits<double>::max();
        result.max_pivot = 0;
        result.min_abs_pivot = std::numeric_limits<double>::max();
        
        for (int i = 0; i < D.size(); ++i) {
            double val = D(i);
            double abs_val = std::abs(val);
            
            // 更新统计
            result.min_pivot = std::min(result.min_pivot, val);
            result.max_pivot = std::max(result.max_pivot, val);
            result.min_abs_pivot = std::min(result.min_abs_pivot, abs_val);
            
            // 惯性计数
            if (val > tolerance) {
                result.n_positive++;
            } else if (val < -tolerance) {
                result.n_negative++;
            } else {
                result.n_zero++;
            }
        }
        
        // 估计条件数
        if (result.min_abs_pivot > 0) {
            result.condition_number_est = result.max_pivot / result.min_abs_pivot;
        } else {
            result.condition_number_est = std::numeric_limits<double>::infinity();
        }
    }
    
    void determineDefiniteness(AnalysisResult& result) {
        result.is_positive_definite = (result.n_negative == 0 && result.n_zero == 0);
        result.is_negative_definite = (result.n_positive == 0 && result.n_zero == 0);
        result.is_indefinite = (result.n_positive > 0 && result.n_negative > 0);
        result.is_singular = (result.n_zero > 0);
        
        if (result.is_singular) {
            result.warnings.push_back("矩阵奇异，有 " + std::to_string(result.n_zero) + " 个零特征值");
        }
        
        if (result.is_indefinite) {
            result.warnings.push_back("矩阵不定，有 " + std::to_string(result.n_negative) + " 个负特征值");
        }
        
        if (result.condition_number_est > 1e12) {
            result.warnings.push_back("矩阵病态，条件数 " + std::to_string(result.condition_number_est));
        }
    }
    
    double computeSuggestedShift(const AnalysisResult& result) {
        if (result.is_positive_definite) {
            // 正定矩阵：只需要很小的移位保证数值稳定
            return std::max(1e-8, result.min_abs_pivot * 1e-6);
        }
        
        if (result.is_indefinite) {
            // 不定矩阵：需要克服负特征值
            // 最小特征值 ≈ 最小的负主元
            double min_negative = 0;
            for (const auto& val : getNegativePivots()) {
                min_negative = std::min(min_negative, val);
            }
            return -min_negative + 1e-6;
        }
        
        if (result.is_singular) {
            // 奇异矩阵：需要克服零特征值
            return result.min_abs_pivot * 0.1 + 1e-6;
        }
        
        // 默认值
        return 1e-4;
    }
    
    std::vector<double> getNegativePivots() {
        std::vector<double> negatives;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        
        if (solver.info() == Eigen::Success) {
            auto D = solver.vectorD();
            for (int i = 0; i < D.size(); ++i) {
                if (D(i) < -tolerance) {
                    negatives.push_back(D(i));
                }
            }
        }
        
        return negatives;
    }
    
    void generateRecommendation(AnalysisResult& result) {
        if (result.is_positive_definite) {
            result.recommendation = "矩阵对称正定，可以直接使用 Cholesky 分解";
            if (result.condition_number_est > 1e10) {
                result.recommendation += "，但病态严重，建议预处理";
            }
        } else if (result.is_negative_definite) {
            result.recommendation = "矩阵负定，建议取负后使用 Cholesky";
        } else if (result.is_indefinite) {
            result.recommendation = "矩阵不定，需要使用移位正则化，建议移位量: " 
                                  + std::to_string(result.suggested_shift);
        } else if (result.is_singular) {
            result.recommendation = "矩阵奇异，需要正则化，建议移位量: " 
                                  + std::to_string(result.suggested_shift);
        }
        
        if (result.decomposition_success && !result.is_positive_definite) {
            result.recommendation += " (LDLT 分解成功，可使用 Bunch-Kaufman 分解)";
        }
    }
};

// 在实际三次正则化中的使用
class CubicRegularSolverWithIndefinite {
private:
    SymmetricIndefiniteAnalyzer analyzer;
    double sigma;
    
public:
    CubicRegularSolverWithIndefinite(const Eigen::SparseMatrix<double>& H, 
                                     double initial_sigma = 1.0)
        : analyzer(H), sigma(initial_sigma) {}
    
    Eigen::VectorXd solveStep(const Eigen::VectorXd& g, const Eigen::SparseMatrix<double>& H) {
        // 1. 分析 Hessian
        auto result = analyzer.analyze();
        
        // 2. 打印诊断信息
        printDiagnostic(result);
        
        // 3. 根据分析结果调整 sigma
        sigma = adjustSigma(result);
        
        // 4. 构造移位后的系统
        auto H_shifted = addShift(H, result.suggested_shift);
        
        // 5. 求解
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H_shifted);
        
        if (solver.info() == Eigen::Success) {
            return solver.solve(-g);
        } else {
            // 如果失败，尝试更大的移位
            sigma *= 2;
            return solveStep(g,H);  // 递归尝试
        }
    }
    
private:
    void printDiagnostic(const SymmetricIndefiniteAnalyzer::AnalysisResult& result) {
        std::cout << "\n=== 对称不定矩阵分析 ===" << std::endl;
        std::cout << "正特征值: " << result.n_positive << std::endl;
        std::cout << "负特征值: " << result.n_negative << std::endl;
        std::cout << "零特征值: " << result.n_zero << std::endl;
        std::cout << "正定性: ";
        if (result.is_positive_definite) std::cout << "正定";
        else if (result.is_negative_definite) std::cout << "负定";
        else if (result.is_indefinite) std::cout << "不定";
        else if (result.is_singular) std::cout << "奇异";
        std::cout << std::endl;
        std::cout << "条件数估计: " << result.condition_number_est << std::endl;
        std::cout << "建议移位: " << result.suggested_shift << std::endl;
        std::cout << "建议: " << result.recommendation << std::endl;
        
        for (const auto& w : result.warnings) {
            std::cout << "警告: " << w << std::endl;
        }
    }
    
    double adjustSigma(const SymmetricIndefiniteAnalyzer::AnalysisResult& result) {
        if (result.is_positive_definite) {
            // 正定：可以适当减小 sigma
            return std::max(sigma * 0.5, 1e-8);
        } else {
            // 非正定：需要足够大的 sigma
            return std::max(sigma, result.suggested_shift * 2);
        }
    }
    
    Eigen::SparseMatrix<double> addShift(const Eigen::SparseMatrix<double>&H, double shift) {
        int n = H.rows();
        std::vector<Eigen::Triplet<double>> triplets;
        
        // 复制原有元素
        for (int k = 0; k < H.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it) {
                int i = it.row();
                int j = it.col();
                double val = it.value();
                
                if (i == j) {
                    triplets.push_back(Eigen::Triplet<double>(i, j, val + shift));
                } else {
                    triplets.push_back(Eigen::Triplet<double>(i, j, val));
                }
            }
        }
        
        // 确保所有对角线存在
        for (int i = 0; i < n; ++i) {
            bool has_diag = false;
            for (const auto& t : triplets) {
                if (t.row() == i && t.col() == i) {
                    has_diag = true;
                    break;
                }
            }
            if (!has_diag) {
                triplets.push_back(Eigen::Triplet<double>(i, i, shift));
            }
        }
        
        Eigen::SparseMatrix<double> H_shifted(n, n);
        H_shifted.setFromTriplets(triplets.begin(), triplets.end());
        return H_shifted;
    }
};

// 测试示例
void testIndefiniteMatrix(Eigen::SparseMatrix<double> &H) {
    // // 创建对称不定矩阵
    // Eigen::SparseMatrix<double> H(3,3);
    // H.insert(0,0) = 4.0;
    // H.insert(0,1) = 2.0;  H.insert(1,0) = 2.0;
    // H.insert(1,1) = -1.0;  // 负对角元
    // H.insert(2,2) = 3.0;
    
    // std::cout << "测试矩阵:" << std::endl;
    // std::cout << "4   2   0" << std::endl;
    // std::cout << "2  -1   0" << std::endl;
    // std::cout << "0   0   3" << std::endl;
    
    // 分析
    SymmetricIndefiniteAnalyzer analyzer(H);
    auto result = analyzer.analyze();
    
    std::cout << "\n分析结果:" << std::endl;
    std::cout << "LDLT 分解成功: " << (result.decomposition_success ? "是" : "否") << std::endl;
    std::cout << "正特征值: " << result.n_positive << std::endl;
    std::cout << "负特征值: " << result.n_negative << std::endl;
    std::cout << "零特征值: " << result.n_zero << std::endl;
    std::cout << "是否正定: " << (result.is_positive_definite ? "是" : "否") << std::endl;
    std::cout << "是否不定: " << (result.is_indefinite ? "是" : "否") << std::endl;
    std::cout << "建议移位: " << result.suggested_shift << std::endl;
    std::cout << "建议: " << result.recommendation << std::endl;
    
    // 快速检查
    auto [is_pd, shift] = analyzer.quickCheck();
    std::cout << "\n快速检查: 正定=" << is_pd << ", 移位=" << shift << std::endl;
    
    // 在三次正则化中使用
    Eigen::VectorXd g(3);
    g << 1.0, 2.0, 3.0;
    
    CubicRegularSolverWithIndefinite solver(H);
    Eigen::VectorXd d = solver.solveStep(g,H);
    
    std::cout << "\n求解步长: " << d.transpose() << std::endl;
}


#endif