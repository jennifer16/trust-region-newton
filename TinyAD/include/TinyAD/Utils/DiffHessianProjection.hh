#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <functional>
namespace TinyAD
{
template<typename T>
class EigenvalueRegularizer {
public:
    
    
    EigenvalueRegularizer(T eps = T(0), HessianProjectionMode m = HessianProjectionMode::AUTO, T smooth = T(0)) 
        : epsilon(eps), mode(m), smoothness(smooth) {
        // 自动设置平滑参数
        if (mode != HessianProjectionMode::ABS_NONDIFF
        && mode != HessianProjectionMode::CLAMP_NONDIFF
        && mode != HessianProjectionMode::CLAMP_ABS_NONDIFF) {
            smoothness = std::max(T(1e-8), eps * T(0.01));
        }   
    }
    
    // 主函数：正则化特征值
    T regularize(T lambda) const {
        // #if DEBUG_OUTPUT
        //     TINYAD_DEBUG_OUT("Projection mode in eval_with_derivatives_add: " << static_cast<int>(mode)); 
        // #endif
        switch (mode) {
            case HessianProjectionMode::AUTO:
                return auto_regularize(lambda);
            case HessianProjectionMode::CLAMP:
                return clamp_regularize(lambda);
            case HessianProjectionMode::SOFT_CLAMP:
                return soft_clamp_regularize(lambda);
            
            case HessianProjectionMode::ABS:
                return abs_regularize(lambda);
            case HessianProjectionMode::SOFT_ABS:
                return soft_abs_regularize(lambda);
            
            case HessianProjectionMode::CLAMP_ABS:
                return clamp_abs_regularize(lambda);
            case HessianProjectionMode::HYBRID:
                return hybrid_regularize(lambda); 
            case HessianProjectionMode::CLAMP_ABS_ADAPTIVE_NONDIFF:
                return clamp_abs_regularize_nondiff(lambda); // 直接复用auto_regularize的非可微版本  
            //nondiff版本：直接clamp或abs，没有平滑过渡
            case HessianProjectionMode::CLAMP_ABS_NONDIFF:
                return clamp_abs_regularize_nondiff(lambda);
            case HessianProjectionMode::ABS_NONDIFF:
                return abs_regularize_nondiff(lambda);
            case HessianProjectionMode::CLAMP_NONDIFF:
                return clamp_regularize_nondiff(lambda);
            
            default:
                return default_regularize(lambda);
        }
    }
    
    // 检查是否需要正则化
    bool needs_regularization(T lambda) const {
        
        switch (mode) {
            case HessianProjectionMode::CLAMP_ABS_NONDIFF:
                if (epsilon < T(0)) {
                    // epsilon < 0: abs 模式
                    return lambda < T(0); // 直接复用ABS的非可微版本的判断条件
                } else {
                    // epsilon > 0: 软 clamp
                    return lambda < epsilon ;
                }
            // case HessianProjectionMode::AUTO:
            //     return lambda < epsilon + smoothness;
            // case HessianProjectionMode::CLAMP:
            // case HessianProjectionMode::SOFT_CLAMP:
            //     return lambda < epsilon;
            // case HessianProjectionMode::ABS:
            // case HessianProjectionMode::SOFT_ABS:
            //     return lambda < T(0);
            // case HessianProjectionMode::CLAMP_ABS:
            // case HessianProjectionMode::HYBRID:
            //     return lambda < std::min(epsilon, T(0));
            // case HessianProjectionMode::CLAMP_ABS_ADAPTIVE_NONDIFF:
            //     return lambda < T(0); // 直接复用auto_regularize的非可微版本的判断条件
            
            case HessianProjectionMode::ABS_NONDIFF:
                return lambda < T(0); // 直接复用ABS的非可微版本的判断条件
            case HessianProjectionMode::CLAMP_NONDIFF:
                return lambda < epsilon; // 直接复用CLAMP的非可微版本的判断条件
            default:
                return lambda < epsilon + smoothness;
        }
    }
    
private:
    T epsilon;
    T smoothness;
    HessianProjectionMode mode;
    
    // 自动模式：根据 epsilon 值选择策略
    T auto_regularize(T lambda) const {
        if (epsilon < T(0)) {
            // epsilon < 0: abs 模式
            // return abs_regularize(lambda);
            //return sqrt(lambda * lambda + smoothness * smoothness) - smoothness;
            return soft_abs_regularize(lambda);
        } else if (epsilon == T(0)) {
            // epsilon = 0: clamp 负值到 0
            //return clamp_abs_regularize(lambda);
            return soft_clamp_regularize(lambda);
        } else {
            // epsilon > 0: 软 clamp
            //return soft_clamp_regularize(lambda);
            // 其他，使用自适应模式
            return clamp_abs_regularize(lambda);
        }

    }
    
    // Clamp+Abs 策略：负值做 abs，小正值 clamp 到 epsilon
    T clamp_abs_regularize(T lambda) const {
        if (lambda >= epsilon) {
            return lambda;
        } else if (lambda > T(0)) {
            // 小正值：平滑 clamp 到 epsilon
            T t = lambda / epsilon;
            return epsilon * smooth_transition(t);
        } else {
            // 负值：平滑 abs
            return sqrt(lambda * lambda + smoothness * smoothness) - smoothness;
            // return abs_regularize(lambda);
        }
    }
    
    // 软 Clamp 策略：所有小于 epsilon 的值都平滑上升到 epsilon
    T soft_clamp_regularize(T lambda) const {
        if (lambda >= epsilon) {
            return lambda;
        } else {
            // 平滑上升到 epsilon
            T delta = epsilon - lambda;
            T t = T(1) / (T(1) + exp(-delta / smoothness));
            return lambda + delta * t;
        }
    }

    // 直接 Clamp 策略：所有小于 epsilon 的值都直接设置为 epsilon
    // 和 trust region paper 中提到的非可微分clamping方法一致
    T clamp_regularize(T lambda) const {
        return std::max(lambda, epsilon);
    }
    
    // 混合策略：结合多种方法
    T hybrid_regularize(T lambda) const {
        if (lambda > epsilon) {
            return lambda;
        }
        
        // 计算两个正则化结果
        T clamp_result = clamp_abs_regularize(lambda);
        T soft_result = soft_clamp_regularize(lambda);
        
        // 根据 lambda 的值加权平均
        T weight = T(0.5) - T(0.5) * tanh(lambda / smoothness);
        return weight * clamp_result + (T(1) - weight) * soft_result;
    }
    
    T default_regularize(T lambda) const {
        // 默认使用简单的可微分函数
        if (lambda >= epsilon) {
            return lambda;
        } else {
            // f(x) = ε + (x-ε) * sigmoid(k*(x-ε))
            T k = T(10) / smoothness;
            T delta = lambda - epsilon;
            return epsilon + delta / (T(1) + exp(-k * delta));
        }
    }
    
    // 仅仅对负值进行正则化，正值保持不变
    T abs_regularize(T lambda) const {
        if (lambda >= T(0)) {
            return lambda; 
        }
        return sqrt(lambda * lambda + smoothness * smoothness) - smoothness;
    }

    // 软绝对值（可微分近似）
    // 和abs_regularize几乎一样
    T soft_abs_regularize(T lambda) const {
        return sqrt(lambda * lambda + smoothness * smoothness) - smoothness;
    }

    // trust region paper中提到的非可微分clamping方法，直接clamp到epsilon或取绝对值
    T clamp_regularize_nondiff(T lambda) const {
        if (lambda < epsilon) {
            return epsilon;
        } else {
            return lambda;
        }
        //return std::max(lambda, epsilon);
    }

    T abs_regularize_nondiff(T lambda) const {
        if (lambda < T(0)) {
            return -lambda;
        } else {
            return lambda;
        }
        //return sqrt(lambda * lambda + smoothness * smoothness) - smoothness;
    }


    // 平滑过渡函数
    T smooth_transition(T t) const {
        // t ∈ [0, 1]，保证 C¹ 连续
        t = std::max(T(0), std::min(T(1), t));
        return t * t * (T(3) - T(2) * t);
    }

    T clamp_abs_regularize_nondiff(T lambda) const {
        
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_OUT("*in clamp_abs_regularize_nondiff(): lambda(old)=" << static_cast<double>(lambda));
            TINYAD_DEBUG_VAR(static_cast<double>(epsilon));
            //TINYAD_DEBUG_OUT("*in clamp_abs_regularize_nondiff(): epsilon=" << static_cast<double>(epsilon) ); 
        
        #endif

        if (epsilon < T(0)) {
            // project to absolute value
            if (lambda < T(0))
            {
                lambda = -lambda;
                //return -lambda;
            }
        }
        else {
            // project to epsilon
            if (lambda < epsilon)
            {
                lambda = epsilon;
                //return epsilon;
            }
        }
        #if DEBUG_OUTPUT
            TINYAD_DEBUG_VAR(static_cast<double>(lambda));
            //TINYAD_DEBUG_OUT("*in clamp_abs_regularize_nondiff(): lambda(new)=" << static_cast<double>(lambda));
        #endif

        return lambda;       
    }
};

}