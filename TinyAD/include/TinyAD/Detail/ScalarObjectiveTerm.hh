/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/SparseCore>
#include <TinyAD/Scalar.hh>
#include <TinyAD/Detail/Element.hh>
#include <TinyAD/Detail/EvalSettings.hh>
#include <TinyAD/Utils/HessianProjection.hh>

#include "../../../include/ElementDeformationState.h"  // 或项目 include 路径

#ifdef _OPENMP
#include <omp.h>
#endif

namespace TinyAD
{

/**
 * Abstract base type for objective terms.
 * We need this to be able to store multiple different
 * objective term instantiations in a vector in ScalarFunction.
 */
template <typename PassiveT>
struct ScalarObjectiveTermBase
{
    virtual ~ScalarObjectiveTermBase() = default;

    virtual PassiveT eval(
            const Eigen::VectorX<PassiveT>& _x) const = 0;

    virtual void eval_with_gradient_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g) const = 0;

    virtual void eval_with_derivatives_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            std::vector<Eigen::Triplet<PassiveT>>& _H_proj_triplets,
            const bool _project_hessian,
            const PassiveT& _projection_eps) const = 0;

    // 新增虚函数
    virtual void set_projection_mode(HessianProjectionMode mode) {
        // 默认实现：什么也不做（为了兼容性）
    }
    
    virtual HessianProjectionMode get_projection_mode() const {
        return HessianProjectionMode::AUTO;
    }

    //约束处理 zj
    virtual void set_fixed_dofs(const std::vector<bool>& _is_fixed) {}

};

/**
 * Objective term stored in ScalarFunction.
 */
template <int variable_dimension, int element_valence, typename PassiveT, typename VariableHandleT, typename ElementHandleT>
struct ScalarObjectiveTerm : ScalarObjectiveTermBase<PassiveT>
{
    static constexpr int n_element = variable_dimension * element_valence;

    // Scalar types. Either passive (e.g. double), or active (TinyAD::Scalar).
    // Declare separate types for first-order-only and for second-order use cases.
    using PassiveScalarType = PassiveT;
    using ActiveFirstOrderScalarType = TinyAD::Scalar<n_element, PassiveT, false>;
    using ActiveSecondOrderScalarType = TinyAD::Scalar<n_element, PassiveT, true>;

    // Element types. These are passed as argument to the user-provided lambda function.
    using PassiveElementType = Element<variable_dimension, element_valence, 1, PassiveT, PassiveT, VariableHandleT, ElementHandleT, false>;
    using ActiveFirstOrderElementType = Element<variable_dimension, element_valence, 1, PassiveT, ActiveFirstOrderScalarType, VariableHandleT, ElementHandleT, true>;
    using ActiveSecondOrderElementType = Element<variable_dimension, element_valence, 1, PassiveT, ActiveSecondOrderScalarType, VariableHandleT, ElementHandleT, true>;

    // Return types of the user-provided lambda function.
    using PassiveEvalElementReturnType = PassiveScalarType;
    using ActiveFirstOrderEvalElementReturnType = ActiveFirstOrderScalarType;
    using ActiveSecondOrderEvalElementReturnType = ActiveSecondOrderScalarType;

    // Types of the user-provided lambda function.
    using PassiveEvalElementFunction = std::function<PassiveEvalElementReturnType(PassiveElementType&)>;
    using ActiveFirstOrderEvalElementFunction = std::function<ActiveFirstOrderEvalElementReturnType(ActiveFirstOrderElementType&)>;
    using ActiveSecondOrderEvalElementFunction = std::function<ActiveSecondOrderEvalElementReturnType(ActiveSecondOrderElementType&)>;

    template <typename EvalElementFunction>
    ScalarObjectiveTerm(
            const std::vector<ElementHandleT>& _element_handles,
            EvalElementFunction _eval_element,
            const Eigen::Index _n_global,
            const EvalSettings& _settings)
        : n_vars_global(_n_global),
          element_handles(_element_handles),
          settings(_settings)
    {
        static_assert (std::is_same_v<
                std::decay_t<decltype((_eval_element(std::declval<PassiveElementType&>())))>,
                PassiveEvalElementReturnType>,
                "Please make sure that the user-provided lambda function has the signature (const auto& element) -> TINYAD_SCALAR_TYPE(element)");

        // Instantiate _eval_element() for passive and active scalar types
        eval_element_passive = _eval_element;
        eval_element_active_first_order = _eval_element;
        eval_element_active_second_order = _eval_element;

        // is_fixed_global = nullptr;
    }

    PassiveT eval(
            const Eigen::VectorX<PassiveT>& _x) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);

        // Eval elements using plain double type
        std::vector<PassiveT> element_results(element_handles.size());

        #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_handles.size(); ++i_element)
        {
            // Call user code
            PassiveElementType element(element_handles[i_element], _x);
            element_results[i_element] = eval_element_passive(element);
        }

        // Sum up results
        PassiveT f = 0.0;
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_results.size(); ++i_element)
            f += element_results[i_element];

        return f;
    }

    void eval_with_gradient_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);
        TINYAD_ASSERT_EQ(_g.size(), n_vars_global);

        // Eval elements using active scalar type
        std::vector<ActiveFirstOrderElementType> elements(element_handles.size());
        std::vector<ActiveFirstOrderScalarType> element_results(element_handles.size());

        #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_handles.size(); ++i_element)
        {
            // Call user code, which initializes active variables via element.variables(...) and performs computations.
            elements[i_element] = ActiveFirstOrderElementType(element_handles[i_element], _x);
            element_results[i_element] = eval_element_active_first_order(elements[i_element]);

            // Assert that derivatives are finite
            TINYAD_ASSERT_FINITE_MAT(element_results[i_element].grad);
        }

        // Add to global f, g and H
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_handles.size(); ++i_element)
        {
            _f += element_results[i_element].val;

            // Add to global gradient
            for (Eigen::Index i = 0; i < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++i)
                _g[elements[i_element].idx_local_to_global[i]] += element_results[i_element].grad[i];
        }
    }

    void eval_with_derivatives_add(
            const Eigen::VectorX<PassiveT>& _x,
            PassiveT& _f,
            Eigen::VectorX<PassiveT>& _g,
            std::vector<Eigen::Triplet<PassiveT>>& _H_triplets,
            const bool _project_hessian,
            const PassiveT& _projection_eps) const override
    {
        TINYAD_ASSERT_EQ(_x.size(), n_vars_global);
        TINYAD_ASSERT_EQ(_g.size(), n_vars_global);

        // Eval elements using active scalar type
        std::vector<ActiveSecondOrderElementType> elements(element_handles.size());
        std::vector<ActiveSecondOrderScalarType> element_results(element_handles.size());

        #pragma omp parallel for schedule(static) num_threads(get_n_threads(settings))
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_handles.size(); ++i_element)
        {
            // Call user code, which initializes active variables via element.variables(...) and performs computations.
            elements[i_element] = ActiveSecondOrderElementType(element_handles[i_element], _x);
            element_results[i_element] = eval_element_active_second_order(elements[i_element]);

            //to do, 特征值为负的时候再修正
            Eigen::MatrixXd element_Hess = element_results[i_element].Hess;
            auto b_positive = positive_diagonally_dominant<PassiveT>(element_Hess, _projection_eps);
            double J_val = g_element_J[i_element];
            if (_project_hessian && (!b_positive)) //非正定
            {
                auto &H_local = element_results[i_element].Hess;
                const auto& idx_local_to_global = elements[i_element].idx_local_to_global;
                if (is_fixed_global != nullptr) 
                {
                    // === 策略 B：只在自由子空间修正 ===
                    auto& H_local = element_results[i_element].Hess;
                    std::vector<Eigen::Index> free_local = get_free_local_indices(idx_local_to_global);
                    int n_free = free_local.size();
                    int n_all = idx_local_to_global.size();

                    // TINYAD_DEBUG_OUT("modify partial free element: " << i_element); 
                    // TINYAD_DEBUG_OUT("n_free: " << n_free); 

                    if (n_free > 0 && n_free < n_all) { // 混合单元：提取自由子块
                        Eigen::MatrixX<PassiveT> H_AA(n_free, n_free);
                        for (int i = 0; i < n_free; ++i)
                            for (int j = 0; j < n_free; ++j)
                                H_AA(i, j) = H_local(free_local[i], free_local[j]);
                          
                        // if (i_element == 660 // free 3
                        //     || i_element == 663   
                        //     || i_element == 661   // free 6
                        //     || i_element == 1
                        //     || i_element == 2 // free 9
                        //     || i_element == 924) 
                        if (i_element == 660 ) 
                        {
                            TINYAD_DEBUG_OUT("modify partial free element: " << i_element); 
                            TINYAD_DEBUG_OUT("n_free: " << n_free); 
                            TINYAD_DEBUG_OUT("H_local(original): " << element_results[i_element].Hess); 
                        }
                        // // 对 H_AA 做修正
                        // #ifdef DIFF_PROJECTED_NEWTON
                        //     project_positive_definite_diff<PassiveT>(H_AA, _projection_eps, get_projection_mode());
                        // #else
                        //     project_positive_definite<PassiveT>(H_AA, _projection_eps);
                        // #endif
                        // 替换掉对 project_positive_definite_diff 的调用
                        // 自己调用 project_positive_definite 的运行期版本（如果存在的话）

                        // 或者直接对 H_AA 做特征分解和修正
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<PassiveT>> eigensolver(H_AA);
                        Eigen::VectorX<PassiveT> eigenvalues = eigensolver.eigenvalues();
                        const auto& eigenvectors = eigensolver.eigenvectors();
                        

                        auto mode  = get_projection_mode();
                        
                        
                        if (mode == HessianProjectionMode::ABS_NONDIFF)
                        {
                            for (int k = 0; k < eigenvalues.size(); ++k) {
                                if (eigenvalues[k] < 0) {
                                    eigenvalues[k] = -eigenvalues[k];  // abs
                                }
                            }
                        }
                        else if (mode == HessianProjectionMode::CLAMP_ABS_NONDIFF
                            || mode == HessianProjectionMode::CLAMP_ABS_BLENDING)
                        {
                            if (_projection_eps < 0)
                            {
                                for (int k = 0; k < eigenvalues.size(); ++k) {
                                    if (eigenvalues[k] < 0) {
                                        eigenvalues[k] = -eigenvalues[k];  // abs
                                    }   
                                }
                            }
                            else
                            {
                                for (int k = 0; k < eigenvalues.size(); ++k) {
                                    if (eigenvalues[k] < 0) {
                                        eigenvalues[k] = 0;  // clamp
                                    }   
                                }
                            }
                            
                        }
                        else // clamp
                        {
                            for (int k = 0; k < eigenvalues.size(); ++k) {
                                if (eigenvalues[k] < 0) {
                                    eigenvalues[k] = 0;  
                                }   
                            }
                        }
                        
                        H_AA = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
                        H_AA = 0.5 * (H_AA + H_AA.transpose());  // 对称化
                    
                        // 嵌入回去
                        for (int i = 0; i < n_free; ++i)
                            for (int j = 0; j < n_free; ++j)
                                H_local(free_local[i], free_local[j]) = H_AA(i, j);

                        // if (i_element == 660 // free 3
                        //     || i_element == 663   
                        //     || i_element == 661   // free 6
                        //     || i_element == 1
                        //     || i_element == 2 // free 9
                        //     || i_element == 924) 
                        if (i_element == 660 ) 
                        { 
                            TINYAD_DEBUG_OUT("modify partial free element: " << i_element); 
                            TINYAD_DEBUG_OUT("n_free: " << n_free); 
                            TINYAD_DEBUG_OUT("H_local(modified): " << element_results[i_element].Hess); 
                        }
                                
                    } 
                    else if (n_free == n_all) //全free
                    {
                        #ifdef DIFF_PROJECTED_NEWTON
                            // #if DEBUG_OUTPUT
                            // //TINYAD_DEBUG_OUT("Projection mode in eval_with_derivatives_add: " << static_cast<int>(get_projection_mode())); 
                            // #endif
                            project_positive_definite_diff<n_element, PassiveT>(element_results[i_element].Hess, 
                                element_results[i_element].grad,   // ← 新增：传入真实梯度
                                _projection_eps, get_projection_mode(), J_val);
                        #else       
                            // project_positive_definite_with_inv_g<n_element, PassiveT>(element_results[i_element].Hess, element_results[i_element].grad, _projection_eps);
                            project_positive_definite<n_element, PassiveT>(element_results[i_element].Hess, _projection_eps);
                        #endif
                    }
                    //全fixed,不修正，后续会被清0
                    
                }
                else 
                {
                    #ifdef DIFF_PROJECTED_NEWTON
                        
                        // if (i_element == 660 )    
                        // {
                        //     TINYAD_DEBUG_OUT("modify partial free element: " << i_element); 
                        //     TINYAD_DEBUG_OUT("H_local(original): " << element_results[i_element].Hess); 
                        // }
                        project_positive_definite_diff<n_element, PassiveT>(element_results[i_element].Hess, 
                            element_results[i_element].grad,   // ← 新增：传入真实梯度
                            _projection_eps, get_projection_mode(), J_val);
                    
                         
                        // if (i_element == 660 ) 
                        // {
                        //     TINYAD_DEBUG_OUT("modify partial free element: " << i_element); 
                        //     TINYAD_DEBUG_OUT("H_local(modified): " << element_results[i_element].Hess); 
                        // }

                    #else       
                        // project_positive_definite_with_inv_g<n_element, PassiveT>(element_results[i_element].Hess, element_results[i_element].grad, _projection_eps);
                        project_positive_definite<n_element, PassiveT>(element_results[i_element].Hess, _projection_eps);
                    #endif
                    
                }
               
            }
               
            // Assert that derivatives are finite
            TINYAD_ASSERT_FINITE_MAT(element_results[i_element].grad);
            TINYAD_ASSERT_FINITE_MAT(element_results[i_element].Hess);
        }

        // Add to global f, g and H
        for (Eigen::Index i_element = 0; i_element < (Eigen::Index)element_handles.size(); ++i_element)
        {
            _f += element_results[i_element].val;

            // Add to global gradient
            for (Eigen::Index i = 0; i < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++i)
                _g[elements[i_element].idx_local_to_global[i]] += element_results[i_element].grad[i];

            // Add to global Hessian
            for (Eigen::Index i = 0; i < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++i)
            {
                for (Eigen::Index j = 0; j < (Eigen::Index)elements[i_element].idx_local_to_global.size(); ++j)
                {
                    _H_triplets.push_back(Eigen::Triplet<PassiveT>(
                               elements[i_element].idx_local_to_global[i],
                               elements[i_element].idx_local_to_global[j],
                               element_results[i_element].Hess(i, j)));
                }
            }
        }
    }

    // 新增：设置投影模式的方法
    void set_projection_mode(HessianProjectionMode mode) {
        projection_mode_ = mode;
    }
    
    HessianProjectionMode get_projection_mode() const {
        return projection_mode_;
    }

    // 处理约束顶点zj
    

    void set_fixed_dofs(const std::vector<bool>& _is_fixed) override {
        is_fixed_global = &_is_fixed;
    }

    // 提取自由局部索引的辅助函数
    std::vector<Eigen::Index> get_free_local_indices(
        const std::vector<Eigen::Index>& idx_local_to_global) const
    {
        std::vector<Eigen::Index> free_local;
        if (is_fixed_global != nullptr)
        {
            for (Eigen::Index local_i = 0; local_i < (Eigen::Index)idx_local_to_global.size(); ++local_i) {
                Eigen::Index global_i = idx_local_to_global[local_i];
                if (global_i < (*is_fixed_global).size()
                && !(*is_fixed_global)[global_i]) {
                    free_local.push_back(local_i);
                }
            }
        }
        
        return free_local;
    }

private:
    const Eigen::Index n_vars_global;

    const std::vector<ElementHandleT> element_handles;
    const EvalSettings& settings;

    // 新增：投影模式成员变量
    HessianProjectionMode projection_mode_ = HessianProjectionMode::AUTO;
    // 新增：全局固定自由度标记的引用
    const std::vector<bool>* is_fixed_global = nullptr;

    // Instantiations of user-provided lambda
    PassiveEvalElementFunction eval_element_passive;
    ActiveFirstOrderEvalElementFunction eval_element_active_first_order;
    ActiveSecondOrderEvalElementFunction eval_element_active_second_order;
};

};
