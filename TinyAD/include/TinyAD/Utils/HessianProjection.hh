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
        //#if DIFF_PROJECTED_NEWTON
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
        for (Eigen::Index i = 0; i < k; ++i) {
            PassiveT old_val = eigenvalues(i);
            
            if (regularizer.needs_regularization(old_val)) {

                eigenvalues(i) = regularizer.regularize(old_val);
                modified = true;

                #if DEBUG_OUTPUT
                    //TINYAD_DEBUG_OUT("Projection mode in project_positive_definite_diff: " << static_cast<int>(_mode));
                    TINYAD_DEBUG_OUT("ev("<<i<<"):" << static_cast<double>(old_val) << " -> " << static_cast<double>(eigenvalues(i))); 
                    //TINYAD_DEBUG_OUT("*evec("<<i<<")=" << eigenvectors.col(i).transpose()); 
                #endif
            }
        }
        
        if (!modified) {
            return; // 如果没有任何特征值需要修改，直接返回
        }
        
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




