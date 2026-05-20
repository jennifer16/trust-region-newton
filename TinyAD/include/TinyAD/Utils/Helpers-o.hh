/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <TinyAD/Utils/Out.hh>

namespace TinyAD
{

    /**
     * Create vector of indices from 0 to n-1.
     */
    inline std::vector<Eigen::Index> range(
            const Eigen::Index _n)
    {
        TINYAD_ASSERT_GEQ(_n, 0);

        std::vector<Eigen::Index> r(_n);
        for (Eigen::Index i = 0; i < _n; ++i)
            r[i] = i;

        return r;
    }

    /**
     * Count elements in range.
     * (This exists because std::distance cannot handle
     * different iterator types for begin and end.)
     */
    template <typename RangeT>
    Eigen::Index count(
            const RangeT& _range)
    {
        Eigen::Index n = 0;
        for (const auto& r : _range)
            ++n;

        return n;
    }

    /**
     * Assemble matrix from column vectors.
     */
    template <typename Derived>
    auto col_mat(
            const Eigen::MatrixBase<Derived>& _v0,
            const Eigen::MatrixBase<Derived>& _v1)
    {
        using T = typename Derived::Scalar;
        Eigen::Matrix<T, Derived::RowsAtCompileTime, 2 * Derived::ColsAtCompileTime> M;

        M << _v0, _v1;

        return M;
    }

    /**
     * Assemble matrix from column vectors.
     */
    template <typename Derived>
    auto col_mat(
            const Eigen::MatrixBase<Derived>& _v0,
            const Eigen::MatrixBase<Derived>& _v1,
            const Eigen::MatrixBase<Derived>& _v2)
    {
        using T = typename Derived::Scalar;
        Eigen::Matrix<T, Derived::RowsAtCompileTime, 3 * Derived::ColsAtCompileTime> M;

        M << _v0, _v1, _v2;

        return M;
    }

    /**
     * Sparse identity matrix.
     */
    template <typename PassiveT>
    Eigen::SparseMatrix<PassiveT> identity(
            const Eigen::Index _n)
    {
        Eigen::SparseMatrix<PassiveT> Id(_n, _n);
        Id.setIdentity();

        return Id;
    }

    // 在 TinyAD 的头文件中添加
    enum class HessianProjectionMode {
        AUTO = 0,      // 自动选择
        SOFT_ABS = 1,   // 软 abs
        ABS = 2,       // 直接取绝对值
        ABS_NONDIFF = 3,       // 直接取绝对值

        CLAMP = 4,     // 直接 clamp 到 epsilon
        SOFT_CLAMP = 5, // 软 clamp
        CLAMP_NONDIFF = 6,     // 直接 clamp 到 epsilon

        CLAMP_ABS = 7, // clamp 到 epsilon + abs 负值
        HYBRID = 8,    // 混合策略 
        CLAMP_ABS_NONDIFF = 9, //  trust region newton

        CLAMP_ABS_ADAPTIVE_NONDIFF = 10, // 自适应选择的非可微版本

        CUBIC = 11,
        ABS_SHIFT_CLAMP_NONDIFF = 12,
        CLAMP_ABS_BLENDING = 13,
        SMOOTH_TR = 14,
        CLAMP_ABS_BLENDING_SHEAR = 15,
        CLAMP_ABS_BLENDING_SMOOTH = 16,
        CLAMP_ABS_BLENDING2 = 17,
        CLAMP_ABS_BLENDING4 = 18,
        CLAMP_ABS_BLENDING3 = 19,
        CLAMP_ABS_BLENDING5 = 20,
    };

    // // 构造函数：可以配置不同的行为
    //     enum class Mode {
    //         AUTO,      // 自动选择
    //         CLAMP_ABS, // clamp 到 epsilon + abs 负值
    //         SOFT_CLAMP, // 软 clamp
    //         SOFT_ABS,   // 软 abs
    //         ABS,       // 直接取绝对值
    //         CLAMP,     // 直接 clamp 到 epsilon
    //         HYBRID     // 混合策略
    //         ABS_NONDIFF,       // 直接取绝对值
    //         CLAMP_NONDIFF,     // 直接 clamp 到 epsilon
    //     };

    // inline变量可以直接在头文件中定义
    inline int g_MU = 1;
    inline double g_LAMBDA = 1;

    inline const double EPS_1E_8 = 1e-8;   // 单精度浮点误差，通常够用
    inline const double EPS_1E_12 = 1e-12; // 双精度浮点误差
    inline const double EPS_MACHINE = std::numeric_limits<double>::epsilon();  // 约 2.22e-16
    inline const double COND_THRESHOLD = 1e10;
    inline const double ZERO = EPS_1E_8;

    // 相对容差（考虑数值大小）
    inline bool isZero(double x, double tolerance = ZERO) {
        return std::abs(x) <= tolerance;
    }

    inline bool isEqual(double a, double b, double tolerance = ZERO) {
        return std::abs(a - b) < tolerance;
    }

    //为负
    inline bool isNegative(double x, double tolerance = ZERO) {
        return x < -tolerance;
    }

    //为正
    inline bool isPostive(double x, double tolerance = ZERO) {
        return x > tolerance;
    }

    //非正
    inline bool isNonPostive(double x, double tolerance = ZERO) {
        return x < tolerance;
    }

}
