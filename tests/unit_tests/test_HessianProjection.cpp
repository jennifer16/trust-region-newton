// ./tests/test_HessianProjection
// test_hessian_projection.cpp
// #include <gtest/gtest.h>
// #include <Eigen/Dense>
// #include <cmath>
// #include "HessianProjection.hh"

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
// #include <vector>
// #include <iostream>
// #include <iomanip>
// #include <chrono>
// #include <mutex>
// #include <sstream>
#include "../../TinyAD/include/TinyAD/Utils/HessianProjection.hh"

// ========== 定义全局变量 ==========
// 这些变量应该在主程序中被定义，但测试时需要提供定义
int g_j_mode = 1;
int g_kappa_mode = 1;
int g_grad_mode = 1;
int g_eta_mode = 1;
int g_pos_mode = 1;
int g_neg_mode = 1;
int g_update_gamma_mode = 1;
double g_MU = 1.0;
double g_LAMBDA = 1.0;


// 测试前设置全局变量默认值
class HessianProjectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 保存原始全局变量值
        original_g_j_mode = g_j_mode;
        original_g_kappa_mode = g_kappa_mode;
        original_g_grad_mode = g_grad_mode;
        original_g_eta_mode = g_eta_mode;
        original_g_pos_mode = g_pos_mode;
        original_g_neg_mode = g_neg_mode;
        original_g_update_gamma_mode = g_update_gamma_mode;
        original_g_MU = TinyAD::g_MU;
        original_g_LAMBDA = TinyAD::g_LAMBDA;
        
        // 设置测试默认值
        g_j_mode = 1;
        g_kappa_mode = 1;
        g_grad_mode = 1;
        g_eta_mode = 1;
        g_pos_mode = 1;
        g_neg_mode = 1;
        g_update_gamma_mode = 1;
        TinyAD::g_MU = 1.0;
        TinyAD::g_LAMBDA = 1.0;
    }
    
    void TearDown() override {
        // 恢复原始全局变量值
        g_j_mode = original_g_j_mode;
        g_kappa_mode = original_g_kappa_mode;
        g_grad_mode = original_g_grad_mode;
        g_eta_mode = original_g_eta_mode;
        g_pos_mode = original_g_pos_mode;
        g_neg_mode = original_g_neg_mode;
        g_update_gamma_mode = original_g_update_gamma_mode;
        TinyAD::g_MU = original_g_MU;
        TinyAD::g_LAMBDA = original_g_LAMBDA;
    }
    
private:
    int original_g_j_mode;
    int original_g_kappa_mode;
    int original_g_grad_mode;
    int original_g_eta_mode;
    int original_g_pos_mode;
    int original_g_neg_mode;
    int original_g_update_gamma_mode;
    double original_g_MU;
    double original_g_LAMBDA;
};

// ==================== clamp01 测试 ====================
TEST(clamp01, NormalValues) {
    EXPECT_DOUBLE_EQ(TinyAD::clamp01(0.5), 0.5);
    EXPECT_DOUBLE_EQ(TinyAD::clamp01(0.0), 0.0);
    EXPECT_DOUBLE_EQ(TinyAD::clamp01(1.0), 1.0);
}

TEST(clamp01, OutOfRangeValues) {
    EXPECT_DOUBLE_EQ(TinyAD::clamp01(-0.5), 0.0);
    EXPECT_DOUBLE_EQ(TinyAD::clamp01(1.5), 1.0);
    EXPECT_DOUBLE_EQ(TinyAD::clamp01(-100.0), 0.0);
    EXPECT_DOUBLE_EQ(TinyAD::clamp01(100.0), 1.0);
}

// ==================== computeKappa 测试 ====================
TEST_F(HessianProjectionTest, ComputeKappa_Method1_PositiveAndNegative) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -2.0, 1.0, 3.0;
    
    double kappa = TinyAD::computeKappa(1, eigenvalues, 1e-8);
    // λ_max = 3, λ_min = -2
    // nominator = 5, denominator = 5 + |1| + eps ≈ 6
    // kappa ≈ 5/6 ≈ 0.833
    EXPECT_NEAR(kappa, 0.83333, 1e-4);
    EXPECT_LE(kappa, 1.0);
    EXPECT_GE(kappa, 0.0);
}

TEST_F(HessianProjectionTest, ComputeKappa_Method1_AllPositive) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << 1.0, 2.0, 3.0;
    
    double kappa = TinyAD::computeKappa(1, eigenvalues, 1e-8);
    // λ_max = 3, λ_min = 1, 同号分支
    // nominator = 2, denominator = |3|+|1|+eps ≈ 4
    // kappa = 0.5
    EXPECT_NEAR(kappa, 0.5, 1e-4);
}

TEST_F(HessianProjectionTest, ComputeKappa_Method2_AllPositive) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << 1.0, 2.0, 3.0;
    
    double kappa = TinyAD::computeKappa(2, eigenvalues, 1e-8);
    // 全正特征值，lambda_min > 0，应返回 0.0
    EXPECT_DOUBLE_EQ(kappa, 0.0);
}

TEST_F(HessianProjectionTest, ComputeKappa_Method2_MixedSign) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -2.0, 1.0, 3.0;
    
    double kappa = TinyAD::computeKappa(2, eigenvalues, 1e-8);
    // sum_abs_all = 2+1+3=6, sum_abs_neg=2, neg_ratio=1/3≈0.333
    // kappa = 1 - exp(-4*0.333) ≈ 1 - 0.264 = 0.736
    EXPECT_NEAR(kappa, 0.736, 1e-3);
    EXPECT_LE(kappa, 1.0);
    EXPECT_GE(kappa, 0.0);
}

TEST_F(HessianProjectionTest, ComputeKappa_Method3_AllPositive) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << 1.0, 2.0, 3.0;
    
    double kappa = TinyAD::computeKappa(3, eigenvalues, 1e-8);
    // lambda_max=3, lambda_min=1, 同号分支
    // nominator=2, denominator=2+|4|+eps≈6, kappa=0.333
    EXPECT_NEAR(kappa, 0.33333, 1e-4);
}

TEST_F(HessianProjectionTest, ComputeKappa_Method4_MixedSign) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -2.0, 1.0, 3.0;
    
    double kappa = TinyAD::computeKappa(4, eigenvalues, 1e-8);
    EXPECT_LE(kappa, 1.0);
    EXPECT_GE(kappa, 0.0);
}

// ==================== computeAlpha_J 测试 ====================
TEST_F(HessianProjectionTest, ComputeAlphaJ_Method1_PositiveJ) {
    double J = 1.2;
    double alpha_J = TinyAD::computeAlpha_J(J, 1e-8);
    // J=1.2 在拉伸侧，应该得到合理的 alpha_J
    EXPECT_LE(alpha_J, 1.0);
    EXPECT_GE(alpha_J, 0.0);
}

TEST_F(HessianProjectionTest, ComputeAlphaJ_Method1_NegativeJ) {
    double J = -0.5;
    double alpha_J = TinyAD::computeAlpha_J(J, 1e-8);
    // J<0 时应该激活 safe 因子，alpha_J 可能较小
    EXPECT_LE(alpha_J, 1.0);
    EXPECT_GE(alpha_J, 0.0);
}

TEST_F(HessianProjectionTest, ComputeAlphaJ_Method2) {
    g_j_mode = 2;
    double J = 1.8;
    double alpha_J = TinyAD::computeAlpha_J(J, 1e-8);
    // J=1.8 > J_threshold=1.5
    // alpha_J = (1.8-1.5)/(2.5-1.5)=0.3
    EXPECT_NEAR(alpha_J, 0.3, 1e-4);
}

TEST_F(HessianProjectionTest, ComputeAlphaJ_Method2_NegativeJ) {
    g_j_mode = 2;
    double J = -0.5;
    double alpha_J = TinyAD::computeAlpha_J(J, 1e-8);
    // J<0: alpha_J = |1+J| = 0.5
    EXPECT_NEAR(alpha_J, 0.5, 1e-4);
}

TEST_F(HessianProjectionTest, ComputeAlphaJ_Method2_SmallJ) {
    g_j_mode = 2;
    double J = 0.5;
    double alpha_J = TinyAD::computeAlpha_J(J, 1e-8);
    // J < J_threshold=1.5, alpha_J = 0
    EXPECT_DOUBLE_EQ(alpha_J, 0.0);
}

TEST_F(HessianProjectionTest, ComputeAlphaJ_Method3) {
    g_j_mode = 3;
    double J = 1.2;
    double alpha_J = TinyAD::computeAlpha_J(J, 1e-8);
    // alpha = (1.2-0.9)^2 = 0.09
    EXPECT_NEAR(alpha_J, 0.09, 1e-4);
}

TEST_F(HessianProjectionTest, ComputeAlphaJ_Default) {
    g_j_mode = 99;  // 触发 default
    double J = 1.2;
    double alpha_J = TinyAD::computeAlpha_J(J, 1e-8);
    EXPECT_LE(alpha_J, 1.0);
    EXPECT_GE(alpha_J, 0.0);
}

// ==================== updateGamma 测试 ====================
TEST_F(HessianProjectionTest, UpdateGamma_Method1) {
    g_update_gamma_mode = 1;
    double gamma = 0.5;
    double J = 0.8;
    TinyAD::updateGamma(gamma, J);
    // gamma = |J|^(1/3) / g_MU = 0.9283 / 1.0 = 0.9283
    EXPECT_NEAR(gamma, std::pow(0.8, 1.0/3.0), 1e-4);
}

TEST_F(HessianProjectionTest, UpdateGamma_Method2) {
    g_update_gamma_mode = 2;
    double gamma = 0.5;
    double J = 0.8;
    double coeff = std::pow(0.8, 1.0/3.0);
    double expected = 0.5 * coeff;
    TinyAD::updateGamma(gamma, J);
    EXPECT_NEAR(gamma, expected, 1e-4);
}

TEST_F(HessianProjectionTest, UpdateGamma_Default) {
    g_update_gamma_mode = 99;
    double gamma = 0.5;
    double original_gamma = gamma;
    double J = 0.8;
    TinyAD::updateGamma(gamma, J);
    // default: gamma 保持不变
    EXPECT_DOUBLE_EQ(gamma, original_gamma);
}

// ==================== computeAlpha_hessian 测试 ====================
TEST(ComputeAlphaHessianTest, AllPositiveEigenvalues) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << 1.0, 2.0, 3.0;
    
    double alpha = TinyAD::computeAlpha_hessian(eigenvalues, 1e-8);
    // sum_lambda=6, sum_neg_lambda=0, r=0, alpha=0
    EXPECT_DOUBLE_EQ(alpha, 0.0);
}

TEST(ComputeAlphaHessianTest, AllNegativeEigenvalues) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -1.0, -2.0, -3.0;
    
    double alpha = TinyAD::computeAlpha_hessian(eigenvalues, 1e-8);
    // sum_lambda=6, sum_neg_lambda=6, r=1, alpha=1-exp(-4)=0.9817
    EXPECT_NEAR(alpha, 0.981684, 1e-4);
}

TEST(ComputeAlphaHessianTest, MixedSignEigenvalues) {
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -2.0, 1.0, 3.0;
    
    double alpha = TinyAD::computeAlpha_hessian(eigenvalues, 1e-8);
    // sum_lambda=6, sum_neg_lambda=2, r=0.333, alpha=1-exp(-1.333)=0.736
    EXPECT_NEAR(alpha, 0.736, 1e-3);
}

TEST(ComputeAlphaHessianTest, EmptyEigenvalues) {
    Eigen::VectorXd eigenvalues(0);
    double alpha = TinyAD::computeAlpha_hessian(eigenvalues, 1e-8);
    // sum_lambda=0, r=0/(0+eps)=0, alpha=0
    EXPECT_DOUBLE_EQ(alpha, 0.0);
}

// ==================== optimizeVpnEigenvalue_neg 测试 ====================
class OptimizeNegTest : public ::testing::Test {
protected:
    void SetUp() override {
        original_g_neg_mode = g_neg_mode;
        g_neg_mode = 1;
    }
    void TearDown() override {
        g_neg_mode = original_g_neg_mode;
    }
private:
    int original_g_neg_mode;
};

TEST_F(OptimizeNegTest, Mode1_GradOnly) {
    g_neg_mode = 1;
    double result = TinyAD::optimizeVpnEigenvalue_neg(-5.0, 0.8, 0.5, 0.6, g_neg_mode, 1e-8);
    // alpha_lambda = 0.8, b=5, eps=0.2, result=0.8*5+0.2*eps=4.0
    EXPECT_NEAR(result, 4.0, 1e-6);
}

TEST_F(OptimizeNegTest, Mode2_HessianOnly) {
    g_neg_mode = 2;
    double result = TinyAD::optimizeVpnEigenvalue_neg(-5.0, 0.8, 0.6, 0.6, g_neg_mode, 1e-8);
    // alpha_lambda = 0.6, result=3.0
    EXPECT_NEAR(result, 3.0, 1e-6);
}

TEST_F(OptimizeNegTest, Mode5_Absolute) {
    g_neg_mode = 5;
    double result = TinyAD::optimizeVpnEigenvalue_neg(-5.0, 0.8, 0.5, 0.6, g_neg_mode, 1e-8);
    // alpha_lambda = 1.0, result=5.0
    EXPECT_NEAR(result, 5.0, 1e-6);
}

TEST_F(OptimizeNegTest, DefaultMode_Clamp) {
    g_neg_mode = 99;
    double result = TinyAD::optimizeVpnEigenvalue_neg(-5.0, 0.8, 0.5, 0.6, g_neg_mode, 1e-8);
    // alpha_lambda = 0, result=eps
    EXPECT_NEAR(result, 1e-8, 1e-12);
}

// ==================== optimizeVpnEigenvalue_pos 测试 ====================
class OptimizePosTest : public ::testing::Test {
protected:
    void SetUp() override {
        original_g_pos_mode = g_pos_mode;
        g_pos_mode = 1;
    }
    void TearDown() override {
        g_pos_mode = original_g_pos_mode;
    }
private:
    int original_g_pos_mode;
};

TEST_F(OptimizePosTest, Mode1_FixedWithEps) {
    g_pos_mode = 1;
    double result = TinyAD::optimizeVpnEigenvalue_pos(5.0, 0.8, 0.5, 0.6, g_pos_mode, 1e-8);
    // alpha_lambda=0.8, result=4.0, eps=0.2, 4.0+0.2*eps≈4.0
    EXPECT_NEAR(result, 4.0, 1e-6);
}

TEST_F(OptimizePosTest, Mode2_FixedNoEps) {
    g_pos_mode = 2;
    double result = TinyAD::optimizeVpnEigenvalue_pos(5.0, 0.8, 0.5, 0.6, g_pos_mode, 1e-8);
    // alpha_lambda=0.8, result=4.0, eps=0.0
    EXPECT_DOUBLE_EQ(result, 4.0);
}

TEST_F(OptimizePosTest, DefaultMode) {
    g_pos_mode = 99;
    double result = TinyAD::optimizeVpnEigenvalue_pos(5.0, 0.8, 0.5, 0.6, g_pos_mode, 1e-8);
    // default: alpha_lambda=1.0, result=5.0
    EXPECT_DOUBLE_EQ(result, 5.0);
}

// ==================== optimizeVpnEigenvalue 测试 ====================
class OptimizeEigenvalueTest : public ::testing::Test {
protected:
    void SetUp() override {
        original_g_pos_mode = g_pos_mode;
        original_g_neg_mode = g_neg_mode;
        g_pos_mode = 1;
        g_neg_mode = 1;
    }
    void TearDown() override {
        g_pos_mode = original_g_pos_mode;
        g_neg_mode = original_g_neg_mode;
    }
private:
    int original_g_pos_mode;
    int original_g_neg_mode;
};

TEST_F(OptimizeEigenvalueTest, PositiveLambda) {
    double result = TinyAD::optimizeVpnEigenvalue(5.0, 0.8, 0.5, 0.6, 1e-8);
    // 调用 optimizeVpnEigenvalue_pos，mode=1，result≈4.0
    EXPECT_NEAR(result, 4.0, 1e-6);
}

TEST_F(OptimizeEigenvalueTest, NegativeLambda) {
    double result = TinyAD::optimizeVpnEigenvalue(-5.0, 0.8, 0.5, 0.6, 1e-8);
    // 调用 optimizeVpnEigenvalue_neg，mode=1，result≈4.0
    EXPECT_NEAR(result, 4.0, 1e-6);
}

TEST_F(OptimizeEigenvalueTest, NearZeroLambda) {
    double result = TinyAD::optimizeVpnEigenvalue(1e-9, 0.8, 0.5, 0.6, 1e-8);
    // |lambda| <= m_eps, 返回 m_eps
    EXPECT_DOUBLE_EQ(result, 1e-8);
}

// ==================== computeGradientProjection_vpn 测试 ====================
TEST(ComputeGradientProjectionTest, BasicFunctionality) {
    Eigen::MatrixXd eigenvectors(2, 2);
    eigenvectors << 1.0, 0.0,
                    0.0, 1.0;
    Eigen::VectorXd g(2);
    g << 2.0, 3.0;
    
    Eigen::VectorXd result = TinyAD::computeGradientProjection_vpn(eigenvectors, g);
    // 单位矩阵，投影等于原向量
    EXPECT_DOUBLE_EQ(result(0), 2.0);
    EXPECT_DOUBLE_EQ(result(1), 3.0);
}

TEST(ComputeGradientProjectionTest, RotationMatrix) {
    Eigen::MatrixXd eigenvectors(2, 2);
    eigenvectors << 0.0, 1.0,
                    1.0, 0.0;
    Eigen::VectorXd g(2);
    g << 2.0, 3.0;
    
    Eigen::VectorXd result = TinyAD::computeGradientProjection_vpn(eigenvectors, g);
    // 转置后，坐标交换
    EXPECT_DOUBLE_EQ(result(0), 3.0);
    EXPECT_DOUBLE_EQ(result(1), 2.0);
}

// ==================== computeAlpha_grad_eta 测试 ====================
TEST_F(HessianProjectionTest, ComputeAlphaGradEta_Mode1) {
    g_eta_mode = 1;
    double eta_pos = 0, eta_neg = 0;
    double Kappa = 0.8;
    TinyAD::computeAlpha_grad_eta(eta_pos, eta_neg, Kappa);
    EXPECT_NEAR(eta_pos, 0.7 * 0.8, 1e-6);
    EXPECT_NEAR(eta_neg, 0.2 * 0.8, 1e-6);
}

TEST_F(HessianProjectionTest, ComputeAlphaGradEta_Mode2) {
    g_eta_mode = 2;
    double eta_pos = 0, eta_neg = 0;
    TinyAD::computeAlpha_grad_eta(eta_pos, eta_neg, 0.8);
    EXPECT_DOUBLE_EQ(eta_pos, 0.7);
    EXPECT_DOUBLE_EQ(eta_neg, 0.2);
}

TEST_F(HessianProjectionTest, ComputeAlphaGradEta_Default) {
    g_eta_mode = 99;
    double eta_pos = 0, eta_neg = 0;
    TinyAD::computeAlpha_grad_eta(eta_pos, eta_neg, 0.8);
    EXPECT_DOUBLE_EQ(eta_pos, 0.5);
    EXPECT_DOUBLE_EQ(eta_neg, 0.5);
}

// ==================== computeAlpha_grad_kappa 测试 ====================
TEST_F(HessianProjectionTest, ComputeAlphaGradKappa_Mode1) {
    g_kappa_mode = 1;
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -2.0, 1.0, 3.0;
    double kappa = TinyAD::computeAlpha_grad_kappa(eigenvalues, 1e-8, 1.0);
    EXPECT_LE(kappa, 1.0);
    EXPECT_GE(kappa, 0.0);
}

TEST_F(HessianProjectionTest, ComputeAlphaGradKappa_Mode2) {
    g_kappa_mode = 2;
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -2.0, 1.0, 3.0;
    double kappa = TinyAD::computeAlpha_grad_kappa(eigenvalues, 1e-8, 0.6);
    EXPECT_DOUBLE_EQ(kappa, 0.6);
}

TEST_F(HessianProjectionTest, ComputeAlphaGradKappa_Mode3) {
    g_kappa_mode = 3;
    Eigen::VectorXd eigenvalues(3);
    eigenvalues << -2.0, 1.0, 3.0;
    double kappa = TinyAD::computeAlpha_grad_kappa(eigenvalues, 1e-8, 0.3);
    // Kappa = max(computeKappa, 0.3)
    EXPECT_GE(kappa, 0.3);
    EXPECT_LE(kappa, 1.0);
}

// ==================== computeAlpha_grad 测试 ====================
TEST_F(HessianProjectionTest, ComputeAlphaGrad_Basic) {
    g_grad_mode = 1;
    double alpha_grad_pos = 0, alpha_grad_neg = 0;
    int k = 2;
    Eigen::VectorXd proj_g(2);
    proj_g << 1.0, 2.0;
    Eigen::VectorXd eigenvalues(2);
    eigenvalues << -1.0, 2.0;
    double gamma = 0.5;
    
    TinyAD::computeAlpha_grad(alpha_grad_pos, alpha_grad_neg, k, proj_g, eigenvalues, gamma, 1.0, 1e-8);
    
    EXPECT_LE(alpha_grad_pos, 1.0);
    EXPECT_GE(alpha_grad_pos, 0.0);
    EXPECT_LE(alpha_grad_neg, 1.0);
    EXPECT_GE(alpha_grad_neg, 0.0);
}

// 主函数（如果测试文件独立编译）
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}