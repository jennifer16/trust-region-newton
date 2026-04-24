// ./tests/test_cubic_regular
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <sstream>
#include "cubic_regular.h"


class CubicRegularTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置测试数据
        g.resize(3);
        g << 1.0, 2.0, 3.0;
        
        d.resize(3);
        d << 0.1, 0.2, 0.3;
        
        H.resize(3, 3);
        H.insert(0,0) = 4.0;
        H.insert(0,1) = 1.0;
        H.insert(0,2) = 0.0;
        H.insert(1,0) = 1.0;
        H.insert(1,1) = 4.0;
        H.insert(1,2) = 1.0;
        H.insert(2,0) = 0.0;
        H.insert(2,1) = 1.0;
        H.insert(2,2) = 4.0;
    }

    Eigen::VectorXd g;  // 梯度
    Eigen::VectorXd d;  // 搜索方向
    Eigen::SparseMatrix<double> H;  // 海森矩阵
};

// 测试构造函数和默认值
TEST_F(CubicRegularTest, ConstructorAndDefaults) {
    Cubic_Regular cr;
    
    EXPECT_EQ(cr.get_strategy(), SIGMA_STRATEGIES::FIXED);
    EXPECT_EQ(cr.get_rho_upper_threshhold(), 0.75);
    EXPECT_EQ(cr.get_rho_lower_threshhold(), 0.25);
    EXPECT_EQ(cr.get_level(), 0);
}

// 测试带参数的构造函数
TEST_F(CubicRegularTest, ParameterizedConstructor) {
    Cubic_Regular cr(SIGMA_STRATEGIES::ADAPTIVE, 0.5, 0.1, 2.0, 3);
    
    EXPECT_EQ(cr.get_strategy(), SIGMA_STRATEGIES::ADAPTIVE);
    EXPECT_EQ(cr.get_rho_upper_threshhold(), 0.5);
    EXPECT_EQ(cr.get_rho_lower_threshhold(), 0.1);
    EXPECT_EQ(cr.get_sigma(), 2.0);
    EXPECT_EQ(cr.get_level(), 3);
}

// 测试compute_ratio_rho方法
TEST_F(CubicRegularTest, ComputeRatioRho) {
    Cubic_Regular cr;
    double e0 = 10.0;
    double e1 = 8.0;
    double sigma = 1.0;
    
    double rho = cr.compute_ratio_rho(e1, e0, d, g, H, sigma);
    
    // 验证rho被正确计算和存储
    //EXPECT_GT(rho, 0);
    EXPECT_EQ(cr.get_rho(), rho);
    
    // // 验证异常情况
    // EXPECT_THROW({
    //     Eigen::VectorXd wrong_size(2);
    //     cr.compute_ratio_rho(e1, e0, wrong_size, g, H, sigma);
    // }, std::exception);
}

// 测试FIXED策略
TEST_F(CubicRegularTest, FixedStrategy) {
    Cubic_Regular cr(SIGMA_STRATEGIES::FIXED, 0.75, 0.25, 2.0, 2);
    
    // 对于FIXED策略，update_sigma应该返回多级启动值
    double sigma = cr.update_sigma();
    EXPECT_NEAR(sigma, 0.1, 1e-10);  // level 2对应0.1
    
    cr.set_level(1);
    sigma = cr.update_sigma();
    EXPECT_NEAR(sigma, 1.0, 1e-10);  // level 1对应1.0
    
    cr.set_level(5);
    sigma = cr.update_sigma();
    EXPECT_NEAR(sigma, 100.0, 1e-10);  // level 5对应100.0
}

// 测试ADAPTIVE策略
TEST_F(CubicRegularTest, AdaptiveStrategy) {
    Cubic_Regular cr(SIGMA_STRATEGIES::ADAPTIVE, 0.7, 0.3, 2.0);
    double old_sigma = 10.0;
    
    // 测试rho大于上阈值（应该减小sigma）
    cr.set_rho(0.8);
    double new_sigma = cr.update_sigma(old_sigma);
    EXPECT_NEAR(new_sigma, old_sigma / 2.0, 1e-10);
    
    // 测试rho小于下阈值（应该增大sigma）
    cr.set_rho(0.2);
    new_sigma = cr.update_sigma(old_sigma);
    EXPECT_NEAR(new_sigma, old_sigma * 2.0, 1e-10);
    
    // 测试rho在阈值之间（sigma不变）
    cr.set_rho(0.5);
    new_sigma = cr.update_sigma(old_sigma);
    EXPECT_NEAR(new_sigma, old_sigma, 1e-10);
}

// 测试DEFORM策略
TEST_F(CubicRegularTest, DeformStrategy) {
    Cubic_Regular cr(SIGMA_STRATEGIES::DEFORM);
    
    // 大变形
    double sigma = cr.update_sigma(0.6);
    EXPECT_NEAR(sigma, 100.0, 1e-10);
    
    // 中等变形
    sigma = cr.update_sigma(0.4);
    EXPECT_NEAR(sigma, 10.0, 1e-10);
    
    // 小变形
    sigma = cr.update_sigma(0.2);
    EXPECT_NEAR(sigma, 1.0, 1e-10);
}

// 测试GRADIENT策略
TEST_F(CubicRegularTest, GradientStrategy) {
    Cubic_Regular cr(SIGMA_STRATEGIES::GRADIENT);
    
    double typical_gradient = 5.0;
    double sigma = cr.update_sigma(typical_gradient);
    
    // sigma应该与梯度相关 (0.1 * typical_gradient)
    EXPECT_NEAR(sigma, 0.5, 1e-10);
}

// 测试HESSIAN策略
TEST_F(CubicRegularTest, HessianStrategy) {
    Cubic_Regular cr(SIGMA_STRATEGIES::HESSIAN);
    
    double gradient_norm = 10.0;
    double sigma = cr.update_sigma(gradient_norm);
    
    EXPECT_NEAR(sigma, 10.0, 1e-10);  // 应该是gradient_norm
}

// 测试ENERGY策略
TEST_F(CubicRegularTest, EnergyStrategy) {
    Cubic_Regular cr(SIGMA_STRATEGIES::ENERGY);
    
    double energy = 100.0;
    double size = 20.0;
    double sigma = cr.update_sigma(energy, size);
    
    EXPECT_NEAR(sigma, 5.0, 1e-10);  // energy / size
    
    // // 测试除零保护
    // EXPECT_THROW({
    //     cr.update_sigma(energy, 0.0);
    // }, std::exception);
}

// 测试getter和setter方法
TEST_F(CubicRegularTest, GettersAndSetters) {
    Cubic_Regular cr;
    
    // 测试rho相关
    cr.set_rho(0.5);
    EXPECT_EQ(cr.get_rho(), 0.5);
    
    // 测试sigma相关
    cr.set_sigma(3.0);
    EXPECT_EQ(cr.get_sigma(), 3.0);
    
    // 测试level相关
    cr.set_level(5);
    EXPECT_EQ(cr.get_level(), 5);
    
    // 测试策略相关
    cr.set_strategy(SIGMA_STRATEGIES::HESSIAN);
    EXPECT_EQ(cr.get_strategy(), SIGMA_STRATEGIES::HESSIAN);
    
    // 测试阈值相关
    cr.set_rho_upper_threshhold(0.8);
    EXPECT_EQ(cr.get_rho_upper_threshhold(), 0.8);
    
    cr.set_rho_lower_threshhold(0.2);  // 注意：这里函数名有误，实际设置lower
    EXPECT_EQ(cr.get_rho_lower_threshhold(), 0.2);
}

// 测试边界条件
TEST_F(CubicRegularTest, BoundaryConditions) {
    Cubic_Regular cr;
    
    // 测试rho等于阈值边界
    cr.set_strategy(SIGMA_STRATEGIES::ADAPTIVE);
    cr.set_rho_upper_threshhold(0.5);
    cr.set_rho_upper_threshhold(0.5);  // 设置lower阈值为0.5
    
    cr.set_rho(0.5);  // rho等于阈值
    double sigma = cr.update_sigma(10.0);
    EXPECT_NEAR(sigma, 10.0, 1e-10);  // 应该保持不变
    
    // 测试极端的level值
    cr.set_strategy(SIGMA_STRATEGIES::FIXED);
    cr.set_level(100);  // level超出范围
    sigma = cr.update_sigma();
    EXPECT_NEAR(sigma, 100.0, 1e-10);  // 应该使用最大level对应的值
}

// 测试不同策略的组合
TEST_F(CubicRegularTest, StrategyCombinations) {
    // 测试所有策略是否都能正常调用
    std::vector<SIGMA_STRATEGIES> strategies = {
        SIGMA_STRATEGIES::FIXED,
        SIGMA_STRATEGIES::ADAPTIVE,
        SIGMA_STRATEGIES::DEFORM,
        SIGMA_STRATEGIES::GRADIENT,
        SIGMA_STRATEGIES::ENERGY,
        SIGMA_STRATEGIES::HESSIAN
    };
    
    for (auto strategy : strategies) {
        Cubic_Regular cr(strategy);
        double sigma;
        
        switch (strategy) {
            case SIGMA_STRATEGIES::FIXED:
                sigma = cr.update_sigma();
                break;
            case SIGMA_STRATEGIES::ADAPTIVE:
                cr.set_rho(0.5);
                sigma = cr.update_sigma(10.0);
                break;
            case SIGMA_STRATEGIES::DEFORM:
                sigma = cr.update_sigma(0.3);
                break;
            case SIGMA_STRATEGIES::GRADIENT:
                sigma = cr.update_sigma(5.0);
                break;
            case SIGMA_STRATEGIES::ENERGY:
                sigma = cr.update_sigma(100.0, 10.0);
                break;
            case SIGMA_STRATEGIES::HESSIAN:
                sigma = cr.update_sigma(8.0);
                break;
        }
        
        EXPECT_GE(sigma, 0);  // sigma应该非负
    }
}

// 测试compute_ratio_rho的数值正确性
TEST_F(CubicRegularTest, ComputeRatioRhoNumerical) {
    Cubic_Regular cr;
    
    // 构造一个简单情况来验证数值计算
    Eigen::VectorXd g_simple(1);
    g_simple << 2.0;
    
    Eigen::VectorXd d_simple(1);
    d_simple << 0.5;
    
    Eigen::SparseMatrix<double> H_simple(1, 1);
    H_simple.insert(0,0) = 4.0;
    
    double e0 = 10.0;
    double e1 = 9.0;
    double sigma = 1.0;
    
    double d_norm = d_simple.norm();
    double expected_denominator = -(d_simple.dot(g_simple) + 
                                    0.5 * d_simple.transpose() * H_simple * d_simple + 
                                    std::pow(d_norm, 3) * sigma / 3.0);
    double expected_rho = (e0 - e1) / expected_denominator;
    
    double actual_rho = cr.compute_ratio_rho(e1, e0, d_simple, g_simple, H_simple, sigma);
    
    EXPECT_NEAR(actual_rho, expected_rho, 1e-10);
}

// 主函数（如果使用gtest单独编译）
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}