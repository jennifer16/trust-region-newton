// ./tests/test_mesh_ops
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "mesh_ops.h"
#include <cmath>
#include <iostream>

// #ifndef DEBUG_OUTPUT
//     #define DEBUG_OUTPUT 1
// #endif


class FSFEMDataTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建两个共享面的四面体
        // 四面体0: (0,1,2,3) - 顶点0,1,2,3
        // 四面体1: (4,2,1,3) - 顶点4,2,1,3 (共享面1-2-3)
        V.resize(5, 3);
        V << 0.0, 0.0, 0.0,  // v0
             1.0, 0.0, 0.0,  // v1
             0.0, 1.0, 0.0,  // v2
             0.0, 0.0, 1.0,  // v3
             1.0, 1.0, 0.0;  // v4
        
        T.resize(2, 4);
        T << 0, 1, 2, 3,  // tet0
             4, 2, 1, 3;  // tet1
        
        // 创建第二个测试用例：一个四面体（边界面测试）
        V_single.resize(4, 3);
        V_single << 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0;
        
        T_single.resize(1, 4);
        T_single << 0, 1, 2, 3;
        
        // 创建第三个测试用例：三个四面体组成的复杂网格
        V_complex.resize(6, 3);
        V_complex << 0.0, 0.0, 0.0,  // v0
                     1.0, 0.0, 0.0,  // v1
                     0.0, 1.0, 0.0,  // v2
                     0.0, 0.0, 1.0,  // v3
                     1.0, 1.0, 0.0,  // v4
                     0.0, 1.0, 1.0;  // v5
        
        T_complex.resize(3, 4);
        T_complex << 0, 1, 2, 3,  // tet0
                     4, 2, 1, 3,  // tet1
                     0, 3, 1, 5;  // tet2
    }
    
    // 测试数据
    Eigen::MatrixXd V;
    Eigen::MatrixXi T;
    
    Eigen::MatrixXd V_single;
    Eigen::MatrixXi T_single;
    
    Eigen::MatrixXd V_complex;
    Eigen::MatrixXi T_complex;
    
    FSFEM_5NodeData data;
    
    // 辅助函数：打印调试信息
    void print_debug_info(const FSFEM_5NodeData& d) {
        std::cout << "=== Debug Info ===" << std::endl;
        std::cout << "Number of faces: " << d.face_vertices.rows() << std::endl;
        std::cout << "Number of face_to_tets: " << d.face_to_tets.size() << std::endl;
        
        for (int i = 0; i < d.face_vertices.rows() && i < 5; i++) {
            std::cout << "Face " << i << ": ["
                      << d.face_vertices(i,0) << ", "
                      << d.face_vertices(i,1) << ", "
                      << d.face_vertices(i,2) << ", "
                      << d.face_vertices(i,3) << ", "
                      << d.face_vertices(i,4) << "] tets: ";
            for (auto t : d.face_to_tets[i]) {
                std::cout << t << " ";
            }
            std::cout << std::endl;
        }
    }
};

// ============ 测试1: 面邻接关系构建 ============
TEST_F(FSFEMDataTest, BuildFaceAdjacency) {
    data.build_face_to_tet_adjacency(T);
    
    // 2个四面体，每个4个面，共享1个面 -> 7个唯一面
    EXPECT_EQ(data.face_vertices.rows(), 7);
    EXPECT_EQ(data.face_to_tets.size(), 7);
    
    // 统计内部面和边界面数量
    int internal_count = 0;
    int boundary_count = 0;
    
    for (const auto& tets : data.face_to_tets) {
        if (tets.size() == 2) {
            internal_count++;
        } else if (tets.size() == 1) {
            boundary_count++;
        }
    }
    
    EXPECT_EQ(internal_count, 1);  // 1个内部面
    EXPECT_EQ(boundary_count, 6);  // 6个边界面
    
    // 找到共享面(1,2,3)
    int shared_face_idx = -1;
    for (int i = 0; i < data.face_vertices.rows(); i++) {
        std::set<int> vertices = {
            data.face_vertices(i,1),
            data.face_vertices(i,2),
            data.face_vertices(i,3)
        };
        if (vertices.count(1) && vertices.count(2) && vertices.count(3)) {
            shared_face_idx = i;
            break;
        }
    }
    
    ASSERT_NE(shared_face_idx, -1);
    EXPECT_EQ(data.face_vertices(shared_face_idx, 0), 0);  // v0 = tet0的对顶点
    EXPECT_EQ(data.face_vertices(shared_face_idx, 4), 4);  // v4 = tet1的对顶点
    EXPECT_EQ(data.face_to_tets[shared_face_idx].size(), 2);
    
    // 验证所有顶点索引有效
    for (int i = 0; i < data.face_vertices.rows(); i++) {
        for (int j = 0; j < 5; j++) {
            if (j == 4 && data.face_to_tets[i].size() == 1) {
                // 边界面，v4应该是-1
                EXPECT_EQ(data.face_vertices(i,4), -1);
            } else {
                // 所有顶点索引应该在0-4之间
                EXPECT_GE(data.face_vertices(i,j), 0);
                EXPECT_LE(data.face_vertices(i,j), 4);
            }
        }
    }
}

// ============ 测试2: 体积和逆矩阵计算 ============
TEST_F(FSFEMDataTest, ComputeVolumesAndInverse) {
    data.build_face_to_tet_adjacency(T);
    data.compute_tet_volumes_Mr_inv(V, T);
    
    double expected_vol = 1.0 / 6.0;  // 单位四面体体积
    
    // 验证所有面的体积和权重
    for (int f_idx = 0; f_idx < data.face_vertices.rows(); f_idx++) {
        int num_tets = data.face_to_tets[f_idx].size();
        
        // 验证权重
        if (num_tets == 2) {
            EXPECT_NEAR(data.adj_tet_volume[f_idx][0], 0.25 * expected_vol, 1e-12);
            EXPECT_NEAR(data.adj_tet_volume[f_idx][1], 0.25 * expected_vol, 1e-12);
            EXPECT_NEAR(data.domain_volumes[f_idx], expected_vol * 0.5, 1e-12);
        } else if (num_tets == 1) {
            EXPECT_NEAR(data.adj_tet_volume[f_idx][0], 0.25 * expected_vol, 1e-12);
            EXPECT_EQ(data.adj_tet_volume[f_idx][1], 0.0);
            EXPECT_NEAR(data.domain_volumes[f_idx], expected_vol*0.25, 1e-12);
        }
        
        // 验证逆矩阵正确性 (Dm * Dm_inv = I)
        for (int i = 0; i < num_tets; i++) {
            int t_idx = data.face_to_tets[f_idx][i];
            auto vertices = data.face_vertices.row(f_idx);
            // 重建Dm矩阵
            Eigen::Vector3d a, b, c, d;
            if (i == 0) {
                a = V.row(vertices(0));
                b = V.row(vertices(1));
                c = V.row(vertices( 2));
                d = V.row(vertices( 3));
            } else {
                a = V.row(vertices(4));
                b = V.row(vertices(3));
                c = V.row(vertices(2));
                d = V.row(vertices(1));
            }
            
            Eigen::Matrix3d Dm;
            Dm.col(0) = b - a;
            Dm.col(1) = c - a;
            Dm.col(2) = d - a;
            
            Eigen::Matrix3d I = Dm * data.Mr_inv[f_idx][i];
            
            // 检查是否为单位矩阵
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    double expected = (r == c) ? 1.0 : 0.0;
                    EXPECT_NEAR(I(r,c), expected, 1e-12);
                }
            }
        }
    }
}

// ============ 测试3: 单四面体网格（边界面测试）============
TEST_F(FSFEMDataTest, SingleTetrahedron) {
    data.build_5node_fsfem_data(V_single, T_single);
    
    // 单个四面体有4个面，都是边界面
    EXPECT_EQ(data.face_vertices.rows(), 4);
    EXPECT_EQ(data.face_to_tets.size(), 4);
    
    double expected_vol = 0.25* 1.0 / 6.0;
    
    // 验证所有面都是边界面
    for (int f_idx = 0; f_idx < data.face_vertices.rows(); f_idx++) {
        EXPECT_EQ(data.face_to_tets[f_idx].size(), 1);
        EXPECT_EQ(data.face_vertices(f_idx,4), -1);  // v4应为-1
        
        // 验证权重
        EXPECT_NEAR(data.adj_tet_volume[f_idx][0], expected_vol, 1e-12);
        EXPECT_EQ(data.adj_tet_volume[f_idx][1], 0.0);
        EXPECT_NEAR(data.domain_volumes[f_idx], expected_vol, 1e-12);
        
        // 验证第二个逆矩阵为零矩阵
        EXPECT_TRUE(data.Mr_inv[f_idx][1].isZero(1e-12));
    }
}

// ============ 测试4: 复杂网格测试 ============
TEST_F(FSFEMDataTest, ComplexMesh) {
    data.build_5node_fsfem_data(V_complex, T_complex);
    
    // 3个四面体，每个4面，假设有2个内部面 -> 12 - 2 = 10个唯一面
    EXPECT_EQ(data.face_vertices.rows(), 10);
    
    // 验证每个面都有有效的顶点索引
    for (int i = 0; i < data.face_vertices.rows(); i++) {
        for (int j = 0; j < 3; j++) {  // 面顶点应该在0-5之间
            EXPECT_GE(data.face_vertices(i, j+1), 0);
            EXPECT_LE(data.face_vertices(i, j+1), 5);
        }
        
        // 对顶点应该在0-5之间或者是-1
        if (data.face_to_tets[i].size() == 2) {
            EXPECT_GE(data.face_vertices(i, 0), 0);
            EXPECT_LE(data.face_vertices(i, 0), 5);
            EXPECT_GE(data.face_vertices(i, 4), 0);
            EXPECT_LE(data.face_vertices(i, 4), 5);
        } else {
            EXPECT_GE(data.face_vertices(i, 0), 0);
            EXPECT_LE(data.face_vertices(i, 0), 5);
            EXPECT_EQ(data.face_vertices(i, 4), -1);
        }
    }
    
    // 验证所有体积为正
    for (double vol : data.domain_volumes) {
        EXPECT_GT(vol, 0.0);
    }
    
    // 验证所有权重之和有意义
    double total_weight_sum = 0.0;
    for (const auto& w : data.adj_tet_volume) {
        total_weight_sum += w[0] + w[1];
    }
    EXPECT_GT(total_weight_sum, 0.0);
}

// ============ 测试5: 逆矩阵属性测试 ============
TEST_F(FSFEMDataTest, InverseMatrixProperties) {
    data.build_5node_fsfem_data(V, T);
    
    for (int f_idx = 0; f_idx < data.face_vertices.rows(); f_idx++) {
        for (int i = 0; i < data.face_to_tets[f_idx].size(); i++) {
            const Eigen::Matrix3d& inv = data.Mr_inv[f_idx][i];
            
            
            // 验证逆矩阵的行列式应该接近1/det(Dm)
            int t_idx = data.face_to_tets[f_idx][i];
            std::cout<< f_idx <<","<< i <<","<< t_idx << std::endl;

            // 重建Dm
            auto vertices = data.face_vertices.row(f_idx);
            Eigen::Vector3d a, b, c, d;
            if (i == 0) {
                a = V.row(vertices(0));
                b = V.row(vertices(1));
                c = V.row(vertices(2));
                d = V.row(vertices(3));
            } else {
                a = V.row(vertices(4));
                b = V.row(vertices(3));
                c = V.row(vertices(2));
                d = V.row(vertices(1));
            }
            
            Eigen::Matrix3d Dm;
            Dm.col(0) = b - a;
            Dm.col(1) = c - a;
            Dm.col(2) = d - a;
            
            double det_Dm = Dm.determinant();
            double det_inv = inv.determinant();
            
            EXPECT_NEAR(det_inv, 1.0 / det_Dm, 1e-10);
        }
    }
}

// ============ 测试6: 退化网格检测 ============
TEST_F(FSFEMDataTest, DegenerateMesh) {
    // 创建退化四面体（四点共面）
    Eigen::MatrixXd V_degenerate(4, 3);
    V_degenerate << 0,0,0, 1,0,0, 0,1,0, 1,1,0;  // 所有z=0
    
    Eigen::MatrixXi T_degenerate(1, 4);
    T_degenerate << 0, 1, 2, 3;
    
    FSFEM_5NodeData bad_data;
    bad_data.build_face_to_tet_adjacency(T_degenerate);
    
    // 计算体积时应检测到退化
    EXPECT_DEATH(bad_data.compute_tet_volumes_Mr_inv(V_degenerate, T_degenerate), ".*");
}

// ============ 测试7: 完整工作流程测试 ============
TEST_F(FSFEMDataTest, FullWorkflow) {
    // 测试完整的build_5node_fsfem_data函数
    EXPECT_NO_THROW(data.build_5node_fsfem_data(V, T));
    
    // 验证数据结构完整性
    EXPECT_EQ(data.face_vertices.rows(), data.face_to_tets.size());
    EXPECT_EQ(data.Mr_inv.size(), data.face_to_tets.size());
    EXPECT_EQ(data.adj_tet_volume.size(), data.face_to_tets.size());
    EXPECT_EQ(data.domain_volumes.size(), data.face_to_tets.size());
    
    // 验证每个面的数据结构一致
    for (int f_idx = 0; f_idx < data.face_vertices.rows(); f_idx++) {
        int num_tets = data.face_to_tets[f_idx].size();
        EXPECT_TRUE(num_tets == 1 || num_tets == 2);
        
        // 验证对顶点的存在性
        if (num_tets == 2) {
            // 内部面：v0和v4都应该是有效顶点
            EXPECT_GE(data.face_vertices(f_idx,0), 0);
            EXPECT_LE(data.face_vertices(f_idx,0), 4);
            EXPECT_GE(data.face_vertices(f_idx,4), 0);
            EXPECT_LE(data.face_vertices(f_idx,4), 4);
        } else {
            // 边界面：v0有效，v4为-1
            EXPECT_GE(data.face_vertices(f_idx,0), 0);
            EXPECT_LE(data.face_vertices(f_idx,0), 4);
            EXPECT_EQ(data.face_vertices(f_idx,4), -1);
        }
        
        // 面顶点应该总是有效
        for (int j = 1; j < 4; j++) {
            EXPECT_GE(data.face_vertices(f_idx,j), 0);
            EXPECT_LE(data.face_vertices(f_idx,j), 4);
        }
    }
}

// ============ 测试8: 批量测试多个四面体 ============
TEST_F(FSFEMDataTest, BatchTest) {
    // 生成多个随机四面体进行测试
    const int num_tests = 6;
    
    for (int test = 0; test < num_tests; test++) {
        // 创建随机四面体
        Eigen::MatrixXd V_rand(5 + test, 3);
        Eigen::MatrixXi T_rand(test + 1, 4);
        
        // 填充随机顶点
        for (int i = 0; i < V_rand.rows(); i++) {
            V_rand.row(i) = Eigen::Vector3d::Random();
        }
        
        // 创建随机四面体（确保不退化）
        for (int i = 0; i < T_rand.rows(); i++) {
            T_rand.row(i) << (i*4) % V_rand.rows(),
                              (i*4+1) % V_rand.rows(),
                              (i*4+2) % V_rand.rows(),
                              (i*4+3) % V_rand.rows();
        }

        
        FSFEM_5NodeData test_data;
        
        // 验证构建过程不会崩溃
        std::cout<<"-->EXPECT_NO_THROW"<<std::endl;
        EXPECT_NO_THROW(test_data.build_5node_fsfem_data(V_rand, T_rand));
        std::cout<<"<--EXPECT_NO_THROW"<<std::endl;
        
        // 验证基本属性
        EXPECT_GT(test_data.face_vertices.rows(), 0);
        EXPECT_GT(test_data.domain_volumes.size(), 0);
        
        // 验证所有权重为正
        for (const auto& w : test_data.adj_tet_volume) {
            EXPECT_GE(w[0], 0.0);
            EXPECT_GE(w[1], 0.0);
        }
    }
}

// ============ 主函数 ============
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}