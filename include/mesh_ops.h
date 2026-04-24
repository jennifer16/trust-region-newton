#ifndef MESH_OPS_H
#define MESH_OPS_H

//#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//#include <vector>
#include <fstream>

// #ifndef DEBUG_OUTPUT_MESH
//     #define DEBUG_OUTPUT_MESH
// #endif

struct FSFEM_5NodeData {
    // 5顶点单元列表（只针对贡献面为(v1,v2,v3)的内部面）
    //std::vector<std::array<int, 5>> face_vertices;  // [v0, v1, v2, v3, v4]
    Eigen::MatrixXi face_vertices; // [v0, v1, v2, v3, v4]
    std::vector<std::vector<unsigned int>> face_to_tets; // [t0,[t1]]

    std::vector<std::array<double, 2>> adj_tet_volume;
    std::vector<double> domain_volumes;
    std::vector<std::array<Eigen::Matrix3d, 2>> Mr_inv;
       
    void build_5node_fsfem_data(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T)
    {   
        #if DEBUG_OUTPUT_MESH
            std::cout<< "build_face_to_tet_adjacency:" <<std::endl;
            std::cout<< "n_tets:"<< T.rows() <<std::endl;
            std::cout<< "n_vs:"<< V.rows() <<std::endl;
            for (int i = 0 ; i < T.rows(); i++)
            {
                std::cout<< T.row(i)[0]<< ","<<T.row(i)[1]<<","<<T.row(i)[2]<<","<<T.row(i)[3] <<std::endl;
            }
        #endif
        build_face_to_tet_adjacency(T); // 你需要实现这个函数，输出每个面对应的四面体列表和面的顶点列表
        #if DEBUG_OUTPUT_MESH
            std::cout<< "n_domains:"<<this->face_vertices.rows() << std::endl;
        #endif
        #if DEBUG_OUTPUT_MESH
            std::cout<< "compute_tet_volumes_Mr_inv:" <<std::endl;
        #endif
        compute_tet_volumes_Mr_inv(V, T); // 你需要实现这个函数，计算每个四面体的体积，并存储在data.tet_volumes中 
        #if DEBUG_OUTPUT_MESH
            std::cout<< "=====exit build_5node_fsfem_data()====" <<std::endl;
        #endif
    };

    // 输入：T 是 #T×4 的四面体顶点索引矩阵
    // 输出：face_to_tet[i] = 共享第i个面的四面体索引列表（1或2个) 
    //      face_5vertices[i] = 组成第i个面的5个顶点索引（v0,v1,v2,v3,v4)，其中v1-v3是共享点，v0和v4分别是两个四面体的第4个顶点
    // 2个四面体的顶点分别为v0 v1 v2 v3/ v4 v3 v2 v1
    void build_face_to_tet_adjacency(
        const Eigen::MatrixXi& T)
    {
        // 1. 将每个四面体的4个面（三角形）展开为有向边序列
        // 四面体的4个面对应的顶点三元组
        const int face_vertices_arrange[4][4] = {
            {1, 2, 3,0}, // 与顶点0相对的面
            {0, 3, 2,1}, // 与顶点1相对的面
            {0, 1, 3,2}, // 与顶点2相对的面
            {0, 2, 1,3}  // 与顶点3相对的面
        };

        Eigen::MatrixXi all_faces(T.rows() * 4, 4); // 所有面（每个四面体贡献4个三角形面）
        Eigen::VectorXi face_to_tet_idx(all_faces.rows()); // 每个面对应的四面体ID
        
        for (int i = 0; i < T.rows(); i++) {
            for (int j = 0; j < 4; j++) {
                int row = i * 4 + j;
                // 按规范顺序写入面顶点索引
                all_faces(row, 0) = T(i, face_vertices_arrange[j][0]); //F[i] = [v0,v1,v2], face_to_tet_idx[i] = tet_idx
                all_faces(row, 1) = T(i, face_vertices_arrange[j][1]);
                all_faces(row, 2) = T(i, face_vertices_arrange[j][2]);
                all_faces(row, 3) = T(i, face_vertices_arrange[j][3]); //非共享顶点
                face_to_tet_idx(row) = i;
            }
        }

        // 2. 构建face_to_tet和face_5vertices
        std::map<std::tuple<int,int,int>, int> face_map;//内部面对应的第一个tet
        std::vector<std::array<int, 5>> candidate_face_vertices; 
        this->face_to_tets.clear();
        //this->face_vertices.clear();
        for (int i = 0; i < all_faces.rows(); i++) {
            // 对三个顶点索引排序，使面方向无关
            int v[3] = {all_faces(i,0), all_faces(i,1), all_faces(i,2)};
            std::sort(v, v+3);
            auto key = std::make_tuple(v[0], v[1], v[2]);

            //如果已存在，修改点索引和四面体索引
            int t_idx = face_to_tet_idx(i);
            auto it = face_map.find(key);
            if (it != face_map.end()) { //如果面已存在，增加点索引和四面体索引
                int shared_face_idx = it->second; //key对应的面在 face_to_tets/face_vertices中的下标

                // ✅ 确保索引有效
                assert(shared_face_idx >= 0 && shared_face_idx < this->face_to_tets.size());
                assert(shared_face_idx < candidate_face_vertices.size());

                this->face_to_tets[shared_face_idx].push_back(t_idx);
                // ✅ 确保candidate_face_vectors也同步更新
                // 更新第5个顶点为当前四面体的第4个顶点
                if (shared_face_idx < candidate_face_vertices.size()) {
                    candidate_face_vertices[shared_face_idx][4] = all_faces(i,3);
                } else {
                    // 错误处理
                    std::cerr << "Error: face index out of range" << std::endl;
                }
                //candidate_face_vertices[shared_face_idx][4] = all_faces(i,3); 
            }
            else //如果不存在，创建新face条目，及对应点索引和四面体索引
            {
                //增加face和tet间映射
                std::vector<unsigned int> tets;
                tets.push_back(t_idx);
                this->face_to_tets.push_back(tets);

                std::array<int, 5> vertices = {
                    all_faces(i,3),  // v0: 第一个四面体的对顶点
                    all_faces(i,0),  // v1: 面顶点1
                    all_faces(i,1),  // v2: 面顶点2
                    all_faces(i,2),  // v3: 面顶点3
                    -1               // v4: 第二个四面体的对顶点（暂未知）
                };
                //增加face对应的5个坐标
                candidate_face_vertices.push_back(vertices); // -1占位，后续填充

                int shared_face_idx = this->face_to_tets.size()-1;
                face_map[key] = shared_face_idx; //记录这个唯一面在face_to_tets中的下标

                assert(candidate_face_vertices.size()==face_map.size());
                assert(face_map.size()==this->face_to_tets.size());
            }
        }

        #if DEBUG_OUTPUT_MESH
            // ✅ 安全检查：确保所有边界面的v4为-1
            for (int i = 0; i < this->face_to_tets.size(); i++) {
                std::cout << "face domain " << i <<":"
                    << candidate_face_vertices[i][0]
                    << candidate_face_vertices[i][1]
                    << candidate_face_vertices[i][2]
                    << candidate_face_vertices[i][3]
                    << candidate_face_vertices[i][4]
                    << std::endl;
                if (this->face_to_tets[i].size() == 1) {
                    assert(candidate_face_vertices[i][4] == -1);
                    //candidate_face_vertices[i][4] = -1;
                }
                else
                {
                    std::cout << "<-(shared)" << std::endl;
                    assert(candidate_face_vertices[i][4] != -1);
                    // std::cout << "Shared Face " << i <<":"
                    // << candidate_face_vertices[i][0]
                    // << candidate_face_vertices[i][1]
                    // << candidate_face_vertices[i][2]
                    // << candidate_face_vertices[i][3]
                    // << candidate_face_vertices[i][4]
                    // << std::endl;
                }
                
            }
        #endif

        this->face_vertices.resize(candidate_face_vertices.size(), 5);
        for (int i = 0; i < candidate_face_vertices.size(); i++) {
            this->face_vertices.row(i) << candidate_face_vertices[i][0], 
                                    candidate_face_vertices[i][1], 
                                    candidate_face_vertices[i][2], 
                                    candidate_face_vertices[i][3], 
                                    candidate_face_vertices[i][4];
        }

        

        return;
    };

    void compute_tet_volumes_Mr_inv(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T)
    {
        unsigned int num_domains = this->face_to_tets.size();
        // ✅ 安全检查
        assert(num_domains == this->face_vertices.rows());

        this->Mr_inv.resize(num_domains);
        this->adj_tet_volume.resize(num_domains);
        this->domain_volumes.resize(num_domains);

        // Pre-compute triangle rest shapes in local coordinate systems
        std::vector<std::array<Eigen::Matrix3d, 2>> rest_shapes(num_domains);
        
        // 四面体顶点顺序约定：
        // 四面体0: [v0, v1, v2, v3]  - v0是对顶点
        // 四面体1: [v4, v3, v2, v1]  - v4是对顶点
        const int tet_vertice_order[2][4] = {
            {0,1,2,3}, // 与顶点0相对的面
            {4,3,2,1}
        };

        

        // 遍历每个面和对应的四面体，计算参考构型矩阵
        for (int f_idx = 0; f_idx < num_domains; ++f_idx)
        {
            // ✅ 安全检查：确保不会越界
            assert(f_idx < this->face_vertices.rows());
            assert(f_idx < this->face_to_tets.size());

            
            auto vertices = this->face_vertices.row(f_idx);
            auto n_adj_tet = this->face_to_tets[f_idx].size();
            if (n_adj_tet > 1)
            {
                assert(this->face_vertices(f_idx, 4) != -1);  // 内部面必须有v4
            }

            for (int t_local_idx=0; t_local_idx < n_adj_tet; t_local_idx++) {//边界面只有一个邻接tet
            
                //int t_idx = this->face_to_tets[f_idx][i];
                // Get 3D vertex positions
                Eigen::Vector3d ar = V.row(vertices(tet_vertice_order[t_local_idx][0]));
                Eigen::Vector3d br = V.row(vertices(tet_vertice_order[t_local_idx][1]));
                Eigen::Vector3d cr = V.row(vertices(tet_vertice_order[t_local_idx][2]));
                Eigen::Vector3d dr = V.row(vertices(tet_vertice_order[t_local_idx][3]));

                // Save 3-by-3 matrix with edge vectors as columns
                rest_shapes[f_idx][t_local_idx].col(0) = br-ar;
                rest_shapes[f_idx][t_local_idx].col(1) = cr-ar;
                rest_shapes[f_idx][t_local_idx].col(2) = dr-ar;
                //= TinyAD::col_mat(br - ar, cr - ar, dr - ar);

                #if DEBUG_OUTPUT_MESH
                    
                    std::cout<<"f_id:" << f_idx << ", t_local_idx:"<< t_local_idx << std::endl;
                    std::cout<< "v_id: ";
                    for (int j =0; j< vertices.size(); j++)
                    {
                        if (vertices(j) != -1)
                        {
                            std::cout<< vertices(j) << ",";
                        }    
                    }
                    std::cout << std::endl;

                    // assert(rest_shapes[f_idx][t_local_idx].col(0).norm() !=0);
                    // assert(rest_shapes[f_idx][t_local_idx].col(1).norm() !=0);
                    // //assert(V.row(vertices(tet_vertice_order[i][2])) !=V.row(vertices(tet_vertice_order[i][0])));
                    // assert(rest_shapes[f_idx][t_local_idx].col(2).norm() !=0);
                    // std::cout << std::endl;
                #endif
                
            }
        }

        // 预计算每个四面体体积和逆矩阵
        for (int f_idx = 0; f_idx < num_domains; ++f_idx) {

            int n_tets = this->face_to_tets[f_idx].size();
            this->Mr_inv[f_idx] = {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
            this->adj_tet_volume[f_idx] = {0.0, 0.0};
            this->domain_volumes[f_idx] = 0.0;

            for (int t_local_idx=0; t_local_idx< n_tets; t_local_idx++) { //per tet
                Eigen::Matrix3d Mr = rest_shapes[f_idx][t_local_idx];
            
                // 解析逆 + 行列式（一次计算）
                double det = Mr(0,0) * (Mr(1,1) * Mr(2,2) - Mr(2,1) * Mr(1,2))
                            - Mr(0,1) * (Mr(1,0) * Mr(2,2) - Mr(2,0) * Mr(1,2))
                            + Mr(0,2) * (Mr(1,0) * Mr(2,1) - Mr(2,0) * Mr(1,1));
                
                double det2 = Mr.determinant();
                assert(det==det2);
                #if DEBUG_OUTPUT_MESH
                if (det< 1e-8)
                {
                    std::cout << "warning: f:"<< f_idx << ", t_local_idx:" << t_local_idx <<","
                    << "v:" << this->face_vertices(f_idx,0) <<","<< this->face_vertices(f_idx,1)<<","
                    << this->face_vertices(f_idx,2) <<","<< this->face_vertices(f_idx,3)<<","
                    << this->face_vertices(f_idx,4) << ","
                    << "det=" <<det <<",det2="<< det2 
                    <<std::endl;

                    // std::cout << "Mr:("
                    // << Mr(0,0) << ", " << Mr(0,1) <<"," << Mr(0,2)<<"),("
                    // << Mr(1,0) << ", " << Mr(1,1) <<"," << Mr(1,2)<<"),("
                    // << Mr(2,0) << ", " << Mr(2,1) <<"," << Mr(2,2)<<")"
                    // <<std::endl;
                }
                #endif
                assert(det!=0.0); //行列式有可能为负数，但后续能量计算中，inv_Mr*M就把负号抵消掉了。
                
                double inv_det = 1.0 / det;
                Eigen::Matrix3d inv;
                inv(0,0) =  (Mr(1,1) * Mr(2,2) - Mr(2,1) * Mr(1,2)) * inv_det;
                inv(0,1) = -(Mr(0,1) * Mr(2,2) - Mr(2,1) * Mr(0,2)) * inv_det;
                inv(0,2) =  (Mr(0,1) * Mr(1,2) - Mr(1,1) * Mr(0,2)) * inv_det;
                inv(1,0) = -(Mr(1,0) * Mr(2,2) - Mr(2,0) * Mr(1,2)) * inv_det;
                inv(1,1) =  (Mr(0,0) * Mr(2,2) - Mr(2,0) * Mr(0,2)) * inv_det;
                inv(1,2) = -(Mr(0,0) * Mr(1,2) - Mr(1,0) * Mr(0,2)) * inv_det;
                inv(2,0) =  (Mr(1,0) * Mr(2,1) - Mr(2,0) * Mr(1,1)) * inv_det;
                inv(2,1) = -(Mr(0,0) * Mr(2,1) - Mr(2,0) * Mr(0,1)) * inv_det;
                inv(2,2) =  (Mr(0,0) * Mr(1,1) - Mr(1,0) * Mr(0,1)) * inv_det;

                this->Mr_inv[f_idx][t_local_idx] = inv;
                
                this->adj_tet_volume[f_idx][t_local_idx]= std::abs(0.25 * det / 6.0);//必须为正
                this->domain_volumes[f_idx] += this->adj_tet_volume[f_idx][t_local_idx];
            }

        }    
    };

};



#endif