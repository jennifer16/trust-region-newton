/**
 *  "信赖域启发式的投影牛顿线搜索法"：

    ✅ 使用信赖域比率评估模型质量

    ✅ 基于比率自适应调整数值方法

    ❌ 没有显式信赖域半径

    ❌ 没有信赖域约束子问题

    ✅ 用线搜索替代步长控制
    线搜索：固定方向，调整步长 α
    信赖域：在球内优化，调整方向 d 和步长
 * 
 */

 #ifndef SHEAR_PROJECTED_NEWTON
  #define SHEAR_PROJECTED_NEWTON 1
#endif

 #ifndef CUBIC_DEFORM_PROJECTED_NEWTON
  #define CUBIC_DEFORM_PROJECTED_NEWTON 1
#endif

 #ifndef DEBUG_OUTPUT_CUBIC
  #define DEBUG_OUTPUT_CUBIC 1
#endif

 #ifndef CUBIC_PROJECTED_NEWTON
  #define CUBIC_PROJECTED_NEWTON 1
 #endif

#ifndef FS_PROJECTED_NEWTON
  #define FS_PROJECTED_NEWTON 1
#endif

#ifndef DIFF_PROJECTED_NEWTON
  #define DIFF_PROJECTED_NEWTON 1
#endif

// #ifndef DEBUG_OUTPUT
//   #define DEBUG_OUTPUT 1
// #endif

#ifndef DEBUG_OUTPUT_BLEND_TR
  #define DEBUG_OUTPUT_BLEND_TR 0
#endif
#ifndef SHEAR_DEBUG_OUTPUT
  #define SHEAR_DEBUG_OUTPUT 1
#endif

// #ifndef STB_IMAGE_WRITE_IMPLEMENTATION
//   #define STB_IMAGE_WRITE_IMPLEMENTATION
  // #include <stb_image_write.h>
// #endif

// #include <stb_image_write.h>

#include <igl/readMESH.h>
#include <igl/writeOBJ.h>
#include <igl/writeMESH.h>

#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/Helpers.hh>

#include <igl/boundary_facets.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>
#include <thread>
#include <mutex>
#include <string>
#include <filesystem>

#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <GL/gl.h>

#include "cxxopts.hpp"
#include "fixed_point_constraints.h"
#include "setup_initial_deformation.h"
#include "mesh_ops.h"
#include "cubic_regular.h"



// void stop_recording(int frame_count_, const std::string& output_dir_) {
    
//     std::cout << "Recording stopped. " << frame_count_ << " frames saved to " 
//               << output_dir_ << std::endl;
//     std::cout << "Run this command to create video:" << std::endl;
//     std::cout << "ffmpeg -r " << 30 << " -i " << output_dir_ 
//               << "/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4" 
//               << std::endl;
// }

void save_ppm(const std::string& filename, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);
    
    // 读取屏幕像素（OpenGL 标准函数）
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // 保存为 PPM 格式
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";
    
    // OpenGL 读取是从下到上，需要翻转
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            file << (int)pixels[idx] << " " 
                 << (int)pixels[idx+1] << " " 
                 << (int)pixels[idx+2] << "\n";
        }
    }
    file.close();
}

int setAggressiveMode(double ratio, double upper_threshold, double lower_threshold) {
    int mode = 0; // 0: normal, 1: aggressive, -1: conservative
    double aggressive = 0.5;
    aggressive+= (ratio < lower_threshold) ? 0.5 : (ratio > upper_threshold) ? -0.5 : 0.0; // 信赖域比率小于lower_threshold时，增加激进程度；大于upper_threshold时，降低激进程度
    // aggressive += (rho < 0.25) ? 1.0 : (rho > 0.9) ? -0.5 : 0.0;
    // aggressive += (kappa > 0.8) ? 1.0 : (kappa < 0.2) ? -0.5 : 0.0;
    // aggressive += (alpha > 0.7) ? 1.0 : (alpha < 0.3) ? -0.5 : 0.0;

    if (aggressive > 0.5) {
        mode = 1; // 激进模式
    } else if (aggressive < 0.1) {
        // 保守模式
        mode = -1;
    } else {
        // 正常模式
        mode = 0;
    }
    return mode;
}

void adjustLineSearchParamsByMode(int mode, double& decay)
{
    switch (mode)
    {
    case 1: // 激进模式
        decay = 0.4;
        //c1 = 0.15;
        break;
    case -1: // 保守模式
        decay = 0.8;
        //c1 = 0.5;
        break;
    case 0: // 正常模式
        decay = 0.75;
        //c1 = 0.35;
        break;
    default:
      break;
    }
}


// 模板方法：追加单个vector到同一行
template<typename T>
void appendVectorToSameLine(const std::string& filename, const std::vector<T>& vec) {
    std::ofstream outFile(filename, std::ios::app);
    if (!outFile) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < vec.size(); i++) {
        outFile << vec[i];
        if (i < vec.size() - 1) outFile << ",";
    }
    
    outFile.close();
}

// 完整的实现
class CSVLineBuilder {
private:
    std::stringstream buffer;
    bool hasData;
    
public:
    CSVLineBuilder() : hasData(false) {}
    
    template<typename T>
    CSVLineBuilder& add(const std::vector<T>& vec) {
        if (hasData) {
            buffer << ",";
        }
        
        for (size_t i = 0; i < vec.size(); i++) {
            buffer << vec[i];
            if (i < vec.size() - 1) buffer << ",";
        }
        
        hasData = true;
        return *this;
    }
    
    void writeToFile(const std::string& filename, bool addNewLine = true) {
        std::ofstream outFile(filename, std::ios::app);
        if (outFile && hasData) {
            outFile << buffer.str();
            if (addNewLine) {
                outFile << std::endl;
            }
        }
    }
    
    std::string str() const {
        return buffer.str();
    }
};

std::string get_time_str() {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        return oss.str();
}

// compute the trust region ratio
double compute_trust_region_ratio(double e1, double e0, 
                                  const Eigen::VectorXd &d, 
                                  const Eigen::VectorXd &g,
                                  const Eigen::SparseMatrix<double> &H){
    assert(d.size() == g.size());
    assert(d.size() == H.rows());
    assert(d.size() == H.cols());
    return (e0 - e1) / (0.0 - (d.dot(g) + 0.5 * d.transpose() * H * d));
}

// 从字符串解析模式
TinyAD::HessianProjectionMode parse_projection_mode(const std::string& mode_str) {
    static const std::map<std::string, TinyAD::HessianProjectionMode> mode_map = {
        {"auto", TinyAD::HessianProjectionMode::AUTO},
        {"clamp_abs", TinyAD::HessianProjectionMode::CLAMP_ABS},
        {"soft_clamp", TinyAD::HessianProjectionMode::SOFT_CLAMP},
        {"hybrid", TinyAD::HessianProjectionMode::HYBRID},
        {"cubic", TinyAD::HessianProjectionMode::CUBIC},
        {"clamp", TinyAD::HessianProjectionMode::CLAMP},
        {"soft_abs", TinyAD::HessianProjectionMode::SOFT_ABS},
        {"abs", TinyAD::HessianProjectionMode::ABS},
        {"abs_nondiff", TinyAD::HessianProjectionMode::ABS_NONDIFF},
        {"clamp_nondiff", TinyAD::HessianProjectionMode::CLAMP_NONDIFF},
        {"clamp_abs_nondiff", TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF}, // 直接复用CLAMP_ABS的非可微版本
        {"clamp_abs_adaptive_nondiff", TinyAD::HessianProjectionMode::CLAMP_ABS_ADAPTIVE_NONDIFF}, // 直接复用auto_regularize的非可微版本;
        {"abs_shift_clamp_nondiff", TinyAD::HessianProjectionMode::ABS_SHIFT_CLAMP_NONDIFF},
        {"clamp_abs_blending", TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING},
        {"clamp_abs_blending2", TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING2},
        {"clamp_abs_blending4", TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING4},
        {"clamp_abs_blending5", TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING5},
        {"clamp_abs_blending3", TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING3},
        {"clamp_abs_blending_smooth", TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH},
        {"smooth_tr", TinyAD::HessianProjectionMode::SMOOTH_TR},
        {"clamp_abs_blending_shear", TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SHEAR}, // 直接复用CLAMP_ABS_BLENDING
    };
    auto it = mode_map.find(mode_str);
    if (it != mode_map.end()) {
        return it->second;
    }
    
    // 默认值
    return TinyAD::HessianProjectionMode::AUTO;
}

// 从字符串解析模式 
unsigned int parse_smooth_mode(const std::string& mode_str) {
    static const std::map<std::string, int> mode_map = {
        {"none", 0},
        {"face", 1},
        {"default", 1}, // 直接复用auto_regularize的非可微版本;
    };
    auto it = mode_map.find(mode_str);
    if (it != mode_map.end()) {
        return it->second;
    }
    
    // 默认值
    return 1;
}

int diff_projected_newton(int argc, char** argv);
int fs_projected_newton(int argc, char** argv);
int cubic_projected_newton(int argc, char** argv);
int cubic_deform_projected_newton(int argc, char** argv)
{
  // parse command line arguments
  cxxopts::Options options("Projected Newton with a trust region", "Choose eigenvalue filtering method: adaptive, clamp, abs");

  options.add_options()
    ("smooth_mode", "add differentiable eigenvalue projection strategy ", cxxopts::value<std::string>()->default_value("none")) // 若设置diff，后续可微的clamp和abs等
    ("diff", "add differentiable eigenvalue projection strategy ", cxxopts::value<bool>()->default_value("false")) // 若设置diff，后续可微的clamp和abs等
    ("diff_mode", "differentiable eigenvalue projection mode", cxxopts::value<std::string>()->default_value("auto")) // 若设置diff，后续可微的clamp和abs等
    ("abs", "use absolute eigenvalue projection strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("clamp", "use eigenvalue clamping strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("p,epsilon", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("-0.5"))
    ("n,mesh_name", "Mesh name", cxxopts::value<std::string>()->default_value("bimba"))
    ("l,pose_label", "Pose label", cxxopts::value<std::string>()->default_value("stretch"))
    ("g,deformation_magnitude", "The magnitude of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("t,deformation_ratio", "The ratio of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("b,fixed_boundary_range", "The range of fixed vertices on the boundary", cxxopts::value<double>()->default_value("0.1"))
    ("c,convergence_eps", "The convergence threshold", cxxopts::value<double>()->default_value("1e-5"))
    ("ym", "Young's modulus (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("1e8"))
    ("pr", "Poisson's ratio (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("0.495"))
    ("tr", "trust region ratio threshold", cxxopts::value<double>()->default_value("0.01"))
    ("experiment_name", "experiment name", cxxopts::value<std::string>()->default_value(""))
    ("rotate_ratio", "The ratio of the rotation", cxxopts::value<double>()->default_value("0.5"))
    ("h,help", "show help")
  ;
  
  auto result = options.parse(argc, argv);
  if (result.count("help"))
  {
      std::cout << options.help() << "\n";
      return 0;
  }

  const bool diff = result["diff"].as<bool>();
  //const int _mode = result["diff_mode"].as<int>();
  std::string diff_mode_str = result["diff_mode"].as<std::string>();
  TinyAD::HessianProjectionMode _diff_mode = parse_projection_mode(diff_mode_str);

  const bool abs = result["abs"].as<bool>();
  const bool clamp = result["clamp"].as<bool>();
  std::string eps_str = result["epsilon"].as<std::string>();
  double eps = std::stod(eps_str);
  std::string mesh_name = result["mesh_name"].as<std::string>();
  std::string pose_label = result["pose_label"].as<std::string>();
  const double deformation_magnitude = result["deformation_magnitude"].as<double>();
  const double deformation_ratio = result["deformation_ratio"].as<double>();
  const double fixed_boundary_range = result["fixed_boundary_range"].as<double>();
  const double convergence_eps = result["convergence_eps"].as<double>();
  const double YM = result["ym"].as<double>();
  const double PR = result["pr"].as<double>();
  const double tr_threshold = result["tr"].as<double>(); // 信赖域接受步长的阈值，通常设置为0.01
  std::string experiment_folder = result["experiment_name"].as<std::string>() == "" ? ("figure_" + mesh_name) : result["experiment_name"].as<std::string>();
  experiment_folder = experiment_folder + get_time_str();
  const double rotate_ratio = result["rotate_ratio"].as<double>();

  /*
  * 0: clamp
  * -1: abs
  * -0.5: adaptive (default)
  * 0.5: differentiable  (if diff is set)
  */
  if (diff)
  {
      eps_str =  diff_mode_str;
      //下面三种情况,eps值有特殊用处
      if (_diff_mode == TinyAD::HessianProjectionMode::ABS_NONDIFF) {
        eps_str = "abs";
        eps = -1;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_NONDIFF) {
        eps_str = "clamp";
        eps = 0;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF) {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
      }
      else{
         //非特殊情况,eps必须大于等于0
        if (eps < 0)
        {
          eps = 0;
        }
      }
  }
  else {
    if (clamp || eps == 0) {
        // we use eps = 0 as a flag for clamp projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "clamp";
        eps = 0; 
        diff_mode_str = "clamp_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_NONDIFF; 
    }
    else if (abs || eps == -1) {
        // we use eps = -1 as a flag for abs projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "abs";
        eps = -1;
        diff_mode_str = "abs_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::ABS_NONDIFF; 
    }
    else {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
        diff_mode_str = "clamp_abs_nondiff"; // trust region newton;
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF;
    }
  }
    
  if (!std::filesystem::exists("../results/"))
    std::filesystem::create_directory("../results/");
  if (!std::filesystem::exists("../results/" + experiment_folder))
    std::filesystem::create_directory("../results/" + experiment_folder);

  const std::string output_folder = "../results/" + experiment_folder + "/" 
    + pose_label + "_YM_" + std::to_string(YM) + "_PR_" + std::to_string(PR)
    + "_deform_mag_" + std::to_string(deformation_magnitude) + "_deform_ratio_" 
    + std::to_string(deformation_ratio) + "_fixed_boundary_range_" 
    + std::to_string(fixed_boundary_range)+"_diff_mode_" + diff_mode_str+ "/";
  {
    // record the statistics
    if (!std::filesystem::exists(output_folder))
      std::filesystem::create_directory(output_folder);
    if (!std::filesystem::exists(output_folder + "obj/"))
      std::filesystem::create_directory(output_folder + "obj/");
    if (!std::filesystem::exists(output_folder + "obj/" + mesh_name + "_" + eps_str + "/"))
      std::filesystem::create_directory(output_folder + "obj/" + mesh_name + "_" + eps_str + "/");
    if (!std::filesystem::exists(output_folder + "hist/"))
      std::filesystem::create_directory(output_folder + "hist/");
    if (!std::filesystem::exists(output_folder + "iter/"))
      std::filesystem::create_directory(output_folder + "iter/");
    if (!std::filesystem::exists(output_folder + "trust_region_ratio/"))
      std::filesystem::create_directory(output_folder + "trust_region_ratio/");
    if (!std::filesystem::exists(output_folder + "trust_region_eps/"))
      std::filesystem::create_directory(output_folder + "trust_region_eps/");
    if (!std::filesystem::exists(output_folder + "line_search_alpha/"))
      std::filesystem::create_directory(output_folder + "line_search_alpha/");
    if (!std::filesystem::exists(output_folder + "line_search_iter/"))
      std::filesystem::create_directory(output_folder + "line_search_iter/");
    
    #if DEBUG_OUTPUT
    if (!std::filesystem::exists(output_folder + "debug_log/"))
    {
      std::filesystem::create_directory(output_folder + "debug_log/");
      TINYAD_INIT_DEBUG_LOG(output_folder + "debug_log/log"+get_time_str()+".txt");
    }
    #endif

    
  }

  const std::string output_tag = "mesh_" + mesh_name + "_eps_" + eps_str;

  // set up viewer
  igl::opengl::glfw::Viewer viewer;

  // compute the Lame parameters
  const double MU = YM / (2 * (1 + PR));
  const double LAMBDA = YM * PR / ((1 + PR) * (1 - 2 * PR));
  const double lambda_mu_ratio = LAMBDA / MU;
 
  // print out the configuration
  {
    TINYAD_DEBUG_OUT("Diff flag: " << diff);
    TINYAD_DEBUG_OUT("*Eigenvalue Differentiable filtering strategy: " << diff_mode_str);
    TINYAD_DEBUG_OUT("Eigenvalue filtering strategy: " << eps_str);
    TINYAD_DEBUG_OUT("mu: " << MU);
    TINYAD_DEBUG_OUT("lambda: " << LAMBDA);
    TINYAD_DEBUG_OUT("*Projection threshold: " << eps);
    TINYAD_DEBUG_OUT("Mesh name: " << mesh_name);
    TINYAD_DEBUG_OUT("Pose label: " << pose_label);
    TINYAD_DEBUG_OUT("Lambda / Mu ratio: " << lambda_mu_ratio);
    TINYAD_DEBUG_OUT("Deformation magnitude: " << deformation_magnitude);
    TINYAD_DEBUG_OUT("Deformation ratio: " << deformation_ratio);
    TINYAD_DEBUG_OUT("Fixed vertices boundary range: " << fixed_boundary_range);
    TINYAD_DEBUG_OUT("Convergence threshold: " << convergence_eps);
    TINYAD_DEBUG_OUT("Young's modulus: " << YM);
    TINYAD_DEBUG_OUT("Poisson's ratio: " << PR);
    TINYAD_DEBUG_OUT("Trust region threshold: " << tr_threshold);
    TINYAD_DEBUG_OUT("Experiment folder: " << experiment_folder);
    TINYAD_DEBUG_OUT("Rotate ratio: " << rotate_ratio);
  }

  Eigen::MatrixXd V, U; // #V-by-3 3D vertex positions,V为初始,U为当前
  Eigen::MatrixXi F, FF; // #T-by-4 indices into V
  Eigen::VectorXi TriTag, TetTag;
  if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh"))
    igl::readMESH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh", V, F, FF);
  else if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh"))
    igl::readMSH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh", V, FF, F, TriTag, TetTag);
  else {
    std::cout << "Mesh " << mesh_name << " not found!" << std::endl;
    exit(1);
  }

  TINYAD_DEBUG_OUT("Read mesh with " << V.rows() << " vertices and " << F.rows() << " tetrahedrons.");

  // get boundary vertices
  igl::boundary_facets(F, FF);
  FF = FF.rowwise().reverse().eval();

  TINYAD_DEBUG_OUT("Boundary has " << FF.rows() << " faces.");

  // normalize the mesh V
  Eigen::RowVector3d mean = V.colwise().mean();
  V.rowwise() -= mean;
  double max_norm = V.rowwise().norm().maxCoeff();
  V /= max_norm;

  // set up mesh U
  U = V;

  // fixed point constraints
  Eigen::SparseMatrix<double> P;
  std::vector<unsigned int> indices_fixed;

  // set up fixed point constraints and initial deformation
  setup_initial_deformation(V, F, pose_label, deformation_magnitude, deformation_ratio, rotate_ratio, fixed_boundary_range, U, indices_fixed);
  fixed_point_constraints(P, 3*V.rows(), 3, indices_fixed);

  TINYAD_DEBUG_OUT("Finish setting up fixed point constraints.");

  bool redraw = false;
  std::mutex m;
  std::thread optimization_thread(
    [&]()
    {
      // Pre-compute triangle rest shapes in local coordinate systems
      std::vector<Eigen::Matrix3d> rest_shapes(F.rows());
      for (int f_idx = 0; f_idx < F.rows(); ++f_idx)
      {
        // Get 3D vertex positions
        Eigen::Vector3d ar = V.row(F(f_idx, 0));
        Eigen::Vector3d br = V.row(F(f_idx, 1));
        Eigen::Vector3d cr = V.row(F(f_idx, 2));
        Eigen::Vector3d dr = V.row(F(f_idx, 3));

        // Save 3-by-3 matrix with edge vectors as columns
        rest_shapes[f_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
      };

      /*
          TinyAD::scalar_function<3>: 创建一个标量函数，每个变量是3维向量（例如3D顶点坐标）
          TinyAD::range(V.rows()): 定义变量的范围，V.rows() 是顶点数量
          auto func: 自动推断函数对象类型
      */ 
      // TinyAD::scalar_function(n)：这是一个函数，可以计算它的函数值、梯度、Hessian的函数，其自变量索引的范围为n
      // Set up function with 3d vertex positions as variables.
      auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

      // func.add_elements<n>(element_range, energy_function); 
      // 定义元素的能量项，1.n为元素的dim，其中2.element_range是元素索引的范围，
      // 3.energy_function是一个lambda函数，接受一个元素对象，返回该元素的能量值
      // 4. lamda函数 [捕获变量] (输入参数) -> 返回值 
      // 5. 所有参与自动微分计算的变量和中间结果，都必须使用 TINYAD_SCALAR_TYPE 或其衍生的类型
      // 6. element.handle对应element_range中的当前元素的索引，element.variables()对应输入变量，想要访问输入变量的第i个变量
      // 7. 在这个地方，输入变量是顶点数组，element_range是F数组，即按F计算能量
      // Add objective term per element. Each connecting 4 vertices.
      // "neo-hooken energy" is defined on each tetrahedron, so element_range is F.rows() and element.variables() gives the vertex positions of the current tetrahedron.
      func.add_elements<4>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element) {
          // Evaluate element using either double or TinyAD::Double
          using T = TINYAD_SCALAR_TYPE(element);

          // Get variable 3d vertex positions
          Eigen::Index f_idx = element.handle;
          Eigen::Vector3<T> a = element.variables(F(f_idx, 0));
          Eigen::Vector3<T> b = element.variables(F(f_idx, 1));
          Eigen::Vector3<T> c = element.variables(F(f_idx, 2));
          Eigen::Vector3<T> d = element.variables(F(f_idx, 3));

          Eigen::Matrix3<T> M = TinyAD::col_mat(b - a, c - a, d - a);
          Eigen::Matrix3d Mr = rest_shapes[f_idx];
          Eigen::Matrix3<T> J = M * Mr.inverse(); // 可以放外边 to be optimized by zj

          // Compute the stable Neo-Hookean energy [Smith et al. 2018]
          double A = 0.5 * Mr.determinant(); //可以放外边 to be optimized by zj
          const double mu = MU;
          const double lambda = LAMBDA;
          auto Ic = (J.transpose() * J).trace();
          auto detF = J.determinant();
          double alpha = 1.0 + mu / lambda;
          auto W = mu / 2.0 * (Ic - 3.0) + lambda / 2.0 * (detF - alpha) * (detF - alpha);
          return A * W;
      });

      // to be optimized by zj: 可以考虑把 rest_shapes 以及 pre-computed Mr.inverse() 放到外边，作为常量传入 lambda 函数中，这样就不需要每次迭代都计算 rest_shapes 和 Mr.inverse() 了
      // to be optimized by zj: 还可以考虑把 A 也放到外边，因为 A 只和 rest_shapes 相关，而 rest_shapes 是不变的
      // to be optimized by zj: 还可以考虑把 lambda 和 mu 放到外边，因为它们也是不变的
      // to be modified by zj: 动态状态下的PD，增加local step，即每个元素的能量项不仅依赖于当前状态，还依赖于上一个状态，这样可以增加稳定性，尤其是在使用abs投影时

      // func.x_from_data() 是 TinyAD 提供的数据格式转换工具，将外部数据（如顶点矩阵）转换为优化所需的展平向量格式，使代码更简洁、更安全。
      // v_idx 的取值范围是由 TinyAD::scalar_function<k>(n_variables) 中的 n_variables 决定的，即函数的输入变量的维度。
      // 在这个例子中，n_variables 是 V.rows()，即顶点数量，因此 v_idx 的取值范围是 [0, V.rows()-1]。
      // Assemble inital x vector from U matrix.
      // x_from_data(...) takes a lambda function that maps
      // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) {
          return U.row(v_idx);
      });

      TINYAD_DEBUG_OUT("Initial energy: " << func.eval(x));

      // Projected Newton
      TinyAD::LinearSolver solver; //Eigen::SimplicialLDLT*,ConjugateGradient,SparseLU
      int max_iters = 200; // 迭代次数可以根据需要调整
      Eigen::VectorXd d;
      Eigen::SparseMatrix<double> H;
      std::vector<double> hist;
      // record the trust region ratio
      // 衡量模型预测的准确性 ρ = (实际下降) / (预测下降), 
      // ρ < 0         → 实际目标函数值增加（拒绝步长）
      // 0 ≤ ρ < 0.25  → 模型质量差，缩小信赖域
      // 0.25 ≤ ρ < 0.75 → 模型质量一般，保持信赖域
      // ρ ≥ 0.75      → 模型质量好，可以放大信赖域
      // double delta;  // 当前信赖域半径
      // double eta1;   // 接受步长的阈值（通常0.25）
      // double eta2;   // 放大半径的阈值（通常0.75）
      std::vector<double> hist_trust_region_ratio;
      hist_trust_region_ratio.push_back(0.0);

      std::vector<double> hist_trust_region_eps; // 记录每次迭代使用的eps值，以观察自适应策略的变化
      std::vector<double> hist_line_search_alpha; // 记录每次迭代的线搜索步长
      std::vector<int> hist_line_search_iter; // 记录每次迭代的线搜索迭代次数

      double cubic_sigma = 0.0;
      
      //Eigen::VectorXd cubic_lambda_vec = Eigen::VectorXd::Zero(x.rows());
      //Eigen::VectorXd cubic_lambda_vec;

      // initial cubic sigma
      if (_diff_mode == TinyAD::HessianProjectionMode::CUBIC)
      {
        auto f0 = func.eval(x);
        int n_v = x.rows();
        Eigen::VectorXd perturbation = 1e-6 * Eigen::VectorXd::Ones(n_v,x.cols());
        double f_perturbed = func.eval(x + perturbation);
        double coeff = perturbation.norm() / n_v;
        double relative_change = std::abs(f_perturbed - f0) / (std::abs(f0) + 1e-12) ;
        relative_change /= (coeff * coeff);
        
        if (relative_change > 0.1) 
        {
            cubic_sigma = 10.0;  // 高度非线性，强正则化
        } 
        else if (relative_change > 0.01) {
            cubic_sigma =  1.0;   // 中等非线性
        } else {
            cubic_sigma = 0.1;   // 接近线性，弱正则化
        }
      }
      double cubic_lambda = cubic_sigma / (x.rows()*x.cols());
      
      //当global matrix发生变化时，进行下面的牛顿投影和线搜索步骤
      for (int i = 0; i < max_iters; ++i)
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".mesh", U, F, FF);
        TINYAD_DEBUG_OUT("===iteration("<< i << ")===");
        TINYAD_DEBUG_OUT("cubic_sigma: "<< cubic_sigma);
        TINYAD_DEBUG_OUT("cubic_lambda: " << cubic_lambda);
        //TINYAD_DEBUG_OUT("x.rows: "<< x.rows()<<" x.cols "<<x.cols());

        // switch between clamp or abs depending on whether the trust region ratio is close to 1
        //if(eps_str == "adaptive") {
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF
          || _diff_mode == TinyAD::HessianProjectionMode::CUBIC) {
          eps = (std::fabs(hist_trust_region_ratio.back() - 1.0) < tr_threshold) ? 0.0 : -1; // tr_threshold信赖域接受步长的阈值，通常设置为0.01,大于接受

          if (eps == 0.0) {
            TINYAD_DEBUG_OUT("Switch to clamp");  
          }
          else {
            TINYAD_DEBUG_OUT("Switch to abs");
          }
          hist_trust_region_eps.push_back(eps);
        }

        auto [f, g, H_proj] = func.eval_with_hessian_proj(x, eps, _diff_mode); // 
        Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量

        if (cubic_lambda > H_proj.norm())
        {
          cubic_lambda = H_proj.norm();
          TINYAD_DEBUG_OUT("cubic_lambda > H_proj.norm("<<H_proj.norm()<<")");
        }

        if (_diff_mode == TinyAD::HessianProjectionMode::CUBIC)
        {
          for (int i = 0; i < H_proj.rows(); ++i) {
            bool found = false;
            for (int k = 0; k < H_proj.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(H_proj, k); it; ++it) {
                    if (it.row() == i && it.col() == i) {
                        it.valueRef() += cubic_lambda;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (!found) {
                H_proj.insert(i, i) = cubic_lambda;
            }
          }
          TINYAD_DEBUG_OUT(" H_proj + "<< cubic_lambda);
        }
        
        // record the energy
        hist.push_back(f);

        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);

        H = H_proj;

        // TINYAD_DEBUG_OUT(" H0 ");
        // testIndefiniteMatrix(H0);
        // TINYAD_DEBUG_OUT(" H_proj ");
        // testIndefiniteMatrix(H);
        
        // // Compute the Newton direction
        // if (_diff_mode == TinyAD::HessianProjectionMode::CUBIC
        //   && cubic_sigma <= 0.1)//linear deform
        // {
        //   TINYAD_DEBUG_OUT("==basic newton method===" );
        //   TINYAD_DEBUG_OUT("cubic_sigma " << cubic_sigma << "<=0.1");
        //   auto H_cubic2 = func.eval_hessian(x); // 
        //   H = H_cubic2;
        // }
        
        Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
        Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
        d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
        d = P.transpose() * d;

        TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
        TINYAD_DEBUG_OUT("d norm " << i << " =  " << d.norm());
        TINYAD_DEBUG_OUT("g norm " << i << " =  " << g.norm());

        // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
        // 优化方案1： 相对收敛
        // if (newton_decrement < convergence_eps * initial_decrement) break;  
        // 方案3：梯度范数 
        // if (g.norm() < convergence_eps) break;
        // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
        // 方案5：根据材料参数调整收敛标准
        //  LAMBDA是材料参数，不是尺度参数
        // // 使用特征长度进行无量纲化
        // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
        // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
        // // 物理意义的收敛判据
        // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
        if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
          break;

        // line search
        Eigen::VectorXd x_prev = x;
        x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 100, 1e-8);
        double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
        hist_line_search_alpha.push_back(alpha);

        int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
        hist_line_search_iter.push_back(line_search_iter);

        // compute the trust region ratio
        double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
        if (trust_region_ratio >= INFINITY)
        {
          TINYAD_WARNING("Trust region ratio: " << trust_region_ratio << " -> 0");
          trust_region_ratio = 0.0;
        }
        hist_trust_region_ratio.push_back(trust_region_ratio);

        TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);

        if (_diff_mode == TinyAD::HessianProjectionMode::CUBIC)
        {
          auto f0 = func.eval(x);
          int n_v = x.rows();
          Eigen::VectorXd perturbation = 1e-6 * Eigen::VectorXd::Ones(n_v,x.cols());
          double f_perturbed = func.eval(x + perturbation);
          double coeff = perturbation.norm() / n_v;
          double relative_change = std::abs(f_perturbed - f0) / (std::abs(f0) + 1e-12) ;
          relative_change /= (coeff * coeff);
          
          if (relative_change > 0.1) 
          {
              cubic_sigma = 10.0;  // 高度非线性，强正则化
          } 
          else if (relative_change > 0.01) {
              cubic_sigma =  1.0;   // 中等非线性
          } else {
              cubic_sigma = 0.1;   // 接近线性，弱正则化
          }

          
          if (cubic_sigma >= 10.0) //大变形，正则化
          {
            cubic_lambda = cubic_sigma * (alpha*d).norm();
            //TINYAD_DEBUG_OUT(" ===will enter cubic newton method=== " );
          } // 中变形，使用上一次的；小变形，不正则化，投影即可
          // TINYAD_DEBUG_OUT("cubic_sigma(" << i << ") =  " << cubic_sigma);
          //TINYAD_DEBUG_OUT("cubic_lambda(" << i << ") =  " << cubic_lambda);
          
        }
        
      // Write final x vector to U matrix.
      // x_to_data(...) takes a lambda function that writes the final value
      // of each variable (Eigen::Vector2d) back to our U matrix.
        func.x_to_data(x, [&] (int v_idx, const Eigen::Vector3d& p) {
            U.row(v_idx) = p;
            });
        {
          std::lock_guard<std::mutex> lock(m);
          redraw = true; 
        }
      }

      TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
      hist.push_back(func.eval(x));

      // output all the optimization statistics
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".mesh", U, F, FF);
        
        std::ofstream output_file_fixed(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_fixed_vid.txt");
        std::ostream_iterator<int> output_iterator_fixed(output_file_fixed, "\n");
        std::copy(std::begin(indices_fixed), std::end(indices_fixed), output_iterator_fixed);

        std::ofstream output_file(output_folder + "hist/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(std::begin(hist), std::end(hist), output_iterator);

        std::ofstream output_file_trust_region_ratio(output_folder + "trust_region_ratio/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_ratio(output_file_trust_region_ratio, "\n");
        std::copy(std::begin(hist_trust_region_ratio), std::end(hist_trust_region_ratio), output_iterator_trust_region_ratio);

        std::ofstream output_file_trust_region_eps(output_folder + "trust_region_eps/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_eps(output_file_trust_region_eps, "\n");
        std::copy(std::begin(hist_trust_region_eps), std::end(hist_trust_region_eps), output_iterator_trust_region_eps);

        std::ofstream output_file_line_search_alpha(output_folder + "line_search_alpha/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_line_search_alpha(output_file_line_search_alpha, "\n");
        std::copy(std::begin(hist_line_search_alpha), std::end(hist_line_search_alpha), output_iterator_line_search_alpha);

        std::ofstream output_file_line_search_iter(output_folder + "line_search_iter/" + output_tag + ".txt");
        std::ostream_iterator<int> output_iterator_line_search_iter(output_file_line_search_iter, "\n");
        std::copy(std::begin(hist_line_search_iter), std::end(hist_line_search_iter), output_iterator_line_search_iter);

        std::ofstream output_file_iter(output_folder + "iter/" + output_tag + ".txt");
        output_file_iter << (hist.size()-1) << std::endl;
      }

      TINYAD_DEBUG_OUT("======== The End ========");
      // comment this out later
      // close the viewer


      exit(0);
    });

  // Plot mesh
  viewer.core().is_animating = true;
  viewer.data().set_mesh(U, FF);
  viewer.core().align_camera_center(U);
  viewer.data().show_lines = false;

  viewer.callback_pre_draw = [&] (igl::opengl::glfw::Viewer& viewer)
  {
    if(redraw)
    {
      viewer.data().set_vertices(U);
      viewer.core().align_camera_center(U);
      {
        std::lock_guard<std::mutex> lock(m);
        redraw = false;
      }
    }
    return false;
  };

  // 在初始化回调中设置标题
  viewer.callback_init = [&](igl::opengl::glfw::Viewer& v)
  {
      // 获取 GLFW 窗口指针并设置标题
      glfwSetWindowTitle(v.window, diff_mode_str.c_str());
      return false;
  };

  viewer.launch();
  if(optimization_thread.joinable())
  {
    optimization_thread.join();
  }

  #if DEBUG_OUTPUT
    TINYAD_CLOSE_DEBUG_LOG();
  #endif
  return 0;
}

int shear_projected_newton(int argc, char** argv)
{
  TINYAD_DEBUG_OUT("#shear_projected_newton " );
  // parse command line arguments
  cxxopts::Options options("Projected Newton with a trust region", "Choose eigenvalue filtering method: adaptive, clamp, abs");

  options.add_options()
    ("smooth_mode", "add differentiable eigenvalue projection strategy ", cxxopts::value<std::string>()->default_value("none")) // 若设置diff，后续可微的clamp和abs等
    ("diff", "add differentiable eigenvalue projection strategy ", cxxopts::value<bool>()->default_value("false")) // 若设置diff，后续可微的clamp和abs等
    ("diff_mode", "differentiable eigenvalue projection mode", cxxopts::value<std::string>()->default_value("auto")) // 若设置diff，后续可微的clamp和abs等
    // ("delta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("0.001"))  // 过渡区间宽度
   // ("beta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("10.0"))// 光滑参数
    ("abs", "use absolute eigenvalue projection strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("clamp", "use eigenvalue clamping strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("p,epsilon", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("-0.5"))
    ("n,mesh_name", "Mesh name", cxxopts::value<std::string>()->default_value("bimba"))
    ("l,pose_label", "Pose label", cxxopts::value<std::string>()->default_value("stretch"))
    ("g,deformation_magnitude", "The magnitude of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("t,deformation_ratio", "The ratio of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("b,fixed_boundary_range", "The range of fixed vertices on the boundary", cxxopts::value<double>()->default_value("0.1"))
    ("c,convergence_eps", "The convergence threshold", cxxopts::value<double>()->default_value("1e-5"))
    ("ym", "Young's modulus (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("1e8"))
    ("pr", "Poisson's ratio (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("0.495"))
    ("tr", "trust region ratio threshold", cxxopts::value<double>()->default_value("0.01"))
    ("experiment_name", "experiment name", cxxopts::value<std::string>()->default_value(""))
    ("rotate_ratio", "The ratio of the rotation", cxxopts::value<double>()->default_value("0.5"))
    ("h,help", "show help")
    ("adaptive", "adaptive according to rho ", cxxopts::value<double>()->default_value("1e-5")) // >0 为adaptive
  ;
  
  auto result = options.parse(argc, argv);
  if (result.count("help"))
  {
      std::cout << options.help() << "\n";
      return 0;
  }

  const double adaptive = result["adaptive"].as<double>();
  // bool b_adaptive = false;
  // if ( TinyAD::isPostive(adaptive))
  // {
  //   b_adaptive = true;
  // }
  const bool diff = result["diff"].as<bool>();
  //const int _mode = result["diff_mode"].as<int>();
  std::string diff_mode_str = result["diff_mode"].as<std::string>();
  TinyAD::HessianProjectionMode _diff_mode = parse_projection_mode(diff_mode_str);
  // std::string delta_str = result["delta"].as<std::string>();
  // double delta = std::stod(delta_str);
  // std::string beta_str = result["beta"].as<std::string>();
  // double beta = std::stod(beta_str);


  const bool abs = result["abs"].as<bool>();
  const bool clamp = result["clamp"].as<bool>();
  std::string eps_str = result["epsilon"].as<std::string>();
  double eps = std::stod(eps_str);
  std::string mesh_name = result["mesh_name"].as<std::string>();
  std::string pose_label = result["pose_label"].as<std::string>();
  const double deformation_magnitude = result["deformation_magnitude"].as<double>();
  const double deformation_ratio = result["deformation_ratio"].as<double>();
  const double fixed_boundary_range = result["fixed_boundary_range"].as<double>();
  const double convergence_eps = result["convergence_eps"].as<double>();
  const double YM = result["ym"].as<double>();
  const double PR = result["pr"].as<double>();
  const double tr_threshold = result["tr"].as<double>(); // 信赖域接受步长的阈值，通常设置为0.01
  std::string experiment_folder = result["experiment_name"].as<std::string>() == "" ? ("figure_" + mesh_name) : result["experiment_name"].as<std::string>();
  std::string time_str = get_time_str();
  experiment_folder = experiment_folder + time_str;
  const double rotate_ratio = result["rotate_ratio"].as<double>();

  /*
  * 0: clamp
  * -1: abs
  * -0.5: adaptive (default)
  * 0.5: differentiable  (if diff is set)
  */
  if (diff)
  {
      eps_str =  diff_mode_str;
      //下面三种情况,eps值有特殊用处
      if (_diff_mode == TinyAD::HessianProjectionMode::ABS_NONDIFF) {
        eps_str = "abs";
        eps = -1;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_NONDIFF) {
        eps_str = "clamp";
        eps = 0;
        //eps = TinyAD::EPS_1E_8;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF
      || _diff_mode == TinyAD::HessianProjectionMode::ABS_SHIFT_CLAMP_NONDIFF) {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
      }
      else{
         //非特殊情况,eps必须大于等于0
        if (eps < 0)
        {
          eps = 0;
          eps = TinyAD::EPS_1E_8;
        }
      }
  }
  else {
    if (clamp || eps == 0) {
        // we use eps = 0 as a flag for clamp projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "clamp";
        eps = 0; 
        diff_mode_str = "clamp_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_NONDIFF; 
    }
    else if (abs || eps == -1) {
        // we use eps = -1 as a flag for abs projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "abs";
        eps = -1;
        diff_mode_str = "abs_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::ABS_NONDIFF; 
    }
    else {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
        diff_mode_str = "clamp_abs_nondiff"; // trust region newton;
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF;
    }
  }
   
  if (!std::filesystem::exists("../results/"))
    std::filesystem::create_directory("../results/");
  if (!std::filesystem::exists("../results/" + experiment_folder))
    std::filesystem::create_directory("../results/" + experiment_folder);

  const std::string output_folder = "../results/" + experiment_folder + "/" 
    + pose_label + "_YM_" + std::to_string(YM) + "_PR_" + std::to_string(PR)
    + "_deform_mag_" + std::to_string(deformation_magnitude) + "_deform_ratio_" 
    + std::to_string(deformation_ratio) + "_fixed_boundary_range_" 
    + std::to_string(fixed_boundary_range)+"_diff_mode_" + diff_mode_str+ "/";
  {
    // record the statistics
    if (!std::filesystem::exists(output_folder))
      std::filesystem::create_directory(output_folder);
    if (!std::filesystem::exists(output_folder + "obj/"))
      std::filesystem::create_directory(output_folder + "obj/");
    if (!std::filesystem::exists(output_folder + "obj/" + mesh_name + "_" + eps_str + "/"))
      std::filesystem::create_directory(output_folder + "obj/" + mesh_name + "_" + eps_str + "/");
    if (!std::filesystem::exists(output_folder + "hist/"))
      std::filesystem::create_directory(output_folder + "hist/");
    if (!std::filesystem::exists(output_folder + "iter/"))
      std::filesystem::create_directory(output_folder + "iter/");
    if (!std::filesystem::exists(output_folder + "trust_region_ratio/"))
      std::filesystem::create_directory(output_folder + "trust_region_ratio/");
    if (!std::filesystem::exists(output_folder + "trust_region_eps/"))
      std::filesystem::create_directory(output_folder + "trust_region_eps/");
    if (!std::filesystem::exists(output_folder + "line_search_alpha/"))
      std::filesystem::create_directory(output_folder + "line_search_alpha/");
    if (!std::filesystem::exists(output_folder + "line_search_iter/"))
      std::filesystem::create_directory(output_folder + "line_search_iter/");
    
    #if DEBUG_OUTPUT
    if (!std::filesystem::exists(output_folder + "debug_log/"))
    {
      std::filesystem::create_directory(output_folder + "debug_log/");
      TINYAD_INIT_DEBUG_LOG(output_folder + "debug_log/log"+get_time_str()+".txt");
    }
    #endif

    
  }

  const std::string output_tag = "mesh_" + mesh_name + "_eps_" + eps_str;

  // set up viewer
  igl::opengl::glfw::Viewer viewer;

  // compute the Lame parameters
  const double MU = YM / (2 * (1 + PR));
  const double LAMBDA = YM * PR / ((1 + PR) * (1 - 2 * PR));
  const double lambda_mu_ratio = LAMBDA / MU;
 
  // print out the configuration
  {
    TINYAD_DEBUG_OUT("Diff flag: " << diff);
    TINYAD_DEBUG_OUT("*Eigenvalue Differentiable filtering strategy: " << diff_mode_str);
    TINYAD_DEBUG_OUT("Eigenvalue filtering strategy: " << eps_str);
    TINYAD_DEBUG_OUT("mu: " << MU);
    TINYAD_DEBUG_OUT("lambda: " << LAMBDA);
    TINYAD_DEBUG_OUT("*Projection threshold: " << eps);
    TINYAD_DEBUG_OUT("Mesh name: " << mesh_name);
    TINYAD_DEBUG_OUT("Pose label: " << pose_label);
    TINYAD_DEBUG_OUT("Lambda / Mu ratio: " << lambda_mu_ratio);
    TINYAD_DEBUG_OUT("Deformation magnitude: " << deformation_magnitude);
    TINYAD_DEBUG_OUT("Deformation ratio: " << deformation_ratio);
    TINYAD_DEBUG_OUT("Fixed vertices boundary range: " << fixed_boundary_range);
    TINYAD_DEBUG_OUT("Convergence threshold: " << convergence_eps);
    TINYAD_DEBUG_OUT("Young's modulus: " << YM);
    TINYAD_DEBUG_OUT("Poisson's ratio: " << PR);
    TINYAD_DEBUG_OUT("Trust region threshold: " << tr_threshold);
    TINYAD_DEBUG_OUT("Experiment folder: " << experiment_folder);
    TINYAD_DEBUG_OUT("Rotate ratio: " << rotate_ratio);
    TINYAD_DEBUG_OUT("Adaptive method: " << adaptive);
  }

  Eigen::MatrixXd V, U; // #V-by-3 3D vertex positions,V为初始,U为当前
  Eigen::MatrixXi F, FF; // #T-by-4 indices into V
  Eigen::VectorXi TriTag, TetTag;
  if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh"))
    igl::readMESH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh", V, F, FF);
  else if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh"))
    igl::readMSH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh", V, FF, F, TriTag, TetTag);
  else {
    std::cout << "Mesh " << mesh_name << " not found!" << std::endl;
    exit(1);
  }

  TINYAD_DEBUG_OUT("Read mesh with " << V.rows() << " vertices and " << F.rows() << " tetrahedrons.");

  // get boundary vertices
  igl::boundary_facets(F, FF);
  FF = FF.rowwise().reverse().eval();

  TINYAD_DEBUG_OUT("Boundary has " << FF.rows() << " faces.");

  // normalize the mesh V
  Eigen::RowVector3d mean = V.colwise().mean();
  V.rowwise() -= mean;
  double max_norm = V.rowwise().norm().maxCoeff();
  V /= max_norm;

  // set up mesh U
  U = V;

  // fixed point constraints
  Eigen::SparseMatrix<double> P;
  std::vector<unsigned int> indices_fixed;

  // set up fixed point constraints and initial deformation
  setup_initial_deformation(V, F, pose_label, deformation_magnitude, deformation_ratio, rotate_ratio, fixed_boundary_range, U, indices_fixed);
  fixed_point_constraints(P, 3*V.rows(), 3, indices_fixed);

  TINYAD_DEBUG_OUT("Finish setting up fixed point constraints.");

  bool redraw = false;
  std::mutex m;
  std::thread optimization_thread(
    [&]()
    {
      // Pre-compute triangle rest shapes in local coordinate systems
      std::vector<Eigen::Matrix3d> rest_shapes(F.rows());
      for (int f_idx = 0; f_idx < F.rows(); ++f_idx)
      {
        // Get 3D vertex positions
        Eigen::Vector3d ar = V.row(F(f_idx, 0));
        Eigen::Vector3d br = V.row(F(f_idx, 1));
        Eigen::Vector3d cr = V.row(F(f_idx, 2));
        Eigen::Vector3d dr = V.row(F(f_idx, 3));

        // Save 3-by-3 matrix with edge vectors as columns
        rest_shapes[f_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
      };

      /*
          TinyAD::scalar_function<3>: 创建一个标量函数，每个变量是3维向量（例如3D顶点坐标）
          TinyAD::range(V.rows()): 定义变量的范围，V.rows() 是顶点数量
          auto func: 自动推断函数对象类型
      */ 
      // TinyAD::scalar_function(n)：这是一个函数，可以计算它的函数值、梯度、Hessian的函数，其自变量索引的范围为n
      // Set up function with 3d vertex positions as variables.
      auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

      // func.add_elements<n>(element_range, energy_function); 
      // 定义元素的能量项，1.n为元素的dim，其中2.element_range是元素索引的范围，
      // 3.energy_function是一个lambda函数，接受一个元素对象，返回该元素的能量值
      // 4. lamda函数 [捕获变量] (输入参数) -> 返回值 
      // 5. 所有参与自动微分计算的变量和中间结果，都必须使用 TINYAD_SCALAR_TYPE 或其衍生的类型
      // 6. element.handle对应element_range中的当前元素的索引，element.variables()对应输入变量，想要访问输入变量的第i个变量
      // 7. 在这个地方，输入变量是顶点数组，element_range是F数组，即按F计算能量
      // Add objective term per element. Each connecting 4 vertices.
      // "neo-hooken energy" is defined on each tetrahedron, so element_range is F.rows() and element.variables() gives the vertex positions of the current tetrahedron.
      func.add_elements<4>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element) {
          // Evaluate element using either double or TinyAD::Double
          using T = TINYAD_SCALAR_TYPE(element);

          // Get variable 3d vertex positions
          Eigen::Index f_idx = element.handle;
          Eigen::Vector3<T> a = element.variables(F(f_idx, 0));
          Eigen::Vector3<T> b = element.variables(F(f_idx, 1));
          Eigen::Vector3<T> c = element.variables(F(f_idx, 2));
          Eigen::Vector3<T> d = element.variables(F(f_idx, 3));

          Eigen::Matrix3<T> M = TinyAD::col_mat(b - a, c - a, d - a);
          Eigen::Matrix3d Mr = rest_shapes[f_idx];
          Eigen::Matrix3<T> J = M * Mr.inverse(); // 可以放外边 to be optimized by zj

          // Compute the stable Neo-Hookean energy [Smith et al. 2018]
          double A = 0.5 * Mr.determinant(); //可以放外边 to be optimized by zj
          const double mu = MU;
          const double lambda = LAMBDA;
          auto Ic = (J.transpose() * J).trace();
          auto detF = J.determinant();
          double alpha = 1.0 + mu / lambda;
          auto W = mu / 2.0 * (Ic - 3.0) + lambda / 2.0 * (detF - alpha) * (detF - alpha);
          return A * W;
      });

      // to be optimized by zj: 可以考虑把 rest_shapes 以及 pre-computed Mr.inverse() 放到外边，作为常量传入 lambda 函数中，这样就不需要每次迭代都计算 rest_shapes 和 Mr.inverse() 了
      // to be optimized by zj: 还可以考虑把 A 也放到外边，因为 A 只和 rest_shapes 相关，而 rest_shapes 是不变的
      // to be optimized by zj: 还可以考虑把 lambda 和 mu 放到外边，因为它们也是不变的
      // to be modified by zj: 动态状态下的PD，增加local step，即每个元素的能量项不仅依赖于当前状态，还依赖于上一个状态，这样可以增加稳定性，尤其是在使用abs投影时

      // func.x_from_data() 是 TinyAD 提供的数据格式转换工具，将外部数据（如顶点矩阵）转换为优化所需的展平向量格式，使代码更简洁、更安全。
      // v_idx 的取值范围是由 TinyAD::scalar_function<k>(n_variables) 中的 n_variables 决定的，即函数的输入变量的维度。
      // 在这个例子中，n_variables 是 V.rows()，即顶点数量，因此 v_idx 的取值范围是 [0, V.rows()-1]。
      // Assemble inital x vector from U matrix.
      // x_from_data(...) takes a lambda function that maps
      // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) {
          return U.row(v_idx);
      });

      TINYAD_DEBUG_OUT("Initial energy: " << func.eval(x));

      // Projected Newton
      TinyAD::LinearSolver solver; //Eigen::SimplicialLDLT*,ConjugateGradient,SparseLU
      int max_iters = 200; // 迭代次数可以根据需要调整
      Eigen::VectorXd d;
      Eigen::SparseMatrix<double> H;
      std::vector<double> hist;
      // record the trust region ratio
      // 衡量模型预测的准确性 ρ = (实际下降) / (预测下降), 
      // ρ < 0         → 实际目标函数值增加（拒绝步长）
      // 0 ≤ ρ < 0.25  → 模型质量差，缩小信赖域
      // 0.25 ≤ ρ < 0.75 → 模型质量一般，保持信赖域
      // ρ ≥ 0.75      → 模型质量好，可以放大信赖域
      // double delta;  // 当前信赖域半径
      // double eta1;   // 接受步长的阈值（通常0.25）
      // double eta2;   // 放大半径的阈值（通常0.75）
      std::vector<double> hist_trust_region_ratio;
      hist_trust_region_ratio.push_back(0.0);

      std::vector<double> hist_trust_region_eps; // 记录每次迭代使用的eps值，以观察自适应策略的变化
      std::vector<double> hist_line_search_alpha; // 记录每次迭代的线搜索步长
      std::vector<int> hist_line_search_iter; // 记录每次迭代的线搜索迭代次数

      //double cubic_sigma = 0.0;
      //Eigen::VectorXd cubic_lambda_vec = Eigen::VectorXd::Zero();
      
      //当global matrix发生变化时，进行下面的牛顿投影和线搜索步骤

      if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING2
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING3
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING5
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING4)
      {
        eps = 1.0;
      }
      for (int i = 0; i < max_iters; ++i)
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".mesh", U, F, FF);
      
        // switch between clamp or abs depending on whether the trust region ratio is close to 1
        //if(eps_str == "adaptive") {
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF
        ||_diff_mode == TinyAD::HessianProjectionMode::ABS_SHIFT_CLAMP_NONDIFF) {
          eps = (std::fabs(hist_trust_region_ratio.back() - 1.0) < tr_threshold) ? 0.0 : -1; // tr_threshold信赖域接受步长的阈值，通常设置为0.01,大于接受

          if (eps == 0.0) {
            //eps = TinyAD::EPS_1E_8; // 避免eps为0导致的数值问题
            TINYAD_DEBUG_OUT("Switch to clamp");  
          }
          else {
            TINYAD_DEBUG_OUT("Switch to abs");
          }
          hist_trust_region_eps.push_back(eps);
        }

        //ok, beta0 adaptive
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING 
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING2
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING3
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING5
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING4) //adaptive mode
        {
          if (TinyAD::isPostive(adaptive + TinyAD::EPS_1E_8)) // >0
          {
            double prev_ratio = hist_trust_region_ratio.back();
            if (std::fabs(prev_ratio-1.0) < tr_threshold) //不需要改变beta0
            {
              eps = 0; 
            }
            else //需要改变beta0
            {
              if (TinyAD::isZero(eps))
              {
                TINYAD_DEBUG_OUT("Switch to abs");
                eps = 1.0;
              }

              if(TinyAD::isPostive(prev_ratio, 0.9))// >0.9模型激进，缩小beta0 <1
              {
                eps *= 0.9;
              }
              else if (TinyAD::isPostive(0.25, prev_ratio))// < 0.2 模型保守，放大beta0 >1
              {
                if (TinyAD::isNonPostive(eps, 1.0))
                {
                  eps = 1.0;
                }
                else{
                  eps = eps * 1.02;
                } 
              }
                
            }
            
            eps = std::max(0.0, std::min(1.2, eps)); 
          }
          else // not adaptive [0 or 1]
          {
            eps = 1.0; 
            double prev_ratio = hist_trust_region_ratio.back();
            if (std::fabs(prev_ratio-1.0) < tr_threshold) //不需要改变beta0
            {
              eps = 0; 
            }
            
          }
            
          TINYAD_DEBUG_OUT("eps: "<<eps); 
          hist_trust_region_eps.push_back(eps);
        }
      

        auto [f, g, H_proj] = func.eval_with_hessian_proj(x, eps, _diff_mode); // 
        Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量

        // record the energy
        hist.push_back(f);

        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);

        H = H_proj;
        

        Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
        Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
        d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
        d = P.transpose() * d;

        TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
        TINYAD_DEBUG_OUT("d norm " << i << " =  " << d.norm());
        TINYAD_DEBUG_OUT("g norm " << i << " =  " << g.norm());

        // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
        // 优化方案1： 相对收敛
        // if (newton_decrement < convergence_eps * initial_decrement) break;  
        // 方案3：梯度范数 
        // if (g.norm() < convergence_eps) break;
        // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
        // 方案5：根据材料参数调整收敛标准
        //  LAMBDA是材料参数，不是尺度参数
        // // 使用特征长度进行无量纲化
        // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
        // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
        // // 物理意义的收敛判据
        // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
        if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
          break;

        // line search
        Eigen::VectorXd x_prev = x;
        x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 100, 1e-8);
        double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
        hist_line_search_alpha.push_back(alpha);

        int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
        hist_line_search_iter.push_back(line_search_iter);
        
        // compute the trust region ratio
        double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
        hist_trust_region_ratio.push_back(trust_region_ratio);

        TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);

        // Write final x vector to U matrix.
        // x_to_data(...) takes a lambda function that writes the final value
        // of each variable (Eigen::Vector2d) back to our U matrix.
        func.x_to_data(x, [&] (int v_idx, const Eigen::Vector3d& p) {
            U.row(v_idx) = p;
            });
        {
          std::lock_guard<std::mutex> lock(m);
          redraw = true; 
        }
      }

      TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
      hist.push_back(func.eval(x));

      // output all the optimization statistics
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".mesh", U, F, FF);
        
        std::ofstream output_file_fixed(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_fixed_vid.txt");
        std::ostream_iterator<int> output_iterator_fixed(output_file_fixed, "\n");
        std::copy(std::begin(indices_fixed), std::end(indices_fixed), output_iterator_fixed);

        std::ofstream output_file(output_folder + "hist/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(std::begin(hist), std::end(hist), output_iterator);

        std::ofstream output_file_trust_region_ratio(output_folder + "trust_region_ratio/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_ratio(output_file_trust_region_ratio, "\n");
        std::copy(std::begin(hist_trust_region_ratio), std::end(hist_trust_region_ratio), output_iterator_trust_region_ratio);

        std::ofstream output_file_trust_region_eps(output_folder + "trust_region_eps/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_eps(output_file_trust_region_eps, "\n");
        std::copy(std::begin(hist_trust_region_eps), std::end(hist_trust_region_eps), output_iterator_trust_region_eps);

        std::ofstream output_file_line_search_alpha(output_folder + "line_search_alpha/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_line_search_alpha(output_file_line_search_alpha, "\n");
        std::copy(std::begin(hist_line_search_alpha), std::end(hist_line_search_alpha), output_iterator_line_search_alpha);

        std::ofstream output_file_line_search_iter(output_folder + "line_search_iter/" + output_tag + ".txt");
        std::ostream_iterator<int> output_iterator_line_search_iter(output_file_line_search_iter, "\n");
        std::copy(std::begin(hist_line_search_iter), std::end(hist_line_search_iter), output_iterator_line_search_iter);

        std::ofstream output_file_iter(output_folder + "iter/" + output_tag + ".txt");
        output_file_iter << (hist.size()-1) << std::endl;
      }

      TINYAD_DEBUG_OUT("======== The End ========");
      // comment this out later
      // close the viewer
      exit(0);
    });

  // Plot mesh
  viewer.core().is_animating = true;
  viewer.data().set_mesh(U, FF);
  viewer.core().align_camera_center(U);
  viewer.data().show_lines = true;
  viewer.core().depth_test = true;
  viewer.data().line_width = 1.0;

  // 设置透明材质
  Eigen::Vector3d color(0.2, 0.2, 0.8);
  viewer.data().uniform_colors(
      //Eigen::Vector3d(color3(0), color3(1), color3(2)),  // 漫反射
      Eigen::Vector3d(color(0), color(1), color(2)),  // 漫反射
      Eigen::Vector3d(0.2, 0.2, 0.2),  // 环境光
      Eigen::Vector3d(0.0, 0.0, 0.0)   // 镜面反射
  );
    
  // 设置面的透明度 - 使用材质属性
  viewer.data().face_based = true;  // 基于面的渲染

  viewer.callback_pre_draw = [&] (igl::opengl::glfw::Viewer& viewer)
  {
    if(redraw)
    {
      viewer.data().set_vertices(U);
      viewer.core().align_camera_center(U);
      {
        std::lock_guard<std::mutex> lock(m);
        redraw = false;
      }
    }
    return false;
  };

  // 在初始化回调中设置标题
  viewer.callback_init = [&](igl::opengl::glfw::Viewer& v)
  {
      // 获取 GLFW 窗口指针并设置标题
      glfwSetWindowTitle(v.window, diff_mode_str.c_str());
      return false;
  };

  viewer.launch();
  if(optimization_thread.joinable())
  {
    optimization_thread.join();
  }

  #if DEBUG_OUTPUT
    TINYAD_CLOSE_DEBUG_LOG();
  #endif
  return 0;
}

int shear_reg_projected_newton(int argc, char** argv)
{
  TINYAD_DEBUG_OUT("#shear_projected_newton " );
  // parse command line arguments
  cxxopts::Options options("Projected Newton with a trust region", "Choose eigenvalue filtering method: adaptive, clamp, abs");

  options.add_options()
    ("smooth_mode", "add differentiable eigenvalue projection strategy ", cxxopts::value<std::string>()->default_value("none")) // 若设置diff，后续可微的clamp和abs等
    ("diff", "add differentiable eigenvalue projection strategy ", cxxopts::value<bool>()->default_value("false")) // 若设置diff，后续可微的clamp和abs等
    ("diff_mode", "differentiable eigenvalue projection mode", cxxopts::value<std::string>()->default_value("auto")) // 若设置diff，后续可微的clamp和abs等
    // ("delta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("0.001"))  // 过渡区间宽度
   // ("beta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("10.0"))// 光滑参数
    ("abs", "use absolute eigenvalue projection strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("clamp", "use eigenvalue clamping strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("p,epsilon", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("-0.5"))
    ("n,mesh_name", "Mesh name", cxxopts::value<std::string>()->default_value("bimba"))
    ("l,pose_label", "Pose label", cxxopts::value<std::string>()->default_value("stretch"))
    ("g,deformation_magnitude", "The magnitude of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("t,deformation_ratio", "The ratio of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("b,fixed_boundary_range", "The range of fixed vertices on the boundary", cxxopts::value<double>()->default_value("0.1"))
    ("c,convergence_eps", "The convergence threshold", cxxopts::value<double>()->default_value("1e-5"))
    ("ym", "Young's modulus (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("1e8"))
    ("pr", "Poisson's ratio (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("0.495"))
    ("tr", "trust region ratio threshold", cxxopts::value<double>()->default_value("0.01"))
    ("experiment_name", "experiment name", cxxopts::value<std::string>()->default_value(""))
    ("rotate_ratio", "The ratio of the rotation", cxxopts::value<double>()->default_value("0.5"))
    ("h,help", "show help")
    ("adaptive", "adaptive according to rho ", cxxopts::value<double>()->default_value("1e-5")) // >0 为adaptive
  ;
  
  auto result = options.parse(argc, argv);
  if (result.count("help"))
  {
      std::cout << options.help() << "\n";
      return 0;
  }

  const double adaptive = result["adaptive"].as<double>();
  // bool b_adaptive = false;
  // if ( TinyAD::isPostive(adaptive))
  // {
  //   b_adaptive = true;
  // }
  const bool diff = result["diff"].as<bool>();
  //const int _mode = result["diff_mode"].as<int>();
  std::string diff_mode_str = result["diff_mode"].as<std::string>();
  TinyAD::HessianProjectionMode _diff_mode = parse_projection_mode(diff_mode_str);
  // std::string delta_str = result["delta"].as<std::string>();
  // double delta = std::stod(delta_str);
  // std::string beta_str = result["beta"].as<std::string>();
  // double beta = std::stod(beta_str);


  const bool abs = result["abs"].as<bool>();
  const bool clamp = result["clamp"].as<bool>();
  std::string eps_str = result["epsilon"].as<std::string>();
  double eps = std::stod(eps_str);
  std::string mesh_name = result["mesh_name"].as<std::string>();
  std::string pose_label = result["pose_label"].as<std::string>();
  const double deformation_magnitude = result["deformation_magnitude"].as<double>();
  const double deformation_ratio = result["deformation_ratio"].as<double>();
  const double fixed_boundary_range = result["fixed_boundary_range"].as<double>();
  const double convergence_eps = result["convergence_eps"].as<double>();
  const double YM = result["ym"].as<double>();
  const double PR = result["pr"].as<double>();
  const double tr_threshold = result["tr"].as<double>(); // 信赖域接受步长的阈值，通常设置为0.01
  std::string experiment_folder = result["experiment_name"].as<std::string>() == "" ? ("figure_" + mesh_name) : result["experiment_name"].as<std::string>();
  std::string time_str = get_time_str();
  experiment_folder = experiment_folder + time_str;
  const double rotate_ratio = result["rotate_ratio"].as<double>();

  /*
  * 0: clamp
  * -1: abs
  * -0.5: adaptive (default)
  * 0.5: differentiable  (if diff is set)
  */
  if (diff)
  {
      eps_str =  diff_mode_str;
      //下面三种情况,eps值有特殊用处
      if (_diff_mode == TinyAD::HessianProjectionMode::ABS_NONDIFF) {
        eps_str = "abs";
        eps = -1;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_NONDIFF) {
        eps_str = "clamp";
        eps = 0;
        //eps = TinyAD::EPS_1E_8;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF
      || _diff_mode == TinyAD::HessianProjectionMode::ABS_SHIFT_CLAMP_NONDIFF) {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
      }
      else{
         //非特殊情况,eps必须大于等于0
        if (eps < 0)
        {
          eps = 0;
          eps = TinyAD::EPS_1E_8;
        }
      }
  }
  else {
    if (clamp || eps == 0) {
        // we use eps = 0 as a flag for clamp projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "clamp";
        eps = 0; 
        diff_mode_str = "clamp_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_NONDIFF; 
    }
    else if (abs || eps == -1) {
        // we use eps = -1 as a flag for abs projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "abs";
        eps = -1;
        diff_mode_str = "abs_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::ABS_NONDIFF; 
    }
    else {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
        diff_mode_str = "clamp_abs_nondiff"; // trust region newton;
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF;
    }
  }
   
  if (!std::filesystem::exists("../results/"))
    std::filesystem::create_directory("../results/");
  if (!std::filesystem::exists("../results/" + experiment_folder))
    std::filesystem::create_directory("../results/" + experiment_folder);

  const std::string output_folder = "../results/" + experiment_folder + "/" 
    + pose_label + "_YM_" + std::to_string(YM) + "_PR_" + std::to_string(PR)
    + "_deform_mag_" + std::to_string(deformation_magnitude) + "_deform_ratio_" 
    + std::to_string(deformation_ratio) + "_fixed_boundary_range_" 
    + std::to_string(fixed_boundary_range)+"_diff_mode_" + diff_mode_str+ "/";
  {
    // record the statistics
    if (!std::filesystem::exists(output_folder))
      std::filesystem::create_directory(output_folder);
    if (!std::filesystem::exists(output_folder + "obj/"))
      std::filesystem::create_directory(output_folder + "obj/");
    if (!std::filesystem::exists(output_folder + "obj/" + mesh_name + "_" + eps_str + "/"))
      std::filesystem::create_directory(output_folder + "obj/" + mesh_name + "_" + eps_str + "/");
    if (!std::filesystem::exists(output_folder + "hist/"))
      std::filesystem::create_directory(output_folder + "hist/");
    if (!std::filesystem::exists(output_folder + "iter/"))
      std::filesystem::create_directory(output_folder + "iter/");
    if (!std::filesystem::exists(output_folder + "trust_region_ratio/"))
      std::filesystem::create_directory(output_folder + "trust_region_ratio/");
    if (!std::filesystem::exists(output_folder + "trust_region_eps/"))
      std::filesystem::create_directory(output_folder + "trust_region_eps/");
    if (!std::filesystem::exists(output_folder + "line_search_alpha/"))
      std::filesystem::create_directory(output_folder + "line_search_alpha/");
    if (!std::filesystem::exists(output_folder + "line_search_iter/"))
      std::filesystem::create_directory(output_folder + "line_search_iter/");
    
    #if DEBUG_OUTPUT
    if (!std::filesystem::exists(output_folder + "debug_log/"))
    {
      std::filesystem::create_directory(output_folder + "debug_log/");
      TINYAD_INIT_DEBUG_LOG(output_folder + "debug_log/log"+get_time_str()+".txt");
    }
    #endif

    
  }

  const std::string output_tag = "mesh_" + mesh_name + "_eps_" + eps_str;

  // set up viewer
  igl::opengl::glfw::Viewer viewer;
  bool g_paused = false;      // 暂停标志
  bool recording_started = false; //录制
  int iter_i = 0;
 
  // compute the Lame parameters
  const double MU = YM / (2 * (1 + PR));
  const double LAMBDA = YM * PR / ((1 + PR) * (1 - 2 * PR));
  const double lambda_mu_ratio = LAMBDA / MU;
 
  // print out the configuration
  {
    TINYAD_DEBUG_OUT("Diff flag: " << diff);
    TINYAD_DEBUG_OUT("*Eigenvalue Differentiable filtering strategy: " << diff_mode_str);
    TINYAD_DEBUG_OUT("Eigenvalue filtering strategy: " << eps_str);
    TINYAD_DEBUG_OUT("mu: " << MU);
    TINYAD_DEBUG_OUT("lambda: " << LAMBDA);
    TINYAD_DEBUG_OUT("*Projection threshold: " << eps);
    TINYAD_DEBUG_OUT("Mesh name: " << mesh_name);
    TINYAD_DEBUG_OUT("Pose label: " << pose_label);
    TINYAD_DEBUG_OUT("Lambda / Mu ratio: " << lambda_mu_ratio);
    TINYAD_DEBUG_OUT("Deformation magnitude: " << deformation_magnitude);
    TINYAD_DEBUG_OUT("Deformation ratio: " << deformation_ratio);
    TINYAD_DEBUG_OUT("Fixed vertices boundary range: " << fixed_boundary_range);
    TINYAD_DEBUG_OUT("Convergence threshold: " << convergence_eps);
    TINYAD_DEBUG_OUT("Young's modulus: " << YM);
    TINYAD_DEBUG_OUT("Poisson's ratio: " << PR);
    TINYAD_DEBUG_OUT("Trust region threshold: " << tr_threshold);
    TINYAD_DEBUG_OUT("Experiment folder: " << experiment_folder);
    TINYAD_DEBUG_OUT("Rotate ratio: " << rotate_ratio);
    TINYAD_DEBUG_OUT("Adaptive method: " << adaptive);
  }

  Eigen::MatrixXd V, U; // #V-by-3 3D vertex positions,V为初始,U为当前
  Eigen::MatrixXi F, FF; // #T-by-4 indices into V
  Eigen::VectorXi TriTag, TetTag;
  if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh"))
    igl::readMESH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh", V, F, FF);
  else if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh"))
    igl::readMSH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh", V, FF, F, TriTag, TetTag);
  else {
    std::cout << "Mesh " << mesh_name << " not found!" << std::endl;
    exit(1);
  }

  TINYAD_DEBUG_OUT("Read mesh with " << V.rows() << " vertices and " << F.rows() << " tetrahedrons.");

  // get boundary vertices
  igl::boundary_facets(F, FF);
  FF = FF.rowwise().reverse().eval();

  TINYAD_DEBUG_OUT("Boundary has " << FF.rows() << " faces.");

  // normalize the mesh V
  Eigen::RowVector3d mean = V.colwise().mean();
  V.rowwise() -= mean;
  double max_norm = V.rowwise().norm().maxCoeff();
  V /= max_norm;

  // set up mesh U
  U = V;

  // fixed point constraints
  Eigen::SparseMatrix<double> P;
  std::vector<unsigned int> indices_fixed;

  // set up fixed point constraints and initial deformation
  setup_initial_deformation(V, F, pose_label, deformation_magnitude, deformation_ratio, rotate_ratio, fixed_boundary_range, U, indices_fixed);
  fixed_point_constraints(P, 3*V.rows(), 3, indices_fixed);

  TINYAD_DEBUG_OUT("Finish setting up fixed point constraints.");

  bool redraw = false;
  std::mutex m;
  std::thread optimization_thread(
    [&]()
    {
      // Pre-compute triangle rest shapes in local coordinate systems
      std::vector<Eigen::Matrix3d> rest_shapes(F.rows());
      for (int f_idx = 0; f_idx < F.rows(); ++f_idx)
      {
        // Get 3D vertex positions
        Eigen::Vector3d ar = V.row(F(f_idx, 0));
        Eigen::Vector3d br = V.row(F(f_idx, 1));
        Eigen::Vector3d cr = V.row(F(f_idx, 2));
        Eigen::Vector3d dr = V.row(F(f_idx, 3));

        // Save 3-by-3 matrix with edge vectors as columns
        rest_shapes[f_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
      };

      /*
          TinyAD::scalar_function<3>: 创建一个标量函数，每个变量是3维向量（例如3D顶点坐标）
          TinyAD::range(V.rows()): 定义变量的范围，V.rows() 是顶点数量
          auto func: 自动推断函数对象类型
      */ 
      // TinyAD::scalar_function(n)：这是一个函数，可以计算它的函数值、梯度、Hessian的函数，其自变量索引的范围为n
      // Set up function with 3d vertex positions as variables.
      auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

      // func.add_elements<n>(element_range, energy_function); 
      // 定义元素的能量项，1.n为元素的dim，其中2.element_range是元素索引的范围，
      // 3.energy_function是一个lambda函数，接受一个元素对象，返回该元素的能量值
      // 4. lamda函数 [捕获变量] (输入参数) -> 返回值 
      // 5. 所有参与自动微分计算的变量和中间结果，都必须使用 TINYAD_SCALAR_TYPE 或其衍生的类型
      // 6. element.handle对应element_range中的当前元素的索引，element.variables()对应输入变量，想要访问输入变量的第i个变量
      // 7. 在这个地方，输入变量是顶点数组，element_range是F数组，即按F计算能量
      // Add objective term per element. Each connecting 4 vertices.
      // "neo-hooken energy" is defined on each tetrahedron, so element_range is F.rows() and element.variables() gives the vertex positions of the current tetrahedron.
      func.add_elements<4>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element) {
          // Evaluate element using either double or TinyAD::Double
          using T = TINYAD_SCALAR_TYPE(element);

          // Get variable 3d vertex positions
          Eigen::Index f_idx = element.handle;
          Eigen::Vector3<T> a = element.variables(F(f_idx, 0));
          Eigen::Vector3<T> b = element.variables(F(f_idx, 1));
          Eigen::Vector3<T> c = element.variables(F(f_idx, 2));
          Eigen::Vector3<T> d = element.variables(F(f_idx, 3));

          Eigen::Matrix3<T> M = TinyAD::col_mat(b - a, c - a, d - a);
          Eigen::Matrix3d Mr = rest_shapes[f_idx];
          Eigen::Matrix3<T> J = M * Mr.inverse(); // 可以放外边 to be optimized by zj

          // Compute the stable Neo-Hookean energy [Smith et al. 2018]
          double A = 0.5 * Mr.determinant(); //可以放外边 to be optimized by zj
          const double mu = MU;
          const double lambda = LAMBDA;
          auto Ic = (J.transpose() * J).trace();
          auto detF = J.determinant();
          double alpha = 1.0 + mu / lambda;
          auto W = mu / 2.0 * (Ic - 3.0) + lambda / 2.0 * (detF - alpha) * (detF - alpha);
          return A * W;
      });

      // to be optimized by zj: 可以考虑把 rest_shapes 以及 pre-computed Mr.inverse() 放到外边，作为常量传入 lambda 函数中，这样就不需要每次迭代都计算 rest_shapes 和 Mr.inverse() 了
      // to be optimized by zj: 还可以考虑把 A 也放到外边，因为 A 只和 rest_shapes 相关，而 rest_shapes 是不变的
      // to be optimized by zj: 还可以考虑把 lambda 和 mu 放到外边，因为它们也是不变的
      // to be modified by zj: 动态状态下的PD，增加local step，即每个元素的能量项不仅依赖于当前状态，还依赖于上一个状态，这样可以增加稳定性，尤其是在使用abs投影时

      // func.x_from_data() 是 TinyAD 提供的数据格式转换工具，将外部数据（如顶点矩阵）转换为优化所需的展平向量格式，使代码更简洁、更安全。
      // v_idx 的取值范围是由 TinyAD::scalar_function<k>(n_variables) 中的 n_variables 决定的，即函数的输入变量的维度。
      // 在这个例子中，n_variables 是 V.rows()，即顶点数量，因此 v_idx 的取值范围是 [0, V.rows()-1]。
      // Assemble inital x vector from U matrix.
      // x_from_data(...) takes a lambda function that maps
      // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) {
          return U.row(v_idx);
      });

      TINYAD_DEBUG_OUT("Initial energy: " << func.eval(x));

      // Projected Newton
      TinyAD::LinearSolver solver; //Eigen::SimplicialLDLT*,ConjugateGradient,SparseLU
      int max_iters = 200; // 迭代次数可以根据需要调整
      Eigen::VectorXd d;
      Eigen::SparseMatrix<double> H;
      std::vector<double> hist;
      // record the trust region ratio
      // 衡量模型预测的准确性 ρ = (实际下降) / (预测下降), 
      // ρ < 0         → 实际目标函数值增加（拒绝步长）
      // 0 ≤ ρ < 0.25  → 模型质量差，缩小信赖域
      // 0.25 ≤ ρ < 0.75 → 模型质量一般，保持信赖域
      // ρ ≥ 0.75      → 模型质量好，可以放大信赖域
      // double delta;  // 当前信赖域半径
      // double eta1;   // 接受步长的阈值（通常0.25）
      // double eta2;   // 放大半径的阈值（通常0.75）
      std::vector<double> hist_trust_region_ratio;
      hist_trust_region_ratio.push_back(0.0);

      std::vector<double> hist_trust_region_eps; // 记录每次迭代使用的eps值，以观察自适应策略的变化
      std::vector<double> hist_line_search_alpha; // 记录每次迭代的线搜索步长
      std::vector<int> hist_line_search_iter; // 记录每次迭代的线搜索迭代次数

      std::vector<double> hist_energy_injection_ratio;
      hist_energy_injection_ratio.push_back(0.0);

      //double cubic_sigma = 0.0;
      //Eigen::VectorXd cubic_lambda_vec = Eigen::VectorXd::Zero();
      
      //当global matrix发生变化时，进行下面的牛顿投影和线搜索步骤
      bool useClamp = false;
      int increase_count = 0;
     
      if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING2
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING3
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING5
      || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING4
      ) // 从abs开始
      {
        eps = 1.0;
        TINYAD_DEBUG_OUT("Switch to abs");
      }
      if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SHEAR)
      {
        hist_trust_region_ratio.push_back(1.0);
      }

      for (int i = 0; i < max_iters; ++i)
      {
        double prev_ratio = hist_trust_region_ratio.back(); //初值为0
        iter_i = i;
        recording_started = false;
        if (g_paused )
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          i--;
          recording_started = false;
          continue;
        }

        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".mesh", U, F, FF);
      
        // switch between clamp or abs depending on whether the trust region ratio is close to 1
        //if(eps_str == "adaptive") {
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF
        ||_diff_mode == TinyAD::HessianProjectionMode::ABS_SHIFT_CLAMP_NONDIFF 
        ||_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SHEAR 
        ||_diff_mode == TinyAD::HessianProjectionMode::SMOOTH_TR) 
        {
          eps = (std::fabs(hist_trust_region_ratio.back() - 1.0) < tr_threshold) ? 0.0 : -1; // tr_threshold信赖域接受步长的阈值，通常设置为0.01,大于接受

          if (eps == 0.0) {
            //eps = TinyAD::EPS_1E_8; // 避免eps为0导致的数值问题
            TINYAD_DEBUG_OUT("Switch to clamp");  
          }
          else {
            TINYAD_DEBUG_OUT("Switch to abs");
          }
          hist_trust_region_eps.push_back(eps);
        }

        //ok, beta0 adaptive
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING 
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SMOOTH
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING2
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING3
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING5
        || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING4) //adaptive mode
        {
          #if 0
          if (TinyAD::isPostive(adaptive + TinyAD::EPS_1E_8)) // >0
          {
            double prev_ratio = hist_trust_region_ratio.back(); //初值为0
            if (std::fabs(prev_ratio-1.0) < tr_threshold) //不需要改变beta0
            {
              TINYAD_DEBUG_OUT("Switch to clamp");
              eps = 0; 
              //useClamp = true;
            }
            else //需要改变beta0
            {
              if (TinyAD::isZero(eps))
              {
                TINYAD_DEBUG_OUT("Switch to abs");
                eps = 1.0;
              }
               
              if(TinyAD::isPostive(prev_ratio, 0.9))// >0.9模型激进，缩小beta0 <1
              {
                eps *= 0.9;
              }
              else if (TinyAD::isPostive(0.25, prev_ratio))// < 0.2 模型保守，放大beta0 >1
              {
                
                if (TinyAD::isNonPostive(eps, 1.0))
                {
                  eps = 1.0;
                }
                else
                {
                  eps = eps * 1.02;
                } 
              }
                
            }
            
            eps = std::max(0.0, std::min(1.2, eps)); 
          }
          #endif 
          
          if (TinyAD::isPostive(adaptive + TinyAD::EPS_1E_8)) // >0
          {
            
            if (std::fabs(prev_ratio-1.0) < tr_threshold) //不需要改变beta0
            {
              if (!useClamp) //switch mode
              {
                TINYAD_DEBUG_OUT("Switch to clamp"); 
              }
              eps = 0; 
              useClamp = true;
              
            }
            else //需要改变beta0
            {
              
              if(useClamp) //switch mode
              {
                TINYAD_DEBUG_OUT("Switch to abs");
                eps = 1.0;
                useClamp = false;
              }
              if(TinyAD::isPostive(prev_ratio, 0.9))// >0.9模型激进，缩小beta0 <1
              {
                eps *= 0.9;
              }
              else if (TinyAD::isPostive(0.25, prev_ratio))// < 0.2 模型保守，放大beta0 >1
              {
                if (i > 0) // 第一次不调整
                {
                  eps = eps * 1.02; 
                }  
              }
                
            }
            
            eps = std::max(0.0, std::min(1.2, eps)); 
          }
          else // not adaptive [0 or 1]
          {
            eps = 1.0; 
            // double prev_ratio = hist_trust_region_ratio.back();
            // if (std::fabs(prev_ratio-1.0) < tr_threshold) //不需要改变beta0
            // {
            //   eps = 0; 
            // }
            
          }
            
          TINYAD_DEBUG_OUT("eps: "<<eps); 
          hist_trust_region_eps.push_back(eps);
        }
      
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SHEAR )
        {
          eps = hist_trust_region_ratio.back(); 
          TINYAD_DEBUG_OUT("last tr_ratio: "<<eps);
        }
        
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x, eps, _diff_mode); // 
        Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量

        // record the energy
        hist.push_back(f);

        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);

        H = H_proj;
        

        Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
        Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
        d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
        d = P.transpose() * d;

        TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
        TINYAD_DEBUG_OUT("d norm " << i << " =  " << d.norm());
        TINYAD_DEBUG_OUT("g norm " << i << " =  " << g.norm());

        // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
        // 优化方案1： 相对收敛
        // if (newton_decrement < convergence_eps * initial_decrement) break;  
        // 方案3：梯度范数 
        // if (g.norm() < convergence_eps) break;
        // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
        // 方案5：根据材料参数调整收敛标准
        //  LAMBDA是材料参数，不是尺度参数
        // // 使用特征长度进行无量纲化
        // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
        // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
        // // 物理意义的收敛判据
        // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
        if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
          break;

        // line search
        Eigen::VectorXd x_prev = x;
        double decay = 0.8; // clamp默认值,每次线搜索迭代的步长衰减因子，通常设置为0.5到0.8之间

        // if (_diff_mode == TinyAD::HessianProjectionMode::ABS_NONDIFF 
        //   || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING_SHEAR
        //   || _diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF
        // )
        // {
        //   int mode = setAggressiveMode(hist_trust_region_ratio.back(), 1.0-tr_threshold, 0.25);
        //   TINYAD_DEBUG_OUT("Line search mode: " << mode);
        //   adjustLineSearchParamsByMode(mode, decay);
        //   TINYAD_DEBUG_OUT("Line search decay: " << decay);
        // }

        x = TinyAD::line_search(x, d, f, g, func, 1.0, decay, 100, 1e-8);
        double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
        hist_line_search_alpha.push_back(alpha);

        int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
        hist_line_search_iter.push_back(line_search_iter);
        
        // compute the trust region ratio
        double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
        hist_trust_region_ratio.push_back(trust_region_ratio);

        TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);

        //能量注入
        double energy_inj = x.transpose() * (H_proj - H0)* x;
        double total_energy = hist.back();
        if (total_energy < 1e-8)
        {
          total_energy  +=  1e-8; 
        }
        hist_energy_injection_ratio.push_back(energy_inj/total_energy);
        TINYAD_DEBUG_OUT("Energy injected, energy injected ratio: " << energy_inj <<","<< hist_energy_injection_ratio.back());


        // if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_BLENDING2)
        // {
        //   if (trust_region_ratio >= prev_ratio + tr_threshold) {
        //     increase_count = 1;
        //     TINYAD_DEBUG_OUT("beta_max is good.");
        //   } else if (trust_region_ratio < prev_ratio - tr_threshold) {
        //     increase_count = -1;
        //     TINYAD_DEBUG_OUT("beta_max is poor.");
        //   } else {
        //     increase_count = 0;
        //     TINYAD_DEBUG_OUT("beta_max is acceptable.");
        //   }
        // }

        // Write final x vector to U matrix.
        // x_to_data(...) takes a lambda function that writes the final value
        // of each variable (Eigen::Vector2d) back to our U matrix.
        func.x_to_data(x, [&] (int v_idx, const Eigen::Vector3d& p) {
            U.row(v_idx) = p;
            });
        {
          std::lock_guard<std::mutex> lock(m);
          redraw = true; 
          recording_started = true;
        }

        TINYAD_DEBUG_OUT("---------------------------------------------------");
      }

      TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
      hist.push_back(func.eval(x));

      // output all the optimization statistics
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".mesh", U, F, FF);
        
        std::ofstream output_file_fixed(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_fixed_vid.txt");
        std::ostream_iterator<int> output_iterator_fixed(output_file_fixed, "\n");
        std::copy(std::begin(indices_fixed), std::end(indices_fixed), output_iterator_fixed);

        std::ofstream output_file(output_folder + "hist/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(std::begin(hist), std::end(hist), output_iterator);

        std::ofstream output_file_trust_region_ratio(output_folder + "trust_region_ratio/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_ratio(output_file_trust_region_ratio, "\n");
        std::copy(std::begin(hist_trust_region_ratio), std::end(hist_trust_region_ratio), output_iterator_trust_region_ratio);

        std::ofstream output_file_trust_region_eps(output_folder + "trust_region_eps/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_eps(output_file_trust_region_eps, "\n");
        std::copy(std::begin(hist_trust_region_eps), std::end(hist_trust_region_eps), output_iterator_trust_region_eps);

        std::ofstream output_file_line_search_alpha(output_folder + "line_search_alpha/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_line_search_alpha(output_file_line_search_alpha, "\n");
        std::copy(std::begin(hist_line_search_alpha), std::end(hist_line_search_alpha), output_iterator_line_search_alpha);

        std::ofstream output_file_line_search_iter(output_folder + "line_search_iter/" + output_tag + ".txt");
        std::ostream_iterator<int> output_iterator_line_search_iter(output_file_line_search_iter, "\n");
        std::copy(std::begin(hist_line_search_iter), std::end(hist_line_search_iter), output_iterator_line_search_iter);

        std::ofstream output_file_energy_injection(output_folder + "energy_injection/" + output_tag + ".txt");
        std::ostream_iterator<int> output_energy_injection(output_file_energy_injection, "\n");
        std::copy(std::begin(hist_energy_injection_ratio), std::end(hist_energy_injection_ratio), output_energy_injection);


        std::ofstream output_file_iter(output_folder + "iter/" + output_tag + ".txt");
        output_file_iter << (hist.size()-1) << std::endl;
      }

      
      //写到同一文件,方便方案对比
      std::vector<std::string> arr_time_str = {time_str};
      std::vector<std::string> arr_mesh_name = {mesh_name};
      std::vector<std::string> arr_pos = {pose_label};
      std::vector<long> arr_n_v = {V.rows()};
      std::vector<long> arr_n_t = {F.rows()};
      std::vector<double> arr_pr = {PR};
      std::vector<double> arr_ym = {YM};
      std::vector<double> arr_deformation_ratio = {deformation_ratio};
      std::vector<double> arr_deformation_magnitude = {deformation_magnitude};
      std::vector<double> arr_rotate_ratio = {rotate_ratio};
      std::vector<std::string> arr_diff_mode_str = {diff_mode_str};
      std::vector<int> arr_iter = {static_cast<int>(hist.size()-1)};
      
      // 使用
      const std::string results_file_csv = "../results/results_compare.csv";

      CSVLineBuilder builder;
      builder.add(arr_time_str)
            .add(arr_mesh_name)
            .add(arr_n_v)
            .add(arr_n_t)
            .add(arr_pos)
            .add(arr_pr)
            .add(arr_ym)
            .add(arr_deformation_ratio)
            .add(arr_deformation_magnitude)
            .add(arr_rotate_ratio)
            .add(arr_diff_mode_str)
            .add(arr_iter)
            .add(hist) //energy
            .add(hist_line_search_iter)
            .add(hist_energy_injection_ratio)
            .writeToFile(results_file_csv, true);  // true表示换行

      TINYAD_DEBUG_OUT("======== The End ========");
      // comment this out later
      // close the viewer
      exit(0);
    });

  // Plot mesh
  viewer.core().is_animating = true;
  viewer.data().set_mesh(U, FF);
  viewer.core().align_camera_center(U);
  viewer.data().show_lines = true;
  viewer.core().depth_test = true;
  viewer.data().line_width = 1.0;

  // 设置透明材质
  Eigen::Vector3d color(0.2, 0.2, 0.8);
  viewer.data().uniform_colors(
      //Eigen::Vector3d(color3(0), color3(1), color3(2)),  // 漫反射
      Eigen::Vector3d(color(0), color(1), color(2)),  // 漫反射
      Eigen::Vector3d(0.2, 0.2, 0.2),  // 环境光
      Eigen::Vector3d(0.0, 0.0, 0.0)   // 镜面反射
  );
    
  // 设置面的透明度 - 使用材质属性
  viewer.data().face_based = true;  // 基于面的渲染
  
  std::cout << "\n>>> " << (g_paused ? "⏸ 已暂停" : "▶ 继续迭代") 
                        << " (按空格键切换)\n";
  std::cout << "\n>>> " << (recording_started ? "⏺ 录制中" : "⏸ 已暂停") 
                        << " (按 'r' 键切换)\n";
  // ========================================
  // 键盘回调：控制暂停/继续
  // ========================================
  viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, 
                                  unsigned int key, int modifiers) -> bool {
      switch (key) {
          case 32:  // 空格键：暂停/继续
              g_paused = !g_paused;
              std::cout << "\n>>> " << (g_paused ? "⏸ 已暂停" : "▶ 继续迭代") 
                        << " (按空格键切换)\n";
              return true;
          case 114:  // 'r' 键：暂停/继续
          case 82:  // 'R' 键：暂停/继续
              recording_started = !recording_started;
              std::cout << "\n>>> " << (recording_started ? "⏺ 录制中" : "⏸ 已暂停") 
                        << " (按 'r/R' 键切换)\n";
              return true;
              
          default:
              return false;  // 未处理的按键交给查看器默认处理
      }
  };

  // 使用 callback_post_draw 替代 callback_pre_draw
  // 录像参数
  static int frame_count = 0;
  static int warmup_frames = 10;  // 等待5帧再开始录制
  const int fps = 30;
  auto last_time = std::chrono::steady_clock::now();
  
  
  std::cout << "========================================" << std::endl;
  std::cout << "Warming up... Recording will start in " << warmup_frames << " frames" << std::endl;
  std::cout << "========================================" << std::endl;
  if (!std::filesystem::exists(output_folder + "frames"))
      std::filesystem::create_directory(output_folder + "frames");

  std::string title = experiment_folder.erase(0,7).erase(experiment_folder.size()-15)+ "_" + diff_mode_str.c_str();

  viewer.callback_pre_draw = [&] (igl::opengl::glfw::Viewer& viewer)
  {
    // 检查是否应该执行迭代
    if(!g_paused&& redraw)
    {
      viewer.data().set_vertices(U);
      viewer.core().align_camera_center(U);
      {
        std::lock_guard<std::mutex> lock(m);
        redraw = false;
      }

      // 获取 GLFW 窗口指针并设置标题
      glfwSetWindowTitle(viewer.window, (title +" iter " +std::to_string(iter_i)).c_str() );
      
      #if 0
      // 自动开始录制
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
      if (elapsed >= (1000 / fps)) {
        last_time = now;
        
        int w = viewer.core().viewport[2];
        int h = viewer.core().viewport[3];
        
        if (w > 0 && h > 0) {
            char filename[256];
            snprintf(filename, 256, "%sframes/frame_%04d.png", output_folder.c_str(), frame_count++);
            save_ppm(filename, w, h);
            
            if (frame_count % 30 == 0) {
                std::cout << "Recorded " << frame_count << " frames" << std::endl;
            }
        }
        
      }
      #endif
      
    }
    return false;

  };

  // 在初始化回调中设置标题
  viewer.callback_init = [&](igl::opengl::glfw::Viewer& v)
  {
      // 获取 GLFW 窗口指针并设置标题
      glfwSetWindowTitle(v.window, (title  + " iter " +std::to_string(iter_i)).c_str() );
      return false;
  };

  viewer.launch();
  if(optimization_thread.joinable())
  {
    optimization_thread.join();
  }

  #if DEBUG_OUTPUT
    TINYAD_CLOSE_DEBUG_LOG();
  #endif
  return 0;
}
 

int main(int argc, char** argv)
{
  #if SHEAR_PROJECTED_NEWTON
    //std::cout <<  << std::endl;
    TINYAD_DEBUG_OUT("Using shear projected Newton method with trust region heuristic.");
    //shear_projected_newton(argc, argv);
    shear_reg_projected_newton(argc, argv);
  #elif CUBIC_DEFORM_PROJECTED_NEWTON
    //std::cout <<  << std::endl;
    TINYAD_DEBUG_OUT("Using cubic projected Newton method with trust region heuristic.");
    cubic_deform_projected_newton(argc, argv);
  #elif CUBIC_PROJECTED_NEWTON
    std::cout << "Using cubic projected Newton method with trust region heuristic." << std::endl;
    cubic_projected_newton(argc, argv);
    //cubic_deform_projected_newton(argc, argv);
  #elif FS_PROJECTED_NEWTON
    std::cout << "Using face smoothed projected Newton method with trust region heuristic." << std::endl;
    fs_projected_newton(argc, argv);
  #else 
    std::cout << "Using Differentiable projected Newton method." << std::endl;
    diff_projected_newton(argc, argv);  
  #endif
}

int diff_projected_newton(int argc, char** argv)
{
  // parse command line arguments
  cxxopts::Options options("Projected Newton with a trust region", "Choose eigenvalue filtering method: adaptive, clamp, abs");

  options.add_options()
    ("smooth_mode", "add differentiable eigenvalue projection strategy ", cxxopts::value<std::string>()->default_value("none")) // 若设置diff，后续可微的clamp和abs等
    ("diff", "add differentiable eigenvalue projection strategy ", cxxopts::value<bool>()->default_value("false")) // 若设置diff，后续可微的clamp和abs等
    ("diff_mode", "differentiable eigenvalue projection mode", cxxopts::value<std::string>()->default_value("auto")) // 若设置diff，后续可微的clamp和abs等
    // ("delta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("0.001"))  // 过渡区间宽度
   // ("beta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("10.0"))// 光滑参数
    ("abs", "use absolute eigenvalue projection strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("clamp", "use eigenvalue clamping strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("p,epsilon", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("-0.5"))
    ("n,mesh_name", "Mesh name", cxxopts::value<std::string>()->default_value("bimba"))
    ("l,pose_label", "Pose label", cxxopts::value<std::string>()->default_value("stretch"))
    ("g,deformation_magnitude", "The magnitude of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("t,deformation_ratio", "The ratio of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("b,fixed_boundary_range", "The range of fixed vertices on the boundary", cxxopts::value<double>()->default_value("0.1"))
    ("c,convergence_eps", "The convergence threshold", cxxopts::value<double>()->default_value("1e-5"))
    ("ym", "Young's modulus (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("1e8"))
    ("pr", "Poisson's ratio (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("0.495"))
    ("tr", "trust region ratio threshold", cxxopts::value<double>()->default_value("0.01"))
    ("experiment_name", "experiment name", cxxopts::value<std::string>()->default_value(""))
    ("rotate_ratio", "The ratio of the rotation", cxxopts::value<double>()->default_value("0.5"))
    ("h,help", "show help")
   // ("adaptive", "adaptive accoring to rho ", cxxopts::value<bool>()->default_value("false")) // 
  ;
  
  auto result = options.parse(argc, argv);
  if (result.count("help"))
  {
      std::cout << options.help() << "\n";
      return 0;
  }

  //const bool b_adaptive = result["adaptive"].as<bool>();
  const bool diff = result["diff"].as<bool>();
  //const int _mode = result["diff_mode"].as<int>();
  std::string diff_mode_str = result["diff_mode"].as<std::string>();
  TinyAD::HessianProjectionMode _diff_mode = parse_projection_mode(diff_mode_str);
  // std::string delta_str = result["delta"].as<std::string>();
  // double delta = std::stod(delta_str);
  // std::string beta_str = result["beta"].as<std::string>();
  // double beta = std::stod(beta_str);


  const bool abs = result["abs"].as<bool>();
  const bool clamp = result["clamp"].as<bool>();
  std::string eps_str = result["epsilon"].as<std::string>();
  double eps = std::stod(eps_str);
  std::string mesh_name = result["mesh_name"].as<std::string>();
  std::string pose_label = result["pose_label"].as<std::string>();
  const double deformation_magnitude = result["deformation_magnitude"].as<double>();
  const double deformation_ratio = result["deformation_ratio"].as<double>();
  const double fixed_boundary_range = result["fixed_boundary_range"].as<double>();
  const double convergence_eps = result["convergence_eps"].as<double>();
  const double YM = result["ym"].as<double>();
  const double PR = result["pr"].as<double>();
  const double tr_threshold = result["tr"].as<double>(); // 信赖域接受步长的阈值，通常设置为0.01
  std::string experiment_folder = result["experiment_name"].as<std::string>() == "" ? ("figure_" + mesh_name) : result["experiment_name"].as<std::string>();
  experiment_folder = experiment_folder + get_time_str();
  const double rotate_ratio = result["rotate_ratio"].as<double>();

  /*
  * 0: clamp
  * -1: abs
  * -0.5: adaptive (default)
  * 0.5: differentiable  (if diff is set)
  */
  if (diff)
  {
      eps_str =  diff_mode_str;
      //下面三种情况,eps值有特殊用处
      if (_diff_mode == TinyAD::HessianProjectionMode::ABS_NONDIFF) {
        eps_str = "abs";
        eps = -1;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_NONDIFF) {
        eps_str = "clamp";
        eps = 0;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF) {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
      }
      else{
         //非特殊情况,eps必须大于等于0
        if (eps < 0)
        {
          eps = 0;
        }
      }
  }
  else {
    if (clamp || eps == 0) {
        // we use eps = 0 as a flag for clamp projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "clamp";
        eps = 0; 
        diff_mode_str = "clamp_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_NONDIFF; 
    }
    else if (abs || eps == -1) {
        // we use eps = -1 as a flag for abs projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "abs";
        eps = -1;
        diff_mode_str = "abs_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::ABS_NONDIFF; 
    }
    else {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
        diff_mode_str = "clamp_abs_nondiff"; // trust region newton;
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF;
    }
  }
    
  if (!std::filesystem::exists("../results/"))
    std::filesystem::create_directory("../results/");
  if (!std::filesystem::exists("../results/" + experiment_folder))
    std::filesystem::create_directory("../results/" + experiment_folder);

  const std::string output_folder = "../results/" + experiment_folder + "/" 
    + pose_label + "_YM_" + std::to_string(YM) + "_PR_" + std::to_string(PR)
    + "_deform_mag_" + std::to_string(deformation_magnitude) + "_deform_ratio_" 
    + std::to_string(deformation_ratio) + "_fixed_boundary_range_" 
    + std::to_string(fixed_boundary_range)+"_diff_mode_" + diff_mode_str+ "/";
  {
    // record the statistics
    if (!std::filesystem::exists(output_folder))
      std::filesystem::create_directory(output_folder);
    if (!std::filesystem::exists(output_folder + "obj/"))
      std::filesystem::create_directory(output_folder + "obj/");
    if (!std::filesystem::exists(output_folder + "obj/" + mesh_name + "_" + eps_str + "/"))
      std::filesystem::create_directory(output_folder + "obj/" + mesh_name + "_" + eps_str + "/");
    if (!std::filesystem::exists(output_folder + "hist/"))
      std::filesystem::create_directory(output_folder + "hist/");
    if (!std::filesystem::exists(output_folder + "iter/"))
      std::filesystem::create_directory(output_folder + "iter/");
    if (!std::filesystem::exists(output_folder + "trust_region_ratio/"))
      std::filesystem::create_directory(output_folder + "trust_region_ratio/");
    if (!std::filesystem::exists(output_folder + "trust_region_eps/"))
      std::filesystem::create_directory(output_folder + "trust_region_eps/");
    if (!std::filesystem::exists(output_folder + "line_search_alpha/"))
      std::filesystem::create_directory(output_folder + "line_search_alpha/");
    if (!std::filesystem::exists(output_folder + "line_search_iter/"))
      std::filesystem::create_directory(output_folder + "line_search_iter/");
    
    #if DEBUG_OUTPUT
    if (!std::filesystem::exists(output_folder + "debug_log/"))
    {
      std::filesystem::create_directory(output_folder + "debug_log/");
      TINYAD_INIT_DEBUG_LOG(output_folder + "debug_log/log"+get_time_str()+".txt");
    }
    #endif

    
  }

  const std::string output_tag = "mesh_" + mesh_name + "_eps_" + eps_str;

  // set up viewer
  igl::opengl::glfw::Viewer viewer;

  // compute the Lame parameters
  const double MU = YM / (2 * (1 + PR));
  const double LAMBDA = YM * PR / ((1 + PR) * (1 - 2 * PR));
  const double lambda_mu_ratio = LAMBDA / MU;
  TinyAD::g_MU = MU;
  TinyAD::g_LAMBDA = LAMBDA;
 
  // print out the configuration
  {
    TINYAD_DEBUG_OUT("Diff flag: " << diff);
    TINYAD_DEBUG_OUT("*Eigenvalue Differentiable filtering strategy: " << diff_mode_str);
    TINYAD_DEBUG_OUT("Eigenvalue filtering strategy: " << eps_str);
    TINYAD_DEBUG_OUT("mu: " << MU);
    TINYAD_DEBUG_OUT("lambda: " << LAMBDA);
    TINYAD_DEBUG_OUT("*Projection threshold: " << eps);
    TINYAD_DEBUG_OUT("Mesh name: " << mesh_name);
    TINYAD_DEBUG_OUT("Pose label: " << pose_label);
    TINYAD_DEBUG_OUT("Lambda / Mu ratio: " << lambda_mu_ratio);
    TINYAD_DEBUG_OUT("Deformation magnitude: " << deformation_magnitude);
    TINYAD_DEBUG_OUT("Deformation ratio: " << deformation_ratio);
    TINYAD_DEBUG_OUT("Fixed vertices boundary range: " << fixed_boundary_range);
    TINYAD_DEBUG_OUT("Convergence threshold: " << convergence_eps);
    TINYAD_DEBUG_OUT("Young's modulus: " << YM);
    TINYAD_DEBUG_OUT("Poisson's ratio: " << PR);
    TINYAD_DEBUG_OUT("Trust region threshold: " << tr_threshold);
    TINYAD_DEBUG_OUT("Experiment folder: " << experiment_folder);
    TINYAD_DEBUG_OUT("Rotate ratio: " << rotate_ratio);
    //TINYAD_DEBUG_OUT("Adaptive flag: " << b_adaptive);
  }

  Eigen::MatrixXd V, U; // #V-by-3 3D vertex positions,V为初始,U为当前
  Eigen::MatrixXi F, FF; // #T-by-4 indices into V
  Eigen::VectorXi TriTag, TetTag;
  if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh"))
    igl::readMESH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh", V, F, FF);
  else if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh"))
    igl::readMSH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh", V, FF, F, TriTag, TetTag);
  else {
    std::cout << "Mesh " << mesh_name << " not found!" << std::endl;
    exit(1);
  }

  TINYAD_DEBUG_OUT("Read mesh with " << V.rows() << " vertices and " << F.rows() << " tetrahedrons.");

  // get boundary vertices
  igl::boundary_facets(F, FF);
  FF = FF.rowwise().reverse().eval();

  TINYAD_DEBUG_OUT("Boundary has " << FF.rows() << " faces.");

  // normalize the mesh V
  Eigen::RowVector3d mean = V.colwise().mean();
  V.rowwise() -= mean;
  double max_norm = V.rowwise().norm().maxCoeff();
  V /= max_norm;

  // set up mesh U
  U = V;

  // fixed point constraints
  Eigen::SparseMatrix<double> P;
  std::vector<unsigned int> indices_fixed;

  // set up fixed point constraints and initial deformation
  setup_initial_deformation(V, F, pose_label, deformation_magnitude, deformation_ratio, rotate_ratio, fixed_boundary_range, U, indices_fixed);
  fixed_point_constraints(P, 3*V.rows(), 3, indices_fixed);

  TINYAD_DEBUG_OUT("Finish setting up fixed point constraints.");

  bool redraw = false;
  std::mutex m;
  std::thread optimization_thread(
    [&]()
    {
      // Pre-compute triangle rest shapes in local coordinate systems
      std::vector<Eigen::Matrix3d> rest_shapes(F.rows());
      for (int f_idx = 0; f_idx < F.rows(); ++f_idx)
      {
        // Get 3D vertex positions
        Eigen::Vector3d ar = V.row(F(f_idx, 0));
        Eigen::Vector3d br = V.row(F(f_idx, 1));
        Eigen::Vector3d cr = V.row(F(f_idx, 2));
        Eigen::Vector3d dr = V.row(F(f_idx, 3));

        // Save 3-by-3 matrix with edge vectors as columns
        rest_shapes[f_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
      };

      /*
          TinyAD::scalar_function<3>: 创建一个标量函数，每个变量是3维向量（例如3D顶点坐标）
          TinyAD::range(V.rows()): 定义变量的范围，V.rows() 是顶点数量
          auto func: 自动推断函数对象类型
      */ 
      // TinyAD::scalar_function(n)：这是一个函数，可以计算它的函数值、梯度、Hessian的函数，其自变量索引的范围为n
      // Set up function with 3d vertex positions as variables.
      auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

      // func.add_elements<n>(element_range, energy_function); 
      // 定义元素的能量项，1.n为元素的dim，其中2.element_range是元素索引的范围，
      // 3.energy_function是一个lambda函数，接受一个元素对象，返回该元素的能量值
      // 4. lamda函数 [捕获变量] (输入参数) -> 返回值 
      // 5. 所有参与自动微分计算的变量和中间结果，都必须使用 TINYAD_SCALAR_TYPE 或其衍生的类型
      // 6. element.handle对应element_range中的当前元素的索引，element.variables()对应输入变量，想要访问输入变量的第i个变量
      // 7. 在这个地方，输入变量是顶点数组，element_range是F数组，即按F计算能量
      // Add objective term per element. Each connecting 4 vertices.
      // "neo-hooken energy" is defined on each tetrahedron, so element_range is F.rows() and element.variables() gives the vertex positions of the current tetrahedron.
      func.add_elements<4>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element) {
          // Evaluate element using either double or TinyAD::Double
          using T = TINYAD_SCALAR_TYPE(element);

          // Get variable 3d vertex positions
          Eigen::Index f_idx = element.handle;
          Eigen::Vector3<T> a = element.variables(F(f_idx, 0));
          Eigen::Vector3<T> b = element.variables(F(f_idx, 1));
          Eigen::Vector3<T> c = element.variables(F(f_idx, 2));
          Eigen::Vector3<T> d = element.variables(F(f_idx, 3));

          Eigen::Matrix3<T> M = TinyAD::col_mat(b - a, c - a, d - a);
          Eigen::Matrix3d Mr = rest_shapes[f_idx];
          Eigen::Matrix3<T> J = M * Mr.inverse(); // 可以放外边 to be optimized by zj

          // Compute the stable Neo-Hookean energy [Smith et al. 2018]
          double A = 0.5 * Mr.determinant(); //可以放外边 to be optimized by zj
          const double mu = MU;
          const double lambda = LAMBDA;
          auto Ic = (J.transpose() * J).trace();
          auto detF = J.determinant();
          double alpha = 1.0 + mu / lambda;
          auto W = mu / 2.0 * (Ic - 3.0) + lambda / 2.0 * (detF - alpha) * (detF - alpha);
          return A * W;
      });

      // to be optimized by zj: 可以考虑把 rest_shapes 以及 pre-computed Mr.inverse() 放到外边，作为常量传入 lambda 函数中，这样就不需要每次迭代都计算 rest_shapes 和 Mr.inverse() 了
      // to be optimized by zj: 还可以考虑把 A 也放到外边，因为 A 只和 rest_shapes 相关，而 rest_shapes 是不变的
      // to be optimized by zj: 还可以考虑把 lambda 和 mu 放到外边，因为它们也是不变的
      // to be modified by zj: 动态状态下的PD，增加local step，即每个元素的能量项不仅依赖于当前状态，还依赖于上一个状态，这样可以增加稳定性，尤其是在使用abs投影时

      // func.x_from_data() 是 TinyAD 提供的数据格式转换工具，将外部数据（如顶点矩阵）转换为优化所需的展平向量格式，使代码更简洁、更安全。
      // v_idx 的取值范围是由 TinyAD::scalar_function<k>(n_variables) 中的 n_variables 决定的，即函数的输入变量的维度。
      // 在这个例子中，n_variables 是 V.rows()，即顶点数量，因此 v_idx 的取值范围是 [0, V.rows()-1]。
      // Assemble inital x vector from U matrix.
      // x_from_data(...) takes a lambda function that maps
      // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) {
          return U.row(v_idx);
      });

      TINYAD_DEBUG_OUT("Initial energy: " << func.eval(x));

      // Projected Newton
      TinyAD::LinearSolver solver; //Eigen::SimplicialLDLT*,ConjugateGradient,SparseLU
      int max_iters = 200; // 迭代次数可以根据需要调整
      Eigen::VectorXd d;
      Eigen::SparseMatrix<double> H;
      std::vector<double> hist;
      // record the trust region ratio
      // 衡量模型预测的准确性 ρ = (实际下降) / (预测下降), 
      // ρ < 0         → 实际目标函数值增加（拒绝步长）
      // 0 ≤ ρ < 0.25  → 模型质量差，缩小信赖域
      // 0.25 ≤ ρ < 0.75 → 模型质量一般，保持信赖域
      // ρ ≥ 0.75      → 模型质量好，可以放大信赖域
      // double delta;  // 当前信赖域半径
      // double eta1;   // 接受步长的阈值（通常0.25）
      // double eta2;   // 放大半径的阈值（通常0.75）
      std::vector<double> hist_trust_region_ratio;
      hist_trust_region_ratio.push_back(0.0);

      std::vector<double> hist_trust_region_eps; // 记录每次迭代使用的eps值，以观察自适应策略的变化
      std::vector<double> hist_line_search_alpha; // 记录每次迭代的线搜索步长
      std::vector<int> hist_line_search_iter; // 记录每次迭代的线搜索迭代次数

      //double cubic_sigma = 0.0;
      //Eigen::VectorXd cubic_lambda_vec = Eigen::VectorXd::Zero();
      
      //当global matrix发生变化时，进行下面的牛顿投影和线搜索步骤
      for (int i = 0; i < max_iters; ++i)
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".mesh", U, F, FF);
      
        // switch between clamp or abs depending on whether the trust region ratio is close to 1
        //if(eps_str == "adaptive") {
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF) {
          eps = (std::fabs(hist_trust_region_ratio.back() - 1.0) < tr_threshold) ? 0.0 : -1; // tr_threshold信赖域接受步长的阈值，通常设置为0.01,大于接受

          if (eps == 0.0) {
            TINYAD_DEBUG_OUT("Switch to clamp");  
          }
          else {
            TINYAD_DEBUG_OUT("Switch to abs");
          }
          hist_trust_region_eps.push_back(eps);
        }

        auto [f, g, H_proj] = func.eval_with_hessian_proj(x, eps, _diff_mode); // 
        Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量

        // record the energy
        hist.push_back(f);

        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);

        H = H_proj;
        

        Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
        Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
        d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
        d = P.transpose() * d;

        TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
        TINYAD_DEBUG_OUT("d norm " << i << " =  " << d.norm());
        TINYAD_DEBUG_OUT("g norm " << i << " =  " << g.norm());

        // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
        // 优化方案1： 相对收敛
        // if (newton_decrement < convergence_eps * initial_decrement) break;  
        // 方案3：梯度范数 
        // if (g.norm() < convergence_eps) break;
        // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
        // 方案5：根据材料参数调整收敛标准
        //  LAMBDA是材料参数，不是尺度参数
        // // 使用特征长度进行无量纲化
        // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
        // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
        // // 物理意义的收敛判据
        // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
        if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
          break;

        // line search
        Eigen::VectorXd x_prev = x;
        x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 100, 1e-8);
        double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
        hist_line_search_alpha.push_back(alpha);

        int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
        hist_line_search_iter.push_back(line_search_iter);

        // compute the trust region ratio
        double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
        hist_trust_region_ratio.push_back(trust_region_ratio);

        TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);


      // Write final x vector to U matrix.
      // x_to_data(...) takes a lambda function that writes the final value
      // of each variable (Eigen::Vector2d) back to our U matrix.
        func.x_to_data(x, [&] (int v_idx, const Eigen::Vector3d& p) {
            U.row(v_idx) = p;
            });
        {
          std::lock_guard<std::mutex> lock(m);
          redraw = true; 
        }
      }

      TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
      hist.push_back(func.eval(x));

      // output all the optimization statistics
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".mesh", U, F, FF);
        
        std::ofstream output_file_fixed(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_fixed_vid.txt");
        std::ostream_iterator<int> output_iterator_fixed(output_file_fixed, "\n");
        std::copy(std::begin(indices_fixed), std::end(indices_fixed), output_iterator_fixed);

        std::ofstream output_file(output_folder + "hist/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(std::begin(hist), std::end(hist), output_iterator);

        std::ofstream output_file_trust_region_ratio(output_folder + "trust_region_ratio/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_ratio(output_file_trust_region_ratio, "\n");
        std::copy(std::begin(hist_trust_region_ratio), std::end(hist_trust_region_ratio), output_iterator_trust_region_ratio);

        std::ofstream output_file_trust_region_eps(output_folder + "trust_region_eps/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_eps(output_file_trust_region_eps, "\n");
        std::copy(std::begin(hist_trust_region_eps), std::end(hist_trust_region_eps), output_iterator_trust_region_eps);

        std::ofstream output_file_line_search_alpha(output_folder + "line_search_alpha/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_line_search_alpha(output_file_line_search_alpha, "\n");
        std::copy(std::begin(hist_line_search_alpha), std::end(hist_line_search_alpha), output_iterator_line_search_alpha);

        std::ofstream output_file_line_search_iter(output_folder + "line_search_iter/" + output_tag + ".txt");
        std::ostream_iterator<int> output_iterator_line_search_iter(output_file_line_search_iter, "\n");
        std::copy(std::begin(hist_line_search_iter), std::end(hist_line_search_iter), output_iterator_line_search_iter);

        std::ofstream output_file_iter(output_folder + "iter/" + output_tag + ".txt");
        output_file_iter << (hist.size()-1) << std::endl;
      }

      TINYAD_DEBUG_OUT("======== The End ========");
      // comment this out later
      // close the viewer
      exit(0);
    });

  // Plot mesh
  viewer.core().is_animating = true;
  viewer.data().set_mesh(U, FF);
  viewer.core().align_camera_center(U);
  viewer.data().show_lines = false;

  viewer.callback_pre_draw = [&] (igl::opengl::glfw::Viewer& viewer)
  {
    if(redraw)
    {
      viewer.data().set_vertices(U);
      viewer.core().align_camera_center(U);
      {
        std::lock_guard<std::mutex> lock(m);
        redraw = false;
      }
    }
    return false;
  };

  // 在初始化回调中设置标题
  viewer.callback_init = [&](igl::opengl::glfw::Viewer& v)
  {
      // 获取 GLFW 窗口指针并设置标题
      glfwSetWindowTitle(v.window, diff_mode_str.c_str());
      return false;
  };

  viewer.launch();
  if(optimization_thread.joinable())
  {
    optimization_thread.join();
  }

  #if DEBUG_OUTPUT
    TINYAD_CLOSE_DEBUG_LOG();
  #endif
  return 0;
}



int fs_projected_newton(int argc, char** argv)
{
  // parse command line arguments
  cxxopts::Options options("Projected Newton with a trust region", "Choose eigenvalue filtering method: adaptive, clamp, abs");

  options.add_options()
    ("smooth_mode", "add differentiable eigenvalue projection strategy ", cxxopts::value<std::string>()->default_value("none")) // 若设置diff，后续可微的clamp和abs等
    ("diff", "add differentiable eigenvalue projection strategy ", cxxopts::value<bool>()->default_value("false")) // 若设置diff，后续可微的clamp和abs等
    ("diff_mode", "differentiable eigenvalue projection mode", cxxopts::value<std::string>()->default_value("auto")) // 若设置diff，后续可微的clamp和abs等
    // ("delta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("0.001"))  // 过渡区间宽度
   // ("beta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("10.0"))// 光滑参数
    ("abs", "use absolute eigenvalue projection strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("clamp", "use eigenvalue clamping strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("p,epsilon", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("-0.5"))
    ("n,mesh_name", "Mesh name", cxxopts::value<std::string>()->default_value("bimba"))
    ("l,pose_label", "Pose label", cxxopts::value<std::string>()->default_value("stretch"))
    ("g,deformation_magnitude", "The magnitude of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("t,deformation_ratio", "The ratio of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("b,fixed_boundary_range", "The range of fixed vertices on the boundary", cxxopts::value<double>()->default_value("0.1"))
    ("c,convergence_eps", "The convergence threshold", cxxopts::value<double>()->default_value("1e-5"))
    ("ym", "Young's modulus (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("1e8"))
    ("pr", "Poisson's ratio (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("0.495"))
    ("tr", "trust region ratio threshold", cxxopts::value<double>()->default_value("0.01"))
    ("experiment_name", "experiment name", cxxopts::value<std::string>()->default_value(""))
    ("rotate_ratio", "The ratio of the rotation", cxxopts::value<double>()->default_value("0.5"))
    ("h,help", "show help")
  ;
  
  auto result = options.parse(argc, argv);
  if (result.count("help"))
  {
      std::cout << options.help() << "\n";
      return 0;
  }

  std::string smooth_mode_str = result["smooth_mode"].as<std::string>();
  unsigned int _smooth_mode = parse_smooth_mode(smooth_mode_str);
  if (_smooth_mode == 0) //非光滑模式
  {
    return diff_projected_newton(argc, argv);
  }

  const bool diff = result["diff"].as<bool>();
  std::string diff_mode_str = result["diff_mode"].as<std::string>();
  TinyAD::HessianProjectionMode _diff_mode = parse_projection_mode(diff_mode_str);
  

  const bool abs = result["abs"].as<bool>();
  const bool clamp = result["clamp"].as<bool>();
  std::string eps_str = result["epsilon"].as<std::string>();
  double eps = std::stod(eps_str);
  std::string mesh_name = result["mesh_name"].as<std::string>();
  std::string pose_label = result["pose_label"].as<std::string>();
  const double deformation_magnitude = result["deformation_magnitude"].as<double>();
  const double deformation_ratio = result["deformation_ratio"].as<double>();
  const double fixed_boundary_range = result["fixed_boundary_range"].as<double>();
  const double convergence_eps = result["convergence_eps"].as<double>();
  const double YM = result["ym"].as<double>();
  const double PR = result["pr"].as<double>();
  const double tr_threshold = result["tr"].as<double>(); // 信赖域接受步长的阈值，通常设置为0.01
  std::string experiment_folder = result["experiment_name"].as<std::string>() == "" ? ("figure_" + mesh_name) : result["experiment_name"].as<std::string>();
  experiment_folder = experiment_folder + get_time_str();
  const double rotate_ratio = result["rotate_ratio"].as<double>();

  /*
  * 0: clamp
  * -1: abs
  * -0.5: adaptive (default)
  * 0.5: differentiable  (if diff is set)
  */
  if (diff)
  {
      eps_str =  diff_mode_str;
      //下面三种情况,eps值有特殊用处
      if (_diff_mode == TinyAD::HessianProjectionMode::ABS_NONDIFF) {
        eps_str = "abs";
        eps = -1;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_NONDIFF) {
        eps_str = "clamp";
        eps = 0;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF) {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
      }
      else{
         //非特殊情况,eps必须大于等于0
        if (eps < 0)
        {
          eps = 0;
        }
      }
  }
  else {
    if (clamp || eps == 0) {
        // we use eps = 0 as a flag for clamp projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "clamp";
        eps = 0; 
        diff_mode_str = "clamp_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_NONDIFF; 
    }
    else if (abs || eps == -1) {
        // we use eps = -1 as a flag for abs projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "abs";
        eps = -1;
        diff_mode_str = "abs_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::ABS_NONDIFF; 
    }
    else {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
        diff_mode_str = "clamp_abs_nondiff"; // trust region newton;
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF;
    }
  }
    
  if (!std::filesystem::exists("../results/"))
    std::filesystem::create_directory("../results/");
  if (!std::filesystem::exists("../results/" + experiment_folder))
    std::filesystem::create_directory("../results/" + experiment_folder);

  const std::string output_folder = "../results/" + experiment_folder + "/" 
    + pose_label + "_YM_" + std::to_string(YM) + "_PR_" + std::to_string(PR)
    + "_deform_mag_" + std::to_string(deformation_magnitude) + "_deform_ratio_" 
    + std::to_string(deformation_ratio) + "_fixed_boundary_range_" 
    + std::to_string(fixed_boundary_range)+"_diff_" + diff_mode_str+"_smooth_" + smooth_mode_str+ "/";
  {
    // record the statistics
    if (!std::filesystem::exists(output_folder))
      std::filesystem::create_directory(output_folder);
    if (!std::filesystem::exists(output_folder + "obj/"))
      std::filesystem::create_directory(output_folder + "obj/");
    if (!std::filesystem::exists(output_folder + "obj/" + mesh_name + "_" + eps_str + "/"))
      std::filesystem::create_directory(output_folder + "obj/" + mesh_name + "_" + eps_str + "/");
    if (!std::filesystem::exists(output_folder + "hist/"))
      std::filesystem::create_directory(output_folder + "hist/");
    if (!std::filesystem::exists(output_folder + "iter/"))
      std::filesystem::create_directory(output_folder + "iter/");
    if (!std::filesystem::exists(output_folder + "trust_region_ratio/"))
      std::filesystem::create_directory(output_folder + "trust_region_ratio/");
    if (!std::filesystem::exists(output_folder + "trust_region_eps/"))
      std::filesystem::create_directory(output_folder + "trust_region_eps/");
    if (!std::filesystem::exists(output_folder + "line_search_alpha/"))
      std::filesystem::create_directory(output_folder + "line_search_alpha/");
    if (!std::filesystem::exists(output_folder + "line_search_iter/"))
      std::filesystem::create_directory(output_folder + "line_search_iter/");
    
    #if DEBUG_OUTPUT
    if (!std::filesystem::exists(output_folder + "debug_log/"))
    {
      std::filesystem::create_directory(output_folder + "debug_log/");
      TINYAD_INIT_DEBUG_LOG(output_folder + "debug_log/log"+get_time_str()+".txt");
    }
    #endif  
  }

  const std::string output_tag = "mesh_" + mesh_name + "_eps_" + eps_str;

  // set up viewer
  igl::opengl::glfw::Viewer viewer;

  // compute the Lame parameters
  const double MU = YM / (2 * (1 + PR));
  const double LAMBDA = YM * PR / ((1 + PR) * (1 - 2 * PR));
  const double lambda_mu_ratio = LAMBDA / MU;
 
  // print out the configuration
  {
    TINYAD_DEBUG_OUT("Diff flag: " << diff);
    TINYAD_DEBUG_OUT("*Eigenvalue Differentiable filtering strategy: " << diff_mode_str);
    TINYAD_DEBUG_OUT("Eigenvalue filtering strategy: " << eps_str);
    TINYAD_DEBUG_OUT("mu: " << MU);
    TINYAD_DEBUG_OUT("lambda: " << LAMBDA);
    TINYAD_DEBUG_OUT("*Projection threshold: " << eps);
    TINYAD_DEBUG_OUT("Mesh name: " << mesh_name);
    TINYAD_DEBUG_OUT("Pose label: " << pose_label);
    TINYAD_DEBUG_OUT("Lambda / Mu ratio: " << lambda_mu_ratio);
    TINYAD_DEBUG_OUT("Deformation magnitude: " << deformation_magnitude);
    TINYAD_DEBUG_OUT("Deformation ratio: " << deformation_ratio);
    TINYAD_DEBUG_OUT("Fixed vertices boundary range: " << fixed_boundary_range);
    TINYAD_DEBUG_OUT("Convergence threshold: " << convergence_eps);
    TINYAD_DEBUG_OUT("Young's modulus: " << YM);
    TINYAD_DEBUG_OUT("Poisson's ratio: " << PR);
    TINYAD_DEBUG_OUT("Trust region threshold: " << tr_threshold);
    TINYAD_DEBUG_OUT("Experiment folder: " << experiment_folder);
    TINYAD_DEBUG_OUT("Rotate ratio: " << rotate_ratio);
  }

  Eigen::MatrixXd V, U; // #V-by-3 3D vertex positions
  Eigen::MatrixXi F, FF; // #T-by-4 indices into V
  Eigen::VectorXi TriTag, TetTag;
  if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh"))
    igl::readMESH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh", V, F, FF);
  else if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh"))
    igl::readMSH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh", V, FF, F, TriTag, TetTag);
  else {
    std::cout << "Mesh " << mesh_name << " not found!" << std::endl;
    exit(1);
  }

  TINYAD_DEBUG_OUT("Read mesh with " << V.rows() << " vertices and " << F.rows() << " tetrahedrons.");

  // get boundary vertices
  igl::boundary_facets(F, FF);
  FF = FF.rowwise().reverse().eval();

  TINYAD_DEBUG_OUT("Boundary has " << FF.rows() << " faces.");

  // normalize the mesh V
  Eigen::RowVector3d mean = V.colwise().mean();
  V.rowwise() -= mean;
  double max_norm = V.rowwise().norm().maxCoeff();
  V /= max_norm;

  // set up mesh U
  U = V;

  // fixed point constraints
  Eigen::SparseMatrix<double> P;
  std::vector<unsigned int> indices_fixed;

  // set up fixed point constraints and initial deformation
  setup_initial_deformation(V, F, pose_label, deformation_magnitude, deformation_ratio, rotate_ratio, fixed_boundary_range, U, indices_fixed);
  fixed_point_constraints(P, 3*V.rows(), 3, indices_fixed);
  // 预处理阶段
  FSFEM_5NodeData elementData;
  elementData.build_5node_fsfem_data(V,F);
  
  TINYAD_DEBUG_OUT("Finish setting up fixed point constraints.");

  bool redraw = false;
  std::mutex m;
  std::thread optimization_thread(
    [&]()
    {
      /*
          TinyAD::scalar_function<3>: 创建一个标量函数，每个变量是3维向量（例如3D顶点坐标）
          TinyAD::range(V.rows()): 定义变量的范围，V.rows() 是顶点数量
          auto func: 自动推断函数对象类型
      */ 
      // TinyAD::scalar_function(n)：这是一个函数，可以计算它的函数值、梯度、Hessian的函数，其自变量索引的范围为n
      // Set up function with 3d vertex positions as variables.
      auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));
      //Eigen::MatrixXi
      // func.add_elements<n>(element_range, energy_function); 
      // 定义元素的能量项，
      // 1.n为元素的dim，其中
      // 2.element_range是元素索引的范围，
      // 3.energy_function是一个lambda函数，接受一个元素对象，返回该元素的能量值
      // 4. lamda函数 [捕获变量] (输入参数) -> 返回值 
      // 5. 所有参与自动微分计算的变量和中间结果，都必须使用 TINYAD_SCALAR_TYPE 或其衍生的类型
      // 6. element.handle对应element_range中的当前元素的索引，element.variables()对应输入变量，想要访问输入变量的第i个变量
      // 7. 在这个地方，输入变量是顶点数组，element_range是F数组，即按F计算能量
      // Add objective term per element. Each connecting 5 vertices.
      // "neo-hooken energy" is defined on each tetrahedron, so element_range is F.rows() and element.variables() gives the vertex positions of the current tetrahedron.
      //Eigen::MatrixXi &face_vertices = elementData.face_vertices;
      func.add_elements<5>(TinyAD::range(elementData.face_vertices.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element) {
          // Evaluate element using either double or TinyAD::Double
          using T = TINYAD_SCALAR_TYPE(element);
          //int n = 3;

          // Get variable 3d vertex positions
          Eigen::Index f_idx = element.handle;

          // 获取5个顶点
          const auto& verts = elementData.face_vertices.row(f_idx);

          Eigen::Vector3<T> v0 = element.variables(verts[0]);
          Eigen::Vector3<T> v1 = element.variables(verts[1]);
          Eigen::Vector3<T> v2 = element.variables(verts[2]);
          Eigen::Vector3<T> v3 = element.variables(verts[3]);
          Eigen::Vector3<T> v4 = element.variables(verts[3]);

          // 计算四面体A的变形梯度
          // 注意：四面体A的参考构型中，顶点顺序必须与预处理一致！
          // 假设预处理时四面体A的顶点顺序是 (v0, v1, v2, v3)
          Eigen::Matrix3<T> M_a = TinyAD::col_mat(v1 - v0, v2 - v0, v3 - v0);
          Eigen::Matrix3<T> Fe_a = M_a * elementData.Mr_inv[f_idx][0].cast<T>();
          T w_a = elementData.adj_tet_volume[f_idx][0];
          
          // 计算四面体B的变形梯度
          // 假设预处理时四面体B的顶点顺序是 (v4, v3, v2, v1)
          Eigen::Matrix3<T> M_b = Eigen::Matrix3<T>::Zero();
          T w_b = T(0.0);
          if (elementData.face_to_tets[f_idx].size()>1)
          {
            v4 = element.variables(verts[4]);
            M_b = TinyAD::col_mat(v3 - v4, v2 - v4, v1 - v4);
            w_b = elementData.adj_tet_volume[f_idx][1];
          }
          Eigen::Matrix3<T> Fe_b = M_b * elementData.Mr_inv[f_idx][1].cast<T>(); //有可能为0
          
          // 加权平均得到光滑变形梯度
          T Vf = elementData.domain_volumes[f_idx];
          Eigen::Matrix3<T> F_tilde = (w_a * Fe_a + w_b * Fe_b)/Vf; // to be optimized by zj 去掉旋转后再加权
          
          // Neo-Hookean应变能密度
          T Ic = (F_tilde.transpose() * F_tilde).trace();
          T detF = F_tilde.determinant();
          
          const double mu = MU;
          const double lambda = LAMBDA;
          const double alpha = 1.0 + mu / lambda;
          T W = mu/2.0 * (Ic - 3.0) + lambda/2.0 * (detF - alpha) * (detF - alpha);
          
          // 返回面光滑域总能量
          return Vf * W;
      });

      // to be optimized by zj: 可以考虑把 rest_shapes 以及 pre-computed Mr.inverse() 放到外边，作为常量传入 lambda 函数中，这样就不需要每次迭代都计算 rest_shapes 和 Mr.inverse() 了
      // to be optimized by zj: 还可以考虑把 A 也放到外边，因为 A 只和 rest_shapes 相关，而 rest_shapes 是不变的
      // to be optimized by zj: 还可以考虑把 lambda 和 mu 放到外边，因为它们也是不变的
      // to be modified by zj: 动态状态下的PD，增加local step，即每个元素的能量项不仅依赖于当前状态，还依赖于上一个状态，这样可以增加稳定性，尤其是在使用abs投影时

      // func.x_from_data() 是 TinyAD 提供的数据格式转换工具，将外部数据（如顶点矩阵）转换为优化所需的展平向量格式，使代码更简洁、更安全。
      // v_idx 的取值范围是由 TinyAD::scalar_function<k>(n_variables) 中的 n_variables 决定的，即函数的输入变量的维度。
      // 在这个例子中，n_variables 是 V.rows()，即顶点数量，因此 v_idx 的取值范围是 [0, V.rows()-1]。
      // Assemble inital x vector from U matrix.
      // x_from_data(...) takes a lambda function that maps
      // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) {
          return U.row(v_idx);
      });

      TINYAD_DEBUG_OUT("Initial energy: " << func.eval(x));

      // Projected Newton
      TinyAD::LinearSolver solver; //Eigen::SimplicialLDLT*,ConjugateGradient,SparseLU
      int max_iters = 200; // 迭代次数可以根据需要调整
      Eigen::VectorXd d;
      Eigen::SparseMatrix<double> H;
      std::vector<double> hist;
      // record the trust region ratio
      // 衡量模型预测的准确性 ρ = (实际下降) / (预测下降), 
      // ρ < 0         → 实际目标函数值增加（拒绝步长）
      // 0 ≤ ρ < 0.25  → 模型质量差，缩小信赖域
      // 0.25 ≤ ρ < 0.75 → 模型质量一般，保持信赖域
      // ρ ≥ 0.75      → 模型质量好，可以放大信赖域
      // double delta;  // 当前信赖域半径
      // double eta1;   // 接受步长的阈值（通常0.25）
      // double eta2;   // 放大半径的阈值（通常0.75）
      std::vector<double> hist_trust_region_ratio;
      hist_trust_region_ratio.push_back(0.0);

      std::vector<double> hist_trust_region_eps; // 记录每次迭代使用的eps值，以观察自适应策略的变化
      std::vector<double> hist_line_search_alpha; // 记录每次迭代的线搜索步长
      std::vector<int> hist_line_search_iter; // 记录每次迭代的线搜索迭代次数
      
      //当global matrix发生变化时，进行下面的牛顿投影和线搜索步骤
      for (int i = 0; i < max_iters; ++i)
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".mesh", U, F, FF);
      
        // switch between clamp or abs depending on whether the trust region ratio is close to 1
        //if(eps_str == "adaptive") {
        if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF) {
          eps = (std::fabs(hist_trust_region_ratio.back() - 1.0) < tr_threshold) ? 0.0 : -1; // tr_threshold信赖域接受步长的阈值，通常设置为0.01,大于接受

          if (eps == 0.0) {
            TINYAD_DEBUG_OUT("Switch to clamp");  
          }
          else {
            TINYAD_DEBUG_OUT("Switch to abs");
          }
          hist_trust_region_eps.push_back(eps);
        }
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x, eps, _diff_mode); // 
        Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量

        // record the energy
        hist.push_back(f);

        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);

        // Compute the Newton direction
        H = H_proj;
        Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
        Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
        d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
        d = P.transpose() * d;

        TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
        TINYAD_DEBUG_OUT("d norm " << i << " =  " << d.norm());
        TINYAD_DEBUG_OUT("g norm " << i << " =  " << g.norm());

        // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
        // 优化方案1： 相对收敛
        // if (newton_decrement < convergence_eps * initial_decrement) break;  
        // 方案3：梯度范数 
        // if (g.norm() < convergence_eps) break;
        // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
        // 方案5：根据材料参数调整收敛标准
        //  LAMBDA是材料参数，不是尺度参数
        // // 使用特征长度进行无量纲化
        // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
        // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
        // // 物理意义的收敛判据
        // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
        if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
          break;

        // line search
        Eigen::VectorXd x_prev = x;
        x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 100, 1e-8);
        double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
        hist_line_search_alpha.push_back(alpha);

        int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
        hist_line_search_iter.push_back(line_search_iter);

        // compute the trust region ratio
        double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
        hist_trust_region_ratio.push_back(trust_region_ratio);

        TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);

      // Write final x vector to U matrix.
      // x_to_data(...) takes a lambda function that writes the final value
      // of each variable (Eigen::Vector2d) back to our U matrix.
        func.x_to_data(x, [&] (int v_idx, const Eigen::Vector3d& p) {
            U.row(v_idx) = p;
            });
        {
          std::lock_guard<std::mutex> lock(m);
          redraw = true; 
        }
      }

      TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
      hist.push_back(func.eval(x));

      // output all the optimization statistics
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".mesh", U, F, FF);
        
        std::ofstream output_file_fixed(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_fixed_vid.txt");
        std::ostream_iterator<int> output_iterator_fixed(output_file_fixed, "\n");
        std::copy(std::begin(indices_fixed), std::end(indices_fixed), output_iterator_fixed);

        std::ofstream output_file(output_folder + "hist/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(std::begin(hist), std::end(hist), output_iterator);

        std::ofstream output_file_trust_region_ratio(output_folder + "trust_region_ratio/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_ratio(output_file_trust_region_ratio, "\n");
        std::copy(std::begin(hist_trust_region_ratio), std::end(hist_trust_region_ratio), output_iterator_trust_region_ratio);

        std::ofstream output_file_trust_region_eps(output_folder + "trust_region_eps/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_eps(output_file_trust_region_eps, "\n");
        std::copy(std::begin(hist_trust_region_eps), std::end(hist_trust_region_eps), output_iterator_trust_region_eps);

        std::ofstream output_file_line_search_alpha(output_folder + "line_search_alpha/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_line_search_alpha(output_file_line_search_alpha, "\n");
        std::copy(std::begin(hist_line_search_alpha), std::end(hist_line_search_alpha), output_iterator_line_search_alpha);

        std::ofstream output_file_line_search_iter(output_folder + "line_search_iter/" + output_tag + ".txt");
        std::ostream_iterator<int> output_iterator_line_search_iter(output_file_line_search_iter, "\n");
        std::copy(std::begin(hist_line_search_iter), std::end(hist_line_search_iter), output_iterator_line_search_iter);

        std::ofstream output_file_iter(output_folder + "iter/" + output_tag + ".txt");
        output_file_iter << (hist.size()-1) << std::endl;
      }

      TINYAD_DEBUG_OUT("======== The End ========");
      // comment this out later
      // close the viewer
      exit(0);
    });

  // Plot mesh
  viewer.core().is_animating = true;
  viewer.data().set_mesh(U, FF);
  viewer.core().align_camera_center(U);
  viewer.data().show_lines = false;

  viewer.callback_pre_draw = [&] (igl::opengl::glfw::Viewer& viewer)
  {
    if(redraw)
    {
      viewer.data().set_vertices(U);
      viewer.core().align_camera_center(U);
      {
        std::lock_guard<std::mutex> lock(m);
        redraw = false;
      }
    }
    return false;
  };

  // 在初始化回调中设置标题
  viewer.callback_init = [&](igl::opengl::glfw::Viewer& v)
  {
      // 获取 GLFW 窗口指针并设置标题
      glfwSetWindowTitle(v.window, diff_mode_str.c_str());
      return false;
  };

  viewer.launch();
  if(optimization_thread.joinable())
  {
    optimization_thread.join();
  }

  #if DEBUG_OUTPUT
    TINYAD_CLOSE_DEBUG_LOG();
  #endif
  return 0;
}

int cubic_projected_newton(int argc, char** argv)
{
  // parse command line arguments
  cxxopts::Options options("Projected Newton with a trust region", "Choose eigenvalue filtering method: adaptive, clamp, abs");

  options.add_options()
    ("smooth_mode", "add differentiable eigenvalue projection strategy ", cxxopts::value<std::string>()->default_value("none")) // 若设置diff，后续可微的clamp和abs等
    ("diff", "add differentiable eigenvalue projection strategy ", cxxopts::value<bool>()->default_value("false")) // 若设置diff，后续可微的clamp和abs等
    ("diff_mode", "differentiable eigenvalue projection mode", cxxopts::value<std::string>()->default_value("auto")) // 若设置diff，后续可微的clamp和abs等
    // ("delta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("0.001"))  // 过渡区间宽度
   // ("beta", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("10.0"))// 光滑参数
    ("abs", "use absolute eigenvalue projection strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("clamp", "use eigenvalue clamping strategy instead", cxxopts::value<bool>()->default_value("false"))
    ("p,epsilon", "Projection threshold for the eigenvalue projection", cxxopts::value<std::string>()->default_value("-0.5"))
    ("n,mesh_name", "Mesh name", cxxopts::value<std::string>()->default_value("bimba"))
    ("l,pose_label", "Pose label", cxxopts::value<std::string>()->default_value("stretch"))
    ("g,deformation_magnitude", "The magnitude of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("t,deformation_ratio", "The ratio of the deformation", cxxopts::value<double>()->default_value("2.0"))
    ("b,fixed_boundary_range", "The range of fixed vertices on the boundary", cxxopts::value<double>()->default_value("0.1"))
    ("c,convergence_eps", "The convergence threshold", cxxopts::value<double>()->default_value("1e-5"))
    ("ym", "Young's modulus (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("1e8"))
    ("pr", "Poisson's ratio (need to set both YM and PR to enable this option, otherwise lambda_mu_ratio is used instead)", cxxopts::value<double>()->default_value("0.495"))
    ("tr", "trust region ratio threshold", cxxopts::value<double>()->default_value("0.01"))
    ("experiment_name", "experiment name", cxxopts::value<std::string>()->default_value(""))
    ("rotate_ratio", "The ratio of the rotation", cxxopts::value<double>()->default_value("0.5"))
    ("h,help", "show help")
  ;
  
  auto result = options.parse(argc, argv);
  if (result.count("help"))
  {
      std::cout << options.help() << "\n";
      return 0;
  }

  const bool diff = result["diff"].as<bool>();
  //const int _mode = result["diff_mode"].as<int>();
  std::string diff_mode_str = result["diff_mode"].as<std::string>();
  TinyAD::HessianProjectionMode _diff_mode = parse_projection_mode(diff_mode_str);
  // std::string delta_str = result["delta"].as<std::string>();
  // double delta = std::stod(delta_str);
  // std::string beta_str = result["beta"].as<std::string>();
  // double beta = std::stod(beta_str);


  const bool abs = result["abs"].as<bool>();
  const bool clamp = result["clamp"].as<bool>();
  std::string eps_str = result["epsilon"].as<std::string>();
  double eps = std::stod(eps_str);
  std::string mesh_name = result["mesh_name"].as<std::string>();
  std::string pose_label = result["pose_label"].as<std::string>();
  const double deformation_magnitude = result["deformation_magnitude"].as<double>();
  const double deformation_ratio = result["deformation_ratio"].as<double>();
  const double fixed_boundary_range = result["fixed_boundary_range"].as<double>();
  const double convergence_eps = result["convergence_eps"].as<double>();
  const double YM = result["ym"].as<double>();
  const double PR = result["pr"].as<double>();
  const double tr_threshold = result["tr"].as<double>(); // 信赖域接受步长的阈值，通常设置为0.01
  std::string experiment_folder = result["experiment_name"].as<std::string>() == "" ? ("figure_" + mesh_name) : result["experiment_name"].as<std::string>();
  experiment_folder = experiment_folder + get_time_str();
  const double rotate_ratio = result["rotate_ratio"].as<double>();

  /*
  * 0: clamp
  * -1: abs
  * -0.5: adaptive (default)
  * 0.5: differentiable  (if diff is set)
  */
  if (diff)
  {
      eps_str =  diff_mode_str;
      //下面三种情况,eps值有特殊用处
      if (_diff_mode == TinyAD::HessianProjectionMode::ABS_NONDIFF) {
        eps_str = "abs";
        eps = -1;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_NONDIFF) {
        eps_str = "clamp";
        eps = 0;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF) {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
      }
      else if (_diff_mode == TinyAD::HessianProjectionMode::CUBIC) {
        // set the default to adaptive
        eps_str = "cubic";
        eps = -1;
      }
      else{
         //非特殊情况,eps必须大于等于0
        if (eps < 0)
        {
          eps = 0;
        }
      }
  }
  else {
    if (clamp || eps == 0) {
        // we use eps = 0 as a flag for clamp projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "clamp";
        eps = 0; 
        diff_mode_str = "clamp_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_NONDIFF; 
    }
    else if (abs || eps == -1) {
        // we use eps = -1 as a flag for abs projection, see lines 71-78 in our modified `TinyAD/include/TinyAD/Utils/HessianProjection.hh`
        eps_str = "abs";
        eps = -1;
        diff_mode_str = "abs_nondiff"; 
        _diff_mode = TinyAD::HessianProjectionMode::ABS_NONDIFF; 
    }
    else {
        // set the default to adaptive
        eps_str = "adaptive";
        eps = -0.5;
        diff_mode_str = "clamp_abs_nondiff"; // trust region newton;
        _diff_mode = TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF;
    }
  }
    
  if (!std::filesystem::exists("../results/"))
    std::filesystem::create_directory("../results/");
  if (!std::filesystem::exists("../results/" + experiment_folder))
    std::filesystem::create_directory("../results/" + experiment_folder);

  const std::string output_folder = "../results/" + experiment_folder + "/" 
    + pose_label + "_YM_" + std::to_string(YM) + "_PR_" + std::to_string(PR)
    + "_deform_mag_" + std::to_string(deformation_magnitude) + "_deform_ratio_" 
    + std::to_string(deformation_ratio) + "_fixed_boundary_range_" 
    + std::to_string(fixed_boundary_range)+"_diff_mode_" + diff_mode_str+ "/";
  {
    // record the statistics
    if (!std::filesystem::exists(output_folder))
      std::filesystem::create_directory(output_folder);
    if (!std::filesystem::exists(output_folder + "obj/"))
      std::filesystem::create_directory(output_folder + "obj/");
    if (!std::filesystem::exists(output_folder + "obj/" + mesh_name + "_" + eps_str + "/"))
      std::filesystem::create_directory(output_folder + "obj/" + mesh_name + "_" + eps_str + "/");
    if (!std::filesystem::exists(output_folder + "hist/"))
      std::filesystem::create_directory(output_folder + "hist/");
    if (!std::filesystem::exists(output_folder + "iter/"))
      std::filesystem::create_directory(output_folder + "iter/");
    if (!std::filesystem::exists(output_folder + "trust_region_ratio/"))
      std::filesystem::create_directory(output_folder + "trust_region_ratio/");
    if (!std::filesystem::exists(output_folder + "cubic_rho/"))
      std::filesystem::create_directory(output_folder + "cubic_rho/");
    if (!std::filesystem::exists(output_folder + "trust_region_eps/"))
      std::filesystem::create_directory(output_folder + "trust_region_eps/");
    if (!std::filesystem::exists(output_folder + "line_search_alpha/"))
      std::filesystem::create_directory(output_folder + "line_search_alpha/");
    if (!std::filesystem::exists(output_folder + "line_search_iter/"))
      std::filesystem::create_directory(output_folder + "line_search_iter/");
    
    #if DEBUG_OUTPUT
    if (!std::filesystem::exists(output_folder + "debug_log/"))
    {
      std::filesystem::create_directory(output_folder + "debug_log/");
      TINYAD_INIT_DEBUG_LOG(output_folder + "debug_log/log"+get_time_str()+".txt");
    }
    #endif

    
  }

  const std::string output_tag = "mesh_" + mesh_name + "_eps_" + eps_str;

  // set up viewer
  igl::opengl::glfw::Viewer viewer;

  // compute the Lame parameters
  const double MU = YM / (2 * (1 + PR));
  const double LAMBDA = YM * PR / ((1 + PR) * (1 - 2 * PR));
  const double lambda_mu_ratio = LAMBDA / MU;
 
  // print out the configuration
  {
    TINYAD_DEBUG_OUT("Diff flag: " << diff);
    TINYAD_DEBUG_OUT("*Eigenvalue Differentiable filtering strategy: " << diff_mode_str);
    TINYAD_DEBUG_OUT("Eigenvalue filtering strategy: " << eps_str);
    TINYAD_DEBUG_OUT("mu: " << MU);
    TINYAD_DEBUG_OUT("lambda: " << LAMBDA);
    TINYAD_DEBUG_OUT("*Projection threshold: " << eps);
    TINYAD_DEBUG_OUT("Mesh name: " << mesh_name);
    TINYAD_DEBUG_OUT("Pose label: " << pose_label);
    TINYAD_DEBUG_OUT("Lambda / Mu ratio: " << lambda_mu_ratio);
    TINYAD_DEBUG_OUT("Deformation magnitude: " << deformation_magnitude);
    TINYAD_DEBUG_OUT("Deformation ratio: " << deformation_ratio);
    TINYAD_DEBUG_OUT("Fixed vertices boundary range: " << fixed_boundary_range);
    TINYAD_DEBUG_OUT("Convergence threshold: " << convergence_eps);
    TINYAD_DEBUG_OUT("Young's modulus: " << YM);
    TINYAD_DEBUG_OUT("Poisson's ratio: " << PR);
    TINYAD_DEBUG_OUT("Trust region threshold: " << tr_threshold);
    TINYAD_DEBUG_OUT("Experiment folder: " << experiment_folder);
    TINYAD_DEBUG_OUT("Rotate ratio: " << rotate_ratio);
  }

  Eigen::MatrixXd V, U; // #V-by-3 3D vertex positions,V为初始,U为当前
  Eigen::MatrixXi F, FF; // #T-by-4 indices into V
  Eigen::VectorXi TriTag, TetTag;
  if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh"))
    igl::readMESH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".mesh", V, F, FF);
  else if (std::filesystem::exists(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh"))
    igl::readMSH(std::string(SOURCE_PATH) + "/data/" + mesh_name + ".msh", V, FF, F, TriTag, TetTag);
  else {
    std::cout << "Mesh " << mesh_name << " not found!" << std::endl;
    exit(1);
  }

  TINYAD_DEBUG_OUT("Read mesh with " << V.rows() << " vertices and " << F.rows() << " tetrahedrons.");

  // get boundary vertices
  igl::boundary_facets(F, FF);
  FF = FF.rowwise().reverse().eval();

  TINYAD_DEBUG_OUT("Boundary has " << FF.rows() << " faces.");

  // normalize the mesh V
  Eigen::RowVector3d mean = V.colwise().mean();
  V.rowwise() -= mean;
  double max_norm = V.rowwise().norm().maxCoeff();
  V /= max_norm;

  // set up mesh U
  U = V;

  // fixed point constraints
  Eigen::SparseMatrix<double> P;
  std::vector<unsigned int> indices_fixed;

  // set up fixed point constraints and initial deformation
  setup_initial_deformation(V, F, pose_label, deformation_magnitude, deformation_ratio, rotate_ratio, fixed_boundary_range, U, indices_fixed);
  fixed_point_constraints(P, 3*V.rows(), 3, indices_fixed);

  TINYAD_DEBUG_OUT("Finish setting up fixed point constraints.");

  bool redraw = false;
  std::mutex m;
  std::thread optimization_thread(
    [&]()
    {
      // Pre-compute triangle rest shapes in local coordinate systems
      std::vector<Eigen::Matrix3d> rest_shapes(F.rows());
      for (int f_idx = 0; f_idx < F.rows(); ++f_idx)
      {
        // Get 3D vertex positions
        Eigen::Vector3d ar = V.row(F(f_idx, 0));
        Eigen::Vector3d br = V.row(F(f_idx, 1));
        Eigen::Vector3d cr = V.row(F(f_idx, 2));
        Eigen::Vector3d dr = V.row(F(f_idx, 3));

        // Save 3-by-3 matrix with edge vectors as columns
        rest_shapes[f_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
      };

      /*
          TinyAD::scalar_function<3>: 创建一个标量函数，每个变量是3维向量（例如3D顶点坐标）
          TinyAD::range(V.rows()): 定义变量的范围，V.rows() 是顶点数量
          auto func: 自动推断函数对象类型
      */ 
      // TinyAD::scalar_function(n)：这是一个函数，可以计算它的函数值、梯度、Hessian的函数，其自变量索引的范围为n
      // Set up function with 3d vertex positions as variables.
      auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

      // func.add_elements<n>(element_range, energy_function); 
      // 定义元素的能量项，1.n为元素的dim，其中2.element_range是元素索引的范围，
      // 3.energy_function是一个lambda函数，接受一个元素对象，返回该元素的能量值
      // 4. lamda函数 [捕获变量] (输入参数) -> 返回值 
      // 5. 所有参与自动微分计算的变量和中间结果，都必须使用 TINYAD_SCALAR_TYPE 或其衍生的类型
      // 6. element.handle对应element_range中的当前元素的索引，element.variables()对应输入变量，想要访问输入变量的第i个变量
      // 7. 在这个地方，输入变量是顶点数组，element_range是F数组，即按F计算能量
      // Add objective term per element. Each connecting 4 vertices.
      // "neo-hooken energy" is defined on each tetrahedron, so element_range is F.rows() and element.variables() gives the vertex positions of the current tetrahedron.
      func.add_elements<4>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element) {
          // Evaluate element using either double or TinyAD::Double
          using T = TINYAD_SCALAR_TYPE(element);

          // Get variable 3d vertex positions
          Eigen::Index f_idx = element.handle;
          Eigen::Vector3<T> a = element.variables(F(f_idx, 0));
          Eigen::Vector3<T> b = element.variables(F(f_idx, 1));
          Eigen::Vector3<T> c = element.variables(F(f_idx, 2));
          Eigen::Vector3<T> d = element.variables(F(f_idx, 3));

          Eigen::Matrix3<T> M = TinyAD::col_mat(b - a, c - a, d - a);
          Eigen::Matrix3d Mr = rest_shapes[f_idx];
          Eigen::Matrix3<T> J = M * Mr.inverse(); // 可以放外边 to be optimized by zj

          // Compute the stable Neo-Hookean energy [Smith et al. 2018]
          double A = 0.5 * Mr.determinant(); //可以放外边 to be optimized by zj
          const double mu = MU;
          const double lambda = LAMBDA;
          auto Ic = (J.transpose() * J).trace();
          auto detF = J.determinant();
          double alpha = 1.0 + mu / lambda;
          auto W = mu / 2.0 * (Ic - 3.0) + lambda / 2.0 * (detF - alpha) * (detF - alpha);
          return A * W;
      });

      // to be optimized by zj: 可以考虑把 rest_shapes 以及 pre-computed Mr.inverse() 放到外边，作为常量传入 lambda 函数中，这样就不需要每次迭代都计算 rest_shapes 和 Mr.inverse() 了
      // to be optimized by zj: 还可以考虑把 A 也放到外边，因为 A 只和 rest_shapes 相关，而 rest_shapes 是不变的
      // to be optimized by zj: 还可以考虑把 lambda 和 mu 放到外边，因为它们也是不变的
      // to be modified by zj: 动态状态下的PD，增加local step，即每个元素的能量项不仅依赖于当前状态，还依赖于上一个状态，这样可以增加稳定性，尤其是在使用abs投影时

      // func.x_from_data() 是 TinyAD 提供的数据格式转换工具，将外部数据（如顶点矩阵）转换为优化所需的展平向量格式，使代码更简洁、更安全。
      // v_idx 的取值范围是由 TinyAD::scalar_function<k>(n_variables) 中的 n_variables 决定的，即函数的输入变量的维度。
      // 在这个例子中，n_variables 是 V.rows()，即顶点数量，因此 v_idx 的取值范围是 [0, V.rows()-1]。
      // Assemble inital x vector from U matrix.
      // x_from_data(...) takes a lambda function that maps
      // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
      Eigen::VectorXd x = func.x_from_data([&] (int v_idx) {
          return U.row(v_idx);
      });

      TINYAD_DEBUG_OUT("Initial energy: " << func.eval(x));

      // Projected Newton
      TinyAD::LinearSolver solver; //Eigen::SimplicialLDLT*,ConjugateGradient,SparseLU
      int max_iters = 200; // 迭代次数可以根据需要调整
      Eigen::VectorXd d;
      Eigen::SparseMatrix<double> H;
      std::vector<double> hist;
      // record the trust region ratio
      // 衡量模型预测的准确性 ρ = (实际下降) / (预测下降), 
      // ρ < 0         → 实际目标函数值增加（拒绝步长）
      // 0 ≤ ρ < 0.25  → 模型质量差，缩小信赖域
      // 0.25 ≤ ρ < 0.75 → 模型质量一般，保持信赖域
      // ρ ≥ 0.75      → 模型质量好，可以放大信赖域
      // double delta;  // 当前信赖域半径
      // double eta1;   // 接受步长的阈值（通常0.25）
      // double eta2;   // 放大半径的阈值（通常0.75）
      std::vector<double> hist_trust_region_ratio;
      hist_trust_region_ratio.push_back(0.0);
      std::vector<double> hist_cubic_rho;
      hist_cubic_rho.push_back(1.0);

      std::vector<double> hist_trust_region_eps; // 记录每次迭代使用的eps值，以观察自适应策略的变化
      std::vector<double> hist_line_search_alpha; // 记录每次迭代的线搜索步长
      std::vector<int> hist_line_search_iter; // 记录每次迭代的线搜索迭代次数
      
      //strategy = SIGMA_STRATEGIES::FIXED,  
      // upper_threshhold = 0.75,  lower_threshhold = 0.25,  step = 2,  level = 0
      // rho = 1.0, sigma = 1.0
      Cubic_Regular cr;//
      //cr.set_rho(1.0);
      //cr.set_sigma(1.0);
      //当global matrix发生变化时，进行下面的牛顿投影和线搜索步骤
      for (int i = 0; i < max_iters; ++i)
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(i) + ".mesh", U, F, FF);
      
        // switch between clamp or abs depending on whether the trust region ratio is close to 1
        //if(eps_str == "adaptive") {
        

        double f;
        Eigen::VectorXd g;
        Eigen::MatrixXd H_proj;
        if (_diff_mode == TinyAD::HessianProjectionMode::CUBIC) {
          if (cr.get_rho() < cr.get_rho_upper_threshhold()) // projected
          {
            //if (_diff_mode == TinyAD::HessianProjectionMode::CLAMP_ABS_NONDIFF) {
            eps = (std::fabs(hist_trust_region_ratio.back() - 1.0) < tr_threshold) ? 0.0 : -1; // tr_threshold信赖域接受步长的阈值，通常设置为0.01,大于接受

            if (eps == 0.0) {
              TINYAD_DEBUG_OUT("Switch to clamp");  
            }
            else {
              TINYAD_DEBUG_OUT("Switch to abs");
            }
            hist_trust_region_eps.push_back(eps);
            TinyAD::HessianProjectionMode _diff_mode_temp = TinyAD::HessianProjectionMode::ABS_NONDIFF; 
            auto [f, g, H_proj] = func.eval_with_hessian_proj(x, eps, _diff_mode_temp); //
            Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量vvv
            
            // record the energy
            hist.push_back(f);

            TINYAD_DEBUG_OUT("Energy of trust region newton in iteration " << i << ": " << f);

            // Compute the Newton direction
            H = H_proj;
            
            Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
            Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
            d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
            d = P.transpose() * d;

            TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
            TINYAD_DEBUG_OUT("d norm " << i << " =  " << d.norm());
            TINYAD_DEBUG_OUT("g norm " << i << " =  " << g.norm());

            // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
            // 优化方案1： 相对收敛
            // if (newton_decrement < convergence_eps * initial_decrement) break;  
            // 方案3：梯度范数 
            // if (g.norm() < convergence_eps) break;
            // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
            // 方案5：根据材料参数调整收敛标准
            //  LAMBDA是材料参数，不是尺度参数
            // // 使用特征长度进行无量纲化
            // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
            // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
            // // 物理意义的收敛判据
            // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
            if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
              break;

            // line search
            Eigen::VectorXd x_prev = x;
            x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 100, 1e-8);
            double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
            hist_line_search_alpha.push_back(alpha);

            int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
            hist_line_search_iter.push_back(line_search_iter);

            // compute the trust region ratio
            double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
            hist_trust_region_ratio.push_back(trust_region_ratio);

            TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);
            
            double rho = cr.compute_ratio_rho(func.eval(x), f, alpha*d, g, H0, cr.get_sigma());
            hist_cubic_rho.push_back(rho);
            hist_cubic_rho.push_back(rho);
            cr.update_sigma();
            cr.update_lambda(alpha*d);
            
          }
          else // cubic 避免矩阵分解
          {
             auto [f, g, H_cr] = func.eval_with_derivatives(x);
             double lambda = cr.get_lambda(); //d初始为0
              Eigen::VectorXd lambda_vec = Eigen::VectorXd::Constant(H_cr.rows(), lambda);
              H_cr += lambda_vec.asDiagonal();
             
              Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量vvv
              
              // record the energy
              hist.push_back(f);

              TINYAD_DEBUG_OUT("Energy of cubic newton in iteration " << i << ": " << f);

              // Compute the Newton direction
              H = H_cr;
              
              Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
              Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
              d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
              d = P.transpose() * d;

              TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
              TINYAD_DEBUG_OUT("d norm of cubic newton " << i << " =  " << d.norm());
              TINYAD_DEBUG_OUT("g norm of cubic newton" << i << " =  " << g.norm());
              TINYAD_DEBUG_OUT("H norm of cubic newton" << i << " =  " << H.norm());

              #if DEBUG_OUTPUT_CUBIC
                auto [f2, g2, H_proj2] = func.eval_with_hessian_proj(x, eps, TinyAD::HessianProjectionMode::ABS_NONDIFF); //
                Eigen::SparseMatrix<double> PHP2 = P* H_proj2 *P.transpose();
                auto d2 = TinyAD::newton_direction(Pg, PHP2, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
                d2 = P.transpose() * d2;
                TINYAD_DEBUG_OUT("d2 norm of projected newton" << i << " =  " << d2.norm());
                TINYAD_DEBUG_OUT("g2 norm of projected newton" << i << " =  " << g2.norm());
                TINYAD_DEBUG_OUT("H2 norm of projected newton" << i << " =  " << H_proj2.norm());
              #endif

              // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
              // 优化方案1： 相对收敛
              // if (newton_decrement < convergence_eps * initial_decrement) break;  
              // 方案3：梯度范数 
              // if (g.norm() < convergence_eps) break;
              // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
              // 方案5：根据材料参数调整收敛标准
              //  LAMBDA是材料参数，不是尺度参数
              // // 使用特征长度进行无量纲化
              // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
              // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
              // // 物理意义的收敛判据
              // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
              if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
                break;

              // line search
              Eigen::VectorXd x_prev = x;
              x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 100, 1e-8);
              double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
              hist_line_search_alpha.push_back(alpha);

              int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
              hist_line_search_iter.push_back(line_search_iter);

              // compute the trust region ratio
              double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
              hist_trust_region_ratio.push_back(trust_region_ratio);

              TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);
              
              double rho = cr.compute_ratio_rho(func.eval(x), f, alpha*d, g, H0, cr.get_sigma());
              hist_cubic_rho.push_back(rho);
              cr.update_sigma();
              cr.update_lambda(alpha*d);
          }

          
        } // non cubic mode
        else{

          auto [f, g, H_proj] = func.eval_with_hessian_proj(x, eps, _diff_mode); //
          Eigen::SparseMatrix<double> H0 = func.eval_hessian(x); // 计算未投影的Hessian，用于后续计算牛顿下降量和牛顿下降量vvv
          
          // record the energy
          hist.push_back(f);

          TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);

          // Compute the Newton direction
          H = H_proj;
          
          Eigen::VectorXd Pg = P*g; // 固定点约束下的梯度
          Eigen::SparseMatrix<double> PHP = P*H*P.transpose();
          d = TinyAD::newton_direction(Pg, PHP, solver); //用于计算牛顿方向的函数,sovler是线性方程求解器实例
          d = P.transpose() * d;

          TINYAD_DEBUG_OUT("Newton decrement " << i << " =  " << TinyAD::newton_decrement(d, g));
          TINYAD_DEBUG_OUT("d norm " << i << " =  " << d.norm());
          TINYAD_DEBUG_OUT("g norm " << i << " =  " << g.norm());

          // 对于刚性材料（LAMBDA大）：收敛标准更严格; 对于柔软材料（LAMBDA小）：收敛标准更宽松
          // 优化方案1： 相对收敛
          // if (newton_decrement < convergence_eps * initial_decrement) break;  
          // 方案3：梯度范数 
          // if (g.norm() < convergence_eps) break;
          // 方案4：牛顿下降量（Newton decrement）  Newton decrement = sqrt(d' * H * d)，它衡量了沿着牛顿方向的预期下降量
          // 方案5：根据材料参数调整收敛标准
          //  LAMBDA是材料参数，不是尺度参数
          // // 使用特征长度进行无量纲化
          // double characteristic_length = compute_mesh_size(V); // 计算网格特征尺寸
          // double characteristic_volume = characteristic_length * characteristic_length * characteristic_length;
          // // 物理意义的收敛判据
          // if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA * characteristic_volume) break;
          if (std::fabs(TinyAD::newton_decrement(d, g)) < convergence_eps * LAMBDA)
            break;

          // line search
          Eigen::VectorXd x_prev = x;
          x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 100, 1e-8);
          double alpha = (x - x_prev).norm() / d.norm(); // 计算实际步长与牛顿方向的比值，反映线搜索的收缩程度
          hist_line_search_alpha.push_back(alpha);

          int line_search_iter = std::lround(std::log(alpha) / std::log(0.8)) + 1; // 计算线搜索迭代次数，基于初始步长和最终步长的比值
          hist_line_search_iter.push_back(line_search_iter);

          // compute the trust region ratio
          double trust_region_ratio = compute_trust_region_ratio(func.eval(x), f, alpha*d, g, H0); // 计算信赖域比率，评估模型预测的准确性
          hist_trust_region_ratio.push_back(trust_region_ratio);

          TINYAD_DEBUG_OUT("Trust region ratio: " << trust_region_ratio);
        }
         
        

      // Write final x vector to U matrix.
      // x_to_data(...) takes a lambda function that writes the final value
      // of each variable (Eigen::Vector2d) back to our U matrix.
        func.x_to_data(x, [&] (int v_idx, const Eigen::Vector3d& p) {
            U.row(v_idx) = p;
            });
        {
          std::lock_guard<std::mutex> lock(m);
          redraw = true; 
        }
      }

      TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
      hist.push_back(func.eval(x));

      // output all the optimization statistics
      {
        igl::writeOBJ(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".obj", U, FF);
        igl::writeMESH(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_iter_" + std::to_string(hist.size()-1) + ".mesh", U, F, FF);
        
        std::ofstream output_file_fixed(output_folder + "obj/" + mesh_name + "_" + eps_str + "/" + output_tag + "_fixed_vid.txt");
        std::ostream_iterator<int> output_iterator_fixed(output_file_fixed, "\n");
        std::copy(std::begin(indices_fixed), std::end(indices_fixed), output_iterator_fixed);

        std::ofstream output_file(output_folder + "hist/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(std::begin(hist), std::end(hist), output_iterator);

        std::ofstream output_file_trust_region_ratio(output_folder + "trust_region_ratio/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_ratio(output_file_trust_region_ratio, "\n");
        std::copy(std::begin(hist_trust_region_ratio), std::end(hist_trust_region_ratio), output_iterator_trust_region_ratio);

        std::ofstream output_file_cubic_rho(output_folder + "cubic_rho/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_cubic_rho(output_file_cubic_rho, "\n");
        std::copy(std::begin(hist_cubic_rho), std::end(hist_cubic_rho), output_iterator_cubic_rho);

        std::ofstream output_file_trust_region_eps(output_folder + "trust_region_eps/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_trust_region_eps(output_file_trust_region_eps, "\n");
        std::copy(std::begin(hist_trust_region_eps), std::end(hist_trust_region_eps), output_iterator_trust_region_eps);

        std::ofstream output_file_line_search_alpha(output_folder + "line_search_alpha/" + output_tag + ".txt");
        std::ostream_iterator<double> output_iterator_line_search_alpha(output_file_line_search_alpha, "\n");
        std::copy(std::begin(hist_line_search_alpha), std::end(hist_line_search_alpha), output_iterator_line_search_alpha);

        std::ofstream output_file_line_search_iter(output_folder + "line_search_iter/" + output_tag + ".txt");
        std::ostream_iterator<int> output_iterator_line_search_iter(output_file_line_search_iter, "\n");
        std::copy(std::begin(hist_line_search_iter), std::end(hist_line_search_iter), output_iterator_line_search_iter);

        std::ofstream output_file_iter(output_folder + "iter/" + output_tag + ".txt");
        output_file_iter << (hist.size()-1) << std::endl;
      }

      TINYAD_DEBUG_OUT("======== The End ========");
      // comment this out later
      // close the viewer
      exit(0);
    });

  // Plot mesh
  viewer.core().is_animating = true;
  viewer.data().set_mesh(U, FF);
  viewer.core().align_camera_center(U);
  viewer.data().show_lines = true;
  viewer.core().depth_test = true;
  viewer.data().line_width = 1.0;

  // // // Plot mesh
  // // viewer.core().is_animating = true;
  // // viewer.core().depth_test = true;

  // // viewer.data().set_mesh(U, FF);
  // // viewer.core().align_camera_center(U);
  
  // // Eigen::Vector3d color(0.2, 0.2, 0.8);
  
  // // 设置顶点颜色，为第一个网格设置带透明度的颜色（使用RGBA颜色）
  // // viewer.data().set_colors(color_blue);

  // viewer.data().show_lines = true;  // 显示网格线
  // // 设置透明度（需要启用深度剥离或混合）
  // viewer.data().line_width = 1.0;
  // viewer.data().point_size = 3.0;  // 显示顶点
  
  // 设置透明材质
  Eigen::Vector3d color(0.2, 0.2, 0.8);
  viewer.data().uniform_colors(
      //Eigen::Vector3d(color3(0), color3(1), color3(2)),  // 漫反射
      Eigen::Vector3d(color(0), color(1), color(2)),  // 漫反射
      Eigen::Vector3d(0.2, 0.2, 0.2),  // 环境光
      Eigen::Vector3d(0.0, 0.0, 0.0)   // 镜面反射
  );
    
  // 设置面的透明度 - 使用材质属性
  viewer.data().face_based = true;  // 基于面的渲染

  // // 设置背景颜色
  // viewer.core().background_color << 0.9, 0.9, 0.9, 0.8;

  viewer.callback_pre_draw = [&] (igl::opengl::glfw::Viewer& viewer)
  {
    if(redraw)
    {
      viewer.data().set_vertices(U);
      viewer.core().align_camera_center(U);
      {
        std::lock_guard<std::mutex> lock(m);
        redraw = false;
      }
    }
    return false;
  };

  // 在初始化回调中设置标题
  viewer.callback_init = [&](igl::opengl::glfw::Viewer& v)
  {
      // 获取 GLFW 窗口指针并设置标题
      glfwSetWindowTitle(v.window, diff_mode_str.c_str());
      return false;
  };
  

  viewer.launch();
  if(optimization_thread.joinable())
  {
    optimization_thread.join();
  }

  #if DEBUG_OUTPUT
    TINYAD_CLOSE_DEBUG_LOG();
  #endif
  return 0;
}


