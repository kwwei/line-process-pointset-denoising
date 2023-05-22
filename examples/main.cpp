#include <assert.h>
#include <igl/unproject_onto_mesh.h>
#include <cstdlib>
#include "postprocessing.h"
#include "utils.h"
#include "IO.h"
#include "neighbors.h"
#include "denoising.h"

int main(int argc, char *argv[])
{
  if ( argc != 16 ) {
    cerr << "Usage: " << endl;
    cerr << "16 params to be inserted in order, but only few needs to be tuned!" << endl;
    cerr << "input_dir: string, input directory" << endl;
    cerr << "output_dir: string, output directory" << endl;
    cerr << "w_a: double, weight parameter for local fitting term. The default value is 1 (in our paper and in all our experiments)." << endl;
    cerr << "w_b: double, weight parameter for stitching term, referred as lambda in our paper. The default value is 5000 (in all of our experiments)." << endl;
    cerr << "w_c: double, weight parameter for piecewise smoothness term, referred as eta in our paper. TO BE TUNED" << endl;
    cerr << "mu_fit: double, selectivity of line processes in local fitting stage, referred as mu_l in our paper. The default value is 5e-9. TO BE TUNED." << endl;
    cerr << "mu_smooth: double, selectivity of line processes in piecewise smoothness stage, reffered as mu_m in our paper, can be interpreted as the minimum normal angle change. The default value is 30.0 (in most of our experiments.)" << endl;
    cerr << "lp_threshold: double, threshold value for line process parameters. Determine whether a point is an outlier. Used for post-processing step. The default value is 0.3" << endl;
    cerr << "k_neighbor: int, neighborhood size. TO BE TUNED" << endl;
    cerr << "max_smooth_iter: int, max iteration step for piecewise smoothness optimization, referred as innerIter in Algorithm 1 in our paper. The default value is 2 (5 for stress tests)" << endl;
    cerr << "need_line_process: int (0 or 1), whether post-processing step is needed" << endl;
    cerr << "smooth_comparison: NULL" << endl;
    cerr << "reconstruction_method: int, 1 - Scale Space surface reconstruction 2 - Advancing Front surface reconstruction, others - no surface reconstruction. The default value is 2." << endl;
    cerr << "need_s: int (0 or 1), whether the scalar coefficients sij's are needed in the optimization (See figure 4). The default value is 1." << endl;

    return __LINE__;
  }
  std::string input_dir = argv[1];
  std::string output_dir = argv[2];
  std::string object_name = argv[3];
  double w_a = std::stod(argv[4]);
  double w_b = std::stod(argv[5]);
  double w_c = std::stod(argv[6]);
  double mu_fit = std::stod(argv[7]);
  double mu_smooth = std::stod(argv[8]);
  double lp_threshold = std::stod(argv[9]); // line process parameter threshold 
  int k_neighbor = std::stoi(argv[10]);
  int max_smooth_iter = std::stoi(argv[11]);
  bool need_post_process = boost::lexical_cast<bool>(argv[12]);
  std::string smooth_comparison = argv[13];
  int reconstruction_method = std::stoi(argv[14]);
  int need_s = std::stoi(argv[15]);
  std::string save_dir = output_dir+object_name;
  if (!fs::is_directory(save_dir) || !fs::exists(save_dir)) {
    fs::create_directories(save_dir);
  }
  std::cout<<save_dir<<std::endl;

  // Mesh with per-face color
  Eigen::MatrixXd V, C;
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F;
  // Read points
  readXYZ((input_dir+"/"+object_name+".xyz").c_str(), V);
  std::cout<<"points : "<<V.rows()<<" "<<V.cols()<<std::endl;
  //normalize points
  double bbox_diag_length;
  Eigen::RowVectorXd mean_x;
  normalize_shape(V, bbox_diag_length, mean_x);


  // Initialize
  std::vector<int> edge_v;
  edge_v.clear();
  // std::cout<<w_a<<" "<<w_b<<" "<<w_c<<" "<<mu_fit<<" "<<mu_smooth<<std::endl;
  mu_smooth = (2 - 2 * cos(mu_smooth * M_PI / 180.0)) * 0.5;

  if (smooth_comparison != "NULL") {
    compute_mesh_dirichlet(smooth_comparison);
  }
  else {
      std::cout<<"no smooth input...."<<std::endl;
  }
  Eigen::MatrixXd c;
  kd_tree tree(3, std::cref(V), 10);
  tree.index -> buildIndex();
  std::vector<Eigen::MatrixXd> inter_res; // store intermediate results

  denoising(V, c, inter_res, tree, k_neighbor, w_a, w_b, w_c, mu_fit, mu_smooth, lp_threshold, max_smooth_iter, save_dir, edge_v, need_s);

  Eigen::MatrixXd V_updated = post_processing(V, bbox_diag_length, mean_x, tree, c, edge_v, 2*k_neighbor, save_dir, need_post_process);
  put_back_shape(V_updated, bbox_diag_length, mean_x);
  put_back_shape(V, bbox_diag_length, mean_x);
  char result_name[160];
  std::sprintf(result_name, "/result_k_%i_wa_%.2f_wb_%.2f_wc_%.5f_mus_%.5f", k_neighbor, w_a, w_b, w_c, mu_smooth);

  saveXYZ((save_dir+result_name+".xyz").c_str(), V_updated);
  if (need_post_process)
    saveXYZ((save_dir+result_name+"_with_post_process.xyz").c_str(), V);
  std::cout<<(save_dir+result_name+".xyz").c_str()<<std::endl;
  if (reconstruction_method == 1) {
    save_surface((save_dir+result_name+".off").c_str(), V_updated);
    if (need_post_process)
      save_surface((save_dir+result_name+"_with_post_process.off").c_str(), V);    
  }
  else if (reconstruction_method == 2) {
    save_surface_advance_front((save_dir+result_name+".off").c_str(), V_updated);
    if (need_post_process)
      save_surface_advance_front((save_dir+result_name+"_with_post_process.off").c_str(), V);    
  }


  // output intermediate results
  // char inter_res_name[80];
  // for (int i=0; i<inter_res.size(); ++i) {
  //   Eigen::MatrixXd res = inter_res[i];
  //   std::sprintf(inter_res_name,"/result_iter_%i.xyz", i);
  //   put_back_shape(res, bbox_diag_length, mean_x);
  //   saveXYZ((save_dir+inter_res_name).c_str(), res);
  // }

  return 0;
}


