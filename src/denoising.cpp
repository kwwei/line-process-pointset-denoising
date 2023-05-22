#include "denoising.h"
#include "utils.h"
#include "neighbors.h"
#include "IO.h"
#include "pcg.h"
#include "postprocessing.h"
void initialize_params(const Eigen::MatrixXd &x,
                       const kd_tree &tree,
                       const int k_neighbor,
                       std::vector<double>& alpha,
                       Eigen::SparseMatrix<double> &m,
                       Eigen::SparseMatrix<double> &l,
                       Eigen::SparseMatrix<double> &r,
                       Eigen::SparseMatrix<double> &s) {
  std::vector<T> tripletList_l;
  std::vector<T> tripletList_s;
  std::vector<T> tripletList_r;
  Eigen::SparseMatrix<double> N;

  const int n = x.rows();
  
  // initialize m
  m = find_neighbors_k(x, tree, k_neighbor); 
  N = find_neighbors_k(x, tree, k_neighbor);
  // initialize l and s
  std::vector<int> neighbor_size(n, 0);
  std::vector<int> l_neighbor_size(n, 0);
  for (int k=0; k<N.outerSize(); ++k) {
    for (int iter = N.outerIndexPtr()[k]; iter < N.outerIndexPtr()[k+1]; ++iter) {
      int i = N.innerIndexPtr()[iter];
      int j = k;

      // fill alpha
      alpha[k] += (x.row(k) - x.row(i)).squaredNorm();
      neighbor_size[k] += 1;

      if (i == j)
        continue;
      l_neighbor_size[k] += 1;
      tripletList_l.push_back(T(std::max(i, j), std::min(i, j), 1.0));
      tripletList_s.push_back(T(std::max(i, j), std::min(i, j), 1.0));
    }
  }

  std::transform(alpha.begin(), alpha.end(), neighbor_size.begin(), alpha.begin(), std::divides<double>());
  
  l.setFromTriplets(tripletList_l.begin(), tripletList_l.end());
  s.setFromTriplets(tripletList_s.begin(), tripletList_s.end());
  if ( !l.isCompressed() ) {
    l.makeCompressed();
  }
  if ( !s.isCompressed() ) {
    s.makeCompressed();
  }  
  std::fill(s.valuePtr(), s.valuePtr()+s.nonZeros(), 1);
  std::fill(l.valuePtr(), l.valuePtr()+l.nonZeros(), 1);

  // beta_ij: area for smoothness integral
  for (int k=0; k<l.outerSize(); ++k) {
    for (int iter = l.outerIndexPtr()[k]; iter < l.outerIndexPtr()[k+1]; ++iter) {
      int i = l.innerIndexPtr()[iter];
      int j = k;
      double beta_ij = alpha[i] / l_neighbor_size[i] + alpha[j] / l_neighbor_size[j];
      double wij = beta_ij / ((x.row(i) - x.row(j)).squaredNorm()+1e-8);
      tripletList_r.push_back(T(i, j, wij));
    }
  }
  r.setFromTriplets(tripletList_r.begin(), tripletList_r.end());
  if ( !r.isCompressed() ) {
    r.makeCompressed();
  }

  for (size_t iter = 0; iter < l.nonZeros(); ++iter) {
    if ( s.innerIndexPtr()[iter] != l.innerIndexPtr()[iter] ) {
      std::cout << "error" << std::endl;
      exit(0);
    }
  }
  for (size_t iter = 0; iter < l.nonZeros(); ++iter) {
    if ( r.innerIndexPtr()[iter] != l.innerIndexPtr()[iter] ) {
      std::cout << "error" << std::endl;
      exit(0);
    }
  }
}


Eigen::SparseMatrix<double> assembleA(const unsigned int n, const std::vector<double>& alpha_weight) {
  if ( alpha_weight.size() != n ) {
    exit(0);
  }
  return Eigen::SparseMatrix<double>(Eigen::Map<const Eigen::VectorXd>(&alpha_weight[0], alpha_weight.size()).asDiagonal());
}

Eigen::SparseMatrix<double>
assemble_smooth_matrix(const unsigned int n,
                       const Eigen::SparseMatrix<double>&s,
                       const Eigen::SparseMatrix<double>&l,
                       const Eigen::SparseMatrix<double>&r) {
  Eigen::SparseMatrix<double> B(n, n);

  // mutual edges
  std::vector<T> tripletList;
  for (int k=0; k<l.outerSize(); ++k) {
    for (int iter = l.outerIndexPtr()[k]; iter < l.outerIndexPtr()[k+1]; ++iter) {
      int row = l.innerIndexPtr()[iter];
      int col = k;

      double val = l.valuePtr()[iter] * r.valuePtr()[iter];
      double sij = s.valuePtr()[iter];

      tripletList.push_back(T(row, row, val));
      tripletList.push_back(T(col, col, sij*sij*val));
      tripletList.push_back(T(row, col, -sij*val));
      tripletList.push_back(T(col, row, -sij*val));
    }
  }
  B.setFromTriplets(tripletList.begin(), tripletList.end());
  if ( !B.isCompressed() ) {
    B.makeCompressed();
  }
  return B;
}

double geman_mcclure(const double mu, const double z) {
  return mu * pow(sqrt(z)-1, 2);
}

double eval_total_energy(const Eigen::MatrixXd &p_x,
                         const Eigen::MatrixXd &c,
                         const Eigen::MatrixXd &h,
                         const Eigen::SparseMatrix<double>& m,
                         const Eigen::SparseMatrix<double>& s,
                         const Eigen::SparseMatrix<double>& r,
                         const Eigen::SparseMatrix<double>& l,
                         const std::vector<double> &alpha,
                         const double w_b,
                         const double w_c,
                         const double mu_fit,
                         const double mu_smooth) {
  double e_local(0.0), e_stitch(0.0), e_smooth(0.0);
  // compute e_local
  for (int j = 0; j < m.outerSize(); ++j) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(m, j); it; ++it) {
      int nei = it.row();
      double mij = it.value();
      Eigen::VectorXd pj = p_x.row(nei);
      e_local += alpha[j] * (mij * pow(h.row(j).dot(pj), 2) + geman_mcclure(mu_fit, mij));
    }
  }
  e_local /= 2.0;
    
    // compute e_stitch
  for (int i=0; i<c.rows(); ++i) {
    e_stitch += alpha[i] * pow((c.row(i)-h.row(i)).norm(), 2);
  }
  e_stitch *= (w_b/2.0);

    //compute e_smooth
  for (int k=0; k<l.outerSize(); ++k) {
    for (int iter = l.outerIndexPtr()[k]; iter < l.outerIndexPtr()[k+1]; ++iter) {
      int nei = l.innerIndexPtr()[iter];
      double lij = l.valuePtr()[iter];
      double beta_ij = r.valuePtr()[iter];
      double sij = s.valuePtr()[iter];
      e_smooth += beta_ij * (lij * pow((c.row(k) - sij * c.row(nei)).norm(), 2) + geman_mcclure(mu_smooth, lij));
    }
  }
  e_smooth *= (w_c/2.0);

  return (e_local+e_stitch+e_smooth);
}

void denoising(Eigen::MatrixXd &x,
               Eigen::MatrixXd &c,
               std::vector<Eigen::MatrixXd> &inter_rs,
               const kd_tree &tree,
               const int k_neighbor,
               const double w_a,
               const double w_b,
               const double w_c,
               const double mu_fit,
               const double mu_smooth,
               const double lp_threshold,
               const int max_smooth_iter,
               const std::string& output_dir,
               std::vector<int>& edge_v,
               const int need_s) {
  const int n = x.rows(), d = 4;
  std::cout<<"n="<<n<<" d="<<d<<std::endl;
  inter_rs.clear();
  char results[100];
  // construct P
  Eigen::MatrixXd p_x = Eigen::MatrixXd::Ones(n, d);
  p_x.leftCols(d-1) = x;
  Eigen::SparseMatrix<double> m(n, n), l(n, n), r(n, n), s(n, n), m_to_visualize(n, n);
  std::vector<double> alpha(n, 0.0);
  initialize_params(x, tree, k_neighbor, alpha, m, l, r, s);

  // initialize A and B
  const auto &A = assembleA(n, alpha);

  // temp variables 
  Eigen::SparseMatrix<double> LHS;
  Eigen::MatrixXd g, h_prev;
  Eigen::SparseMatrix<double> s_prev;

  // start iteration
  Eigen::MatrixXd h = Eigen::MatrixXd::Random(n, d);
  // int save_cnt = 0;
  // Eigen::MatrixXd normals = h.leftCols(3);
  // normals.rowwise().normalize();
  // std::sprintf(results, "/result_normals_iter_%i.vtk", save_cnt);
  // line2vtk((output_dir+results).c_str(), x, x+0.03*normals);
  // save_cnt ++;

  c = Eigen::MatrixXd::Zero(n, d);
  if (fs::exists(output_dir+"/c.bin")) {  
    EigenSup::read_binary((output_dir+"/c.bin").c_str(), c);
    EigenSup::read_binary_sparse((output_dir+"/l.bin").c_str(), l);
  } else {
    int iter_time = 1;
    std::shared_ptr<preconditioner> amg;
    amg = std::make_shared<amg_precon>(1, 1);
    precond_cg_solver pcg(amg);
    pcg.set_tol(1e-9);
    Eigen::MatrixXd inter_res = x;
    while (iter_time <= 5) {      
      int fitting_iter = 1;
      std::cout<<iter_time<<"th global energy val = "<<eval_total_energy(p_x, c, h, m, s, r, l, alpha, w_b, w_c, mu_fit, mu_smooth)<<std::endl;
      while (fitting_iter < 2) {
        const clock_t begin_time = std::clock();
        
        h_prev = h;
        
        #pragma omp parallel for
        for (int j = 0; j < m.outerSize(); ++j) {
          Eigen::MatrixXd S = Eigen::MatrixXd::Zero(d, d);
          for (Eigen::SparseMatrix<double>::InnerIterator it(m, j); it; ++it) {
            int nei = it.row();
            double mij = it.value();
            Eigen::VectorXd pj = p_x.row(nei);
            S += w_a * alpha[j] * mij * (pj * pj.transpose());
          }
          S += w_b * alpha[j] * Eigen::MatrixXd::Identity(d, d);
          Eigen::VectorXd b = c.row(j) * w_b * alpha[j];
          h.row(j) = trustregprob(S, b);
        }

        // update m
        
        #pragma omp parallel for
        for (int k=0; k<m.outerSize(); ++k) {
          for (int iter = m.outerIndexPtr()[k]; iter < m.outerIndexPtr()[k+1]; ++iter) {
            int nei = m.innerIndexPtr()[iter];
            Eigen::VectorXd xj = p_x.row(nei);
            m.valuePtr()[iter] = pow(mu_fit /(mu_fit +
                                              pow(h.row(k).dot(xj), 2)), 2);
          }
        }

        // check if h converges
        if ((h-h_prev).norm() < 1e-8 * h_prev.norm()) {
          break;
        }
        std::cout<<"local fitting time : "<<float( std::clock() - begin_time ) / CLOCKS_PER_SEC <<std::endl;
        std::cout<<iter_time<<" local_iter energy = "<<eval_total_energy(p_x, c, h, m, s, r, l, alpha, w_b, w_c, mu_fit, mu_smooth)<<std::endl;

        ++fitting_iter;
      }

      // global smoothing
      int smooth_iter = 1;
      while (smooth_iter <= max_smooth_iter) {
        const clock_t begin_time_smooth = std::clock();
        s_prev = s;
        
        // update c
        const auto &B = assemble_smooth_matrix(n, s, l, r);
        g = w_b*A*(c-h)+w_c*B*c;
        g *= -1;
        if (g.norm() < 1e-6) {
          std::cout<<"gradient converge"<<std::endl;
          break;
        }

        LHS = w_b*A+w_c*B;
        pcg.factorize(LHS);


        for (int cl = 0; cl < g.cols(); ++cl) {
          c.col(cl) += pcg.solve(g.col(cl));
        }
        // SMOOTH [n, d]
        // update s
        if (need_s) {
          double s_max = -10000.;
          double s_min = 10000.;
#pragma omp parallel for
          for (int k=0; k<s.outerSize(); ++k) {
            for (int iter = s.outerIndexPtr()[k]; iter < s.outerIndexPtr()[k+1]; ++iter) {
              int nei = s.innerIndexPtr()[iter];
              s.valuePtr()[iter] = c.row(nei).dot(c.row(k)) /c.row(nei).dot(c.row(nei));
              
              if (s.valuePtr()[iter] > s_max)
                s_max = s.valuePtr()[iter];
              if (s.valuePtr() [iter]< s_min)
                s_min = s.valuePtr()[iter];
            }
          }
          std::cout<<iter_time<<"th s_max"<<smooth_iter <<" = "<<s_max<<std::endl;
          std::cout<<iter_time<<"th s_min"<<smooth_iter<<" = "<<s_min<<std::endl;
        }

        // update l
#pragma omp parallel for
        for (int k=0; k<l.outerSize(); ++k) {
          for (int iter = l.outerIndexPtr()[k]; iter < l.outerIndexPtr()[k+1]; ++iter) {
            int nei = l.innerIndexPtr()[iter];
            double sij = s.valuePtr()[iter]; //s.coeff(i, j);
            l.valuePtr()[iter] = pow(mu_smooth / (mu_smooth + (c.row(nei)-sij*c.row(k)).squaredNorm()), 2);
          }
        }
      std::cout<<"global fitting time :"<<float( std::clock() - begin_time_smooth ) / CLOCKS_PER_SEC <<std::endl;
      std::cout<<iter_time<<"th energy val = "<<eval_total_energy(p_x, c, h, m, s, r, l, alpha, w_b, w_c, mu_fit, mu_smooth)<<std::endl;
      // Eigen::MatrixXd normals = c.leftCols(3);
      // normals.rowwise().normalize();
      // std::sprintf(results, "/result_normals_iter_%i.vtk", save_cnt);
      // line2vtk((output_dir+results).c_str(), x, x+0.03*normals);
      // save_cnt++;
      ++smooth_iter;
    }
    for (int i=0; i<n; ++i) {
      inter_res.row(i) = project(x.row(i), c.row(i));
    }
    inter_rs.push_back(inter_res);

    // std::sprintf(results, "/result_lij_iter_%i.vtk", iter_time);
    // write_line_with_scalar((output_dir+results).c_str(), x, l);

    // for plotting purposes
    // m_to_visualize = l;
    // for (int k=0; k<m_to_visualize.outerSize(); ++k) {
    //   for (int iter = m_to_visualize.outerIndexPtr()[k]; iter < m_to_visualize.outerIndexPtr()[k+1]; ++iter) {
    //     int nei = m_to_visualize.innerIndexPtr()[iter];
    //     double mij = m.coeffRef(k, nei);
    //     double mji = m.coeffRef(nei, k);
    //     m_to_visualize.valuePtr()[iter] = std::min(mij, mji);
    //   }
    // }
    // std::sprintf(results, "/result_mij_iter_%i.vtk", iter_time);
    // write_line_with_scalar((output_dir+results).c_str(), x, m_to_visualize);
    
    ++iter_time;
    
  }
}

  // for post-processing
  // thresholding: store "outliers" for later use
  Eigen::VectorXd edge_cnt;
  edge_cnt.setZero(n);
  // points on the edges.
  for (int k=0; k<l.outerSize(); ++k) {
    for (int iter = l.outerIndexPtr()[k]; iter < l.outerIndexPtr()[k+1]; ++iter) {
      int nei = l.innerIndexPtr()[iter];
      double lij = l.valuePtr()[iter];
      if (lij <= 0.5) {
        edge_cnt(k) += 1;
        edge_cnt(nei) += 1;
      }
    }
  }

  // outlier points.
  // for (int k=0; k<m_to_visualize.outerSize(); ++k) {
  //   for (int iter = m_to_visualize.outerIndexPtr()[k]; iter < m_to_visualize.outerIndexPtr()[k+1]; ++iter) {
  //     int nei = m_to_visualize.innerIndexPtr()[iter];
  //     double mij = m_to_visualize.valuePtr()[iter];
  //     if (mij <= 0.5) {
  //       edge_cnt(k) += 1;
  //       edge_cnt(nei) += 1;
  //     }
  //   }
  // }

  // edge: for plotting purpose; edge_v: for post-processing

  Eigen::MatrixXd edge;
  edge.setZero(x.rows(), x.cols());
  int cnt = 0;
  for (int i=0; i<n; ++i) {
    if (edge_cnt(i) >= int(lp_threshold * k_neighbor)){
      edge.row(cnt) = x.row(i);
      edge_v.push_back(i);
      cnt += 1;
    }
  } 
}




void compute_mesh_dirichlet(const std::string filename) {
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F2;
  Eigen::MatrixXd N2;
  igl::readOFF(filename, V2, F2);
  igl::per_vertex_normals(V2,F2,N2);

  Eigen::SparseMatrix<double> G; //3#F*#N
  igl::grad(V2, F2, G);

  Eigen::VectorXd area;
  igl::doublearea(V2, F2, area);
  area = area.array()/2;
  Eigen::VectorXd interm_area = area.replicate(3, 1);
  Eigen::SparseMatrix<double> AM;
  AM = interm_area.asDiagonal();
  //interm_area.cols(), interm_area.cols());
  // std::vector<T> tripletListAM;
  // for (size_t i = 0; i < interm_area.cols(); ++i) {
  //   tripletListAM.push_back(T(i, i, interm_area(i)));
  // }
  // AM.setFromTriplets(tripletListAM.begin(), tripletListAM.end());
  double E_comp = 0.;
  double gn = 0;
  for (int i = 0; i < N2.cols(); ++i) {
    Eigen::MatrixXd GradN = G*N2.col(i);
    Eigen::MatrixXd TT = GradN.transpose() * AM * GradN;

    E_comp += TT.norm();
  }

  std::cout<<"mesh gradient E ="<<E_comp<<std::endl;

  Eigen::MatrixXd GN = G*N2;
  std::cout << (GN.transpose()*AM*GN).trace() << std::endl;
}

