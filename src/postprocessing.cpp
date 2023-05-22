#include "postprocessing.h"
#include "utils.h"
#include "IO.h"


// two points on the intersection line
std::pair<Eigen::VectorXd, Eigen::VectorXd> compute_intersection_line(const Eigen::VectorXd& pl1, const Eigen::VectorXd& pl2) {
  std::pair<Eigen::VectorXd, Eigen::VectorXd> pairs;
  Eigen::Vector3d plane_n1 = pl1.head(3);
  Eigen::Vector3d plane_n2 = pl2.head(3);
  
  Eigen::Vector3d line_n = plane_n1.cross(plane_n2);
  if (line_n.norm()<1e-10) {
    // two planes are parallel
    pairs.first = Eigen::VectorXd::Zero(3);
    pairs.second = Eigen::VectorXd::Zero(3);
    return pairs;
  }
  
  Eigen::MatrixXd A(2, 3);
  A.row(0) = plane_n1.transpose();
  A.row(1) = plane_n2.transpose();
  Eigen::VectorXd b(2);
  b[0] = -pl1[3]+plane_n1.dot(line_n);
  b[1] = -pl2[3]+plane_n2.dot(line_n);

  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

  return std::make_pair(x-line_n, x+line_n);
}


Eigen::VectorXd project(const Eigen::VectorXd& pt, const Eigen::VectorXd& normal) {
  double x = pt(0);
  double y = pt(1);
  double z = pt(2);
  double c0 = (normal(0) * x + normal(1) * y + normal(2) * z + normal(3));
  Eigen::VectorXd n = normal.head(3);
  double d = c0 / pow(n.norm(), 2);
  Eigen::VectorXd pt_prime = pt - d * n;
  return pt_prime;
}


Eigen::VectorXd project_onto_line(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::VectorXd& pt) {
  Eigen::Vector3d p = pt;
  Eigen::Vector3d ap = p - a;
  Eigen::Vector3d ab = b - a;
  Eigen::VectorXd result = a + ap.dot(ab) / ab.dot(ab) * ab;
  return result;
  
}


void convert_db_pts(const std::vector<Eigen::VectorXd>& pts, std::vector<dbPoint>& db_pts) {
    int num_points = pts.size();
    dbPoint p;
    p.clusterID = UNCLASSIFIED;
    for (auto pt : pts) {
        p.x = pt(0);
        p.y = pt(1);
        p.z = pt(2);
        db_pts.push_back(p);
    }
}

double compute_Manhattan_dist(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2) {
  double dist = 0.0;
  for (int i=0; i<p1.rows(); ++i) {
    dist += abs(p1[i] - p2[i]);
  }
  return dist;
}


Eigen::MatrixXd post_processing(Eigen::MatrixXd& x, const double bbox_diag_length, const Eigen::RowVector3d mean_x, const kd_tree& tree, const Eigen::MatrixXd& c, const std::vector<int>& edge_v, const int post_process_k, std::string output_dir, const bool need_post_process) {

  if (need_post_process) {
    
    Eigen::MatrixXd x_updated = x;
    for (int i=0; i<x.rows(); ++i) {
        x_updated.row(i) = project(x.row(i), c.row(i));
    }
    
    std::vector<Eigen::VectorXd> new_pos(x.rows());
    std::cout<<"going through post processing ..."<<std::endl;
    std::vector<double> neighb_dist(x.rows(), 0.0);
    nanoflann::SearchParams params;
    for (int i=0; i<x.rows(); ++i) {
      double sum_dist = 0.0;
      std::vector<size_t> ret_indexes(2);
      std::vector<double> out_dists_sqr(2);
      nanoflann::KNNResultSet<double> resultSet(2);
      resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
      std::vector<double> query_pt(3);
      for (int j=0; j<3; j++) query_pt[j] = x(i, j);
      tree.index->findNeighbors(resultSet, &query_pt[0], params);
      for (int t=0; t<2; ++t) {
        sum_dist += (x.row(i) - x.row(ret_indexes[t])).norm();
      }
      neighb_dist[i] = sum_dist / 2;
    }

    
    Eigen::MatrixXd cluster_color;
    cluster_color.setZero(edge_v.size(), 4);
    Eigen::MatrixXd edge_x, interpl1_normal, interpl2_normal;
    edge_x.setZero(edge_v.size(), 3);
    interpl1_normal.setZero(edge_v.size(), 3);
    interpl2_normal.setZero(edge_v.size(), 3);
  
    int cnt_for_clrs = 0;
    std::set<int> e_v(edge_v.begin(), edge_v.end());
    Eigen::SparseMatrix<double> N;
    N = adaptive_find_neighbors_k(x, edge_v, tree, post_process_k);
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> plane_pairs;
    plane_pairs.clear();
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> plane_pairs_backup;
    plane_pairs_backup.clear();
    for (int k=0; k<N.outerSize(); ++k) {
      if (e_v.find(k) == e_v.end())
        continue;
      // cnt: count neighbor size of k-th vertex
      int cnt = 0;
      std::vector<Eigen::VectorXd> test_cluster;
      test_cluster.clear();
      std::vector<int> neighbor_index;
      neighbor_index.clear();
      // for k-th vertex


      for (int iter = N.outerIndexPtr()[k]; iter<N.outerIndexPtr()[k+1]; ++iter) {
        int nei = N.innerIndexPtr()[iter];
        // ignore the outlier points
        if (e_v.find(nei)!=e_v.end())
          continue;
        neighbor_index.push_back(nei);

        // consider the normal orientation
        Eigen::VectorXd oriented_n(4);
        oriented_n = c.row(nei);
        // oriented_n.normalize();
        test_cluster.push_back(oriented_n);
        cnt += 1;
      }
      std::pair<Eigen::VectorXd, Eigen::VectorXd> pair_plane;
      int dim_num = c.cols();
      int point_num = cnt;

      // if there's no "inlier neighbors"
      if (point_num == 0) {
        pair_plane.first = Eigen::VectorXd::Zero(3);
        pair_plane.second = Eigen::VectorXd::Zero(3);
        plane_pairs.push_back(pair_plane);
        continue;
      }
      // clustering - DBSCAN
      std::vector<dbPoint> db_pts;
      convert_db_pts(test_cluster, db_pts);
      double epsilon = 1.0 / sqrt(x.rows());
      DBSCAN ds(2, epsilon, db_pts);
      ds.run();

      // assign clusters
      std::vector<int> num_pts_in_cluster(5, 0);
      std::vector<std::vector<int>> indices(3);
      indices.clear();
      for (int i=0; i<cnt; ++i) {
        int id = ds.m_points[i].clusterID;
        if (id == -1) {
          num_pts_in_cluster[4] += 1;
        }
        else if (id < 4) {
          num_pts_in_cluster[id-1] += 1;
          indices[id-1].push_back(neighbor_index[i]);
        }
        else { 
          num_pts_in_cluster[3] += 1;
        }
      }
      Eigen::MatrixXd c_cl1;
      Eigen::MatrixXd c_cl2;
      Eigen::MatrixXd c_cl3;
      c_cl1.setZero(num_pts_in_cluster[0], 4);
      c_cl2.setZero(num_pts_in_cluster[1], 4);
      c_cl3.setZero(num_pts_in_cluster[2], 4);

      Eigen::MatrixXd x_cl1;
      Eigen::MatrixXd x_cl2;
      Eigen::MatrixXd x_cl3;
      x_cl1.setZero(num_pts_in_cluster[0], 3);
      x_cl2.setZero(num_pts_in_cluster[1], 3);
      x_cl3.setZero(num_pts_in_cluster[2], 3);


      if (num_pts_in_cluster[2] == 0)
        cluster_color.row(cnt_for_clrs) << x.row(k)(0), x.row(k)(1), x.row(k)(2), 1;
      else {
        cluster_color.row(cnt_for_clrs) << x.row(k)(0), x.row(k)(1), x.row(k)(2), 2;
      }
      c_cl1 = c(indices[0], Eigen::all);
      c_cl2 = c(indices[1], Eigen::all);
      c_cl3 = c(indices[2], Eigen::all);

      x_cl1 = x(indices[0], Eigen::all);
      x_cl2 = x(indices[1], Eigen::all);
      x_cl3 = x(indices[2], Eigen::all);


      if ((num_pts_in_cluster[0]+num_pts_in_cluster[1]) == 0 || (num_pts_in_cluster[0]+num_pts_in_cluster[2]) == 0 || (num_pts_in_cluster[1]+num_pts_in_cluster[2]) == 0) {
        pair_plane.first = Eigen::VectorXd::Zero(3);
        pair_plane.second = Eigen::VectorXd::Zero(3);
        plane_pairs.push_back(pair_plane);
        continue;
      }

      
      Eigen::VectorXd inter_pl1, inter_pl2;
      inter_pl1 = c_cl1.colwise().mean();
      inter_pl2 = c_cl2.colwise().mean();
      
      // // filters: if 3 clusters,
      //             1. normal filter (in case fake cluster) 2. base on manhattan distance
      
      if (num_pts_in_cluster[2] > 2) {
        Eigen::VectorXd c_centroid_1 = c_cl1.colwise().mean();
        Eigen::VectorXd c_centroid_2 = c_cl2.colwise().mean();
        Eigen::VectorXd c_centroid_3 = c_cl3.colwise().mean();
        double n_12 = (c_centroid_1-c_centroid_2).norm();
        double n_13 = (c_centroid_1-c_centroid_3).norm();
        double n_23 = (c_centroid_2-c_centroid_3).norm();
        bool nn_12 = false;
        bool nn_13 = false;
        bool nn_23 = false;

        std::vector<double> n_comp;
        n_comp.push_back(n_12);
        n_comp.push_back(n_13);
        n_comp.push_back(n_23);
        std::sort(n_comp.begin(), n_comp.end());

        if (n_comp[1] < 2 * (n_comp[2] - n_comp[0])){
          if (n_comp[0] == n_12) nn_12 = true;
          else if (n_comp[0] == n_13) nn_13 = true;
          else if (n_comp[0] == n_23) nn_23 = true;
        }
        if (nn_12 || nn_13 || nn_23) {

          if (nn_12) {
            inter_pl1 = c_cl1.colwise().mean();
            inter_pl2 = c_cl3.colwise().mean();
          }
          else if (nn_13) {
            inter_pl1 = c_cl2.colwise().mean();
            inter_pl2 = c_cl3.colwise().mean();
          }
          else {
            inter_pl1 = c_cl1.colwise().mean();
            inter_pl2 = c_cl2.colwise().mean();
          }
        }
        else {

          Eigen::VectorXd x_centroid_1 = x_cl1.colwise().mean();
          Eigen::VectorXd x_centroid_2 = x_cl2.colwise().mean();
          Eigen::VectorXd x_centroid_3 = x_cl3.colwise().mean();
          double m1 = compute_Manhattan_dist(x_centroid_1, x.row(k));
          double m2 = compute_Manhattan_dist(x_centroid_2, x.row(k));
          double m3 = compute_Manhattan_dist(x_centroid_3, x.row(k));
          if (m1 >= m2 && m1 >= m3) {
            inter_pl1 = c_cl2.colwise().mean();
            inter_pl2 = c_cl3.colwise().mean();
          }
          else if (m2 >= m1 && m2 >= m3) {
            inter_pl1 = c_cl1.colwise().mean();
            inter_pl2 = c_cl3.colwise().mean();
          }
          else {
            inter_pl1 = c_cl1.colwise().mean();
            inter_pl2 = c_cl2.colwise().mean();
          }
        }
      }

      
      pair_plane = compute_intersection_line(inter_pl1, inter_pl2);
      plane_pairs.push_back(pair_plane);
      Eigen::VectorXd n1 = inter_pl1.head(3);
      interpl1_normal.row(cnt_for_clrs) = n1.normalized();
      Eigen::VectorXd n2 = inter_pl2.head(3);
      interpl2_normal.row(cnt_for_clrs) = n2.normalized();
      edge_x.row(cnt_for_clrs) = x.row(k);
      cnt_for_clrs += 1;
    }

    int cntcnt = 0;

    for (int i = 0; i < x.rows(); ++i) {
      // x.row(i) = new_pos[i];
      Eigen::VectorXd xi_new;
      if (e_v.find(i)==e_v.end())
      {
        x.row(i) =  project(x.row(i), c.row(i));
      }
      else {
        if (plane_pairs[cntcnt].first.norm()==0 || plane_pairs[cntcnt].second.norm()==0) {
          x.row(i) = project(x.row(i), c.row(i));
        }
        xi_new = project_onto_line(plane_pairs[cntcnt].first, plane_pairs[cntcnt].second, x.row(i));
        xi_new = (xi_new.array().isFinite()).select(xi_new, 0);
        if ((xi_new-x.row(i)).norm() < (neighb_dist[i])) {
          x.row(i) = xi_new;
        }
        else {
          x.row(i) = project(x.row(i), c.row(i));
        }
        cntcnt += 1;
      }
    }
    return x_updated;
  }

  else {
    std::cout<<"doesn't need post processing"<<std::endl;
    
    Eigen::MatrixXd x_updated = x;
    for (int i=0; i<x.rows(); ++i) {
        x_updated.row(i) = project(x.row(i), c.row(i));
      }
    

    // remove outliers
    // for (int i=0; i<x.rows(); ++i) {
    //     x.row(i) = project(x.row(i), c.row(i));
    //   }
    // std::vector<int> pts_to_keep;
    // Eigen::MatrixXd pts_with_colors = x;
    // pts_to_keep.clear();
    // for (int i=0; i<x.rows(); ++i) {
    //   Eigen::VectorXd xi = x.row(i);
    //   if (std::find(edge_v.begin(), edge_v.end(), i) != edge_v.end()) {
    //     continue;
    //     pts_with_colors.row(i) << xi(0), xi(1), xi(2), 1.;
    //   }
    //   pts_with_colors.row(i) << xi(0), xi(1), xi(2), 0.;
    //   pts_to_keep.push_back(i);
    // }
    // point2vtk("result.vtk", pts_with_colors);
    // Eigen::MatrixXd x_updated = x(pts_to_keep, Eigen::all);
    
    return x_updated;
  }
}
