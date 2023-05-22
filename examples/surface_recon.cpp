#include "IO.h"
#include <iostream>
int main(int argc, char*argv[])
{
  const char* input_xyz = argv[1];
  const char* output_off = argv[2];
  std::cout << "reconstruct surface" << std::endl;
  Eigen::MatrixXd V;
  readXYZ(input_xyz, V);
  save_surface_advance_front(output_off, V);
  return 0;
}
