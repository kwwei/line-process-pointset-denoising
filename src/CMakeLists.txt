add_library(lineproc SHARED
  dbscan.cpp denoising.cpp
  kmeans.cpp neighbors.cpp
  pcg.cc postprocessing.cpp
  utils.cpp IO.cpp)
target_link_libraries(lineproc igl::core ${Boost_LIBRARIES} ${GMP_LIBRARIES}
  ${MPFR_LIBRARIES})
