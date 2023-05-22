include_directories(${PROJECT_SOURCE_DIR}/src)

# ours
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
  lineproc)

# # bilateral
add_executable(bilateral bilateral.cpp)
target_link_libraries(bilateral
  lineproc)

# surface reconstruction for apss and rimls
add_executable(recon surface_recon.cpp)
target_link_libraries(recon
  lineproc)

