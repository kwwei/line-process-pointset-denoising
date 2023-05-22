#/bin/bash

#############################
#     Demo for Fig.8        #
#############################

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)/2" | bc`
echo "threads=${num_threads}"
export OMP_NUM_THREADS=${num_threads}

prog=../build/examples/Denoising
output_directory=../result/

w_a=1
w_b=5000
mu_fit=5e-9
mu_smooth=30.0
max_smooth_iter=10
needs_post_process=1
smooth_comparison=NULL
lp_threshold=0.3
reconstruction_method=2
need_s=1

file=../data/cad/small_noise/anchor_noise_0.03.xyz
w_c=1 # to be tuned
k_neighbor=20 # to be tuned

input_directory=$(dirname "${file}")
object_name=$(basename "${file}" .xyz)
mkdir -p ${output_directory}/${object_name}

${prog} ${input_directory} ${output_directory} ${object_name} ${w_a} ${w_b} ${w_c} ${mu_fit} ${mu_smooth} ${lp_threshold} ${k_neighbor}  ${max_smooth_iter} ${needs_post_process} ${smooth_comparison} ${reconstruction_method} ${need_s} 2>&1 | tee ${output_directory}/${object_name}/log-k-${k_neighbor}-smooth-${w_a}-${w_b}-${w_c}-musmooth-${mu_smooth}-postprocess-${needs_post_process}.txt
