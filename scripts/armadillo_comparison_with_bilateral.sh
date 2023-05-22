#/bin/bash

#####################################
#        Demo for Fig.3             #
#####################################

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)/2" | bc`
echo "threads=${num_threads}"
export OMP_NUM_THREADS=${num_threads}

prog=../build/examples/
output_directory=../result/

w_a=1
w_b=5000
w_c=0.05
mu_fit=0.001
mu_smooth=30.0
k_neighbor=15
max_smooth_iter=2
needs_post_process=0
smooth_comparison=NULL
lp_threshold=0.3
reconstruction_method=1
need_s=1

file=../data/scan/armadillo.xyz
input_directory=$(dirname "${file}")
object_name=$(basename "${file}" .xyz)

mkdir -p ${output_directory}/${object_name}

# =============== Ours ===============================
${prog}/Denoising ${input_directory} ${output_directory} ${object_name} ${w_a} ${w_b} ${w_c} ${mu_fit} ${mu_smooth} ${lp_threshold} ${k_neighbor}  ${max_smooth_iter} ${needs_post_process} ${smooth_comparison} ${reconstruction_method} ${need_s} 2>&1 | tee ${output_directory}/${object_name}/log-k-${k_neighbor}-smooth-${w_a}-${w_b}-${w_c}-musmooth-${mu_smooth}-postprocess-${needs_post_process}.txt


# ============= bilateral ============================
# prepare normals
# python3 py_file input_file operation parameter(gaussian noise)
# operation: 0 - no operation 1 - meshlab normals + random normals 2 - random normals 3 - all one
#
python3 compute_normals.py ${file} 1 0.5
k=50
angle=60
iter=3
bilateral_xyz=${output_directory}/${object_name}/${object_name}_bilateral.xyz
bilateral_off=${bilateral_xyz/.xyz/.off}
save_iter=0

echo "bilateral filtering ..."
${prog}/bilateral ${file/.xyz/_with_normal.xyz} ${bilateral_xyz} ${bilateral_off} ${k} ${angle} ${iter}

if [[ ! -e ${bilateral_xyz/.xyz/_random_gaussian.xyz} ]]
then
    ${prog}/bilateral ${file/.xyz/_random_gaussian_normal.xyz} ${bilateral_xyz/.xyz/_random_gaussian.xyz} ${bilateral_off/.off/_random_gaussian.off} ${k} ${angle} ${iter}
fi

if [[ ! -e ${bilateral_xyz/.xyz/_one.xyz} ]]
then
    ${prog}/bilateral ${file/.xyz/_one.xyz} ${bilateral_xyz/.xyz/_one.xyz} ${bilateral_off/.off/_one.off} ${k} ${angle} ${iter}
fi

if [[ ! -e ${bilateral_xyz/.xyz/_random.xyz} ]]
then
    ${prog}/bilateral ${file/.xyz/_random.xyz} ${bilateral_xyz/.xyz/_random.xyz} ${bilateral_off/.off/_random.off} ${k} ${angle} ${iter}
fi
