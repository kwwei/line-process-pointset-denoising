#/bin/bash

######################################################
#    Demo for Fig.14 -- check Table 2 for parameters #
#    used for different models.                      #
######################################################

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)/2" | bc`
echo "threads=${num_threads}"
export OMP_NUM_THREADS=${num_threads}

output_directory=../result/
prog=../build/examples

w_a=1
w_b=5000
mu_fit=5e-9 
mu_smooth=30.0
max_smooth_iter=5
needs_post_process=0
smooth_comparison=NULL
lp_threshold=0.3
reconstruction_method=2
need_s=1

# ANCHOR
file=../data/cad/stress_test/anchor_noise_0.08.xyz
w_c=1  # lambda in our paper, to be tuned
k_neighbor=50 # to be tuned

# CAD2
file=../data/cad/stress_test/cad2_noise_0.03.xyz
w_c=2  # lambda in our paper, to be tuned
k_neighbor=150 # to be tuned

# ================================================
input_directory=$(dirname "${file}")
object_name=$(basename "${file}" .xyz)
mkdir -p ${output_directory}/${object_name}
echo "${input_directory}"
echo "${object_name}"

# =================== Ours =======================
${prog}/Denoising ${input_directory} ${output_directory} ${object_name} ${w_a} ${w_b} ${w_c} ${mu_fit} ${mu_smooth} ${lp_threshold} ${k_neighbor}  ${max_smooth_iter} ${needs_post_process} ${smooth_comparison} ${reconstruction_method} ${need_s} 2>&1 | tee ${output_directory}/${object_name}/log-k-${k_neighbor}-smooth-${w_a}-${w_b}-${w_c}-musmooth-${mu_smooth}-postprocess-${needs_post_process}.txt

# ================== bilateral filtering (requires pymeshlab) =====
echo "${file}"
python3 compute_normals.py ${file} 0 0.5    
k=80
angle=60
iter=3
bilateral_xyz=${output_directory}/${object_name}/${object_name}_bilateral.xyz
bilateral_off=${bilateral_xyz/.xyz/.off}
save_iter=0

echo "bilateral filtering ..."
${prog}/bilateral ${file/.xyz/_with_normal.xyz} ${bilateral_xyz} ${bilateral_off} ${k} ${angle} ${iter}

# ============ rimls & apss (requires meshlabserver) ===========
rimls_param=8.0
apss_param=8.0
rimls_output=${output_directory}/${object_name}/${object_name}_rimls_${rimls_param}.xyz
apss_output=${output_directory}/${object_name}/${object_name}_apss_${apss_param}.xyz

echo ${rimls_output}
xmlstarlet ed -u '/FilterScript/filter/Param[@name="FilterScale"]/@value' -v ${rimls_param} script_for_rimls.xml > inter.xml
mv inter.xml script_for_rimls.xml
meshlabserver -i ${file/.xyz/_with_normal.xyz} -o ${rimls_output} -s script_for_rimls.xml

echo ${apss_output}
xmlstarlet ed -u '/FilterScript/filter/Param[@name="FilterScale"]/@value' -v ${apss_param} script_for_apss.xml > inter.xml
mv inter.xml script_for_apss.xml
meshlabserver -i ${file/.xyz/_with_normal.xyz} -o ${apss_output} -s script_for_apss.xml

# reconstruct surface
${prog}/recon ${rimls_output} ${rimls_output/.xyz/.off}
${prog}/recon ${apss_output} ${apss_output/.xyz/.off}
