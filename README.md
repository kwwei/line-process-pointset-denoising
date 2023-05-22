
# Robust Pointset Denoising via Line Processes

This is a reference implementation of our Eurographics'23 paper [Robust Pointset Denoising of Piecewise-Smooth Surfaces through Line Processes](https://jiongchen.github.io/files/lineproc-paper.pdf). 

### Builiding

The project was developed and tested on Ubuntu 20.04 with gcc 9.4.0. 

Install dependencies:
```
apt install libboost-all-dev libgmp-dev libmpfr-dev 
```

Compile: 
```
mkdir build & cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

In directory `scripts`, we provide 3 example bash scripts to reproduce our results in the paper.

```
# assuming you are under ./build
cd ../scripts
bash armadillo_comparison_with_bilateral.sh 
# or
bash cad_models_stress_test.sh 
# or
bash cad_post_processing_demo.sh
```

### Contact information
Jiayi Wei: jiayi.wei@polytechnique.edu
