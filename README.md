# BoxBlur

## Build:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
make -j
```

## Usage:

```
./Release/benchmark --input INPUT --output OUTPUT --ker KER --mode1 MODE1 --mode2 MODE2
```
INPUT - input image (ppm only)
OUTPUT - output image (ppm only)
ker - integer window size
mode1 - first mode to launch (Possible MODE values)
mode2 - second mode to launch (Possible MODE values)

It will launch in MODE1 and then MODE2 to compare time.

Possible MODE values:
* cpu_conv - CPU convolution without optimisation.
* cpu_sep - CPU separable convolution.
* cpu_acc - CPU convolution with accumulation optimization.
* cuda_conv - CUDA convolution.
* cuda_sep - CUDA separable convolution

Example:

```
./Release/benchmark --input ../images/big.PPM --output res.png --ker 20 --mode1 cpu_conv --mode2 cuda_conv
```

Also `./Release/boxblur_cpu` and `./Release/boxblur_cuda` are compiled, you can use them to test particular modes.
