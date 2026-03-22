To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

For the GPU version you need to have a compatible NVIDIA GPU, the instructions are in the CuPy documentation: https://docs.cupy.dev/en/stable/install.html

fbmutil.py: Direct translation of the algorithm used in the original MATLAB code.
fbmfast.py and fbmfast_gpu.py: Implementations of the Davies-Harte method for fBm (instead of the Cholesky method used in fbmutil.py). The GPU version uses CuPy for acceleration.