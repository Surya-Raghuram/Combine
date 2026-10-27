// CBS_GPU_gen.cu  -- dataset generator (Convergent Born Series, 2D, cuFFT)
//
// Build : nvcc -O2 -std=c++17 CBS_GPU_gen.cu -o cbs_gen -lcufft
// Run   : ./cbs_gen <out_dir> <n_samples> <seed> <start_id>
//         e.g. ./cbs_gen cbs_data 1000 0 0
//         Re-running with a larger <start_id> resumes; each sample is
//         seeded from (seed, id), so results are reproducible.
//
// Output per sample : <out_dir>/sample_XXXXX.bin
//     float32, planar, shape [4, 256, 256] = {Re SHI, Im SHI, Re S, Im S}
// Plus              : <out_dir>/metadata.csv  (one row per sample)
//
// Differences from the original CBS_GPU.cu (physics kernels unchanged):
//   * single loop over samples (old code had a redundant outer radius loop and
//     an inner loop that shadowed `radius_um`)
//   * frequency index and sound speed vs are sampled per sample
//   * source centre is sampled so the whole disk lies inside crop [64,192)
//   * one cuFFT plan, created once and destroyed (old code leaked one per sample)
//   * uninitialised ITEmax removed; non-converged samples are flagged
//   * binary output instead of 65536-line text files
//   * no dependency on helper_cuda.h

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <chrono>
#include <filesystem>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
#define CUFFT_CHECK(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1);} } while (0)

// ------------------------------------------------------------------ settings
constexpr int    NN       = 256;          // grid size
constexpr int    GRID_C   = NN / 2;
constexpr int    PML      = 62;           // absorbing layer thickness (px)
constexpr int    PMLM     = 31;
constexpr int    CROP_LO  = 64;           // loader crops [64,192) -> 128x128
constexpr int    CROP_HI  = 192;
constexpr double DX       = 100.0e-9;     // 100 nm
constexpr double MIN_RADIUS_UM = 2.0;     // 20 px
constexpr double MAX_RADIUS_UM = 5.0;     // 50 px
constexpr double VF       = 1500.0;
// Frequency indices (wavelength = NN/idx pixels). Edit freely.
static const int FREQ_LIST[] = {8, 10, 12, 14};
constexpr int    N_FREQ   = sizeof(FREQ_LIST) / sizeof(FREQ_LIST[0]);
// vs must keep |ks^2-kf^2| <= epsilon = 0.8 kf^2  (CBS convergence),
// i.e. vs roughly in [1200, 1800]. Two bands, so contrast is never ~0.
constexpr double VS_LO_A = 1200.0, VS_HI_A = 1400.0;
constexpr double VS_LO_B = 1600.0, VS_HI_B = 1800.0;
constexpr double ERR_TOL = 1e-4;
constexpr int    MAX_ITERS = 2000;
constexpr double PI      = 3.141592653589793;

// ------------------------------------------------------------------- kernels
__global__ void init_SVGG(cuDoubleComplex* S, cuDoubleComplex* V, cuDoubleComplex* gamma,
                          cuDoubleComplex* G, int N, int cx, int cy, int gc, double dx, double a,
                          cuDoubleComplex S_in, cuDoubleComplex S_out,
                          cuDoubleComplex V_in, cuDoubleComplex V_out, double eps, double kf)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N) return;
  int idx = row * N + col;

  double dist = sqrt((double)((row - cy) * (row - cy) + (col - cx) * (col - cx))) * dx;
  if (dist <= a) { S[idx] = S_in;  V[idx] = V_in;  }
  else           { S[idx] = S_out; V[idx] = V_out; }

  gamma[idx].x = -V[idx].y / eps;
  gamma[idx].y =  V[idx].x / eps;

  double ky = 2.0 * PI * (row - gc) / (N * dx);
  double kx = 2.0 * PI * (col - gc) / (N * dx);
  double k2 = kx * kx + ky * ky - kf * kf;
  double den = k2 * k2 + eps * eps;
  G[idx].x = k2 / den;
  G[idx].y = eps / den;
}

__global__ void fftshift_g(cuDoubleComplex* M, int N1, int N2, int gc)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N1 || j >= N2) return;
  if (i < gc && j < gc) {
    long a = (long)i * N2 + j, b = (long)(i + gc) * N2 + (j + gc);
    cuDoubleComplex t = M[b]; M[b] = M[a]; M[a] = t;
  }
  if (i >= gc && j < gc) {
    long a = (long)i * N2 + j, b = (long)(i - gc) * N2 + (j + gc);
    cuDoubleComplex t = M[b]; M[b] = M[a]; M[a] = t;
  }
}

// Equivalent rewrite of absorbinglayer_g_step1/step2: smooth sigmoid ramp in the
// PML bands, min() in the corners, 1 in the interior.
__device__ double ramp(double alpha, double xb) { return 1.0 / (1.0 + exp(-alpha * xb)); }

__global__ void make_abl(double* mask, int N, int pml, int pmlm, double alpha, double dx)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || j >= N) return;
  double b = 1.0, c = 1.0;
  if (i < pml)           b = ramp(alpha, dx * (i - pmlm));
  else if (i >= N - pml) b = ramp(alpha, dx * (N - pmlm - i));
  if (j < pml)           c = ramp(alpha, dx * (j - pmlm));
  else if (j >= N - pml) c = ramp(alpha, dx * (N - pmlm - j));
  mask[i * N + j] = (b < c) ? b : c;
}

__global__ void multiply_g(const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = cuCmul(A[i], B[i]);
}

__global__ void div_real_g(const cuDoubleComplex* A, double d, cuDoubleComplex* C, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) { C[i].x = A[i].x / d; C[i].y = A[i].y / d; }
}

// D = A*B + C
__global__ void multiply_sum_g(const cuDoubleComplex* A, const cuDoubleComplex* B,
                               const cuDoubleComplex* C, cuDoubleComplex* D, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) D[i] = cuCadd(cuCmul(A[i], B[i]), C[i]);
}

// shirfn = ( shirin - (i/eps) * V * (shirin - fft_out/norm) ) * ABL
__global__ void prepare_next_shi(const cuDoubleComplex* fft_out, const cuDoubleComplex* shirin,
                                 const cuDoubleComplex* V, const double* ABL,
                                 cuDoubleComplex eps_inv, double norm, cuDoubleComplex* shirfn, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  cuDoubleComplex scaled = make_cuDoubleComplex(fft_out[i].x / norm, fft_out[i].y / norm);
  cuDoubleComplex r = cuCsub(shirin[i], cuCmul(eps_inv, cuCmul(V[i], cuCsub(shirin[i], scaled))));
  shirfn[i].x = r.x * ABL[i];
  shirfn[i].y = r.y * ABL[i];
}

// ---------------------------------------------------------------------- host
static double rel_error(const cuDoubleComplex* M, const cuDoubleComplex* Q, int n)
{
  double num = 0.0, den = 0.0;
  for (int i = 0; i < n; i++) {
    double dx_ = M[i].x - Q[i].x, dy_ = M[i].y - Q[i].y;
    num += sqrt(dx_ * dx_ + dy_ * dy_);
    den += sqrt(Q[i].x * Q[i].x + Q[i].y * Q[i].y);
  }
  return num / den;
}

int main(int argc, char** argv)
{
  std::string out_dir = (argc > 1) ? argv[1] : "cbs_data";
  int n_samples       = (argc > 2) ? atoi(argv[2]) : 1000;
  unsigned long long seed = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 0ULL;
  int start_id        = (argc > 4) ? atoi(argv[4]) : 0;

  std::filesystem::create_directories(out_dir);
  std::string meta_path = out_dir + "/metadata.csv";
  bool new_meta = !std::filesystem::exists(meta_path);
  FILE* meta = fopen(meta_path.c_str(), "a");
  if (!meta) { fprintf(stderr, "cannot open %s\n", meta_path.c_str()); return 1; }
  if (new_meta) fprintf(meta, "id,radius_um,radius_px,cx,cy,freq_idx,vs,kf,omega,epsilon,iters,err,converged\n");

  const int NPIX = NN * NN;
  const size_t cbytes = sizeof(cuDoubleComplex) * NPIX;

  cuDoubleComplex *d_a, *d_b, *V, *S, *G, *gamma, *fft_in, *fft_out;
  double* ABL;
  CUDA_CHECK(cudaMalloc(&d_a, cbytes));   CUDA_CHECK(cudaMalloc(&d_b, cbytes));
  CUDA_CHECK(cudaMalloc(&V, cbytes));     CUDA_CHECK(cudaMalloc(&S, cbytes));
  CUDA_CHECK(cudaMalloc(&G, cbytes));     CUDA_CHECK(cudaMalloc(&gamma, cbytes));
  CUDA_CHECK(cudaMalloc(&fft_in, cbytes)); CUDA_CHECK(cudaMalloc(&fft_out, cbytes));
  CUDA_CHECK(cudaMalloc(&ABL, sizeof(double) * NPIX));

  cufftHandle plan;
  CUFFT_CHECK(cufftPlan2d(&plan, NN, NN, CUFFT_Z2Z));

  dim3 threads(16, 16), grid(NN / 16, NN / 16);
  int t1 = 256, g1 = (NPIX + t1 - 1) / t1;

  std::vector<cuDoubleComplex> hF(NPIX), hS(NPIX), hRowA(NN), hRowB(NN);
  std::vector<float> outbuf(4 * (size_t)NPIX);

  auto t_begin = std::chrono::high_resolution_clock::now();

  for (int id = start_id; id < start_id + n_samples; id++) {
    std::mt19937_64 rng(seed * 1000003ULL + (unsigned long long)id);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // ---- sample parameters
    double radius_um = MIN_RADIUS_UM + U(rng) * (MAX_RADIUS_UM - MIN_RADIUS_UM);
    double a = radius_um * 1e-6;
    int r_px = (int)ceil(radius_um * 1e-6 / DX);
    int lo = CROP_LO + r_px + 2, hi = CROP_HI - 1 - r_px - 2;   // inclusive range
    int cx = lo + (int)(rng() % (unsigned long long)(hi - lo + 1));
    int cy = lo + (int)(rng() % (unsigned long long)(hi - lo + 1));
    int fidx = FREQ_LIST[rng() % N_FREQ];
    double vs = (U(rng) < 0.5) ? VS_LO_A + U(rng) * (VS_HI_A - VS_LO_A)
                               : VS_LO_B + U(rng) * (VS_HI_B - VS_LO_B);

    // ---- physics constants (as in the original)
    double mu = 1.0, beta = 1.0, Cp = 1.0, I0 = 1.0;
    double kkxy  = 2.0 * PI * fidx / (NN * DX);
    double f     = kkxy * VF / (2.0 * PI);
    double omega = 2.0 * PI * f;
    double kf    = omega / VF;
    double ks    = omega / vs;
    double eps   = 0.8 * kf * kf;
    double Scale = 2.5 * 20.0 / (double)fidx;
    double alpha = Scale * 0.5 * 0.8 * kf;

    cuDoubleComplex S_in  = make_cuDoubleComplex(0.0, -(mu * beta * I0 * omega) / Cp);
    cuDoubleComplex S_out = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex V_in  = make_cuDoubleComplex(ks * ks - kf * kf, -eps);
    cuDoubleComplex V_out = make_cuDoubleComplex(0.0, -eps);

    // ---- setup
    init_SVGG<<<grid, threads>>>(S, V, gamma, G, NN, cx, cy, GRID_C, DX, a,
                                 S_in, S_out, V_in, V_out, eps, kf);
    CUDA_CHECK(cudaMemset(d_a, 0, cbytes));
    CUDA_CHECK(cudaMemset(d_b, 0, cbytes));
    fftshift_g<<<grid, threads>>>(G, NN, NN, GRID_C);
    make_abl<<<grid, threads>>>(ABL, NN, PML, PMLM, alpha, DX);

    cuDoubleComplex* shirin = d_a;
    cuDoubleComplex* shirfn = d_b;

    // initial guess: shirin = gamma * IFFT( G * FFT(S) ) / N^2
    CUFFT_CHECK(cufftExecZ2Z(plan, S, fft_out, CUFFT_FORWARD));
    multiply_g<<<g1, t1>>>(fft_out, G, fft_in, NPIX);
    CUFFT_CHECK(cufftExecZ2Z(plan, fft_in, fft_out, CUFFT_INVERSE));
    multiply_g<<<g1, t1>>>(fft_out, gamma, shirin, NPIX);
    div_real_g<<<g1, t1>>>(shirin, (double)NPIX, shirin, NPIX);

    // ---- CBS iterations
    bool converged = false;
    int iters = 0;
    double err = 1e30;
    for (int it = 0; it < MAX_ITERS; it++) {
      multiply_sum_g<<<g1, t1>>>(shirin, V, S, fft_in, NPIX);
      CUFFT_CHECK(cufftExecZ2Z(plan, fft_in, fft_out, CUFFT_FORWARD));
      multiply_g<<<g1, t1>>>(fft_out, G, fft_in, NPIX);
      CUFFT_CHECK(cufftExecZ2Z(plan, fft_in, fft_out, CUFFT_INVERSE));
      prepare_next_shi<<<g1, t1>>>(fft_out, shirin, V, ABL, make_cuDoubleComplex(0.0, 1.0 / eps),
                                   (double)NPIX, shirfn, NPIX);

      CUDA_CHECK(cudaMemcpy(hRowA.data(), shirfn + GRID_C * NN, sizeof(cuDoubleComplex) * NN, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hRowB.data(), shirin + GRID_C * NN, sizeof(cuDoubleComplex) * NN, cudaMemcpyDeviceToHost));
      err = rel_error(hRowA.data(), hRowB.data(), NN);
      iters = it + 1;
      if (!std::isfinite(err)) break;
      if (err <= ERR_TOL) { converged = true; break; }
      std::swap(shirin, shirfn);
    }
    CUDA_CHECK(cudaGetLastError());

    // latest iterate: shirfn if converged, otherwise it was swapped into shirin
    cuDoubleComplex* result = converged ? shirfn : shirin;
    CUDA_CHECK(cudaMemcpy(hF.data(), result, cbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hS.data(), S, cbytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NPIX; i++) {
      outbuf[0 * (size_t)NPIX + i] = (float)hF[i].x;
      outbuf[1 * (size_t)NPIX + i] = (float)hF[i].y;
      outbuf[2 * (size_t)NPIX + i] = (float)hS[i].x;
      outbuf[3 * (size_t)NPIX + i] = (float)hS[i].y;
    }
    char path[512];
    snprintf(path, sizeof(path), "%s/sample_%05d.bin", out_dir.c_str(), id);
    FILE* fo = fopen(path, "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    fwrite(outbuf.data(), sizeof(float), outbuf.size(), fo);
    fclose(fo);

    fprintf(meta, "%d,%.4f,%d,%d,%d,%d,%.2f,%.6e,%.6e,%.6e,%d,%.3e,%d\n",
            id, radius_um, r_px, cx, cy, fidx, vs, kf, omega, eps, iters, err, converged ? 1 : 0);
    fflush(meta);

    printf("[%d/%d] r=%.2fum c=(%d,%d) f_idx=%d vs=%.0f iters=%d err=%.2e %s\n",
           id - start_id + 1, n_samples, radius_um, cx, cy, fidx, vs, iters, err,
           converged ? "ok" : "NOT CONVERGED");
  }

  fclose(meta);
  cufftDestroy(plan);
  cudaFree(d_a); cudaFree(d_b); cudaFree(V); cudaFree(S); cudaFree(G);
  cudaFree(gamma); cudaFree(fft_in); cudaFree(fft_out); cudaFree(ABL);

  auto t_end = std::chrono::high_resolution_clock::now();
  printf("Total time: %lld s\n",
         (long long)std::chrono::duration_cast<std::chrono::seconds>(t_end - t_begin).count());
  return 0;
}
