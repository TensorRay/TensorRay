/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <cub/cub.cuh>
#include "NEE.cuh"


template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

__global__ void build_tree(float* p0, float* p1, const float* cdf_x, int x_num, int y_num, int z_num, int max_tree_size) {
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    int buf = x_num*y_num*z_num;
    if (tid < buf) {
        float dimy = 1.0f/y_num;
        float dimz = 1.0f/z_num;

        int i = tid % x_num;
        int j = tid / x_num % y_num;
        int k = tid / (x_num*y_num) % z_num;

        p0[tid] = (i!=0) ? cdf_x[i-1] : 0.0f;
        p0[tid+max_tree_size] = j*dimy;
        p0[tid+max_tree_size*2] = k*dimz;

        p1[tid] = cdf_x[i];
        p1[tid+max_tree_size] = (j+1.f)*dimy;
        p1[tid+max_tree_size*2] = (k+1.f)*dimz;
    }
}

__global__ void gen_eval_point( const float* p0, const float* p1, float* out, int npass) {
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    int i = bid * blockDim.x + tid;

    unsigned int seed = tea<4>( i, npass );

    float midx = (p1[bid*3]+p0[bid*3])/2.0f;
    float midy = (p1[bid*3+1]+p0[bid*3+1])/2.0f;
    float midz = (p1[bid*3+2]+p0[bid*3+2])/2.0f;

    float rndx = rnd(seed);
    float rndy = rnd(seed);
    float rndz = rnd(seed);

    switch(tid) {
        case 0:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p0[bid*3+2];
            break;
        case 1:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p1[bid*3+2];
            break;
        case 2:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p0[bid*3+2]+midz;
            break;

        case 3:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p0[bid*3+2];
            break;
        case 4:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p1[bid*3+2];
            break;
        case 5:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p0[bid*3+2]+midz;
            break;

        case 6:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p0[bid*3+1]+midy;
            out[i*3+2] = p0[bid*3+2];
            break;
        case 7:
            out[i*3] = p0[bid*3];
            out[i*3+1] = p0[bid*3+1]+midy;
            out[i*3+2] = p1[bid*3+2];
            break;

        case 8:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p0[bid*3+2];
            break;
        case 9:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p1[bid*3+2];
            break;
        case 10:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p0[bid*3+2]+midz;
            break;

        case 11:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p0[bid*3+2];
            break;
        case 12:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p1[bid*3+2];
            break;
        case 13:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p0[bid*3+2]+midz;
            break;

        case 14:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p0[bid*3+1]+midy;
            out[i*3+2] = p0[bid*3+2];
            break;
        case 15:
            out[i*3] = p1[bid*3];
            out[i*3+1] = p0[bid*3+1]+midy;
            out[i*3+2] = p1[bid*3+2];
            break;

        case 16:
            out[i*3] = p0[bid*3]+midx;
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p0[bid*3+2];
            break;
        case 17:
            out[i*3] = p0[bid*3]+midx;
            out[i*3+1] = p0[bid*3+1];
            out[i*3+2] = p1[bid*3+2];
            break;
        case 18:
            out[i*3] = p0[bid*3]+midx;
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p0[bid*3+2];
            break;
        case 19:
            out[i*3] = p0[bid*3]+midx;
            out[i*3+1] = p1[bid*3+1];
            out[i*3+2] = p1[bid*3+2];
            break;
        
        // random sample for grid
        case 20:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*rndx;
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*rndz;
            break;
        case 21:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*rndx;
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*rndz;
            break;
        case 22:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*rndx;
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*rndz;
            break;
        case 23:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*rndx;
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*rndz;
            break;

        case 24:
            out[i*3] = midx      + (midx-p0[bid*3])*rndx;
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*rndz;
            break;
        case 25:
            out[i*3] = midx      + (midx-p0[bid*3])*rndx;
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*rndz;
            break;
        case 26:
            out[i*3] = midx      + (midx-p0[bid*3])*rndx;
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*rndz;
            break;
        case 27:
            out[i*3] = midx      + (midx-p0[bid*3])*rndx;
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*rndy;
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*rndz;
            break;

        case 28:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;
        case 29:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;
        case 30:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;
        case 31:
            out[i*3] = p0[bid*3] + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;

        case 32:
            out[i*3] = midx      + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;
        case 33:
            out[i*3] = midx      + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = p0[bid*3+1] + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;
        case 34:
            out[i*3] = midx      + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = p0[bid*3+2] + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;
        case 35:
            out[i*3] = midx      + (midx-p0[bid*3])*(1.0f-rndx);
            out[i*3+1] = midy      + (midy-p0[bid*3+1])*(1.0f-rndy);
            out[i*3+2] = midz      + (midz-p0[bid*3+2])*(1.0f-rndz);
            break;
    }
}

__global__ void cut_tree(const float* eval_value, float* p0, float* p1,
                                                  float thold, float wt1, int bound_size, int* global_index) {
    unsigned int seed = tea<4>( blockIdx.x, threadIdx.x);
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid < bound_size) {
        int eid = bid*36 * 3;
        float leaf_size =   (p1[bid]-p0[bid])*
                            (p1[bid+1]-p0[bid+1])*
                            (p1[bid+2]-p0[bid+2]);

        float lag_val = (eval_value[eid] + eval_value[eid+1] + eval_value[eid+3] + eval_value[eid+4] + eval_value[eid+8] + eval_value[eid+9] + eval_value[eid+11] + eval_value[eid+12]) / 8.0f;
        float MC_val = (eval_value[eid+20] + eval_value[eid+21] + eval_value[eid+22] + eval_value[eid+23] + 
                        eval_value[eid+24] + eval_value[eid+25] + eval_value[eid+26] + eval_value[eid+27] +
                        eval_value[eid+28] + eval_value[eid+29] + eval_value[eid+30] + eval_value[eid+31] +
                        eval_value[eid+32] + eval_value[eid+33] + eval_value[eid+34] + eval_value[eid+35]) / 16.0f;
        
        float error = abs(lag_val-MC_val) * leaf_size * wt1;
        if (error> thold) {
            int current_index = atomicAdd(&global_index[0], 1) + bound_size;
            current_index *= 3;
            float dim1a  = (eval_value[eid]+eval_value[eid+1]+eval_value[eid+3]+eval_value[eid+4]+eval_value[eid+16]+eval_value[eid+17]+eval_value[eid+18]+eval_value[eid+19]) / 8.0f;
            float dim1b  = (eval_value[eid+8]+eval_value[eid+9]+eval_value[eid+11]+eval_value[eid+12]+eval_value[eid+16]+eval_value[eid+17]+eval_value[eid+18]+eval_value[eid+19]) / 8.0f;

            float dim2a  = (eval_value[eid]+eval_value[eid+1]+eval_value[eid+6]+eval_value[eid+7]+eval_value[eid+8]+eval_value[eid+9]+eval_value[eid+14]+eval_value[eid+15]) / 8.0f;
            float dim2b  = (eval_value[eid+3]+eval_value[eid+4]+eval_value[eid+6]+eval_value[eid+7]+eval_value[eid+11]+eval_value[eid+12]+eval_value[eid+14]+eval_value[eid+15]) / 8.0f;

            float dim3a  = (eval_value[eid]+eval_value[eid+2]+eval_value[eid+3]+eval_value[eid+5]+eval_value[eid+8]+eval_value[eid+10]+eval_value[eid+11]+eval_value[eid+13]) / 8.0f;
            float dim3b  = (eval_value[eid+1]+eval_value[eid+2]+eval_value[eid+4]+eval_value[eid+5]+eval_value[eid+9]+eval_value[eid+10]+eval_value[eid+12]+eval_value[eid+13]) / 8.0f;

            float MC1a  = (eval_value[eid+20]+eval_value[eid+21]+eval_value[eid+22]+eval_value[eid+23]+eval_value[eid+20+8]+eval_value[eid+21+8]+eval_value[eid+22+8]+eval_value[eid+23+8]) / 8.0f;
            float MC1b  = (eval_value[eid+24]+eval_value[eid+25]+eval_value[eid+26]+eval_value[eid+27]+eval_value[eid+24+8]+eval_value[eid+25+8]+eval_value[eid+26+8]+eval_value[eid+27+8]) / 8.0f;

            float MC2a  = (eval_value[eid+20]+eval_value[eid+21]+eval_value[eid+24]+eval_value[eid+25]+eval_value[eid+20+8]+eval_value[eid+21+8]+eval_value[eid+24+8]+eval_value[eid+25+8]) / 8.0f;
            float MC2b  = (eval_value[eid+22]+eval_value[eid+23]+eval_value[eid+26]+eval_value[eid+27]+eval_value[eid+22+8]+eval_value[eid+23+8]+eval_value[eid+26+8]+eval_value[eid+27+8]) / 8.0f;

            float MC3a  = (eval_value[eid+20]+eval_value[eid+22]+eval_value[eid+24]+eval_value[eid+26]+eval_value[eid+20+8]+eval_value[eid+22+8]+eval_value[eid+24+8]+eval_value[eid+26+8]) / 8.0f;
            float MC3b  = (eval_value[eid+21]+eval_value[eid+23]+eval_value[eid+25]+eval_value[eid+27]+eval_value[eid+21+8]+eval_value[eid+23+8]+eval_value[eid+25+8]+eval_value[eid+27+8]) / 8.0f;;

            float error0 = abs(abs(dim1a-MC1a) - abs(dim1b-MC1b));
            float error1 = abs(abs(dim2a-MC2a) - abs(dim2b-MC2b));
            float error2 = abs(abs(dim3a-MC3a) - abs(dim3b-MC3b));
            if (error0 > error1 && error0 > error2) {
                p1[current_index] = p1[bid];
                p1[current_index+1] = p1[bid+1];
                p1[current_index+2] = p1[bid+2];
                p0[current_index] = (p1[bid] + p0[bid]) / 2.0f;
                p0[current_index+1] = p0[bid+1];
                p0[current_index+2] = p0[bid+2];
                p1[bid] = (p1[bid] + p0[bid]) / 2.0f;
            } else if (error1 > error0 && error1 > error2) {
                p1[current_index] = p1[bid];
                p1[current_index+1] = p1[bid+1];
                p1[current_index+2] = p1[bid+2];
                p0[current_index+1] = (p1[bid+1] + p0[bid+1]) / 2.0f;
                p0[current_index] = p0[bid];
                p0[current_index+2] = p0[bid+2];
                p1[bid+1] = (p1[bid+1] + p0[bid+1]) / 2.0f;
            } else if (error2 > error0 && error2 > error1) {
                p1[current_index] = p1[bid];
                p1[current_index+1] = p1[bid+1];
                p1[current_index+2] = p1[bid+2];
                p0[current_index+2] = (p1[bid+2] + p0[bid+2]) / 2.0f;
                p0[current_index] = p0[bid];
                p0[current_index+1] = p0[bid+1];
                p1[bid+2] = (p1[bid+2] + p0[bid+2]) / 2.0f;
            } else {
                float rnd_num = rnd(seed) * 3.0f;
                if (rnd_num < 1.0f) {
                    p1[current_index] = p1[bid];
                    p1[current_index+1] = p1[bid+1];
                    p1[current_index+2] = p1[bid+2];
                    p0[current_index] = (p1[bid] + p0[bid]) / 2.0f;
                    p0[current_index+1] = p0[bid+1];
                    p0[current_index+2] = p0[bid+2];
                    p1[bid] = (p1[bid] + p0[bid]) / 2.0f;
                } else if (rnd_num < 2.0f) {
                    p1[current_index] = p1[bid];
                    p1[current_index+1] = p1[bid+1];
                    p1[current_index+2] = p1[bid+2];
                    p0[current_index+1] = (p1[bid+1] + p0[bid+1]) / 2.0f;
                    p0[current_index] = p0[bid];
                    p0[current_index+2] = p0[bid+2];
                    p1[bid+1] = (p1[bid+1] + p0[bid+1]) / 2.0f;
                } else {
                    p1[current_index] = p1[bid];
                    p1[current_index+1] = p1[bid+1];
                    p1[current_index+2] = p1[bid+2];
                    p0[current_index+2] = (p1[bid+2] + p0[bid+2]) / 2.0f;
                    p0[current_index] = p0[bid];
                    p0[current_index+1] = p0[bid+1];
                    p1[bid+2] = (p1[bid+2] + p0[bid+2]) / 2.0f;
                }
            }
        }
    }
}


void psdr_cuda::init_tree(const float* cdf_x, const float* cdf_y, const float* cdf_z, int dimx, int dimy, int dimz, float* p0, float* p1, int max_tree_size) {
    int thread_size = 64;
    int block_size = (dimx*dimy*dimz-1) / thread_size + 1;
    build_tree<<<block_size,thread_size>>>( p0, p1, cdf_x, dimx, dimy, dimz, max_tree_size);
}

void psdr_cuda::generate_eval_point(int leaf_size, const float* p0, const float* p1, float* out, int npass) {
    gen_eval_point<<<leaf_size, 36>>>(p0, p1, out, npass);
}

int psdr_cuda::cut_grid(const float* eval_value, float* p0, float* p1,
                                                 int fix_size, float thold, float wt1) {
    int thread_size = 512;
    int block_size = (fix_size-1) / thread_size + 1;

    int* global_index;
    cudaMalloc(&global_index, sizeof(int));
    cudaMemset(global_index, 0, sizeof(int));
    cut_tree<<<block_size, thread_size>>>(eval_value, p0,  p1,
                                            thold, wt1, fix_size, global_index);
    
    int app_size;
    cudaMemcpy(&app_size,global_index,sizeof(int),cudaMemcpyDeviceToHost);
    return app_size;
}

__global__ void get_tree_area(float* out_area, const float* p0, const float* p1, int size) {
    int bid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (bid < size) {
        out_area[bid] = (p1[bid*3]-p0[bid*3])*(p1[bid*3+1]-p0[bid*3+1])*(p1[bid*3+2]-p0[bid*3+2]);
    }
}
void psdr_cuda::get_area(float* out_area, const float* p0, const float* p1, int size) {
    int thread_size = 512;
    int block_size = (size-1) / thread_size + 1;

    get_tree_area<<<block_size, thread_size>>>(out_area, p0, p1, size);
}

__global__ void op_gather(float* result, const float* data, const int* idx, int size) {
    int bid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (bid < size) {
        result[bid*3] = data[idx[bid]*3];
        result[bid*3+1] = data[idx[bid]*3+1];
        result[bid*3+2] = data[idx[bid]*3+2];
    }
}


void psdr_cuda::gather(float* result, const float* data, const int* idx, int size) {
    int thread_size = 512;
    int block_size = (size-1) / thread_size + 1;

    op_gather<<<block_size, thread_size>>>(result, data, idx, size);
}

__global__ void op_aq_sample(float* result, const float* p0, const float* p1, const float* rnd, int size) {
    int bid = (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = bid*3;
    if (bid < size) {
        result[tid]   = p0[tid] + rnd[tid] * (p1[tid] - p0[tid]);
        result[tid+1]   = p0[tid+1] + rnd[tid+1] * (p1[tid+1] - p0[tid+1]);
        result[tid+2]   = p0[tid+2] + rnd[tid+2] * (p1[tid+2] - p0[tid+2]);
    }
}

void psdr_cuda::aq_sample(float* result, const float* p0, const float* p1, const float* rnd, int size) {
    int thread_size = 512;
    int block_size = (size-1) / thread_size + 1;
    op_aq_sample<<<block_size, thread_size>>>(result, p0, p1, rnd, size);
}

