#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <string>
#include <functional>
//#define X64_LOCAL 1
#define ARM_SCHOOL 1
#ifdef X64_LOCAL
#include <emmintrin.h>
#include <immintrin.h>
#endif

#ifdef ARM_SCHOOL
#include <arm_neon.h>
#endif
struct Timer
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;
    int repeat;
    Timer(int num) : repeat(num)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    ~Timer()
    {
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        float ms = duration.count() * 1000.0f / repeat;
        std::cout << ms << std::endl;
    }
};

float **getMatrix(int size)
{
    float **matrix = new float *[size];
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new float[size];
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = 0.0;
        }
        matrix[i][i] = 1.0;
        for (int j = i + 1; j < size; j++)
        {
            matrix[i][j] = rand();
        }
    }
    for (int k = 0; k < size; k++)
    {
        for (int i = k + 1; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrix[i][j] += matrix[k][j];
            }
        }
    }
    return matrix;
}

void delMatrix(float **matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        // free(matrix[i]);
        //_aligned_free(matrix[i]);
        delete[] matrix[i];
    }
    delete[] matrix;
}
//平凡高斯消元法
void ordinarilyGauss(float **matrix, int size)
{
    for (int k = 0; k < size; k++)
    {
        for (int j = k + 1; j < size; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < size; i++)
        {
            for (int j = k + 1; j < size; j++)
            {
                matrix[i][j] -= matrix[k][j] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
}
#ifdef X64_LOCAL
void SIMDIntrinsicSSEGaussWithoutOmp(float **matrix, int size)
{
    for (int k = 0; k < size; k++)
    {
        __m128 divisor = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
        //对第一行并行化
        int j = k + 1;
        for (; j + 3 < size; j += 4)
        {
            __m128 diviend = _mm_loadu_ps(matrix[k] + j);
            diviend = _mm_div_ps(diviend, divisor);
            _mm_storeu_ps(matrix[k] + j, diviend);
        }
        //处理剩余部分
        for (j; j < size; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        matrix[k][k] = 1.0;
        //对子矩阵处理
        for (int i = k + 1; i < size; i++)
        {
            __m128 firstEle = _mm_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
            int t = k + 1;
            for (; t + 3 < size; t += 4)
            {
                __m128 multi = _mm_loadu_ps(matrix[k] + t);
                multi = _mm_mul_ps(firstEle, multi);
                __m128 outcome = _mm_loadu_ps(matrix[i] + t);
                outcome = _mm_sub_ps(outcome, multi);
                _mm_storeu_ps(matrix[i] + t, outcome);
            }
            for (t; t < size; t++)
            {
                matrix[i][t] -= matrix[k][t] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
}
// SIMD指令级并行
void SIMDIntrinsicSSEGauss(float **matrix, int size)
{
    int i, j, k, t;
    __m128 tmp, diviend;
#pragma omp parallel num_threads(4), private(i, j, k, t, tmp, diviend)
    for (k = 0; k < size; k++)
    {
#pragma omp single
        {
            tmp = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
            //对第一行并行化
            for (j = k + 1; j + 3 < size; j += 4)
            {
                diviend = _mm_load_ps(matrix[k] + j);
                diviend = _mm_div_ps(diviend, tmp);
                _mm_store_ps(matrix[k] + j, diviend);
            }
            //处理剩余部分
            for (j; j < size; j++)
            {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        //对子矩阵处理
#pragma omp for
        for (i = k + 1; i < size; i++)
        {
            tmp = _mm_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
            t = k + 1;
            for (; t + 3 < size; t += 4)
            {
                __m128 multi = _mm_load_ps(matrix[k] + t);
                multi = _mm_mul_ps(tmp, multi);
                __m128 outcome = _mm_load_ps(matrix[i] + t);
                outcome = _mm_sub_ps(outcome, multi);
                _mm_store_ps(matrix[i] + t, outcome);
            }
            for (t; t < size; t++)
            {
                matrix[i][t] -= matrix[k][t] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
}
// AVX
void SIMDIntrinsicAVXGaussOmp(float **matrix, int size)
{
    int i, j, k, t;
    __m256 tmp, diviend;
#pragma omp parallel num_threads(4), private(i, j, k, t, tmp, diviend)
    for (k = 0; k < size; k++)
    {
#pragma omp single
        {
            tmp = _mm256_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
            //对第一行并行化
            for (j = k + 1; j + 7 < size; j += 8)
            {
                diviend = _mm256_loadu_ps(matrix[k] + j);
                diviend = _mm256_div_ps(diviend, tmp);
                _mm256_storeu_ps(matrix[k] + j, diviend);
            }
            //处理剩余部分
            for (j; j < size; j++)
            {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        //对子矩阵处理
#pragma omp for
        for (i = k + 1; i < size; i++)
        {
            tmp = _mm256_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
            t = k + 1;
            for (; t + 7 < size; t += 8)
            {
                __m256 multi = _mm256_loadu_ps(matrix[k] + t);
                multi = _mm256_mul_ps(tmp, multi);
                __m256 outcome = _mm256_loadu_ps(matrix[i] + t);
                outcome = _mm256_sub_ps(outcome, multi);
                _mm256_storeu_ps(matrix[i] + t, outcome);
            }
            for (t; t < size; t++)
            {
                matrix[i][t] -= matrix[k][t] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
}

void SIMDIntrinsicAVXGauss(float **matrix, int size)
{
    for (int k = 0; k < size; k++)
    {
        __m256 divisor = _mm256_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
        //对第一行并行化
        int j = k + 1;
        for (; j + 7 < size; j += 8)
        {
            __m256 diviend = _mm256_loadu_ps(matrix[k] + j);
            diviend = _mm256_div_ps(diviend, divisor);
            _mm256_storeu_ps(matrix[k] + j, diviend);
        }
        //处理剩余部分
        for (j; j < size; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        matrix[k][k] = 1.0;
        //对子矩阵处理
        for (int i = k + 1; i < size; i++)
        {
            __m256 firstEle = _mm256_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
            int t = k + 1;
            for (; t + 7 < size; t += 8)
            {
                __m256 multi = _mm256_loadu_ps(matrix[k] + t);
                multi = _mm256_mul_ps(firstEle, multi);
                __m256 outcome = _mm256_loadu_ps(matrix[i] + t);
                outcome = _mm256_sub_ps(outcome, multi);
                _mm256_storeu_ps(matrix[i] + t, outcome);
            }
            for (t; t < size; t++)
            {
                matrix[i][t] -= matrix[k][t] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
}
#endif

#ifdef ARM_SCHOOL
void SIMDIntrinsicNEONGaussWithOmp(float **matrix, int size)
{
    int i, j, k, t;
    float32x4_t tmp, diviend;
#pragma omp parallel num_threads(4), private(i, j, k, t, tmp, diviend)
    for (k = 0; k < size; k++)
    {
#pragma omp single
        {
            tmp = vdupq_n_f32(matrix[k][k]);
            //对第一行并行化
            for (j = k + 1; j + 3 < size; j += 4)
            {
                diviend = vld1q_f32(matrix[k] + j);
                diviend = vdivq_f32(diviend, tmp);
                vst1q_f32(matrix[k] + j, diviend);
            }
            //处理剩余部分
            for (j; j < size; j++)
            {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        //对子矩阵处理
#pragma omp for
        for (i = k + 1; i < size; i++)
        {
            tmp = vdupq_n_f32(matrix[i][k]);
            for (t = k + 1; t + 3 < size; t += 4)
            {
                float32x4_t multi = vld1q_f32(matrix[k] + t);
                multi = vmulq_f32(tmp, multi);
                float32x4_t outcome = vld1q_f32(matrix[i] + t);
                outcome = vsubq_f32(outcome, multi);
                vst1q_f32(matrix[i] + t, outcome);
            }
            for (t; t < size; t++)
            {
                matrix[i][t] -= matrix[k][t] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
}
void SIMDIntrinsicNEONGauss(float **matrix, int size)
{
    for (int k = 0; k < size; k++)
    {
        float32x4_t divisor = vdupq_n_f32(matrix[k][k]);
        //对第一行并行化
        int j = k + 1;
        for (; j + 3 < size; j += 4)
        {
            float32x4_t diviend = vld1q_f32(matrix[k] + j);
            diviend = vdivq_f32(diviend, divisor);
            vst1q_f32(matrix[k] + j, diviend);
        }
        //处理剩余部分
        for (j; j < size; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        matrix[k][k] = 1.0;
        //对子矩阵处理
        for (int i = k + 1; i < size; i++)
        {
            float32x4_t firstEle = vdupq_n_f32(matrix[i][k]);
            int t = k + 1;
            for (; t + 3 < size; t += 4)
            {
                float32x4_t multi = vld1q_f32(matrix[k] + t);
                multi = vmulq_f32(firstEle, multi);
                float32x4_t outcome = vld1q_f32(matrix[i] + t);
                outcome = vsubq_f32(outcome, multi);
                vst1q_f32(matrix[i] + t, outcome);
            }
            for (t; t < size; t++)
            {
                matrix[i][t] -= matrix[k][t] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
}
#endif
//没有指令级并行的openmp并行
void ordinarilyGaussOpenMp2(float **matrix, int size)
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(2), private(i, j, k, tmp)
    for (k = 1; k < size; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
            for (j = k + 1; j < size; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for
        for (i = k + 1; i < size; i++)
        {
            tmp = matrix[i][k];
            for (j = k + 1; j < size; j++)
            {
                matrix[i][j] -= tmp * matrix[k][j];
            }
            matrix[i][k] = 0.0;
        }
    }
}

void ordinarilyGaussOpenMp4(float **matrix, int size)
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(4), private(i, j, k, tmp)
    for (k = 1; k < size; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
            for (j = k + 1; j < size; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for
        for (i = k + 1; i < size; i++)
        {
            tmp = matrix[i][k];
            for (j = k + 1; j < size; j++)
            {
                matrix[i][j] -= tmp * matrix[k][j];
            }
            matrix[i][k] = 0.0;
        }
    }
}
void ordinarilyGaussOpenMp6(float **matrix, int size)
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(6), private(i, j, k, tmp)
    for (k = 1; k < size; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
            for (j = k + 1; j < size; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for
        for (i = k + 1; i < size; i++)
        {
            tmp = matrix[i][k];
            for (j = k + 1; j < size; j++)
            {
                matrix[i][j] -= tmp * matrix[k][j];
            }
            matrix[i][k] = 0.0;
        }
    }
}
void testPerformance(int size, std::function<void(float **, int)> func, const std::string &file_name)
{
    std::fstream fout(file_name, std::ios::in | std::ios::out | std::ios::trunc);
    std::streambuf *coutBackup;
    coutBackup = std::cout.rdbuf(fout.rdbuf());
    int step = 10;
    int repeat = 100;
    for (int scale = 0; scale <= size; scale += step)
    {
        float **matrix = getMatrix(scale);
        {
            Timer timer(repeat);
            for (int j = 0; j < repeat; j++)
            {
                func(matrix, scale);
            }
            std::cout << scale << " ";
        }
        delMatrix(matrix, scale);
        switch (scale)
        {
        case 100:
            repeat = 10;
            step = 100;
            break;
        case 1000:
            repeat = 1;
            step = 1000;
            break;
        }
    }
    std::cout.rdbuf(coutBackup);
    fout.close();
}

int main()
{
    const int TESTSCALE = 1000;
    testPerformance(TESTSCALE, ordinarilyGauss, "ordinary_thread.txt");
    testPerformance(TESTSCALE, ordinarilyGaussOpenMp2, "ordinary_thread2.txt");
    testPerformance(TESTSCALE, ordinarilyGaussOpenMp4, "ordinary_thread4.txt");
    testPerformance(TESTSCALE, ordinarilyGaussOpenMp6, "ordinary_thread6.txt");
}