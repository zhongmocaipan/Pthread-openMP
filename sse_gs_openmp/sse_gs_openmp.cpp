#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <immintrin.h> // 引入AVX指令集头文件

#define N 1000  // 假设矩阵大小为1000x1000

double A[N][N + 1];  // 增广矩阵

int main() {
	struct timespec start, end;

	// 初始化增广矩阵A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;  // 随机初始化值
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &start); // 记录开始时间

	// 高斯消元
#pragma omp parallel for
	for (int k = 0; k < N - 1; k++) {
		for (int i = k + 1; i < N; i++) {
			double factor = A[i][k] / A[k][k];

			// 使用AVX指令集优化内层循环
			__m256d factor_v = _mm256_set1_pd(factor);
			for (int j = k; j < N + 1; j += 4) {
				__m256d A_kj = _mm256_loadu_pd(&A[k][j]);
				__m256d A_ij = _mm256_loadu_pd(&A[i][j]);
				__m256d result = _mm256_add_pd(A_ij, _mm256_mul_pd(factor_v, A_kj));
				_mm256_storeu_pd(&A[i][j], result);
			}
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &end); // 记录结束时间

	double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9; // 转换为秒

	printf("Total time: %f seconds\n", time_taken); // 输出总时间

	// 打印消元后的矩阵A

	return 0;
}
