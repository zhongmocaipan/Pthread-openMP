#include <iostream>
#include <vector>
#include <omp.h>
#include <immintrin.h>
//#include <unistd.h>

#define N 5000  // 矩阵大小为1000x1000

double A[N][N + 1];  // 增广矩阵

void gauss_elimination(int start, int end) {
	for (int k = 0; k < N - 1; k++) {
#pragma omp parallel for
		for (int i = start; i < end; i++) {
			double factor = A[i][k] / A[k][k];
			__m256d factor_v = _mm256_set1_pd(factor);
			for (int j = k; j < N + 1; j += 4) {
				__m256d A_kj = _mm256_loadu_pd(&A[k][j]);
				__m256d A_ij = _mm256_loadu_pd(&A[i][j]);
				__m256d result = _mm256_add_pd(A_ij, _mm256_mul_pd(factor_v, A_kj));
				_mm256_storeu_pd(&A[i][j], result);
			}
		}
#pragma omp barrier
	}
}

int main() {
	double start_time, end_time;
	// 初始化增广矩阵A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;  // 随机初始化值
		}
	}

	// 创建线程
	start_time = omp_get_wtime(); // 获取开始时间
#pragma omp parallel num_threads(4)
	{
		int thread_id = omp_get_thread_num();
		int start = thread_id * (N / 4);
		int end = (thread_id + 1) * (N / 4);
		gauss_elimination(start, end);
	}
	end_time = omp_get_wtime(); // 获取结束时间

	// 打印消元后的矩阵A

	std::cout << "Total time: " << end_time - start_time << " seconds" << std::endl;

	return 0;
}
