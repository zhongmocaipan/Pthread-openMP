#include <iostream>
#include <vector>
#include <immintrin.h>
#include <ctime>
#include <cstdlib>

#define N 1000  // 矩阵大小

double A[N][N + 1];  // 增广矩阵

void gauss_elimination() {
	for (int k = 0; k < N - 1; k++) {
		// 处理主元所在的行
		double main_element = A[k][k];
		__m256d main_element_v = _mm256_set1_pd(main_element);

		// 消去当前列下方的所有元素
		for (int i = k + 1; i < N; i++) {
			double factor = A[i][k] / main_element;
			__m256d factor_v = _mm256_set1_pd(factor);

			for (int j = k; j < N + 1; j += 4) {
				__m256d A_kj = _mm256_loadu_pd(&A[k][j]);
				__m256d A_ij = _mm256_loadu_pd(&A[i][j]);
				__m256d result = _mm256_sub_pd(A_ij, _mm256_mul_pd(factor_v, A_kj));
				_mm256_storeu_pd(&A[i][j], result);
			}
		}
	}
}

int main() {
	// 初始化增广矩阵A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;
		}
	}
	clock_t start = clock();
	gauss_elimination();
	clock_t end = clock();

	double time_taken = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Total time: " << time_taken << " seconds" << std::endl;

	return 0;
}
