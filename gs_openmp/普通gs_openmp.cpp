#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000  // 假设矩阵大小为1000x1000

double A[N][N + 1];  // 增广矩阵

int main() {
	double start_time, end_time;

	// 初始化增广矩阵A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;  // 随机初始化值
		}
	}

	start_time = omp_get_wtime(); // 记录开始时间

	// 高斯消元
	for (int k = 0; k < N - 1; k++) {
#pragma omp parallel for
		for (int i = k + 1; i < N; i++) {
			double factor = A[i][k] / A[k][k];
			for (int j = k; j < N + 1; j++) {
				A[i][j] -= factor * A[k][j];
			}
		}
	}

	end_time = omp_get_wtime(); // 记录结束时间

	printf("Total time: %f seconds\n", end_time - start_time); // 输出总时间

	// 打印消元后的矩阵A

	return 0;
}
