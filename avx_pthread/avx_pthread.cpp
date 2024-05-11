#include <iostream>
#include <vector>
#include <pthread.h>
#include <immintrin.h>
#include <unistd.h>

#define N 1000  // 矩阵大小为1000x1000
#define NUM_THREADS 4  // 使用4个线程

double A[N][N + 1];  // 增广矩阵
pthread_barrier_t barrier;  // 定义屏障

void* gauss_elimination(void* arg) {
	int thread_id = *((int*)arg);
	int start = thread_id * (N / NUM_THREADS);
	int end = (thread_id + 1) * (N / NUM_THREADS);

	for (int k = 0; k < N - 1; k++) {
		for (int i = start + 1; i < end; i++) {
			double factor = A[i][k] / A[k][k];
			__m256d factor_v = _mm256_set1_pd(factor);
			for (int j = k; j < N + 1; j += 4) {
				__m256d A_kj = _mm256_loadu_pd(&A[k][j]);
				__m256d A_ij = _mm256_loadu_pd(&A[i][j]);
				__m256d result = _mm256_add_pd(A_ij, _mm256_mul_pd(factor_v, A_kj));
				_mm256_storeu_pd(&A[i][j], result);
			}
		}
		pthread_barrier_wait(&barrier);  // 等待其他线程完成当前列的消元
	}

	return NULL;
}

int main() {
	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];

	// 初始化增广矩阵A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;  // 随机初始化值
		}
	}

	pthread_barrier_init(&barrier, NULL, NUM_THREADS); // 初始化屏障

	// 创建线程
	for (int i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		pthread_create(&threads[i], NULL, gauss_elimination, &thread_ids[i]);
	}

	// 等待所有线程完成
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	// 打印消元后的矩阵A

	pthread_barrier_destroy(&barrier); // 销毁屏障

	return 0;
}
