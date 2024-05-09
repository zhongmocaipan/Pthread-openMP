#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <emmintrin.h> // 引入SSE指令集头文件

#define N 1000  // 假设矩阵大小为1000x1000
#define NUM_THREADS 4  // 假设使用4个线程

double A[N][N + 1];  // 增广矩阵
pthread_barrier_t barrier;  // 定义屏障

void* eliminate(void* arg) {
	int thread_id = *((int*)arg);
	int start = thread_id * (N / NUM_THREADS);
	int end = (thread_id + 1) * (N / NUM_THREADS);

	for (int k = 0; k < N - 1; k++) {
		for (int i = start + 1; i < end; i++) {
			double factor = A[i][k] / A[k][k];

			// 使用SSE指令集优化内层循环
			__m128d factor_v = _mm_set1_pd(factor);
			for (int j = k; j < N + 1; j += 2) {
				__m128d A_kj = _mm_loadu_pd(&A[k][j]);
				__m128d A_ij = _mm_loadu_pd(&A[i][j]);
				__m128d result = _mm_add_pd(A_ij, _mm_mul_pd(factor_v, A_kj));
				_mm_storeu_pd(&A[i][j], result);
			}
		}
		pthread_barrier_wait(&barrier);  // 等待其他线程完成当前列的消元
	}

	return NULL;
}

int main() {
	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];
	struct timespec start, end;

	// 初始化增广矩阵A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;  // 随机初始化值
		}
	}

	pthread_barrier_init(&barrier, NULL, NUM_THREADS); // 初始化屏障

	clock_gettime(CLOCK_MONOTONIC, &start); // 记录开始时间

	// 创建线程
	for (int i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		pthread_create(&threads[i], NULL, eliminate, &thread_ids[i]);
	}

	// 等待所有线程完成
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	clock_gettime(CLOCK_MONOTONIC, &end); // 记录结束时间

	double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9; // 转换为秒

	printf("Total time: %f seconds\n", time_taken); // 输出总时间

	// 打印消元后的矩阵A

	pthread_barrier_destroy(&barrier); // 销毁屏障

	return 0;
}
