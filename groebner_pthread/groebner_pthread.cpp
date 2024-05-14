#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#define N 4 // 多项式个数
#define M 3 // 变量个数
#define MAX_TERMS 100 // 每个多项式的最高项数
#define EPSILON 1e-10 // 零判定阈值
#define NUM_THREADS 4 // 假设使用4个线程

double coefficients[N][MAX_TERMS][M]; // 多项式系数
double roots[N][M]; // 方程组的根
pthread_barrier_t barrier; // 屏障

void* eliminate(void* arg) {
	int thread_id = *((int*)arg);
	int start = thread_id * (N / NUM_THREADS);
	int end = (thread_id + 1) * (N / NUM_THREADS);

	for (int k = 0; k < M; k++) {
		for (int i = start; i < end; i++) {
			if (fabs(coefficients[i][k][k]) < EPSILON) {
				continue; // 如果主元素为0，则跳过
			}

			for (int j = 0; j < N; j++) {
				if (j == i) {
					continue; // 跳过当前方程
				}

				double factor = coefficients[j][k][k] / coefficients[i][k][k];
				for (int l = k; l < M; l++) {
					coefficients[j][k][l] -= factor * coefficients[i][k][l];
				}
			}
		}
		pthread_barrier_wait(&barrier); // 等待其他线程完成当前列的消元
	}

	return NULL;
}

void compute_roots() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			roots[i][j] = coefficients[i][j][M] / coefficients[i][j][j]; // 计算根
		}
	}
}

int main() {
	srand(time(NULL)); // 设置随机种子

	// 初始化多项式系数
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < M + 1; k++) {
				coefficients[i][j][k] = rand() % 10; // 随机初始化值
			}
		}
	}

	clock_t start_time = clock(); // 记录开始时间

	pthread_barrier_init(&barrier, NULL, NUM_THREADS); // 初始化屏障

	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];

	// 创建线程
	for (int i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		pthread_create(&threads[i], NULL, eliminate, &thread_ids[i]);
	}

	// 等待所有线程完成
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	pthread_barrier_destroy(&barrier); // 销毁屏障

	clock_t end_time = clock(); // 记录结束时间
	double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; // 计算经过的时间

	// 计算并输出根
	compute_roots();
	for (int i = 0; i < N; i++) {
		printf("Roots for equation %d: ", i + 1);
		for (int j = 0; j < M; j++) {
			printf("%f ", roots[i][j]);
		}
		printf("\n");
	}

	printf("Elapsed time: %f seconds\n", elapsed_time);

	return 0;
}
