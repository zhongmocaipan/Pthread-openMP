#include <iostream>
#include <vector>
#include <pthread.h>
#include <cstdlib>

#define N 1000
#define NUM_THREADS 4

double A[N][N + 1];
pthread_barrier_t barrier;

void gauss_elimination(int start, int end) {
	for (int k = 0; k < N - 1; k++) {
		for (int i = start; i < end; i++) {
			double factor = A[i][k] / A[k][k];
			float32x4_t factor_v = vdupq_n_f32(factor);
			for (int j = k; j < N + 1; j += 4) {
				float32x4_t A_kj = vld1q_f32(&A[k][j]);
				float32x4_t A_ij = vld1q_f32(&A[i][j]);
				float32x4_t result = vaddq_f32(A_ij, vmulq_f32(factor_v, A_kj));
				vst1q_f32(&A[i][j], result);
			}
		}
		pthread_barrier_wait(&barrier);
	}
}

void* thread_func(void* arg) {
	int thread_id = *((int*)arg);
	int start = thread_id * (N / NUM_THREADS);
	int end = (thread_id + 1) * (N / NUM_THREADS);
	gauss_elimination(start, end);
	return NULL;
}

int main() {
	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;
		}
	}

	pthread_barrier_init(&barrier, NULL, NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
	}
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
	pthread_barrier_destroy(&barrier);

	return 0;
}
