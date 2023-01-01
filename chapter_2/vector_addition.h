#include <chrono>
#include <cmath>
#include <iostream>

/**
 * @brief CPU version of vector addition
 *
 * @param h_A
 * @param h_B
 * @param h_C
 * @param n
 */
void VecAddCpu(float* h_A, float* h_B, float* h_C, int n);

/**
 * @brief GPU version of vector addition
 *
 * @param h_A
 * @param h_B
 * @param h_C
 * @param n
 */
void VecAddGpu(float* h_A, float* h_B, float* h_C, int n);
