#include <chrono>
#include <cmath>
#include <iostream>

constexpr int kTileWidth = 32;

/**
 * @brief Simple version for square matrix mul
 *
 * @param M
 * @param N
 * @param P
 * @param width
 */
void SimpleMatrixMulGpu(float* M, float* N, float* P, int width);

/**
 * @brief Tiled version for square matrix mul
 *
 * @param M
 * @param N
 * @param P
 * @param width
 */
void TiledMatrixMulGpu(float* M, float* N, float* P, int width);
