// Yuan: This is specifically for the GPU version of the function computeVelocity

// may not need the full duplicate of Particle struct
// Note: use structure of arrays (SoA) for GPU compatibility
// This struct is for demonstration purposes only, which is not used in the code
#include <assert.h>
#include <limits>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <map>
#include <iostream>
#include <algorithm>  // For std::sort  // For std::sort
#include "matrix.h"
#include "SetupParticleSystem.h"
#include "util.h"

// TODO: host only access for now, need a way for device access
static const int INVALID_INT = std::numeric_limits<int>::max();
static const double INVALID_DOUBLE = std::numeric_limits<double>::max();

__device__ void Mat_GPU_inverse2D(double *inv,      // Return Val, size = 2*2 in matrix formata
                                  double *elements) // size = 2*2 in matrix format
{
    // compute determinant
    double det = elements[0] * elements[3] - elements[1] * elements[2];

    // Handle near-singular matrices with a small tolerance
    const double tolerance = 1e-12;
    if (fabs(det) < tolerance) {
        // Set to identity matrix for singular matrices
        inv[0] = 1.0;  // inv[0][0]
        inv[1] = 0.0;  // inv[0][1]
        inv[2] = 0.0;  // inv[1][0]
        inv[3] = 1.0;  // inv[1][1]
        return;
    }

    inv[0] = elements[3] / det;  // inv[0][0]
    inv[1] = -elements[1] / det; // inv[0][1]
    inv[2] = -elements[2] / det; // inv[1][0]
    inv[3] = elements[0] / det;  // inv[1][1]
}

// TODO: 3D inverse function

__device__ void Mat_GPU_mul_mat(double *C,
                                const double *A,
                                const double *B,
                                int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__device__ void Mat_GPU_mul_vec(double *C,
                                const double *A,
                                const double *B,
                                int N)
{
    for (int i = 0; i < N; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < N; ++j)
        {
            sum += A[i * N + j] * B[j];
        }
        C[i] = sum;
    }
}

__device__ void Mat_GPU_add(double *C,
                            const double *A,
                            const double *B,
                            int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

template <int NDIM, int MAX_NEIGHBOR_CAPACITY>
__device__ void computeShapeTensors_GPU(double *shape_tensor0, // Return Val, size = ndim*ndim in matrix format
                                        double *shape_tensor1, // Return Val, size = ndim*ndim in matrix format
                                        double n1,
                                        double n2,
                                        double dx,
                                        double horizon,
                                        int pi_neighbor_local_size,
                                        int *pi_local_neighbor_arr,
                                        double *total_local_particle_volume_arr,
                                        double *total_local_particle_initial_positions_arr,
                                        double *total_local_particle_current_positions_arr,
                                        double *pi_init_pos, // size = ndim
                                        double *pj_init_pos, // size = ndim - CRITICAL: This is the target particle pj!
                                        int total_local_particle_size) // Added missing parameter
{
    // Initialize output tensors to zero
    for (int i = 0; i < NDIM * NDIM; ++i) {
        shape_tensor0[i] = 0.0;
        shape_tensor1[i] = 0.0;
    }

    // Compute bond IJ from pi to pj (this is the target bond for weight calculation)
    double bondIJ[NDIM];
    double length2 = 0.0;
    for (int i = 0; i < NDIM; ++i) {
        bondIJ[i] = pj_init_pos[i] - pi_init_pos[i];
        length2 += bondIJ[i] * bondIJ[i];
    }
    double length = sqrt(length2);
    
    // Debug output for shape tensor computation - trace particle 0 only
    bool debug_this = false; // Disable debugging for now
    if (debug_this) {
        printf("GPU SHAPE DEBUG: pi=[%.6f,%.6f], pj=[%.6f,%.6f]\n", 
               pi_init_pos[0], pi_init_pos[1], pj_init_pos[0], pj_init_pos[1]);
        printf("GPU SHAPE DEBUG: bondIJ=[%.6f,%.6f], length=%.6f\n", 
               bondIJ[0], bondIJ[1], length);
        printf("GPU SHAPE DEBUG: neighbor_count=%d\n", pi_neighbor_local_size);
    }
    
    // Loop through all neighbors of particle pi
    for (int nidx = 0; nidx < pi_neighbor_local_size; nidx++)
    {
        int pi_nb_local_idx = pi_local_neighbor_arr[nidx];
        if (pi_nb_local_idx < 0) continue; // Skip invalid neighbors
        
        // CRITICAL FIX: Add bounds checking for neighbor array access
        if (nidx >= MAX_NEIGHBOR_CAPACITY) {
            if (debug_this) printf("GPU SHAPE ERROR: nidx %d >= MAX_NEIGHBOR_CAPACITY %d\n", nidx, MAX_NEIGHBOR_CAPACITY);
            break;
        }
        
        // CRITICAL FIX: Check if neighbor index is valid before accessing arrays
        if (pi_nb_local_idx >= total_local_particle_size) {
            if (debug_this) printf("GPU SHAPE ERROR: Invalid neighbor idx %d >= total_size %d\n", 
                                  pi_nb_local_idx, total_local_particle_size);
            continue;
        }
        
        double pi_nb_volume = total_local_particle_volume_arr[pi_nb_local_idx];
        double *pi_nb_init_pos_arr = &total_local_particle_initial_positions_arr[pi_nb_local_idx * NDIM];
        double *pi_nb_curr_pos_arr = &total_local_particle_current_positions_arr[pi_nb_local_idx * NDIM];
        
        double bondINb[NDIM];
        double bondINbcurrent[NDIM];
        double lengthNb2 = 0.0;
        double numerator = 0.0;

        for (int i = 0; i < NDIM; ++i)
        {
            bondINb[i] = pi_nb_init_pos_arr[i] - pi_init_pos[i];
            bondINbcurrent[i] = pi_nb_curr_pos_arr[i] - pi_init_pos[i];  // FIXED: should be current positions
            lengthNb2 += bondINb[i] * bondINb[i];
            numerator += bondIJ[i] * bondINb[i];  // For angle calculation
        }

        double lengthNb = sqrt(lengthNb2);
        if (lengthNb <= 0.0 || length <= 0.0) continue; // Skip zero-length bonds

        // Calculate the cosine angle (matching CPU exactly)
        double cosAngle = numerator / (length * lengthNb);
        if (cosAngle > 1.0) cosAngle = 1.0; 
        else if (cosAngle < -1.0) cosAngle = -1.0;

        double lengthRatio = fabs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

        // Debug first few neighbors for detailed analysis
        if (debug_this && nidx < 3) {
            printf("GPU SHAPE DEBUG: neighbor %d: nb_pos=[%.6f,%.6f]\n", 
                   nidx, pi_nb_init_pos_arr[0], pi_nb_init_pos_arr[1]);
            printf("GPU SHAPE DEBUG: neighbor %d: bondINb=[%.6f,%.6f], lengthNb=%.6f\n", 
                   nidx, bondINb[0], bondINb[1], lengthNb);
            printf("GPU SHAPE DEBUG: neighbor %d: cosAngle=%.6f, weight=%.6f, volume=%.6f\n", 
                   nidx, cosAngle, weight, pi_nb_volume);
            printf("GPU SHAPE DEBUG: neighbor %d: contribution=[%.6e,%.6e,%.6e,%.6e]\n", 
                   nidx, weight * bondINb[0] * bondINb[0] * pi_nb_volume,
                   weight * bondINb[0] * bondINb[1] * pi_nb_volume,
                   weight * bondINb[1] * bondINb[0] * pi_nb_volume,
                   weight * bondINb[1] * bondINb[1] * pi_nb_volume);
        }

        // Accumulate shape tensors with proper weighting (matching CPU exactly)
        for (int i = 0; i < NDIM; ++i)
        {
            for (int j = 0; j < NDIM; ++j)
            {
                shape_tensor0[i * NDIM + j] += weight * bondINb[i] * bondINb[j] * pi_nb_volume;
                shape_tensor1[i * NDIM + j] += weight * bondINbcurrent[i] * bondINb[j] * pi_nb_volume;
            }
        }
    }
    
    // Debug final shape tensor values
    if (debug_this) {
        printf("GPU SHAPE DEBUG: final shapeRef=[%.6e,%.6e,%.6e,%.6e]\n", 
               shape_tensor0[0], shape_tensor0[1], shape_tensor0[2], shape_tensor0[3]);
        printf("GPU SHAPE DEBUG: final shapeCur=[%.6e,%.6e,%.6e,%.6e]\n", 
               shape_tensor1[0], shape_tensor1[1], shape_tensor1[2], shape_tensor1[3]);
    }
}

template <int NDIM, int STIFFNESS_TENSOR_SIZE, int MAX_NEIGHBOR_CAPACITY>
__device__ void computeStressTensor_GPU(double *stress_tensor,    // Return Val, size = ndim*ndim in matrix
                                        double *shape_ref_tensor, // size = ndim*ndim in matrix
                                        double *shape_cur_tensor, // size = ndim*ndim in matrix
                                        double *stiffness_tensor, // size = stiffness_tensor_size * stiffness_tensor_size in matrix
                                        int pi_local_Index,
                                        int pj_local_Index)
{
    double deformationGradient[NDIM * NDIM];
    double Imatrix[NDIM * NDIM] = {0.0};
    double strain[NDIM * NDIM];
    double strain_vector[NDIM * NDIM]; // actually a vector of size stiffness_tensor_size
    double stress_vector[NDIM * NDIM]; // actually a vector of size stiffness_tensor_size

    double shape_ref_tensor_inv[NDIM * NDIM];
    Mat_GPU_inverse2D(shape_ref_tensor_inv, shape_ref_tensor);
    Mat_GPU_mul_mat(deformationGradient, shape_cur_tensor, shape_ref_tensor_inv, NDIM);

    // Debug output for particle 0 - trace each step
    if (pi_local_Index == 0) {
        printf("GPU PARTICLE 0 SHAPE DEBUG: shapeRef=[%.12e,%.12e,%.12e,%.12e]\n", 
               shape_ref_tensor[0], shape_ref_tensor[1], shape_ref_tensor[2], shape_ref_tensor[3]);
        printf("GPU PARTICLE 0 SHAPE DEBUG: shapeCur=[%.12e,%.12e,%.12e,%.12e]\n", 
               shape_cur_tensor[0], shape_cur_tensor[1], shape_cur_tensor[2], shape_cur_tensor[3]);
        printf("GPU PARTICLE 0 SHAPE DEBUG: shapeRefInv=[%.12e,%.12e,%.12e,%.12e]\n", 
               shape_ref_tensor_inv[0], shape_ref_tensor_inv[1], shape_ref_tensor_inv[2], shape_ref_tensor_inv[3]);
        printf("GPU PARTICLE 0 DEFORM DEBUG: deformGrad=[%.12e,%.12e,%.12e,%.12e]\n", 
               deformationGradient[0], deformationGradient[1], deformationGradient[2], deformationGradient[3]);
    }

    for (int i = 0; i < NDIM; ++i)
    {
        Imatrix[i * NDIM + i] = 1.0; // Initialize identity matrix
    }

    for (int i = 0; i < NDIM; ++i)
    {
        for (int j = 0; j < NDIM; ++j)
        {
            // FIXED: Correct strain calculation matching CPU
            // strain = 0.5 * (deformationGradient + deformationGradient.transpose()) - I
            strain[i * NDIM + j] = 0.5 * (deformationGradient[i * NDIM + j] + deformationGradient[j * NDIM + i]) - Imatrix[i * NDIM + j];
        }
    }

    // Convert strain matrix to vector
    if (NDIM == 2)
    {
        strain_vector[0] = strain[0]; // e11
        strain_vector[1] = strain[3]; // e22
        strain_vector[2] = strain[1]; // e12
    }
    else if (NDIM == 3)
    {
        strain_vector[0] = strain[0]; // e11
        strain_vector[1] = strain[3]; // e22
        strain_vector[2] = strain[8]; // e33
        strain_vector[3] = strain[5]; // e23
        strain_vector[4] = strain[2]; // e13
        strain_vector[5] = strain[1]; // e12
    }

    Mat_GPU_mul_vec(stress_vector, stiffness_tensor, strain_vector, STIFFNESS_TENSOR_SIZE);

    // Debug output for particle 0 specifically
    if (pi_local_Index == 0) {
        printf("GPU PARTICLE 0 STRESS DEBUG: strain=[%.12e,%.12e,%.12e,%.12e]\n", 
               strain[0], strain[1], strain[2], strain[3]);
        printf("GPU PARTICLE 0 STRESS DEBUG: strainV=[%.12e,%.12e,%.12e]\n", 
               strain_vector[0], strain_vector[1], strain_vector[2]);
        printf("GPU PARTICLE 0 STRESS DEBUG: stiffness=[%.6e,%.6e,%.6e;%.6e,%.6e,%.6e;%.6e,%.6e,%.6e]\n", 
               stiffness_tensor[0], stiffness_tensor[1], stiffness_tensor[2],
               stiffness_tensor[3], stiffness_tensor[4], stiffness_tensor[5],
               stiffness_tensor[6], stiffness_tensor[7], stiffness_tensor[8]);
        printf("GPU PARTICLE 0 STRESS DEBUG: stressV=[%.12e,%.12e,%.12e]\n", 
               stress_vector[0], stress_vector[1], stress_vector[2]);
    }

    // Removed damage calculation - set damage to 0 for simplicity
    double damage = 0.0;

    // FIXED: Convert stress vector to stress tensor matrix (matching CPU exactly)
    if (NDIM == 2)
    {
        // CPU: stress = matrix(2, {stressV[0], stressV[2], stressV[2], stressV[1]});
        stress_tensor[0] = stress_vector[0] * (1.0 - damage); // s11
        stress_tensor[1] = stress_vector[2] * (1.0 - damage); // s12  
        stress_tensor[2] = stress_vector[2] * (1.0 - damage); // s21 (symmetric)
        stress_tensor[3] = stress_vector[1] * (1.0 - damage); // s22
    }
    else if (NDIM == 3)
    {
        stress_tensor[0] = stress_vector[0] * (1.0 - damage); // s11
        stress_tensor[1] = stress_vector[5] * (1.0 - damage); // s12
        stress_tensor[2] = stress_vector[4] * (1.0 - damage); // s13
        stress_tensor[3] = stress_vector[5] * (1.0 - damage); // s21
        stress_tensor[4] = stress_vector[1] * (1.0 - damage); // s22
        stress_tensor[5] = stress_vector[3] * (1.0 - damage); // s23
        stress_tensor[6] = stress_vector[4] * (1.0 - damage); // s31
        stress_tensor[7] = stress_vector[3] * (1.0 - damage); // s32
        stress_tensor[8] = stress_vector[2] * (1.0 - damage); // s33
    }

    // Note: The stress_tensor is now updated with the computed values
}

template <int NDIM, int STIFFNESS_TENSOR_SIZE, int MAX_NEIGHBOR_CAPACITY>
__device__ void computeForceDensityStates_GPU_pair(double *Tvector,
                                              int pi_local_idx,
                                              int pj_local_idx,
                                              double n1,
                                              double n2,
                                              double horizon,
                                              double dx,
                                              double *stiffness_tensor, // size = STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE in matrix

                                              /* Stores neighbor index as local index! */
                                              int total_local_particle_size,           // size = localParticleSize + ghostParticleSize
                                              int *total_local_particle_neighbors_arr, // size = localParticleSize + ghostParticleSize
                                              int *total_local_particle_neighbor_sizes_arr,
                                              int *total_local_particle_core_ID_arr,

                                              double *total_local_particle_volume_arr,            // size = localParticleSize
                                              double *total_local_particle_initial_positions_arr, // size = localParticleSize * ndim
                                              double *total_local_particle_current_positions_arr) // size = localParticleSize * ndim
{
    // Initialize arrays
    double Tmatrix[NDIM * NDIM] = {0.0};
    double bondIJ[NDIM];
    double horizonVolume = 0.0;

    double *pi_local_current_positions = &total_local_particle_current_positions_arr[pi_local_idx * NDIM];
    double *pi_local_initial_positions = &total_local_particle_initial_positions_arr[pi_local_idx * NDIM];
    double *pj_local_initial_positions = &total_local_particle_initial_positions_arr[pj_local_idx * NDIM];

    // Calculate bond vector IJ (from pi to pj initial positions)
    double length2 = 0.0;
    for (int i = 0; i < NDIM; ++i)
    {
        bondIJ[i] = pj_local_initial_positions[i] - pi_local_initial_positions[i];
        length2 += bondIJ[i] * bondIJ[i];
    }
    double length = sqrt(length2);

    int pi_neighbor_local_size = total_local_particle_neighbor_sizes_arr[pi_local_idx];
    int *pi_local_neighbor_arr = &total_local_particle_neighbors_arr[pi_local_idx * MAX_NEIGHBOR_CAPACITY];

    // Check if particle has valid neighbors
    if (pi_neighbor_local_size <= 0) {
        // Set Tvector to zero if no valid neighbors
        for (int i = 0; i < NDIM; ++i)
        {
            Tvector[i] = 0.0;
        }
        return;
    }

    // CRITICAL FIX: Match CPU algorithm exactly
    // CPU: computeShapeTensors(piNeighbors, pi, pj) - for specific bond pi->pj
    double shape_tensor0[NDIM * NDIM] = {0.0};
    double shape_tensor1[NDIM * NDIM] = {0.0};
    
    computeShapeTensors_GPU<NDIM, MAX_NEIGHBOR_CAPACITY>(shape_tensor0, shape_tensor1,
                                  n1, n2, dx, horizon,
                                  pi_neighbor_local_size,
                                  pi_local_neighbor_arr,
                                  total_local_particle_volume_arr,
                                  total_local_particle_initial_positions_arr,
                                  total_local_particle_current_positions_arr,
                                  pi_local_initial_positions,
                                  pj_local_initial_positions,  // Shape tensors for specific bond pi->pj
                                  total_local_particle_size);

    // Compute stress tensor using this specific bond's shape tensors
    double stress_tensor[NDIM * NDIM] = {0.0};
    computeStressTensor_GPU<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY>(
        stress_tensor,
        shape_tensor0,    // Use bond-specific shape tensor
        shape_tensor1,    // Use bond-specific shape tensor  
        stiffness_tensor,
        pi_local_idx, pj_local_idx);

    // Compute shape_tensor0^(-1)
    double shape_tensor0_inv[NDIM * NDIM];
    Mat_GPU_inverse2D(shape_tensor0_inv, shape_tensor0);
    
    // Compute stress * shape_tensor0^(-1)
    double stress_shape_inv_prod[NDIM * NDIM] = {0.0};
    Mat_GPU_mul_mat(stress_shape_inv_prod, stress_tensor, shape_tensor0_inv, NDIM);

    // Initialize Tmatrix and horizon volume for the specific bond
    for (int i = 0; i < NDIM * NDIM; ++i) {
        Tmatrix[i] = 0.0;
    }
    
    // Accumulate from all neighbors of pi (matching CPU exactly)
    for (int nidx = 0; nidx < pi_neighbor_local_size; ++nidx)
    {
        int pi_nb_local_idx = pi_local_neighbor_arr[nidx];
        if (pi_nb_local_idx < 0) continue;
        
        // Add bounds checking
        if (nidx >= MAX_NEIGHBOR_CAPACITY) break;
        if (pi_nb_local_idx >= total_local_particle_size) continue;
        
        double pi_nb_volume = total_local_particle_volume_arr[pi_nb_local_idx];
        double *pi_nb_local_initial_positions = &total_local_particle_initial_positions_arr[pi_nb_local_idx * NDIM];

        // Calculate neighbor bond INb from pi to nb
        double bondINb[NDIM];
        double lengthNb2 = 0.0;
        double numerator = 0.0;

        for (int i = 0; i < NDIM; ++i)
        {
            bondINb[i] = pi_nb_local_initial_positions[i] - pi_local_initial_positions[i];
            lengthNb2 += bondINb[i] * bondINb[i];
            numerator += bondIJ[i] * bondINb[i];  // For angle between IJ and INb
        }

        double lengthNb = sqrt(lengthNb2);
        if (length <= 0.0 || lengthNb <= 0.0) continue;

        // Calculate weight based on angle between bondIJ and bondINb
        double cosAngle = numerator / (length * lengthNb);
        if (cosAngle > 1.0) cosAngle = 1.0; 
        else if (cosAngle < -1.0) cosAngle = -1.0;

        double lengthRatio = fabs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

        // Accumulate: Tmatrix += stress * shapeInv * weight * volume (matching CPU exactly)
        for (int i = 0; i < NDIM; ++i)
        {
            for (int j = 0; j < NDIM; ++j)
            {
                Tmatrix[i * NDIM + j] += stress_shape_inv_prod[i * NDIM + j] * weight * pi_nb_volume;
            }
        }
        horizonVolume += pi_nb_volume;
    }

    // Normalize by horizon volume and compute final force vector
    // CPU: Tmatrix.timeScalar(1.0 / horizonVolume).timeVector(bondIJ)
    if (horizonVolume > 0.0) {
        for (int i = 0; i < NDIM; ++i)
        {
            for (int j = 0; j < NDIM; ++j)
            {
                Tmatrix[i * NDIM + j] /= horizonVolume;
            }
        }
        
        Mat_GPU_mul_vec(Tvector, Tmatrix, bondIJ, NDIM);
    } else {
        // Set Tvector to zero if no valid horizon volume
        for (int i = 0; i < NDIM; ++i) {
            Tvector[i] = 0.0;
        }
    }
}

template <int NDIM, int STIFFNESS_TENSOR_SIZE, int MAX_NEIGHBOR_CAPACITY>
__device__ void computeForceDensityStates_GPU(double *Tvector,
                                              int pi_local_idx,
                                              int pj_local_idx,
                                              double n1,
                                              double n2,
                                              double horizon,
                                              double dx,
                                              double *stiffness_tensor, // size = STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE in matrix

                                              /* Stores neighbor index as local index! */
                                              int total_local_particle_size,           // size = localParticleSize + ghostParticleSize
                                              int *total_local_particle_neighbors_arr, // size = localParticleSize + ghostParticleSize
                                              int *total_local_particle_neighbor_sizes_arr,
                                              int *total_local_particle_core_ID_arr,

                                              double *total_local_particle_volume_arr,            // size = localParticleSize
                                              double *total_local_particle_initial_positions_arr, // size = localParticleSize * ndim
                                              double *total_local_particle_current_positions_arr) // size = localParticleSize * ndim
{
    // Initialize arrays
    double Tmatrix[NDIM * NDIM] = {0.0};
    double bondIJ[NDIM];

    double horizonVolume = 0.0;

    double *pi_local_current_positions = &total_local_particle_current_positions_arr[pi_local_idx * NDIM];
    double *pi_local_initial_positions = &total_local_particle_initial_positions_arr[pi_local_idx * NDIM];
    double *pj_local_initial_positions = &total_local_particle_initial_positions_arr[pj_local_idx * NDIM];

    // Calculate bond vector IJ (from pi to pj initial positions)
    double length2 = 0.0;
    for (int i = 0; i < NDIM; ++i)
    {
        bondIJ[i] = pj_local_initial_positions[i] - pi_local_initial_positions[i];
        length2 += bondIJ[i] * bondIJ[i];
    }
    double length = sqrt(length2);

    int pi_neighbor_local_size = total_local_particle_neighbor_sizes_arr[pi_local_idx];
    int *pi_local_neighbor_arr = &total_local_particle_neighbors_arr[pi_local_idx * MAX_NEIGHBOR_CAPACITY];

    // Check if particle has valid neighbors
    if (pi_neighbor_local_size <= 0) {
        // Set Tvector to zero if no valid neighbors
        for (int i = 0; i < NDIM; ++i)
        {
            Tvector[i] = 0.0;
        }
        return;
    }

    // Initialize Tmatrix and horizon volume
    for (int i = 0; i < NDIM * NDIM; ++i) {
        Tmatrix[i] = 0.0;
    }

    // SIMPLIFIED: Match CPU algorithm exactly
    // For each neighbor nb in piNeighbors:
    for (int nidx = 0; nidx < pi_neighbor_local_size; ++nidx)
    {
        int pi_nb_local_idx = pi_local_neighbor_arr[nidx];
        if (pi_nb_local_idx < 0) continue;
        
        // CRITICAL FIX: Add bounds checking for force density computation
        if (nidx >= MAX_NEIGHBOR_CAPACITY) {
            printf("GPU FORCE ERROR: nidx %d >= MAX_NEIGHBOR_CAPACITY %d\n", nidx, MAX_NEIGHBOR_CAPACITY);
            break;
        }
        
        if (pi_nb_local_idx >= total_local_particle_size) {
            printf("GPU FORCE ERROR: Invalid neighbor idx %d >= total_size %d\n", 
                   pi_nb_local_idx, total_local_particle_size);
            continue;
        }
        
        double pi_nb_volume = total_local_particle_volume_arr[pi_nb_local_idx];
        double *pi_nb_local_initial_positions = &total_local_particle_initial_positions_arr[pi_nb_local_idx * NDIM];

        // Calculate neighbor bond INb from pi to nb
        double bondINb[NDIM];
        double lengthNb2 = 0.0;
        double numerator = 0.0;

        for (int i = 0; i < NDIM; ++i)
        {
            bondINb[i] = pi_nb_local_initial_positions[i] - pi_local_initial_positions[i];
            lengthNb2 += bondINb[i] * bondINb[i];
            numerator += bondIJ[i] * bondINb[i];  // For angle between IJ and INb
        }

        double lengthNb = sqrt(lengthNb2);
        if (length <= 0.0 || lengthNb <= 0.0) continue;

        // Calculate weight based on angle between bondIJ and bondINb
        double cosAngle = numerator / (length * lengthNb);
        if (cosAngle > 1.0) cosAngle = 1.0; 
        else if (cosAngle < -1.0) cosAngle = -1.0;

        double lengthRatio = fabs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

        // CORRECTED: Compute shape tensors FOR the specific bond pi->nb (matching CPU exactly)
        // CPU: computeShapeTensors(piNeighbors, pi, *nb) - for each specific neighbor nb
        double nb_shape_tensor0[NDIM * NDIM] = {0.0};
        double nb_shape_tensor1[NDIM * NDIM] = {0.0};
        
        computeShapeTensors_GPU<NDIM, MAX_NEIGHBOR_CAPACITY>(nb_shape_tensor0, nb_shape_tensor1,
                                      n1, n2, dx, horizon,
                                      pi_neighbor_local_size,
                                      pi_local_neighbor_arr,
                                      total_local_particle_volume_arr,
                                      total_local_particle_initial_positions_arr,
                                      total_local_particle_current_positions_arr,
                                      pi_local_initial_positions,
                                      pi_nb_local_initial_positions,  // Shape tensors for specific bond pi->nb
                                      total_local_particle_size); // Added missing parameter

        // Compute stress tensor using this specific bond's shape tensors
        double nb_stress_tensor[NDIM * NDIM] = {0.0};
        computeStressTensor_GPU<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY>(
            nb_stress_tensor,
            nb_shape_tensor0,    // Use neighbor-specific shape tensor
            nb_shape_tensor1,    // Use neighbor-specific shape tensor
            stiffness_tensor,
            pi_local_idx, nidx);  // FIXED: Use correct particle index

        // Debug output for particle 0 stress computation
        if (pi_local_idx == 0 && nidx < 5) {
            printf("GPU DEBUG P0 STRESS: neighbor %d\n", nidx);
            printf("GPU DEBUG P0 STRESS: shape_tensor0=[%.6e,%.6e,%.6e,%.6e]\n", 
                   nb_shape_tensor0[0], nb_shape_tensor0[1], nb_shape_tensor0[2], nb_shape_tensor0[3]);
            printf("GPU DEBUG P0 STRESS: stress=[%.6e,%.6e,%.6e,%.6e]\n", 
                   nb_stress_tensor[0], nb_stress_tensor[1], nb_stress_tensor[2], nb_stress_tensor[3]);
            printf("GPU DEBUG P0 STRESS: weight=%.6e, volume=%.6e\n", weight, pi_nb_volume);
        }

        // Compute stress * shape_tensor0^(-1) using neighbor-specific shape tensor
        double nb_shape_tensor0_inv[NDIM * NDIM];
        Mat_GPU_inverse2D(nb_shape_tensor0_inv, nb_shape_tensor0);
        double nb_stress_shape_inv_prod[NDIM * NDIM] = {0.0};
        Mat_GPU_mul_mat(nb_stress_shape_inv_prod, nb_stress_tensor, nb_shape_tensor0_inv, NDIM);

        // Accumulate: Tmatrix += stress * shapeInv * weight * volume (matching CPU exactly)
        for (int i = 0; i < NDIM; ++i)
        {
            for (int j = 0; j < NDIM; ++j)
            {
                Tmatrix[i * NDIM + j] += nb_stress_shape_inv_prod[i * NDIM + j] * weight * pi_nb_volume;
            }
        }
        horizonVolume += pi_nb_volume;
    }

    // Normalize by horizon volume and compute final force vector
    // CPU: Tmatrix.timeScalar(1.0 / horizonVolume).timeVector(bondIJ)
    if (horizonVolume > 0.0) {
        for (int i = 0; i < NDIM; ++i)
        {
            for (int j = 0; j < NDIM; ++j)
            {
                Tmatrix[i * NDIM + j] /= horizonVolume;
            }
        }
        // DEBUG: Print Tmatrix before multiplication for first particle
        if (pi_local_idx == 0 && pj_local_idx == 1) {
            printf("GPU DEBUG: Tmatrix=[%.6e,%.6e,%.6e,%.6e]\n", 
                   Tmatrix[0], Tmatrix[1], Tmatrix[2], Tmatrix[3]);
            printf("GPU DEBUG: bondIJ=[%.6e,%.6e], horizonVolume=%.6e\n", 
                   bondIJ[0], bondIJ[1], horizonVolume);
            
            // Manual calculation verification
            double manual_Tvector[2];
            manual_Tvector[0] = Tmatrix[0] * bondIJ[0] + Tmatrix[1] * bondIJ[1];
            manual_Tvector[1] = Tmatrix[2] * bondIJ[0] + Tmatrix[3] * bondIJ[1];
            printf("GPU DEBUG: manual_Tvector=[%.6e,%.6e]\n", 
                   manual_Tvector[0], manual_Tvector[1]);
        }
        
        Mat_GPU_mul_vec(Tvector, Tmatrix, bondIJ, NDIM);
        
        // DEBUG: Print detailed results for first particle
        if (pi_local_idx == 0 && pj_local_idx == 1) {
            printf("GPU DEBUG: pi=%d, pj=%d, bondIJ=[%.6f,%.6f], length=%.6f\n", 
                   pi_local_idx, pj_local_idx, bondIJ[0], bondIJ[1], length);
            printf("GPU DEBUG: horizonVolume=%.6e, neighbor_count=%d\n", 
                   horizonVolume, pi_neighbor_local_size);
            printf("GPU DEBUG: final_Tvector=[%.6e,%.6e]\n", 
                   Tvector[0], Tvector[1]);
        }
    } else {
        // Set Tvector to zero if no valid horizon volume
        for (int i = 0; i < NDIM; ++i) {
            Tvector[i] = 0.0;
        }
    }
}

template <int NDIM, int STIFFNESS_TENSOR_SIZE, int MAX_NEIGHBOR_CAPACITY>
__global__ void compute_velocity_kernel_GPU(int ndim,
                                            double n1,
                                            double n2,
                                            double horizon,
                                            double dx,
                                            double massDensity,
                                            double *stiffness_tensor, // size = STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE in matrix
                                            double stepSize,

                                            int core_particle_size,
                                            int *core_particle_local_ID_arr, // size = localParticleSize

                                            double *core_velocity_arr,              // size = localParticleSize * ndim
                                            double *core_acceleration_arr,          // size = localParticleSize * ndim
                                            double *core_net_force_arr,             // size = localParticleSize * ndim
                                            // Removed: double *core_neighbor_bound_damage_arr, // size = localParticleSize * max_neighbor_capacity

                                            /* Stores neighbor index as local index! */
                                            int total_local_particle_size,           // size = localParticleSize + ghostParticleSize
                                            int *total_local_particle_neighbors_arr, // size = localParticleSize + ghostParticleSize
                                            int *total_local_particle_neighbor_sizes_arr,
                                            int *total_local_particle_core_ID_arr,

                                            double *total_local_particle_volume_arr,       // size = localParticleSize
                                            double *total_local_particle_initial_positions_arr, // size = localParticleSize * ndim
                                            double *total_local_particle_current_positions_arr) // size = localParticleSize * ndim
{
    int cp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (cp_idx >= core_particle_size) return;

    double forceIJ[NDIM];
    double forceJI[NDIM];

    int cp_local_idx = core_particle_local_ID_arr[cp_idx];
    
    // CRITICAL FIX: Add bounds checking for all array accesses
    if (cp_local_idx < 0 || cp_local_idx >= total_local_particle_size) {
        printf("GPU ERROR: Invalid cp_local_idx=%d, total_size=%d\n", cp_local_idx, total_local_particle_size);
        return;
    }
    
    int cp_neighbor_local_size = total_local_particle_neighbor_sizes_arr[cp_local_idx];

    if (cp_neighbor_local_size <= 0) return; // Skip boundary particles
    if (cp_neighbor_local_size > MAX_NEIGHBOR_CAPACITY) {
        printf("GPU ERROR: Too many neighbors: %d > %d for particle %d\n", 
               cp_neighbor_local_size, MAX_NEIGHBOR_CAPACITY, cp_local_idx);
        return;
    }

    int *cp_local_neighbor_arr = &total_local_particle_neighbors_arr[cp_local_idx * MAX_NEIGHBOR_CAPACITY];

    double acc_new[NDIM] = {0.0};
    double net_force[NDIM] = {0.0};

    for (int cp_neighbor_idx = 0; cp_neighbor_idx < cp_neighbor_local_size; ++cp_neighbor_idx)
    {
        // get two hop neighbor list
        int cp_neighbor_local_idx = cp_local_neighbor_arr[cp_neighbor_idx];
        
        // CRITICAL FIX: Add bounds checking for neighbor indices
        if (cp_neighbor_local_idx < 0 || cp_neighbor_local_idx >= total_local_particle_size) {
            printf("GPU ERROR: Invalid neighbor idx=%d at pos %d for particle %d, total_size=%d\n", 
                   cp_neighbor_local_idx, cp_neighbor_idx, cp_local_idx, total_local_particle_size);
            continue;
        }
        
        int cp_nb_neighbor_local_size = total_local_particle_neighbor_sizes_arr[cp_neighbor_local_idx];

        if (cp_nb_neighbor_local_size <= 0) continue; // Skip boundary particles
        if (cp_nb_neighbor_local_size > MAX_NEIGHBOR_CAPACITY) {
            printf("GPU ERROR: Neighbor %d has too many neighbors: %d > %d\n", 
                   cp_neighbor_local_idx, cp_nb_neighbor_local_size, MAX_NEIGHBOR_CAPACITY);
            continue;
        }

        double cp_neighbor_volume = total_local_particle_volume_arr[cp_neighbor_local_idx];

        computeForceDensityStates_GPU_pair<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY>(
            forceIJ, cp_local_idx, cp_neighbor_local_idx,
            n1, n2, horizon, dx,
            stiffness_tensor,
            total_local_particle_size,
            total_local_particle_neighbors_arr,
            total_local_particle_neighbor_sizes_arr,
            total_local_particle_core_ID_arr,
            total_local_particle_volume_arr,
            total_local_particle_initial_positions_arr,
            total_local_particle_current_positions_arr
        );

        computeForceDensityStates_GPU_pair<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY>(
            forceJI, cp_neighbor_local_idx, cp_local_idx,
            n1, n2, horizon, dx,
            stiffness_tensor,
            total_local_particle_size,
            total_local_particle_neighbors_arr,
            total_local_particle_neighbor_sizes_arr,
            total_local_particle_core_ID_arr,
            total_local_particle_volume_arr,
            total_local_particle_initial_positions_arr,
            total_local_particle_current_positions_arr
        );

        // Debug output for particle 0 (local index 0)
        if (cp_local_idx == 0 && cp_neighbor_local_idx <= 5) {
            int cp_global_id = total_local_particle_core_ID_arr[cp_local_idx];
            int nb_global_id = total_local_particle_core_ID_arr[cp_neighbor_local_idx];
            printf("GPU DEBUG P0: pi=%d, pj=%d\n", cp_global_id, nb_global_id);
            printf("GPU DEBUG P0: forceIJ=[%e,%e]\n", forceIJ[0], forceIJ[1]);
            printf("GPU DEBUG P0: forceJI=[%e,%e]\n", forceJI[0], forceJI[1]);
            printf("GPU DEBUG P0: volume=%e\n", cp_neighbor_volume);
            printf("GPU DEBUG P0: force_contrib=[%e,%e]\n", 
                   (forceIJ[0] - forceJI[0]) * cp_neighbor_volume, 
                   (forceIJ[1] - forceJI[1]) * cp_neighbor_volume);
        }

        for (int i = 0; i < NDIM; ++i)
        {
            net_force[i] += (forceIJ[i] - forceJI[i]) * cp_neighbor_volume;
        }
    }

    // Store net force
    for (int i = 0; i < NDIM; ++i)
    {
        core_net_force_arr[cp_local_idx * NDIM + i] = net_force[i];
    }

    // Debug output for particle 0 final results
    if (cp_local_idx == 0) {
        int cp_global_id = total_local_particle_core_ID_arr[cp_local_idx];
        printf("GPU DEBUG P0: final_netF=[%e,%e]\n", net_force[0], net_force[1]);
        printf("GPU DEBUG P0: acceleration=[%e,%e]\n", net_force[0] / massDensity, net_force[1] / massDensity);
    }

    // Compute acceleration and update velocity
    for (int i = 0; i < NDIM; ++i)
    {
        // compute acceleration
        acc_new[i] = core_net_force_arr[cp_local_idx * NDIM + i] / massDensity;

        // update velocity using velocity-verlet integration
        core_velocity_arr[cp_local_idx * NDIM + i] += 0.5 * (core_acceleration_arr[cp_local_idx * NDIM + i] + acc_new[i]) * stepSize;
        
        // update acceleration for next time step
        core_acceleration_arr[cp_local_idx * NDIM + i] = acc_new[i];
    }
}

void compute_velocity_GPU_host(int rank,
                               int ndim,
                               double n1,
                               double n2,
                               double horizon,
                               double dx,
                               double massDensity,
                               matrix &StiffnessTensor,
                               double stepSize,
                               vector<vector<long double>> &velocity,
                               vector<vector<Particle *>> &Neighborslist,
                               vector<vector<long double>> &acce,
                               vector<vector<long double>> &netF,
                               vector<Particle> &localParticles,
                               vector<Particle> &ghostParticles,
                               const unordered_map<int, int> &globalLocalIDmap,
                               const map<int, int> &globalPartitionIDmap,
                               const unordered_map<int, int> &globalGhostIDmap,
                               vector<vector<double>> &bondDamage)
{
    // localParticles and ghostParticles sum up to be all the particles accessed by this rank
    // ghost particle contains two hop neighbor of the localParticles

    // cast every long double to double for GPU compatibility
    for (auto &v : velocity)
    {
        for (auto &val : v)
        {
            val = static_cast<double>(val);
        }
    }

    for (auto &v : acce)
    {
        for (auto &val : v)
        {
            val = static_cast<double>(val);
        }
    }

    for (auto &v : netF)
    {
        for (auto &val : v)
        {
            val = static_cast<double>(val);
        }
    }


    // Prepare data for GPU computation
    int localParticleSize = localParticles.size();
    int ghostParticleSize = ghostParticles.size();

    int total_particle_size = localParticleSize + ghostParticleSize;
    int core_particle_size = localParticleSize;

    constexpr int NDIM = 2;
    constexpr int STIFFNESS_TENSOR_SIZE = 3;  // Assuming 2D, so size is 3 (e11, e22, e12) or 6 for 3D (e11, e22, e33, e23, e13, e12)
    constexpr int MAX_NEIGHBOR_CAPACITY = 30; // Assuming a maximum neighbor capacity, adjust as needed

    vector<Particle> totalParticles; // default first localParticlesSize is core particles, then ghost particles
    totalParticles.reserve(total_particle_size);
    totalParticles.insert(totalParticles.end(), localParticles.begin(), localParticles.end());
    totalParticles.insert(totalParticles.end(), ghostParticles.begin(), ghostParticles.end());

    map<int, int> new_globalLocalIDmap; // Maps global ID to local index in totalParticles
    
    // CRITICAL FIX: Ensure deterministic ordering by sorting particles by global ID
    // This ensures GPU and CPU process particles in the same order
    std::sort(totalParticles.begin(), totalParticles.end(), 
              [](const Particle& a, const Particle& b) {
                  return a.globalID < b.globalID;
              });
    
    for (int i = 0; i < total_particle_size; ++i)
    {
        Particle &p = totalParticles[i];
        // Preserve original global IDs - never overwrite them!
        new_globalLocalIDmap[p.globalID] = i; // Map original global ID to local index
    }

    vector<vector<int>> totalParticleNeighborslist(total_particle_size); // use local idx
    vector<int> totalParticleNeighborSizes(total_particle_size, 0);

    // Fill the neighbor list for each particle with deterministic ordering
    for (int i = 0; i < total_particle_size; ++i)
    {
        int missing_neighbor_count = 0;
        Particle &p = totalParticles[i];
        
        // CRITICAL FIX: Sort neighbors by global ID to ensure deterministic processing order
        vector<int> sorted_neighbors = p.neighbors;
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());
        
        for (int j = 0; j < sorted_neighbors.size(); ++j)
        {
            int neighbor_globalID = sorted_neighbors[j];
            if (new_globalLocalIDmap.find(neighbor_globalID) != new_globalLocalIDmap.end())
            {
                int neighbor_local_idx = new_globalLocalIDmap[neighbor_globalID];
                totalParticleNeighborslist[i].push_back(neighbor_local_idx);
                totalParticleNeighborSizes[i]++;
            }
            else
            {
                // If the neighbor is not found in the map, it might be a ghost particle or an invalid neighbor
                missing_neighbor_count++;
            }
        }

        if (missing_neighbor_count > 0 && missing_neighbor_count < sorted_neighbors.size())
        {
            // Handle the case where some neighbors are missing
            std::cout << "Warning: " << (i < core_particle_size ? "Core " : "Ghost ") << "Particle local ID" << i << " has " << missing_neighbor_count << " missing neighbors." << std::endl;
        }
        else if (missing_neighbor_count == sorted_neighbors.size())
        {
            // If all neighbors are missing, this particle is isolated
            std::cout << "Warning: " << (i < core_particle_size ? "Core " : "Ghost ") << "Particle local ID" << i << " is isolated with no valid neighbors." << std::endl;
        }


        if (missing_neighbor_count == 0) {
            totalParticleNeighborSizes[i] = sorted_neighbors.size(); // Update neighbor size only if no missing neighbors
        } else {
            totalParticleNeighborslist[i].clear(); // Clear the neighbor list if there are missing neighbors
            totalParticleNeighborSizes[i] = 0; // Reset neighbor size if there are missing
        }
    }

    //sanity check, first hop neighbor should have full neighbor list
    for (int core_local_id = 0; core_local_id < core_particle_size; ++core_local_id)
    {
        vector<int> &neighbors = totalParticleNeighborslist[core_local_id];
        for (int core_neighbor_local_id : neighbors)
        {
            if (totalParticleNeighborSizes[core_neighbor_local_id] == 0)
            {
                std::cout << "Error: Core particle local ID " << core_local_id << " has a neighbor with no neighbors." << std::endl;
            }
        }
    }

    // Allocate memory on GPU for all necessary arrays
    double *d_stiffness_tensor;

    int *d_core_particle_local_ID_arr;
    int *d_total_local_particle_core_ID_arr;

    double *d_core_velocity_arr;
    double *d_core_acceleration_arr;
    double *d_core_net_force_arr;
    // Removed: d_core_neighbor_bound_damage_arr

    int *d_total_local_particle_neighbors_arr;
    int *d_total_local_particle_neighbor_sizes_arr;

    double *d_total_local_particle_volume_arr;
    double *d_total_local_particle_initial_positions_arr;
    double *d_total_local_particle_current_positions_arr;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_stiffness_tensor, sizeof(double) * STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE);
    cudaMalloc((void **)&d_core_particle_local_ID_arr, sizeof(int) * core_particle_size);
    cudaMalloc((void **)&d_total_local_particle_core_ID_arr, sizeof(int) * total_particle_size);
    cudaMalloc((void **)&d_core_velocity_arr, sizeof(double) * core_particle_size * NDIM);
    cudaMalloc((void **)&d_core_acceleration_arr, sizeof(double) * core_particle_size * NDIM);
    cudaMalloc((void **)&d_core_net_force_arr, sizeof(double) * core_particle_size * NDIM);
    // Removed: cudaMalloc for d_core_neighbor_bound_damage_arr
    cudaMalloc((void **)&d_total_local_particle_neighbors_arr, sizeof(int) * total_particle_size * MAX_NEIGHBOR_CAPACITY);
    cudaMalloc((void **)&d_total_local_particle_neighbor_sizes_arr, sizeof(int) * total_particle_size);
    cudaMalloc((void **)&d_total_local_particle_volume_arr, sizeof(double) * total_particle_size);
    cudaMalloc((void **)&d_total_local_particle_initial_positions_arr, sizeof(double) * total_particle_size * NDIM);
    cudaMalloc((void **)&d_total_local_particle_current_positions_arr, sizeof(double) * total_particle_size * NDIM);

    // Precalculate the neighbor information
    vector<double> h_stiffness_tensor_arr(STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE);

    vector<int> h_core_particle_local_ID_arr(core_particle_size);
    vector<int> h_total_local_particle_core_ID_arr(total_particle_size);

    vector<double> h_core_velocity_arr(core_particle_size * NDIM);
    vector<double> h_core_acceleration_arr(core_particle_size * NDIM);
    vector<double> h_core_net_force_arr(core_particle_size * NDIM);
    // Removed: h_core_neighbor_bound_damage_arr

    vector<int> h_total_local_particle_neighbors_arr(total_particle_size * MAX_NEIGHBOR_CAPACITY);
    vector<int> h_total_local_particle_neighbor_sizes_arr(total_particle_size);

    vector<double> h_total_local_particle_volume_arr(total_particle_size);
    vector<double> h_total_local_particle_initial_positions_arr(total_particle_size * NDIM);
    vector<double> h_total_local_particle_current_positions_arr(total_particle_size * NDIM);

    // h_stiffness_tensor_arr
    for (int i = 0; i < STIFFNESS_TENSOR_SIZE; ++i)
    {
        for (int j = 0; j < STIFFNESS_TENSOR_SIZE; ++j)
        {
            h_stiffness_tensor_arr[i * STIFFNESS_TENSOR_SIZE + j] = static_cast<double>(StiffnessTensor.elements[i][j]);
        }
    }

    // h_core_particle_local_ID_arr - maps core particle index to local index in totalParticles
    // h_total_local_particle_core_ID_arr - stores global ID for each particle in totalParticles
    for (int i = 0; i < total_particle_size; ++i) {
        h_total_local_particle_core_ID_arr[i] = totalParticles[i].globalID; // Store actual global ID
    }

    // CRITICAL FIX: Map core particles to their positions in the sorted totalParticles array
    for (int i = 0; i < core_particle_size; ++i)
    {
        int local_particle_global_id = localParticles[i].globalID;
        // Find where this local particle ended up in the sorted totalParticles array
        int sorted_local_idx = new_globalLocalIDmap[local_particle_global_id];
        h_core_particle_local_ID_arr[i] = sorted_local_idx;
    }

    // h_core_velocity_arr, h_core_acceleration_arr, h_core_net_force_arr
    // CRITICAL FIX: Map velocity/acceleration/netforce to correct positions in sorted array
    for (int i = 0; i < core_particle_size; ++i)
    {
        Particle &p = localParticles[i];
        for (int j = 0; j < NDIM; ++j)
        {
            h_core_velocity_arr[i * NDIM + j] = velocity[i][j];
            h_core_acceleration_arr[i * NDIM + j] = acce[i][j];
            h_core_net_force_arr[i * NDIM + j] = netF[i][j];
        }
    }

    // Removed bond damage initialization

    for (int i = 0; i < total_particle_size; ++i)
    {
        Particle &p = totalParticles[i];

        h_total_local_particle_volume_arr[i] = p.volume;

        for (int j = 0; j < NDIM; ++j)
        {
            h_total_local_particle_initial_positions_arr[i * NDIM + j] = p.initialPositions[j];
            h_total_local_particle_current_positions_arr[i * NDIM + j] = p.currentPositions[j];
        }

        // Fill neighbor array with proper padding
        for (int j = 0; j < MAX_NEIGHBOR_CAPACITY; ++j)
        {
            if (j < totalParticleNeighborslist[i].size())
            {
                h_total_local_particle_neighbors_arr[i * MAX_NEIGHBOR_CAPACITY + j] = totalParticleNeighborslist[i][j];
            }
            else
            {
                h_total_local_particle_neighbors_arr[i * MAX_NEIGHBOR_CAPACITY + j] = INVALID_INT;
            }
        }

        h_total_local_particle_neighbor_sizes_arr[i] = totalParticleNeighborSizes[i];
    }

    // Copy data from host to device
    cudaMemcpy(d_stiffness_tensor, h_stiffness_tensor_arr.data(), sizeof(double) * STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_particle_local_ID_arr, h_core_particle_local_ID_arr.data(), sizeof(int) * core_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_core_ID_arr, h_total_local_particle_core_ID_arr.data(), sizeof(int) * total_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_velocity_arr, h_core_velocity_arr.data(), sizeof(double) * core_particle_size * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_acceleration_arr, h_core_acceleration_arr.data(), sizeof(double) * core_particle_size * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_net_force_arr, h_core_net_force_arr.data(), sizeof(double) * core_particle_size * NDIM, cudaMemcpyHostToDevice);
    // Removed: cudaMemcpy for d_core_neighbor_bound_damage_arr
    cudaMemcpy(d_total_local_particle_neighbors_arr, h_total_local_particle_neighbors_arr.data(), sizeof(int) * total_particle_size * MAX_NEIGHBOR_CAPACITY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_neighbor_sizes_arr, h_total_local_particle_neighbor_sizes_arr.data(), sizeof(int) * total_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_volume_arr, h_total_local_particle_volume_arr.data(), sizeof(double) * total_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_initial_positions_arr, h_total_local_particle_initial_positions_arr.data(), sizeof(double) * total_particle_size * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_current_positions_arr, h_total_local_particle_current_positions_arr.data(), sizeof(double) * total_particle_size * NDIM, cudaMemcpyHostToDevice); 

    cudaDeviceSynchronize(); // Ensure all data is copied before launching the kernel

    // Launch the kernel with proper grid configuration
    int blockSize = 256; // Adjust as needed - typical values are 128, 256, 512
    int numBlocks = (core_particle_size + blockSize - 1) / blockSize;
    
    if (core_particle_size > 0) {
        compute_velocity_kernel_GPU<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY><<<numBlocks, blockSize>>>(
            ndim, n1, n2, horizon, dx, massDensity,
            d_stiffness_tensor,
            stepSize,
            core_particle_size,
            d_core_particle_local_ID_arr,
            d_core_velocity_arr,
            d_core_acceleration_arr,
            d_core_net_force_arr,
            // Removed: d_core_neighbor_bound_damage_arr,
            total_particle_size,
            d_total_local_particle_neighbors_arr,
            d_total_local_particle_neighbor_sizes_arr,
            d_total_local_particle_core_ID_arr,
            d_total_local_particle_volume_arr,
            d_total_local_particle_initial_positions_arr,
            d_total_local_particle_current_positions_arr
        );

        // Check for kernel launch errors
        cudaError_t kernelError = cudaGetLastError();
        if (kernelError != cudaSuccess) {
            std::cout << "CUDA kernel launch error: " << cudaGetErrorString(kernelError) << std::endl;
            // Free memory before returning
            goto cleanup;
        }

        // Wait for kernel to complete
        cudaDeviceSynchronize();

        // Check for kernel execution errors
        cudaError_t kernelExecError = cudaGetLastError();
        if (kernelExecError != cudaSuccess) {
            std::cout << "CUDA kernel execution error: " << cudaGetErrorString(kernelExecError) << std::endl;
            // Free memory before returning
            goto cleanup;
        }
    }

    // Copy results back from device to host
    cudaMemcpy(h_core_velocity_arr.data(), d_core_velocity_arr, sizeof(double) * core_particle_size * NDIM, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_core_acceleration_arr.data(), d_core_acceleration_arr, sizeof(double) * core_particle_size * NDIM, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_core_net_force_arr.data(), d_core_net_force_arr, sizeof(double) * core_particle_size * NDIM, cudaMemcpyDeviceToHost);

    // Update the original data structures with results from GPU
    for (int i = 0; i < core_particle_size; ++i)
    {
        for (int j = 0; j < NDIM; ++j)
        {
            velocity[i][j] = static_cast<long double>(h_core_velocity_arr[i * NDIM + j]);
            acce[i][j] = static_cast<long double>(h_core_acceleration_arr[i * NDIM + j]);
            netF[i][j] = static_cast<long double>(h_core_net_force_arr[i * NDIM + j]);
        }
    }

cleanup:
    // Free GPU memory
    cudaFree(d_stiffness_tensor);
    cudaFree(d_core_particle_local_ID_arr);
    cudaFree(d_total_local_particle_core_ID_arr);
    cudaFree(d_core_velocity_arr);
    cudaFree(d_core_acceleration_arr);
    cudaFree(d_core_net_force_arr);
    // Removed: cudaFree(d_core_neighbor_bound_damage_arr);
    cudaFree(d_total_local_particle_neighbors_arr);
    cudaFree(d_total_local_particle_neighbor_sizes_arr);
    cudaFree(d_total_local_particle_volume_arr);
    cudaFree(d_total_local_particle_initial_positions_arr);
    cudaFree(d_total_local_particle_current_positions_arr);
}