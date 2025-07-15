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
#include "matrix.h"
#include "SetupParticleSystem.h"
#include "util.h"

// TODO: host only access for now, need a way for device access
static const int INVALID_INT = std::numeric_limits<int>::max();
static const double INVALID_DOUBLE = std::numeric_limits<double>::max();

__device__ void Mat_GPU_inverse2D(long double *inv,      // Return Val, size = 2*2 in matrix format
                                  long double *elements) // size = 2*2 in matrix format
{
    // compute determinant
    double det = elements[0] * elements[3] - elements[1] * elements[2];

    assert(det != 0.0); // Ensure the matrix is invertible

    inv[0] = elements[3] / det;  // inv[0][0]
    inv[1] = -elements[1] / det; // inv[0][1]
    inv[2] = -elements[2] / det; // inv[1][0]
    inv[3] = elements[0] / det;  // inv[1][1]
}

// TODO: 3D inverse function

__device__ void Mat_GPU_mul_mat(long double *C,
                                const long double *A,
                                const long double *B,
                                int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double sum = 0.0L;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__device__ void Mat_GPU_mul_vec(long double *C,
                                const long double *A,
                                const long double *B,
                                int N)
{
    for (int i = 0; i < N; ++i)
    {
        double sum = 0.0L;
        for (int j = 0; j < N; ++j)
        {
            sum += A[i * N + j] * B[j];
        }
        C[i] = sum;
    }
}

__device__ void Mat_GPU_add(long double *C,
                            const long double *A,
                            const long double *B,
                            int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

template <int NDIM>
__device__ void computeShapeTensors_GPU(long double *shape_tensor0, // Return Val, size = ndim*ndim in matrix format
                                        long double *shape_tensor1, // Return Val, size = ndim*ndim in matrix format
                                        double n1,
                                        double n2,
                                        double dx,
                                        double horizon,
                                        int pi_neighbor_local_size,
                                        int *pi_local_neighbor_arr,
                                        long double *total_local_particle_volume_arr,
                                        long double *total_local_particle_initial_positions_arr,
                                        long double *total_local_particle_current_positions_arr,
                                        long double *pi_init_pos, // size = ndim
                                        long double *pi_curr_pos, // size = ndim
                                        long double *pj_init_pos) // size = ndim
{
    double length2 = 0.0;
    double lengthNb2 = 0.0;
    double length = 0.0;
    long double bondIJ[NDIM];
    long double bondINbcurrent[NDIM];
    long double bondINb[NDIM];

    for (int i = 0; i < NDIM; i++)
    {
        bondIJ[i] = pj_init_pos[i] - pi_init_pos[i];
        length2 += bondIJ[i] * bondIJ[i];
    }

    length = sqrt(length2);
    for (int nidx = 0; nidx < pi_neighbor_local_size; nidx++)
    {
        int pi_nb_local_idx = pi_local_neighbor_arr[nidx];
        long double pi_nb_volume = total_local_particle_volume_arr[pi_nb_local_idx];
        long double *pi_nb_init_pos_arr = &total_local_particle_initial_positions_arr[pi_nb_local_idx * NDIM];
        long double *pi_nb_curr_pos_arr = &total_local_particle_current_positions_arr[pi_nb_local_idx * NDIM];
        lengthNb2 = 0.0;
        double numerator = 0.0;

        for (int i = 0; i < NDIM; ++i)
        {
            bondINb[i] = pi_nb_init_pos_arr[i] - pi_init_pos[i];
            bondINbcurrent[i] = pi_nb_curr_pos_arr[i] - pi_curr_pos[i];
            lengthNb2 += bondINb[i] * bondINb[i];
            numerator += bondIJ[i] * bondINb[i];
        }

        double lengthNb = sqrt(lengthNb2);

        double cosAngle = numerator / (length * lengthNb);
        cosAngle = fmin(-1.0, fmax(cosAngle, 1.0)); // Clamp to [-1, 1]

        double lengthRatio = abs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

        for (int i = 0; i < NDIM; ++i)
        {
            for (int j = 0; j < NDIM; ++j)
            {
                shape_tensor0[i * NDIM + j] += weight * bondINb[i] * bondINb[j] * pi_nb_volume;
                shape_tensor1[i * NDIM + j] += weight * bondINbcurrent[i] * bondINb[j] * pi_nb_volume;
            }
        }
    }
}

template <int NDIM, int STIFFNESS_TENSOR_SIZE, int MAX_NEIGHBOR_CAPACITY>
__device__ void computeStressTensor_GPU(long double *stress_tensor,    // Return Val, size = ndim*ndim in matrix
                                        long double *shape_ref_tensor, // size = ndim*ndim in matrix
                                        long double *shape_cur_tensor, // size = ndim*ndim in matrix
                                        long double *stiffness_tensor, // size = stiffness_tensor_size * stiffness_tensor_size in matrix
                                        long double *bond_damage_arr,
                                        int pi_local_Index,
                                        int pj_local_Index)
{
    long double deformationGradient[NDIM * NDIM];
    long double Imatrix[NDIM * NDIM] = {0.0L};
    long double strain[NDIM * NDIM];
    long double strain_vector[NDIM * NDIM]; // actually a vector of size stiffness_tensor_size
    long double stress_vector[NDIM * NDIM]; // actually a vector of size stiffness_tensor_size

    long double shape_ref_tensor_inv[NDIM * NDIM];
    Mat_GPU_inverse2D(shape_ref_tensor_inv, shape_ref_tensor);
    Mat_GPU_mul_mat(deformationGradient, shape_cur_tensor, shape_ref_tensor_inv, NDIM);

    for (int i = 0; i < NDIM; ++i)
    {
        Imatrix[i * NDIM + i] = 1.0L; // Initialize identity matrix
    }

    for (int i = 0; i < NDIM; ++i)
    {
        for (int j = 0; j < NDIM; ++j)
        {
            // matrix strain = (deformationGradient.transpose()).matrixAdd(deformationGradient);
            strain[i * NDIM + j] = 0.5L * (deformationGradient[i * NDIM + j] + deformationGradient[j * NDIM + i]) - Imatrix[i * NDIM + j];

            // stress_tensor = strain.timeScalar(0.5);
            stress_tensor[i * NDIM + j] = strain[i * NDIM + j] * 0.5L;

            // strain = strain.matrixSub(Imatrix);
            strain[i * NDIM + j] -= Imatrix[i * NDIM + j];
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

    double damage = 0; // bond_damage_arr[piIndex * MAX_NEIGHBOR_CAPACITY + pjIndex]; // TODO: set tp 0 for debugging

    // assert(damage != INVALIDlongdouble); // Ensure damage is valid

    if (NDIM == 2)
    {
        stress_tensor[0] = stress_vector[0] * (1.0L - damage); // s11
        stress_tensor[1] = stress_vector[2] * (1.0L - damage); // s12
        stress_tensor[2] = stress_vector[2] * (1.0L - damage); // s21
        stress_tensor[3] = stress_vector[1] * (1.0L - damage); // s22
    }
    else if (NDIM == 3)
    {
        stress_tensor[0] = stress_vector[0] * (1.0L - damage); // s11
        stress_tensor[1] = stress_vector[5] * (1.0L - damage); // s12
        stress_tensor[2] = stress_vector[4] * (1.0L - damage); // s13
        stress_tensor[3] = stress_vector[5] * (1.0L - damage); // s21
        stress_tensor[4] = stress_vector[1] * (1.0L - damage); // s22
        stress_tensor[5] = stress_vector[3] * (1.0L - damage); // s23
        stress_tensor[6] = stress_vector[4] * (1.0L - damage); // s31
        stress_tensor[7] = stress_vector[3] * (1.0L - damage); // s32
        stress_tensor[8] = stress_vector[2] * (1.0L - damage); // s33
    }

    // Note: The stress_tensor is now updated with the computed values
}

template <int NDIM, int STIFFNESS_TENSOR_SIZE, int MAX_NEIGHBOR_CAPACITY>
__device__ void computeForceDensityStates_GPU(long double *Tvector,
                                              int pi_local_idx,
                                              int pj_local_idx,
                                              double n1,
                                              double n2,
                                              double horizon,
                                              double dx,
                                              long double *stiffness_tensor, // size = STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE in matrix

                                              long double *core_neighbor_bound_damage_arr, // size = localParticleSize * max_neighbor_capacity

                                              /* Stores neighbor index as local index! */
                                              int total_local_particle_size,           // size = localParticleSize + ghostParticleSize
                                              int *total_local_particle_neighbors_arr, // size = localParticleSize + ghostParticleSize
                                              int *total_local_particle_neighbor_sizes_arr,
                                              int *total_local_particle_core_ID_arr,

                                              long double *total_local_particle_volume_arr,            // size = localParticleSize
                                              long double *total_local_particle_initial_positions_arr, // size = localParticleSize * ndim
                                              long double *total_local_particle_current_positions_arr) // size = localParticleSize * ndim
{
    // extract what's needed from Particle
    long double Tmatrix[NDIM * NDIM];

    long double bondIJ[NDIM];
    long double bondINb[NDIM];

    long double tmp_shape_tensor0[NDIM * NDIM] = {0.0L};
    long double tmp_shape_tensor1[NDIM * NDIM] = {0.0L};
    long double tmp_stress_tensor[NDIM * NDIM] = {0.0L};
    long double tmp_stress_shape_inv_prod[NDIM * NDIM] = {0.0L};

    double length2 = 0.0;
    double lengthNb2 = 0.0;
    double numerator = 0.0;
    double horizonVolume = 0.0;

    long double *pj_local_current_positions = &total_local_particle_current_positions_arr[pj_local_idx * NDIM];
    long double *pi_local_current_positions = &total_local_particle_current_positions_arr[pi_local_idx * NDIM];

    for (int i = 0; i < NDIM; ++i)
    {
        bondIJ[i] = pj_local_current_positions[i] - pi_local_current_positions[i];
        length2 += bondIJ[i] * bondIJ[i];
    }

    double length = sqrt(length2);

    // TODO revisit the piIndex query later
    int piIndex = 0;

    // TODO: revisit the pjIndex query later
    int pjIndex = 0;

    int pi_neighbor_local_size = total_local_particle_neighbor_sizes_arr[pi_local_idx];
    int *pi_local_neighbor_arr = &total_local_particle_neighbors_arr[pi_local_idx * MAX_NEIGHBOR_CAPACITY];

    long double *pi_local_initial_positions = &total_local_particle_initial_positions_arr[pi_local_idx * NDIM];

    for (int nidx = 0; nidx < pi_neighbor_local_size; ++nidx)
    {
        int pi_nb_local_idx = pi_local_neighbor_arr[nidx];
        long double pi_nb_volume = total_local_particle_volume_arr[pi_nb_local_idx];
        long double *pi_nb_local_initial_positions = &total_local_particle_initial_positions_arr[pi_nb_local_idx * NDIM];

        lengthNb2 = 0.0;
        numerator = 0.0;
        for (int i = 0; i < NDIM; ++i)
        {
            bondINb[i] = pi_nb_local_initial_positions[i] - pi_local_initial_positions[i];
            lengthNb2 += bondINb[i] * bondINb[i];
            numerator += bondIJ[i] * bondINb[i];
        }

        double lengthNb = sqrt(lengthNb2);

        double cosAngle = numerator / (length * lengthNb);
        cosAngle = fmin(-1.0, fmax(cosAngle, 1.0)); // Clamp to [-1, 1]

        double lengthRatio = abs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

        // Compute shape tensors
        computeShapeTensors_GPU<NDIM>(tmp_shape_tensor0, tmp_shape_tensor1,
                                      n1, n2, dx, horizon,
                                      pi_neighbor_local_size,
                                      pi_local_neighbor_arr,
                                      total_local_particle_volume_arr,
                                      total_local_particle_initial_positions_arr,
                                      total_local_particle_current_positions_arr,
                                      pi_local_initial_positions,
                                      pi_local_current_positions,
                                      pi_nb_local_initial_positions);

        computeStressTensor_GPU<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY>(
            tmp_stress_tensor,
            tmp_shape_tensor0,
            tmp_shape_tensor1,
            stiffness_tensor,
            core_neighbor_bound_damage_arr,
            piIndex, pjIndex); // TODO: piIndex and pjIndex are not used in this context as bound damage is skipped for now

        // stress.timeMatrix(shapeTensors[0].inverse2D())
        Mat_GPU_inverse2D(tmp_shape_tensor0, tmp_shape_tensor0);
        Mat_GPU_mul_mat(tmp_stress_shape_inv_prod, tmp_stress_tensor, tmp_shape_tensor0, NDIM);

        for (int i = 0; i < NDIM; ++i)
        {
            for (int j = 0; j < NDIM; ++j)
            {
                Tmatrix[i * NDIM + j] += tmp_stress_shape_inv_prod[i * NDIM + j] * weight * pi_nb_volume;
            }
        }
        horizonVolume += pi_nb_volume;
    }

    for (int i = 0; i < NDIM; ++i)
    {
        for (int j = 0; j < NDIM; ++j)
        {
            Tvector[i * NDIM + j] = Tmatrix[i * NDIM + j] / horizonVolume;
        }
    }
    Mat_GPU_mul_vec(Tvector, Tmatrix, bondIJ, NDIM);
}

template <int NDIM, int STIFFNESS_TENSOR_SIZE, int MAX_NEIGHBOR_CAPACITY>
__global__ void compute_velocity_kernel_GPU(int ndim,
                                            double n1,
                                            double n2,
                                            double horizon,
                                            double dx,
                                            long double massDensity,
                                            long double *stiffness_tensor, // size = STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE in matrix
                                            long double stepSize,

                                            int core_particle_size,
                                            int *core_particle_local_ID_arr, // size = localParticleSize

                                            long double *core_velocity_arr,              // size = localParticleSize * ndim
                                            long double *core_acceleration_arr,          // size = localParticleSize * ndim
                                            long double *core_net_force_arr,             // size = localParticleSize * ndim
                                            long double *core_neighbor_bound_damage_arr, // size = localParticleSize * max_neighbor_capacity

                                            /* Stores neighbor index as local index! */
                                            int total_local_particle_size,           // size = localParticleSize + ghostParticleSize
                                            int *total_local_particle_neighbors_arr, // size = localParticleSize + ghostParticleSize
                                            int *total_local_particle_neighbor_sizes_arr,
                                            int *total_local_particle_core_ID_arr,

                                            long double *total_local_particle_volume_arr,       // size = localParticleSize
                                            double *total_local_particle_initial_positions_arr, // size = localParticleSize * ndim
                                            double *total_local_particle_current_positions_arr) // size = localParticleSize * ndim
{
    int forceIJ[NDIM];
    int forceJI[NDIM];

    // traverse local particles in this partition
    for (int cp_idx = 0; cp_idx < core_particle_size; ++cp_idx)
    {
        int cp_local_idx = core_particle_local_ID_arr[cp_idx];
        int cp_neighbor_local_size = total_local_particle_neighbor_sizes_arr[cp_local_idx];

        assert(cp_neighbor_local_size > 0); // Ensure the particle is not boundary particle

        int *cp_local_neighbor_arr = &total_local_particle_neighbors_arr[cp_local_idx * MAX_NEIGHBOR_CAPACITY];

        long double acc_new[NDIM];
        long double net_force[NDIM];

        for (int cp_neighbor_idx = 0; cp_neighbor_idx < cp_neighbor_local_size; ++cp_neighbor_idx)
        {
            // get two hop neighbor list
            int cp_neighbor_local_idx = cp_local_neighbor_arr[cp_neighbor_idx];
            int cp_nb_neighbor_local_size = total_local_particle_neighbor_sizes_arr[cp_neighbor_local_idx];

            assert(cp_nb_neighbor_local_size > 0); // Ensure the neighbor is not boundary particle

            long double cp_neighbor_volume = total_local_particle_volume_arr[cp_neighbor_local_idx];

            int *cp_nb_neighbor_local_arr = &total_local_particle_neighbors_arr[cp_neighbor_local_idx * MAX_NEIGHBOR_CAPACITY];

            computeForceDensityStates_GPU<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY>(
                forceIJ, cp_local_idx, cp_neighbor_local_idx,
                n1, n2, horizon, dx,
                stiffness_tensor,
                core_neighbor_bound_damage_arr,
                total_local_particle_size,
                total_local_particle_neighbors_arr,
                total_local_particle_neighbor_sizes_arr,
                total_local_particle_core_ID_arr,
                total_local_particle_volume_arr,
                total_local_particle_initial_positions_arr,
                total_local_particle_current_positions_arr);

            computeForceDensityStates_GPU<NDIM, STIFFNESS_TENSOR_SIZE, MAX_NEIGHBOR_CAPACITY>(
                forceJI, cp_neighbor_local_idx, cp_local_idx,
                n1, n2, horizon, dx,
                stiffness_tensor,
                core_neighbor_bound_damage_arr,
                total_local_particle_size,
                total_local_particle_neighbors_arr,
                total_local_particle_neighbor_sizes_arr,
                total_local_particle_core_ID_arr,
                total_local_particle_volume_arr,
                total_local_particle_initial_positions_arr,
                total_local_particle_current_positions_arr);

            for (int i = 0; i < NDIM; ++i)
            {
                net_force[i] = (forceIJ[i] - forceJI[i]) * cp_neighbor_volume;
            }
        }

        for (int i = 0; i < NDIM; ++i)
        {
            core_net_force_arr[cp_local_idx * NDIM + i] = net_force[i];
        }

        for (int i = 0; i < NDIM; ++i)
        {
            // compute acceleration
            acc_new[i] = core_net_force_arr[cp_local_idx * NDIM + i] / massDensity;

            // update velocity
            core_velocity_arr[cp_local_idx * NDIM + i] += 0.5 * (core_acceleration_arr[cp_local_idx * NDIM + i] + acc_new[i]) * stepSize;
        }
    }
}

void compute_velocity_GPU_host(int rank,
                               int ndim,
                               double n1,
                               double n2,
                               double horizon,
                               double dx,
                               long double massDensity,
                               matrix &StiffnessTensor,
                               long double stepSize,
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
    for (int i = 0; i < total_particle_size; ++i)
    {
        Particle &p = totalParticles[i];
        p.globalID = i;                       // Assign global ID based on index in totalParticles
        new_globalLocalIDmap[p.globalID] = i; // Update the map with the new global ID
    }

    vector<vector<int>> totalParticleNeighborslist(total_particle_size); // use local idx
    vector<int> totalParticleNeighborSizes(total_particle_size, 0);

    // Fill the neighbor list for each particle
    for (int i = 0; i < total_particle_size; ++i)
    {
        int missing_neighbor_count = 0;
        Particle &p = totalParticles[i];
        for (int j = 0; j < p.neighbors.size(); ++j)
        {
            int neighbor_globalID = p.neighbors[j];
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

        if (missing_neighbor_count > 0 && missing_neighbor_count < p.neighbors.size())
        {
            // Handle the case where some neighbors are missing
            std::cout << "Warning: " << (i < core_particle_size ? "Core " : "Ghost ") << "Particle local ID" << i << " has " << missing_neighbor_count << " missing neighbors." << std::endl;
        }
        else if (missing_neighbor_count == p.neighbors.size())
        {
            // If all neighbors are missing, this particle is isolated
            std::cout << "Warning: " << (i < core_particle_size ? "Core " : "Ghost ") << "Particle local ID" << i << " is isolated with no valid neighbors." << std::endl;
        }


        if (missing_neighbor_count == 0) {
            totalParticleNeighborSizes[i] = p.neighbors.size(); // Update neighbor size only if no missing neighbors
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
    long double *d_stiffness_tensor;

    int *d_core_particle_local_ID_arr;
    int *d_total_local_particle_core_ID_arr;

    long double *d_core_velocity_arr;
    long double *d_core_acceleration_arr;
    long double *d_core_net_force_arr;
    long double *d_core_neighbor_bound_damage_arr;

    int *d_total_local_particle_neighbors_arr;
    int *d_total_local_particle_neighbor_sizes_arr;

    long double *d_total_local_particle_volume_arr;
    double *d_total_local_particle_initial_positions_arr;
    double *d_total_local_particle_current_positions_arr;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_stiffness_tensor, sizeof(double) * STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE);
    cudaMalloc((void **)&d_core_particle_local_ID_arr, sizeof(int) * core_particle_size);
    cudaMalloc((void **)&d_total_local_particle_core_ID_arr, sizeof(int) * total_particle_size);
    cudaMalloc((void **)&d_core_velocity_arr, sizeof(double) * core_particle_size * NDIM);
    cudaMalloc((void **)&d_core_acceleration_arr, sizeof(double) * core_particle_size * NDIM);
    cudaMalloc((void **)&d_core_net_force_arr, sizeof(double) * core_particle_size * NDIM);
    cudaMalloc((void **)&d_core_neighbor_bound_damage_arr, sizeof(double) * core_particle_size * MAX_NEIGHBOR_CAPACITY);
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
    vector<double> h_core_neighbor_bound_damage_arr(core_particle_size * MAX_NEIGHBOR_CAPACITY);

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

    // h_core_particle_local_ID_arr
    // h_total_local_particle_core_ID_arr
    for (int i = 0; i < total_particle_size; ++i) {
        h_total_local_particle_core_ID_arr[i] = INVALID_INT;
    }

    for (int i = 0; i < core_particle_size; ++i)
    {
        h_core_particle_local_ID_arr[i] = i;
        h_total_local_particle_core_ID_arr[i] = i;
    }

    // h_core_velocity_arr
    // h_core_acceleration_arr
    // h_core_net_force_arr
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

    // h_core_neighbor_bound_damage_arr
    for (int i = 0; i < core_particle_size; ++i)
    {
        Particle &p = localParticles[i];
        for (int j = 0; j < MAX_NEIGHBOR_CAPACITY; ++j)
        {
            if (j < p.neighbors.size())
            {
                int neighbor_globalID = p.neighbors[j];
                h_core_neighbor_bound_damage_arr[i * MAX_NEIGHBOR_CAPACITY + j] = bondDamage[i][j];
            }
            else
            {
                h_core_neighbor_bound_damage_arr[i * MAX_NEIGHBOR_CAPACITY + j] = INVALID_DOUBLE;
            }
        }
    }

    for (int i = 0; i < total_particle_size; ++i)
    {
        Particle &p = totalParticles[i];

        h_total_local_particle_volume_arr[i] = p.volume;

        for (int j = 0; j < NDIM; ++j)
        {
            h_total_local_particle_initial_positions_arr[i * NDIM + j] = p.initialPositions[j];
            h_total_local_particle_current_positions_arr[i * NDIM + j] = p.currentPositions[j];
        }

        for (int j = 0; j < p.neighbors.size(); ++j)
        {
            int neighbor_globalID = p.neighbors[j];
            if (new_globalLocalIDmap.find(neighbor_globalID) != new_globalLocalIDmap.end())
            {
                int neighbor_local_idx = new_globalLocalIDmap[neighbor_globalID];
                h_total_local_particle_neighbors_arr[i * MAX_NEIGHBOR_CAPACITY + j] = neighbor_local_idx;
            }
            else
            {
                h_total_local_particle_neighbors_arr[i * MAX_NEIGHBOR_CAPACITY + j] = INVALID_INT;
            }
        }

        h_total_local_particle_neighbor_sizes_arr[i] = totalParticleNeighborSizes[i];
    }

    // Copy data from host to device
    cudaMemcpy(d_stiffness_tensor, h_stiffness_tensor_arr.data(), sizeof(long double) * STIFFNESS_TENSOR_SIZE * STIFFNESS_TENSOR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_particle_local_ID_arr, h_core_particle_local_ID_arr.data(), sizeof(int) * core_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_core_ID_arr, h_total_local_particle_core_ID_arr.data(), sizeof(int) * total_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_velocity_arr, h_core_velocity_arr.data(), sizeof(long double) * core_particle_size * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_acceleration_arr, h_core_acceleration_arr.data(), sizeof(long double) * core_particle_size * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_net_force_arr, h_core_net_force_arr.data(), sizeof(long double) * core_particle_size * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_neighbor_bound_damage_arr, h_core_neighbor_bound_damage_arr.data(), sizeof(long double) * core_particle_size * MAX_NEIGHBOR_CAPACITY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_neighbors_arr, h_total_local_particle_neighbors_arr.data(), sizeof(int) * total_particle_size * MAX_NEIGHBOR_CAPACITY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_neighbor_sizes_arr, h_total_local_particle_neighbor_sizes_arr.data(), sizeof(int) * total_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_volume_arr, h_total_local_particle_volume_arr.data(), sizeof(long double) * total_particle_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_initial_positions_arr, h_total_local_particle_initial_positions_arr.data(), sizeof(double) * total_particle_size * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_local_particle_current_positions_arr, h_total_local_particle_current_positions_arr.data(), sizeof(double) * total_particle_size * NDIM, cudaMemcpyHostToDevice); 

    cudaDeviceSynchronize(); // Ensure all data is copied before launching the kernel

    // TODO: continue from here
    // Launch the kernel
}