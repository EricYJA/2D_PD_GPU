// Yuan: This is specifically for the GPU version of the function computeVelocity

// may not need the full duplicate of Particle struct
// Note: use structure of arrays (SoA) for GPU compatibility
// This struct is for demonstration purposes only, which is not used in the code
struct Particle_GPU
{

    int globalID;
    int partitionID;
    long double volume;
    double damageStatus;
    int ndim;
    long double *initialPositions; // size = ndim
    long double *currentPositions; // size = ndim
    int *neighbors;                // size = neighborSize
    int *nneighbors;               // TODO: size = ?
};

struct Matrix_GPU
{
    int n_rows; // typically n_rows is the same as n_cols for our computation
    int n_cols;
    long double *data; // size = rows*cols
};

template <int NDIM>
__device__ void computeShapeTensors(long double *shape_tensor0, // Return Val, size = ndim*ndim in matrix format
                                    long double *shape_tensor1, // Return Val, size = ndim*ndim in matrix format
                                    double n1,
                                    double n2,
                                    double dx,
                                    double horizon,
                                    int n_piNeighbors,
                                    long double *piNeighbors_volume_arr,   // size = n_piNeighbors
                                    long double *piNeighbors_init_pos_arr, // size = n_piNeighbors*ndim
                                    long double *piNeighbors_curr_pos_arr, // size = n_piNeighbors*ndim
                                    long double *pi_init_pos,              // size = ndim
                                    long double *pi_curr_pos,              // size = ndim
                                    long double *pj_init_pos,              // size = ndim
                                    long double *pj_curr_pos)              // size = ndim
{
    double length2 = 0.0;
    double lengthNb2 = 0.0;
    double length = 0.0;
    long double bondIJ[NDIM];
    long double bondINbcurrent[NDIM];
    long double bondINb[NDIM];

    for (int i = 0; i < NDIM; i++)
    {
        bondIJ[i] = pj_curr_pos[i] - pi_curr_pos[i];
        length2 += bondIJ[i] * bondIJ[i];
    }

    length = sqrt(length2);
    for (int nidx = 0; nidx < n_piNeighbors; nidx++)
    {
        lengthNb2 = 0.0;
        double numerator = 0.0;

        for (int i = 0; i < NDIM; ++i)
        {
            bondINb[i] = piNeighbors_curr_pos_arr[nidx * NDIM + i] - pi_curr_pos[i];
            bondINbcurrent[i] = piNeighbors_curr_pos_arr[nidx * NDIM + i] - pj_curr_pos[i];
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
                shape_tensor0[i * NDIM + j] += weight * bondINb[i] * bondINb[j] * piNeighbors_volume_arr[nidx];
                shape_tensor1[i * NDIM + j] += weight * bondINbcurrent[i] * bondINb[j] * piNeighbors_volume_arr[nidx];
            }
        }
    }
}