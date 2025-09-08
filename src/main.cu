#include <iostream>
#include <mpi.h>
#include <vector>
#include <map>
#include <unordered_map>
#include "mechanics.h"
#include "mechanics.cuh"

using namespace std;


int main(int argc, char *argv[]){

    printf("Hello\n");

    vector<Particle> allParticles;

    int numParticlesRow = 30, ndim = 2;
    long double dx = 1.0, horizon = 3.0;
    double n1 = 10.0, n2 = 10.0;
    vector<long double> boxlimit(2*ndim);

    // Read mesh file and find neighbors - no partitioning needed for sequential
    //allParticles = computeInitialPositions(numParticlesRow, dx, ndim, boxlimit);
    allParticles = readMeshFile(boxlimit, dx);
    findNeighbor(allParticles, dx, horizon, ndim);
    cout << "findNeighbor completed." << endl;

    // Set all particles to partition 0 for sequential execution
    for (auto& particle : allParticles) {
        particle.partitionID = 0;
    }

    int localNumParticles = allParticles.size();
    cout << "Total number of particles: " << localNumParticles << endl;

    //define the particle ID map (global->local) - sequential case means global == local
    unordered_map<int, int> globalLocalIDmap;
    defineParticleIDmap(allParticles, globalLocalIDmap);


    // No ghost particles needed for sequential execution

    //identify boundary particles
    vector<set<int>> boundarySet(2 * ndim);
    defineBoundarySet(ndim, dx, boxlimit, boundarySet, allParticles);


    vector<vector<long double>> netF(localNumParticles, vector<long double>(ndim, 0.0)), velocity(localNumParticles, vector<long double>(ndim, 0.0)), acce(localNumParticles, vector<long double>(ndim, 0.0));
    vector<long double> dispBC(ndim, 0.0);
    vector<int> boundaryLeft(boundarySet[0].begin(), boundarySet[0].end());
    vector<int> boundaryRight(boundarySet[1].begin(), boundarySet[1].end());
    vector<int> boundaryBottom(boundarySet[2].begin(), boundarySet[2].end());
    vector<int> boundaryTop(boundarySet[3].begin(), boundarySet[3].end());
    vector<vector<Particle*>> Neighborslist;
    vector<vector<double>> bondDamage(localNumParticles + 1);
    vector<vector<double>> bondDamageThreshold(localNumParticles);

    vector<long double> vt = {0.0, 5.0};
    vector<long double> vb = {0.0, 0.0};

    //unit system used in the current simulation
    //  Length - mm
    //  Time - s
    //  Mass - Kg
    //  Stress - MPa

    long double totalTime = 1.0e-2, stepSize = 1.0e-9, massDensity = 2.5e-6;
    long long totalSteps = totalTime / stepSize;
    long double E = 3.5e4, nv = 0.2, tensileStrength = 3.0;
    long double damageThreshold = tensileStrength / E;
    
    matrix StiffnessTensor = getStiffnessTensor(ndim, E, nv);

    //cout << " totalSteps " << totalSteps << endl;

    buildLocalNeighborlist(ndim, dx, horizon, allParticles, globalLocalIDmap, Neighborslist);

    cout << "buildLocalNeighborlist completed." << endl;

    //initialize bondDamageThreshold and bondDamage

    for (int piIndex = 0; piIndex < localNumParticles; ++piIndex){
        bondDamageThreshold[piIndex].resize(Neighborslist[piIndex].size(), damageThreshold);
        bondDamage[piIndex].resize(Neighborslist[piIndex].size(), 0.0);
    }
    bondDamage[localNumParticles].resize(30, 0.0); //to avoid the sengmentation caused by the ghost particle.

    cout << "initialize bondDamageThreshold and bondDamage completed." << endl;

    for (int j = 0; j < totalSteps; ++j){

        cout << "--------time step---------" << j << "--------" << endl;
    
        computeDamageStatus(ndim, n1, n2, dx, horizon, bondDamageThreshold, allParticles, Neighborslist, bondDamage, globalLocalIDmap);
    
        if (boundaryTop.size() > 0) {
            applyVelocityBC(ndim, boundaryTop, vt, velocity, acce);
        }
    
        updatePositions(ndim, allParticles, stepSize, massDensity, velocity, acce, netF);
    
        applyFixedBC(ndim, boundaryBottom, allParticles);
    
        // No need to update ghost particles in sequential version
    
        compute_velocity_GPU_host(ndim, n1, n2, horizon, dx, massDensity, StiffnessTensor, stepSize, velocity, Neighborslist, acce, netF,
                        allParticles, globalLocalIDmap, bondDamage);
        
        storeVelocity(ndim, velocity, "./output/velocity_step_0_after_cuda.txt");
    
        if (j % 5000 == 0) {
            outputLocalParticlesPositions(0, allParticles, ndim, j);
        }

        break; // Remove this line to run all time steps
    }

    return 0;
}
