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

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<Particle> globalParticles;
    vector<Particle> ghostParticles;
    vector<Particle> localParticles;

    int numParticlesRow = 30, ndim = 2;
    long double dx = 1.0, horizon = 3.0;
    double n1 = 10.0, n2 = 10.0;
    vector<long double> boxlimit(2*ndim);


    vector<idx_t> vtxdist(size + 1);
    vector<idx_t> xadj;
    vector<idx_t> adjncy;

    map<int, vector<idx_t>> xadjMap;
    map<int, vector<idx_t>> adjncyMap;

    if (rank == 0){
        //globalParticles = computeInitialPositions(numParticlesRow, dx, ndim, boxlimit);
        globalParticles = readMeshFile(boxlimit, dx);
        findNeighbor(globalParticles, dx, horizon, ndim);
        cout << "findNeighbor completed." << endl;
        idx_t numVertices = globalParticles.size();
        defineVtxdist(numVertices, size, vtxdist, comm);
        buildGraph(globalParticles, size, vtxdist, xadjMap, adjncyMap);
        cout << "buildGraph completed." << endl;
        //outputParticles(rank, globalParticles);
        //for(int i = 0; i < boxlimit.size(); ++i){ cout << " boxlimit " << i << " : " <<  boxlimit[i] << endl;}
    }

    //distribute the boxlimit for determining boundary particles at each processor
    MPI_Bcast(boxlimit.data(), 2*ndim, MPI_LONG_DOUBLE, 0, comm);
    //if(rank == size - 1) for(int i = 0; i < boxlimit.size(); ++i){ cout << " boxlimit " << i << " : " <<  boxlimit[i] << endl;}

    //distribute the vtxdist and send local xadj, adjncy
    MPI_Bcast(vtxdist.data(), size + 1, MPI_INT, 0, comm);
    distributeGraph(xadjMap, adjncyMap, size, rank, xadj, adjncy, comm);
    cout << "distributeGraph completed." << endl;

    int localNumVertices = vtxdist[rank+1] - vtxdist[rank];
    idx_t *part = new idx_t[localNumVertices];

    map<int, int> globalPartitionIDmap;
    partitionGraph(rank, vtxdist, xadj, adjncy, size, part, comm);
    updatePartitionID(globalParticles, rank, size, part, localNumVertices, globalPartitionIDmap, comm);

  
    if(rank == 0){
        cout << "updatePartitionID completed." << endl;
    }


    distributeParticles(rank, size, globalParticles, localParticles, comm);
    int localNumParticles = localParticles.size();
    if(rank == 0){
        cout << "distributeParticles completed." << endl;
    }

    //define the particle ID map (global->local)
    unordered_map<int, int> globalLocalIDmap;
    defineParticleIDmap(localParticles, globalLocalIDmap);
    //outputGlobalLocalIDmap(rank, localParticles, globalLocalIDmap);


    //remove the globalParticles and release memory
    if (rank == 0) {
        globalParticles.clear(); // Removes all elements
        cout << "Rank 0: globalParticles deleted." << endl;
        cout << "Rank 0: globalParticles size after deletion = " << globalParticles.size() << endl;
    }

    createGhostParticles(localParticles, ghostParticles, globalLocalIDmap, globalPartitionIDmap, rank, comm);
    if (rank == 0) {
        cout << "createGhostParticles completed." << endl;
    }

    unordered_map<int, int> globalGhostIDmap;
    defineParticleIDmap(ghostParticles, globalGhostIDmap);

    //outputParticles(rank, localParticles);
    //outputGhostParticles(rank, ghostParticles);

    //identify boundary particles
    vector<set<int>> boundarySet(2 * ndim);
    defineBoundarySet(ndim, dx, boxlimit, boundarySet, localParticles);


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

    buildLocalNeighborlist(rank, ndim, dx, horizon, localParticles, ghostParticles, globalLocalIDmap, globalPartitionIDmap, globalGhostIDmap, Neighborslist);
    MPI_Barrier(comm);
    //outputBuildLocalNeighborList(rank, Neighborslist);

    cout << "buildLocalNeighborlist completed." << endl;

    //initialize bondDamageThreshold and bondDamage

    for (int piIndex = 0; piIndex < localNumParticles; ++piIndex){
        bondDamageThreshold[piIndex].resize(Neighborslist[piIndex].size(), damageThreshold);
        bondDamage[piIndex].resize(Neighborslist[piIndex].size(), 0.0);
    }
    bondDamage[localNumParticles].resize(30, 0.0); //to avoid the sengmentation caused by the ghost particle.

    cout << "initialize bondDamageThreshold and bondDamage completed." << endl;

    // check data structue size
    if (rank == 0) {
        cout << "localParticles size: " << localParticles.size() << endl;
        cout << "ghostParticles size: " << ghostParticles.size() << endl;
        cout << "Neighborslist size: " << Neighborslist.size() << endl;
        cout << "bondDamageThreshold size: " << bondDamageThreshold.size() << endl;
        cout << "bondDamage size: " << bondDamage.size() << endl;
    }


    for (int j = 0; j < totalSteps; ++j){

        if (rank == 0){
            cout << "--------time step---------" << j << "--------" << endl;
        }
    
        if (rank == 0) {
            double t0 = MPI_Wtime();
            computeDamageStatus(ndim, n1, n2, dx, horizon, bondDamageThreshold, localParticles, Neighborslist, bondDamage, globalLocalIDmap);
            double t1 = MPI_Wtime();
            cout << "computeDamageStatus: " << (t1 - t0) << " sec" << endl;
        } else {
            cout << "only 1 process is allowed to computeDamageStatus" << endl;
        }
    
        if (boundaryTop.size() > 0) {
            if (rank == 0) {
                double t0 = MPI_Wtime();
                applyVelocityBC(ndim, boundaryTop, vt, velocity, acce);
                double t1 = MPI_Wtime();
                cout << "applyVelocityBC (Top): " << (t1 - t0) << " sec" << endl;
            } else {
                cout << "only 1 process is allowed to applyVelocityBC (Top)" << endl;
            }
        }
    
        if (rank == 0) {
            double t0 = MPI_Wtime();
            updatePositions(ndim, localParticles, stepSize, massDensity, velocity, acce, netF);
            double t1 = MPI_Wtime();
            cout << "updatePositions: " << (t1 - t0) << " sec" << endl;
        } else {
            cout << "only 1 process is allowed to updatePositions" << endl;
        }
    
        if (rank == 0) {
            double t0 = MPI_Wtime();
            applyFixedBC(ndim, boundaryBottom, localParticles);
            double t1 = MPI_Wtime();
            cout << "applyFixedBC: " << (t1 - t0) << " sec" << endl;
        } else {
            cout << "only 1 process is allowed to applyFixedBC" << endl;
        }
    
        MPI_Barrier(comm);
    
        if (rank == 0) {
            double t0 = MPI_Wtime();
            updateGhostParticlePositions(ndim, rank, ghostParticles, localParticles, globalLocalIDmap, globalGhostIDmap, comm);
            double t1 = MPI_Wtime();
            cout << "updateGhostParticlePositions: " << (t1 - t0) << " sec" << endl;
        } else {
            cout << "only 1 process is allowed to updateGhostParticlePositions" << endl;
        }
    
        MPI_Barrier(comm);
    
        if (rank == 0) {
            storeVelocity(ndim, velocity, "./output/velocity_step_0_before.txt");

            double t0 = MPI_Wtime();
            // computeVelocity(rank, ndim, n1, n2, horizon, dx, massDensity, StiffnessTensor, stepSize, velocity, Neighborslist, acce, netF,
            //                 localParticles, ghostParticles, globalLocalIDmap, globalPartitionIDmap, globalGhostIDmap, bondDamage);
            compute_velocity_GPU_host(rank, ndim, n1, n2, horizon, dx, massDensity, StiffnessTensor, stepSize, velocity, Neighborslist, acce, netF,
                            localParticles, ghostParticles, globalLocalIDmap, globalPartitionIDmap, globalGhostIDmap, bondDamage);

            double t1 = MPI_Wtime();
            cout << "computeVelocity: " << (t1 - t0) << " sec" << endl;

            storeVelocity(ndim, velocity, "./output/velocity_step_0_after.txt");
        } else {
            cout << "only 1 process is allowed to computeVelocity" << endl;
        }
    
        MPI_Barrier(comm);
    
        // if (j % 5000 == 0) {
            if (rank == 0) {
                double t0 = MPI_Wtime();
                outputGatheredPositions(rank, size, ndim, j, localParticles, comm);
                double t1 = MPI_Wtime();
                cout << "outputGatheredPositions: " << (t1 - t0) << " sec" << endl;
            } else {
                outputGatheredPositions(rank, size, ndim, j, localParticles, comm);
            }
        // }

        break; // Remove this line to run all time steps
    }

    MPI_Finalize();
    return 0;
}
