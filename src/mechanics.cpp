#include "mechanics.h"
#include "matrix.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

void applyDispBC(int ndim, vector<int>& boundarySet, vector<long double>& dispBC, vector<Particle>& localParticles, vector<vector<long double>>& velocity, vector<vector<long double>>& acce){
    for (int i = 0; i < boundarySet.size(); ++i){
        int pi = boundarySet[i];
        for (int j = 0; j < ndim; ++j){
            localParticles[pi].currentPositions[j] = localParticles[pi].initialPositions[j] + dispBC[j];
            // cout << " dispbC " << dispBC[j] << endl;
            // cout << " local particles pi " << pi << " currentPositions[j] " << localParticles[pi].currentPositions[j] << endl;
            velocity[pi][j] = 0.0;
            acce[pi][j] = 0.0;
        }
    }
}

void applyVelocityBC(int ndim, vector<int>& boundarySet, vector<long double> v0, vector<vector<long double>>& velocity, vector<vector<long double>>& acce){
    for (int i = 0; i < boundarySet.size(); ++i){
        int pi = boundarySet[i];
        for (int j = 0; j < ndim; ++j){
            velocity[pi][j] = v0[j];
            acce[pi][j] = 0.0;
        }
    }
}

void applyFixedBC(int ndim, vector<int>& boundarySet, vector<Particle>& localParticles){

    for (int i = 0; i < boundarySet.size(); ++i){
        int pi = boundarySet[i];
        for (int j = 0; j < ndim; j++){
            localParticles[pi].currentPositions[j] = localParticles[pi].initialPositions[j];
        }
    }
}


void buildLocalNeighborlist(int rank, int ndim, double dx, double horizon, vector<Particle>& localParticles, vector<Particle>& ghostParticles, const unordered_map<int, int>& globalLocalIDmap, const map<int, int>& globalPartitionIDmap, 
                            const unordered_map<int, int>& globalGhostIDmap, vector<vector<Particle*>>& Neighborslist){
        
    //build Neighborslist for local particles;
    for (int i = 0; i < localParticles.size(); ++i){
        vector<Particle*> piNeighbors;
        for (const int nb : localParticles[i].neighbors){

            if (globalPartitionIDmap.at(nb) == rank){
                int localNbId = globalLocalIDmap.at(nb);
                piNeighbors.push_back(&localParticles[localNbId]);
               
            }

            else if(globalGhostIDmap.find(nb) != globalGhostIDmap.end()){
                int ghostID = globalGhostIDmap.at(nb);
                piNeighbors.push_back(&ghostParticles[ghostID]);

            }
            else{
                cout << "Rank " << rank << " Error in buildLocalNeighborlist: Key " << nb << " not found in either  globalLocalIDmap or globalGhostIDmap." << endl;
                cout << " pi = " << localParticles[i].globalID << endl;
            }

        }
        
        Neighborslist.push_back(piNeighbors);    

    }

}


vector<matrix> computeShapeTensors(int ndim, double n1, double n2, double dx, double horizon, vector<Particle*>& piNeighbors, Particle& pi, Particle& pj){
    
    vector<matrix> shapeTensors = {
        matrix(ndim, vector<long double>(ndim*ndim, 0.0)),
        matrix(ndim, vector<long double>(ndim*ndim, 0.0))
    };

    double length2 = 0.0, lengthNb2;
    vector<long double> bondIJ(ndim), bondINbcurrent(ndim), bondINb(ndim);

    for (int i = 0; i < ndim; ++i){
        bondIJ[i] = pj.initialPositions[i] - pi.initialPositions[i];
        length2 += pow(bondIJ[i], 2);
    }

    double length = sqrt(length2);
    
    // Debug output for shape tensor computation - trace first particle
    bool debug_this = false;
    if (debug_this) {
        cout << "CPU SHAPE DEBUG: pi=[" << pi.initialPositions[0] << "," << pi.initialPositions[1] 
             << "], pj=[" << pj.initialPositions[0] << "," << pj.initialPositions[1] << "]" << endl;
        cout << "CPU SHAPE DEBUG: bondIJ=[" << bondIJ[0] << "," << bondIJ[1] << "], length=" << length << endl;
        cout << "CPU SHAPE DEBUG: neighbor_count=" << piNeighbors.size() << endl;
    }
    
    int neighbor_idx = 0;
    for (const Particle* nb : piNeighbors){

        lengthNb2 = 0.0;
        double numerator = 0.0;

        for (int i = 0; i < ndim; ++i){
            bondINb[i] =  nb->initialPositions[i] - pi.initialPositions[i];
            bondINbcurrent[i] = nb->currentPositions[i] - pi.currentPositions[i];
            lengthNb2 += pow(bondINb[i], 2);
            numerator += bondIJ[i] * bondINb[i];
        }

        double lengthNb = sqrt(lengthNb2);

        //calculate the cos angle

        double cosAngle = numerator / (length * lengthNb);
        if (cosAngle > 1.0) cosAngle = 1.0; else if (cosAngle < -1.0) cosAngle = -1.0;

        double lengthRatio = abs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

        // Debug first few neighbors for detailed analysis
        if (debug_this && neighbor_idx < 3) {
            cout << "CPU SHAPE DEBUG: neighbor " << neighbor_idx << ": nb_pos=[" << nb->initialPositions[0] 
                 << "," << nb->initialPositions[1] << "]" << endl;
            cout << "CPU SHAPE DEBUG: neighbor " << neighbor_idx << ": bondINb=[" << bondINb[0] 
                 << "," << bondINb[1] << "], lengthNb=" << lengthNb << endl;
            cout << "CPU SHAPE DEBUG: neighbor " << neighbor_idx << ": cosAngle=" << cosAngle 
                 << ", weight=" << weight << ", volume=" << nb->volume << endl;
            cout << "CPU SHAPE DEBUG: neighbor " << neighbor_idx << ": contribution=[" << scientific << setprecision(6)
                 << weight * bondINb[0] * bondINb[0] * nb->volume << ","
                 << weight * bondINb[0] * bondINb[1] * nb->volume << ","
                 << weight * bondINb[1] * bondINb[0] * nb->volume << ","
                 << weight * bondINb[1] * bondINb[1] * nb->volume << "]" << endl;
        }

        for (int k = 0; k < ndim; ++k){
            for (int l = 0; l < ndim; ++l){
                shapeTensors[0].elements[k][l] += weight * bondINb[k] * bondINb[l] * nb->volume;
                shapeTensors[1].elements[k][l] += weight * bondINbcurrent[k] * bondINb[l] * nb->volume;
            }
        }
        
        neighbor_idx++;
    }
    
    // Debug final shape tensor values
    if (debug_this) {
        cout << "CPU SHAPE DEBUG: final shapeRef=[" << scientific << setprecision(6)
             << shapeTensors[0].elements[0][0] << "," << shapeTensors[0].elements[0][1] << ","
             << shapeTensors[0].elements[1][0] << "," << shapeTensors[0].elements[1][1] << "]" << endl;
        cout << "CPU SHAPE DEBUG: final shapeCur=[" << scientific << setprecision(6)
             << shapeTensors[1].elements[0][0] << "," << shapeTensors[1].elements[0][1] << ","
             << shapeTensors[1].elements[1][0] << "," << shapeTensors[1].elements[1][1] << "]" << endl;
    }

    return shapeTensors;
}

vector<long double> StrainVector(const matrix& strain){
    vector<long double> strainV;
    if(strain.ndim == 2){
        strainV.resize(3);
        strainV[0] = strain.elements[0][0];
        strainV[1] = strain.elements[1][1];
        strainV[2] = strain.elements[0][1];
    }

    if(strain.ndim == 3){
        strainV.resize(6);
        strainV[0] = strain.elements[0][0];
        strainV[1] = strain.elements[1][1];
        strainV[2] = strain.elements[2][2];
        strainV[3] = strain.elements[1][2];
        strainV[4] = strain.elements[0][2];
        strainV[5] = strain.elements[0][1];
    }

    return strainV;
}

matrix getStiffnessTensor(int ndim, double E, double nv){
    double preFactor = 0.0;
    matrix StiffnessTensor;
    if (ndim == 2){
        //asume plane stress case
        preFactor = E / (1 - pow(nv, 2));
        StiffnessTensor = matrix(3, {1.0, nv, 0.0, nv, 1.0, 0.0, 0.0, 0.0, 1 - nv});
        StiffnessTensor = StiffnessTensor.timeScalar(preFactor);
    }
    else if (ndim == 3){
        preFactor = E / (1 + nv) / (1 - 2 * nv);
        StiffnessTensor = matrix(6, {1.0 - nv, nv, nv, 0.0, 0.0, 0.0,
                                     nv, 1.0 - nv, nv, 0.0, 0.0, 0.0,
                                     nv, nv, 1.0 - nv, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 1 - 2 * nv, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 1 - 2 * nv, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 1 - 2 * nv});
        StiffnessTensor = StiffnessTensor.timeScalar(preFactor);
    }

    return StiffnessTensor;
}


matrix computeStressTensor(matrix& shapeRef, matrix& shapeCur, int ndim, matrix& StiffnessTensor, vector<vector<double>>& bondDamage, int piIndex, int pjIndex){

    matrix stress;
    matrix deformationGradient = shapeCur.timeMatrix(shapeRef.inverse2D());

    // Debug output for particle 0 - trace each step
    if (piIndex == 0) {
        cout << "CPU PARTICLE 0 SHAPE DEBUG: shapeRef=[" << shapeRef.elements[0][0] << "," << shapeRef.elements[0][1] << "," << shapeRef.elements[1][0] << "," << shapeRef.elements[1][1] << "]" << endl;
        cout << "CPU PARTICLE 0 SHAPE DEBUG: shapeCur=[" << shapeCur.elements[0][0] << "," << shapeCur.elements[0][1] << "," << shapeCur.elements[1][0] << "," << shapeCur.elements[1][1] << "]" << endl;
        matrix shapeRefInv = shapeRef.inverse2D();
        cout << "CPU PARTICLE 0 SHAPE DEBUG: shapeRefInv=[" << shapeRefInv.elements[0][0] << "," << shapeRefInv.elements[0][1] << "," << shapeRefInv.elements[1][0] << "," << shapeRefInv.elements[1][1] << "]" << endl;
        cout << "CPU PARTICLE 0 DEFORM DEBUG: deformGrad=[" << deformationGradient.elements[0][0] << "," << deformationGradient.elements[0][1] << "," << deformationGradient.elements[1][0] << "," << deformationGradient.elements[1][1] << "]" << endl;
    }

    matrix Imatrix = matrix(ndim, vector<long double> (ndim*ndim, 0.0));
    for(int i = 0; i < ndim; ++i) {Imatrix.elements[i][i] = 1.0;}

    matrix strain = (deformationGradient.transpose()).matrixAdd(deformationGradient);
    strain = strain.timeScalar(0.5);
    strain = strain.matrixSub(Imatrix);

    vector<long double> strainV = StrainVector(strain);
    vector<long double> stressV = StiffnessTensor.timeVector(strainV);

    // Debug output for particle 0 specifically
    if (piIndex == 0) {
        cout << "CPU PARTICLE 0 STRESS DEBUG: strain=[" << strain.elements[0][0] << "," << strain.elements[0][1] << "," << strain.elements[1][0] << "," << strain.elements[1][1] << "]" << endl;
        cout << "CPU PARTICLE 0 STRESS DEBUG: strainV=[" << strainV[0] << "," << strainV[1] << "," << strainV[2] << "]" << endl;
        cout << "CPU PARTICLE 0 STRESS DEBUG: stiffness=[" << StiffnessTensor.elements[0][0] << "," << StiffnessTensor.elements[0][1] << "," << StiffnessTensor.elements[0][2] << ";" << StiffnessTensor.elements[1][0] << "," << StiffnessTensor.elements[1][1] << "," << StiffnessTensor.elements[1][2] << ";" << StiffnessTensor.elements[2][0] << "," << StiffnessTensor.elements[2][1] << "," << StiffnessTensor.elements[2][2] << "]" << endl;
        cout << "CPU PARTICLE 0 STRESS DEBUG: stressV=[" << stressV[0] << "," << stressV[1] << "," << stressV[2] << "]" << endl;
    }


    double d = 0; // bondDamage[piIndex][pjIndex]; debug use

    if(ndim == 2){
        stress = matrix(2, {stressV[0], stressV[2],
                            stressV[2], stressV[1]});
    }
    else if(ndim == 3){
        stress = matrix(3, {stressV[0], stressV[5], stressV[4],
                            stressV[5], stressV[1], stressV[3],
                            stressV[4], stressV[3], stressV[2]});
    }
    stress.timeScalar(1.0 - d);

    return stress;
}


vector<long double> computeForceDensityStates(int ndim, int rank, double n1, double n2, double horizon, matrix& StiffnessTensor, double dx, vector<Particle*>& piNeighbors, Particle& pi, Particle& pj, const unordered_map<int, int>& globalLocalIDmap, vector<vector<double>>& bondDamage){

    matrix Tmatrix = matrix(ndim, vector<long double>(ndim * ndim, 0.0));
    Particle pnb;
    int localNbId;
    int index = 0;

    vector<long double> bondIJ(ndim), bondINb(ndim);
    double length2 = 0.0, lengthNb2, numerator;

    for (int i = 0; i < ndim; ++i){
        bondIJ[i] = pj.initialPositions[i] - pi.initialPositions[i];
        length2 += pow(bondIJ[i], 2);
    }

    double length = sqrt(length2), horizonVolume = 0.0;
    
    int piIndex;
    if (pi.partitionID == rank) {
        piIndex = globalLocalIDmap.at(pi.globalID);
    }
    else{
        piIndex = bondDamage.size() - 1;
    }

    int pjIndex = 0;

    for (Particle* nb : piNeighbors){

        lengthNb2 = 0.0;
        numerator = 0.0;
        for (int i = 0; i < ndim; ++i){
            bondINb[i] =  nb->initialPositions[i] - pi.initialPositions[i];
            lengthNb2 += pow(bondINb[i], 2);
            numerator += bondIJ[i] * bondINb[i];
        }

        double lengthNb = sqrt(lengthNb2);

        double cosAngle = numerator / (length * lengthNb);
        if (cosAngle > 1.0) cosAngle = 1.0; else if (cosAngle < -1.0) cosAngle = -1.0;

        double lengthRatio = abs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

        vector<matrix> shapeTensors = computeShapeTensors(ndim, n1, n2, dx, horizon, piNeighbors, pi, *nb);

        matrix stress = computeStressTensor(shapeTensors[0], shapeTensors[1], ndim, StiffnessTensor, bondDamage, piIndex, pjIndex);
        if (pi.partitionID == rank) {pjIndex += 1;}

        // Debug output for shape tensors and stress
        if (pi.globalID == 0 && nb->globalID == 1) {
            cout << "CPU DEBUG: shape_tensor0=[" << shapeTensors[0].elements[0][0] << "," << shapeTensors[0].elements[0][1] << "," << shapeTensors[0].elements[1][0] << "," << shapeTensors[0].elements[1][1] << "]" << endl;
            cout << "CPU DEBUG: stress=[" << stress.elements[0][0] << "," << stress.elements[0][1] << "," << stress.elements[1][0] << "," << stress.elements[1][1] << "]" << endl;
            cout << "CPU DEBUG: weight=" << weight << ", volume=" << nb->volume << endl;
        }

        Tmatrix = Tmatrix.matrixAdd((stress.timeMatrix(shapeTensors[0].inverse2D())).timeScalar(weight * nb->volume));
        horizonVolume += nb->volume;
    }

    // Debug output for Tmatrix computation for first particle pair
    if (pi.globalID == 0 && pj.globalID == 1) {
        cout << "CPU DEBUG: Tmatrix=[" << Tmatrix.elements[0][0] << "," << Tmatrix.elements[0][1] << "," << Tmatrix.elements[1][0] << "," << Tmatrix.elements[1][1] << "]" << endl;
        cout << "CPU DEBUG: bondIJ=[" << bondIJ[0] << "," << bondIJ[1] << "], horizonVolume=" << horizonVolume << endl;
        
        // Manual calculation verification
        double manual_x = (Tmatrix.elements[0][0] / horizonVolume) * bondIJ[0] + (Tmatrix.elements[0][1] / horizonVolume) * bondIJ[1];
        double manual_y = (Tmatrix.elements[1][0] / horizonVolume) * bondIJ[0] + (Tmatrix.elements[1][1] / horizonVolume) * bondIJ[1];
        cout << "CPU DEBUG: manual_Tvector=[" << manual_x << "," << manual_y << "]" << endl;
    }

    vector<long double> Tvector = (Tmatrix.timeScalar(1.0 / horizonVolume)).timeVector(bondIJ);

    // Debug output for final Tvector for first particle pair
    if (pi.globalID == 0 && pj.globalID == 1) {
        cout << "CPU DEBUG: final_Tvector=[" << Tvector[0] << "," << Tvector[1] << "]" << endl;
    }

    return Tvector;

}


void updateGhostParticlePositions(int ndim, int rank, vector<Particle>& ghostParticles, const vector<Particle>& localParticles, const unordered_map<int, int>& globalLocalIDmap, const unordered_map<int, int>& globalGhostIDmap, MPI_Comm comm){

    set<int> neighborRanks;
    map<int, vector<int>> particlesRequestVec;

    for (const auto& p : ghostParticles){
        neighborRanks.insert(p.partitionID);
        particlesRequestVec[p.partitionID].push_back(p.globalID);

    }

    map<int, vector<int>> particlesSend;
    for (int nbrank : neighborRanks){
        vector<int> sendparticles = particlesRequestVec[nbrank];
        int sendsize  = sendparticles.size(), recvsize;
        // exchange send size
        MPI_Sendrecv(&sendsize, 1, MPI_INT, nbrank, 0,
                     &recvsize, 1, MPI_INT, nbrank, 0, comm, MPI_STATUS_IGNORE);

        //exchange particle IDs
        vector<int> recvparticles(recvsize);
        MPI_Sendrecv(sendparticles.data(), sendsize, MPI_INT, nbrank, 1,
                     recvparticles.data(), recvsize, MPI_INT, nbrank, 1, comm, MPI_STATUS_IGNORE);

        particlesSend[nbrank] = recvparticles;

    }

    map<int, vector<long double>> ghostPositions;
    for (int nbrank : neighborRanks){
        vector<long double> sendParticlesPositions;
        for (const auto& pid : particlesSend[nbrank]){
            Particle particle = localParticles[globalLocalIDmap.at(pid)];
            sendParticlesPositions.push_back(particle.globalID);
            sendParticlesPositions.insert(sendParticlesPositions.end(), particle.currentPositions.begin(), particle.currentPositions.end());
        }


        // exchange send size
        int sendPdataSize = sendParticlesPositions.size(), recvPdataSize;
        MPI_Sendrecv(&sendPdataSize, 1, MPI_INT, nbrank, 2,
                     &recvPdataSize, 1, MPI_INT, nbrank, 2, comm,  MPI_STATUS_IGNORE);

        // exchange particles data
        vector<long double> recvParticlesPositions(recvPdataSize);
        MPI_Sendrecv(sendParticlesPositions.data(), sendPdataSize, MPI_LONG_DOUBLE, nbrank, 3,
                     recvParticlesPositions.data(), recvPdataSize, MPI_LONG_DOUBLE, nbrank, 3, comm, MPI_STATUS_IGNORE);



        int idx = 0;
        while (idx < recvParticlesPositions.size()){
            int k = recvParticlesPositions[idx++];
            vector<long double> ghostPosition(recvParticlesPositions.begin() + idx, recvParticlesPositions.begin() + idx + ndim);
            ghostPositions[k] = ghostPosition;
            idx += ndim;

        }

    }

    for (const auto& [id, position] : ghostPositions){
        int localID = globalGhostIDmap.at(id);
        ghostParticles[localID].currentPositions = position;
    }

}


void updatePositions(int ndim, vector<Particle>& localParticles, long double stepSize, long double massDensity, vector<vector<long double>>& velocity, vector<vector<long double>>& acce, vector<vector<long double>>& netF){
    for (int i = 0; i < localParticles.size(); ++i){
        for(int m = 0; m < ndim; ++m){
            acce[i][m] = netF[i][m] / massDensity;
            localParticles[i].currentPositions[m] += velocity[i][m] * stepSize + 0.5 * acce[i][m] * pow(stepSize, 2);
        }
    }
}


void computeVelocity(int rank, int ndim, double n1, double n2, double horizon, double dx, long double massDensity, matrix& StiffnessTensor, long double stepSize, vector<vector<long double>>& velocity, vector<vector<Particle*>>& Neighborslist,
                     vector<vector<long double>>& acce, vector<vector<long double>>& netF, vector<Particle>& localParticles, vector<Particle>& ghostParticles, 
                     const unordered_map<int, int>& globalLocalIDmap, const map<int, int>& globalPartitionIDmap, const unordered_map<int, int>& globalGhostIDmap, vector<vector<double>>& bondDamage){
    
    vector<long double> forceIJ, forceJI;

    for (int i = 0; i < localParticles.size(); ++i){

        Particle pi = localParticles[i];
        vector<Particle*>& piNeighbors = Neighborslist[i];
        vector<long double> acceNew(ndim, 0.0);
        
        //compute netForce
        vector<long double> netForce(ndim, 0.0);

        for (Particle* nb : piNeighbors){

            //build pjNeighbors
            vector<Particle*> pjNeighbors;
            if (nb->partitionID == rank){

                int localID = globalLocalIDmap.at(nb->globalID);
                pjNeighbors = Neighborslist[localID];

            }else{
                
                for (auto& nnb : nb->neighbors){
                    if (globalPartitionIDmap.at(nnb) == rank){
                        int localNnbId = globalLocalIDmap.at(nnb);
                        pjNeighbors.push_back(&localParticles[localNnbId]);
                    }
                    else if(globalGhostIDmap.find(nnb) != globalGhostIDmap.end()){
                        int ghostnnbID = globalGhostIDmap.at(nnb);
                        pjNeighbors.push_back(&ghostParticles[ghostnnbID]);
                    }
                    else{
                        cout << "Error in build pjNeighbors: cannot find " << nnb << endl;
                        cout << " pi = " << pi.globalID << " at rank " << rank << endl;
                        cout << " nb = " << nb->globalID << endl;
                    }
                }
            }
            
            // for (auto& nnb : nb->neighbors){
            //     if (globalPartitionIDmap.at(nnb) == rank){
            //         int localNnbId = globalLocalIDmap.at(nnb);
            //         pjNeighbors.push_back(&localParticles[localNnbId]);
            //     }
            //     else{
            //         int ghostnnbID = globalGhostIDmap.at(nnb);
            //         pjNeighbors.push_back(&ghostParticles[ghostnnbID]);
            //     }
            // }

            forceIJ = computeForceDensityStates(ndim, rank, n1, n2, horizon, StiffnessTensor, dx, piNeighbors, pi, *nb, globalLocalIDmap, bondDamage);
            forceJI = computeForceDensityStates(ndim, rank, n1, n2, horizon, StiffnessTensor, dx, pjNeighbors, *nb, pi, globalLocalIDmap, bondDamage);

            // Debug output for particle 0 (index 0)
            if (i == 0 && nb->globalID <= 5) {
                cout << "CPU DEBUG P0: pi=" << pi.globalID << ", pj=" << nb->globalID << endl;
                cout << "CPU DEBUG P0: forceIJ=[" << forceIJ[0] << "," << forceIJ[1] << "]" << endl;
                cout << "CPU DEBUG P0: forceJI=[" << forceJI[0] << "," << forceJI[1] << "]" << endl;
                cout << "CPU DEBUG P0: volume=" << nb->volume << endl;
                cout << "CPU DEBUG P0: force_contrib=[" << (forceIJ[0] - forceJI[0]) * nb->volume << "," << (forceIJ[1] - forceJI[1]) * nb->volume << "]" << endl;
            }

            pjNeighbors.clear();

            for (int j = 0; j < ndim; ++j) {
                netForce[j] += (forceIJ[j] - forceJI[j]) * nb->volume;
            }
        }

        netF[i] = netForce;

        // Debug output for particle 0 final results
        if (i == 0) {
            cout << "CPU DEBUG P0: final_netF=[" << netF[i][0] << "," << netF[i][1] << "]" << endl;
            cout << "CPU DEBUG P0: acceleration=[" << netF[i][0] / massDensity << "," << netF[i][1] / massDensity << "]" << endl;
        }

        // netF[i] = computeNetForce(rank, ndim, horizon, dx, localParticles, ghostParticles,
        //           globalLocalIDmap, globalPartitionIDmap, globalGhostIDmap, pi);

        for(int m = 0; m < ndim; ++m){
            acceNew[m] = netF[i][m] / massDensity;
            velocity[i][m] += 0.5 * (acce[i][m] + acceNew[m]) * stepSize;
        }
    
    }

}


void computeDamageStatus(int ndim, double n1, double n2, double dx, double horizon, vector<vector<double>>& bondDamageThreshold, vector<Particle>& localParticles, const vector<vector<Particle*>>& Neighborslist, vector<vector<double>>& bondDamage, const unordered_map<int, int>& globalLocalIDmap){

    //Mazars Damage model is implemented
    //model parameters
    double alpha_t, alpha_c, d_t = 0.0, d_c = 0.0, d = 0.0;

    //material parameters
    double a_t = 0.87, b_t = 20000, a_c = 0.65, b_c = 2150;

    int piIndex = 0;
    bool damageOccur = false;
    for (auto& pi : localParticles){
        
        int pjIndex = 0;
        
        vector<Particle*> piNeighbors = Neighborslist[piIndex];
        
        pi.damageStatus = 0.0;
        

        for (Particle* nb : piNeighbors){
            alpha_t = 0.0, alpha_c = 0.0;
            vector<matrix> shapeTensors = computeShapeTensors(ndim, n1, n2, dx, horizon, piNeighbors, pi, *nb);
            matrix deformationGradient = shapeTensors[1].timeMatrix(shapeTensors[0].inverse2D());

            matrix Imatrix = matrix(ndim, vector<long double> (ndim*ndim, 0.0));
            for(int i = 0; i < ndim; ++i) {Imatrix.elements[i][i] = 1.0;}

            matrix strain = (deformationGradient.transpose()).matrixAdd(deformationGradient);
            strain = strain.timeScalar(0.5);
            strain = strain.matrixSub(Imatrix);

            Eigen::MatrixXd strain_temp(ndim, ndim);
            Eigen::VectorXd strain_eigenvalues(ndim);
            Eigen::MatrixXd strain_eigenvectors(ndim, ndim);

            Eigen::MatrixXd strain_t(ndim, ndim);
            Eigen::VectorXd strain_t_eigenvalues(ndim);
            Eigen::MatrixXd strain_t_eigenvectors(ndim, ndim);

            Eigen::MatrixXd strain_c(ndim, ndim);
            Eigen::VectorXd strain_c_eigenvalues(ndim);
            Eigen::MatrixXd strain_c_eigenvectors(ndim, ndim);

            for(int i = 0; i < ndim; ++i){
                for(int j = 0; j < ndim; ++j){
                    strain_temp(i,j) = strain.elements[i][j]; 
                }
            }
            
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
            solver.compute(strain_temp);
            strain_eigenvalues = solver.eigenvalues();
            strain_eigenvectors = solver.eigenvectors();

            //compute the equivalent strain
            double strain2 = 0.0, strain_eq = 0.0;
            for (int i = 0; i < ndim; ++i){
                if (strain_eigenvalues(i) > 0.0){
                    strain2 += pow(strain_eigenvalues(i), 2);
                }
            }
            if (strain2 > 0.0) strain_eq = sqrt(strain2);

            //if(pi.globalID == 220) {cout << "strain_eq = " << strain_eq << endl;}
            
            if(strain_eq > bondDamageThreshold[piIndex][pjIndex]){
                damageOccur = true;
                double damageThreshold = bondDamageThreshold[piIndex][pjIndex];
                //compute the strain_t and strain_c
                for (int i = 0; i < ndim; ++i){
                    if (strain_eigenvalues(i) > 0.0){
                        for (int j = 0; j < ndim; ++j){
                            for (int k = 0; k < ndim; ++k){
                                strain_t(j,k) += strain_eigenvalues(i) * strain_eigenvectors(j,i) * strain_eigenvectors(k,i);
                            }
                        }
                    }
                    else if (strain_eigenvalues(i) < 0.0){
                        for (int j = 0; j < ndim; ++j){
                            for (int k = 0; k < ndim; ++k){
                                strain_c(j,k) += strain_eigenvalues(i) * strain_eigenvectors(j,i) * strain_eigenvectors(k,i);
                            }
                        }
                    }
                }

                //compute the eigen values of strain_t and strain_c
                solver.compute(strain_t);
                strain_t_eigenvalues = solver.eigenvalues();

                solver.compute(strain_c);
                strain_c_eigenvalues = solver.eigenvalues();

                //compute alpha_t and alpha_c
                for (int i = 0; i < ndim; ++i){
                    if(strain_t_eigenvalues(i) > 0.0){
                        alpha_t += strain_t_eigenvalues(i) * (strain_t_eigenvalues(i) + strain_c_eigenvalues(i)) / (strain_eq * strain_eq);
                    }
                    if(strain_c_eigenvalues(i) > 0.0){
                        alpha_c += strain_c_eigenvalues(i) * (strain_t_eigenvalues(i) + strain_c_eigenvalues(i)) / (strain_eq * strain_eq);
                    }
                }

                //compute d_t and d_c and d
                d_t = 1.0 - damageThreshold * (1 - a_t) / strain_eq - a_t / exp(b_t * (strain_eq - damageThreshold));
                d_c = 1.0 - damageThreshold * (1 - a_c) / strain_eq - a_c / exp(b_c * (strain_eq - damageThreshold));

                d = alpha_t * d_t + alpha_c * d_c;
                
                bondDamage[piIndex][pjIndex] = d;
                bondDamageThreshold[piIndex][pjIndex] = strain_eq;

            }

            pi.damageStatus += bondDamage[piIndex][pjIndex] / pi.neighbors.size();
            pjIndex += 1;
            
        }
        
        piIndex++;

    }
    bondDamage[piIndex].resize(30, 0.0); //to avoid the sengmentation caused by the ghost particle.
    damageOccur = false;
}