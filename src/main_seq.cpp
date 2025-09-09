#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <iomanip>
#include <Eigen/Dense>
#include "util.h"
#include "matrix.h"
#include "SetupParticleSystem.h"

using namespace std;

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


void buildLocalNeighborlist(int ndim, double dx, double horizon, vector<Particle>& localParticles, const unordered_map<int, int>& globalLocalIDmap, vector<vector<Particle*>>& Neighborslist){
        
    //build Neighborslist for local particles;
    for (int i = 0; i < localParticles.size(); ++i){
        vector<Particle*> piNeighbors;
        for (const int nb : localParticles[i].neighbors){
            // Since this is sequential, all neighbors should be in localParticles
            if (globalLocalIDmap.find(nb) != globalLocalIDmap.end()){
                int localNbId = globalLocalIDmap.at(nb);
                piNeighbors.push_back(&localParticles[localNbId]);
            }
            else{
                cout << "Error in buildLocalNeighborlist: Key " << nb << " not found in globalLocalIDmap." << endl;
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

    // PARTICLE PAIR DEBUG: Print stress tensor function entry
    if (piIndex == 0) {
        cout << "CPU ENTRY computeStressTensor: pi_local=" << piIndex << ", pj_local(neighbor_idx)=" << pjIndex << endl;
    }

    matrix stress;
    
    // DEBUG: Print matrix multiplication inputs
    if (piIndex == 0) {
        matrix shapeRefInv = shapeRef.inverse2D();
        cout << "CPU MATRIX DEBUG: shapeCur=[" << scientific << setprecision(12) 
             << shapeCur.elements[0][0] << "," << shapeCur.elements[0][1] << "," 
             << shapeCur.elements[1][0] << "," << shapeCur.elements[1][1] << "]" << endl;
        cout << "CPU MATRIX DEBUG: shapeRefInv=[" << scientific << setprecision(12) 
             << shapeRefInv.elements[0][0] << "," << shapeRefInv.elements[0][1] << "," 
             << shapeRefInv.elements[1][0] << "," << shapeRefInv.elements[1][1] << "]" << endl;
    }
    
    matrix deformationGradient = shapeCur.timeMatrix(shapeRef.inverse2D());

    // Debug output for particle 0 - trace each step
    // if (piIndex == 0) {
    //     cout << "CPU PARTICLE 0 SHAPE DEBUG: shapeRef=[" << shapeRef.elements[0][0] << "," << shapeRef.elements[0][1] << "," << shapeRef.elements[1][0] << "," << shapeRef.elements[1][1] << "]" << endl;
    //     cout << "CPU PARTICLE 0 SHAPE DEBUG: shapeCur=[" << shapeCur.elements[0][0] << "," << shapeCur.elements[0][1] << "," << shapeCur.elements[1][0] << "," << shapeCur.elements[1][1] << "]" << endl;
    //     matrix shapeRefInv = shapeRef.inverse2D();
    //     cout << "CPU PARTICLE 0 SHAPE DEBUG: shapeRefInv=[" << shapeRefInv.elements[0][0] << "," << shapeRefInv.elements[0][1] << "," << shapeRefInv.elements[1][0] << "," << shapeRefInv.elements[1][1] << "]" << endl;
    //     cout << "CPU PARTICLE 0 DEFORM DEBUG: deformGrad=[" << deformationGradient.elements[0][0] << "," << deformationGradient.elements[0][1] << "," << deformationGradient.elements[1][0] << "," << deformationGradient.elements[1][1] << "]" << endl;
    // }

    matrix Imatrix = matrix(ndim, vector<long double> (ndim*ndim, 0.0));
    for(int i = 0; i < ndim; ++i) {Imatrix.elements[i][i] = 1.0;}

    matrix strain = (deformationGradient.transpose()).matrixAdd(deformationGradient);
    strain = strain.timeScalar(0.5);
    strain = strain.matrixSub(Imatrix);

    vector<long double> strainV = StrainVector(strain);
    vector<long double> stressV = StiffnessTensor.timeVector(strainV);

    // Debug output for particle 0 specifically
    // if (piIndex == 0) {
    //     cout << "CPU PARTICLE 0 STRESS DEBUG: strain=[" << strain.elements[0][0] << "," << strain.elements[0][1] << "," << strain.elements[1][0] << "," << strain.elements[1][1] << "]" << endl;
    //     cout << "CPU PARTICLE 0 STRESS DEBUG: strainV=[" << strainV[0] << "," << strainV[1] << "," << strainV[2] << "]" << endl;
    //     cout << "CPU PARTICLE 0 STRESS DEBUG: stiffness=[" << StiffnessTensor.elements[0][0] << "," << StiffnessTensor.elements[0][1] << "," << StiffnessTensor.elements[0][2] << ";" << StiffnessTensor.elements[1][0] << "," << StiffnessTensor.elements[1][1] << "," << StiffnessTensor.elements[1][2] << ";" << StiffnessTensor.elements[2][0] << "," << StiffnessTensor.elements[2][1] << "," << StiffnessTensor.elements[2][2] << "]" << endl;
    //     cout << "CPU PARTICLE 0 STRESS DEBUG: stressV=[" << stressV[0] << "," << stressV[1] << "," << stressV[2] << "]" << endl;
    // }


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


vector<long double> computeForceDensityStates(int ndim, double n1, double n2, double horizon, matrix& StiffnessTensor, double dx, vector<Particle*>& piNeighbors, Particle& pi, Particle& pj, const unordered_map<int, int>& globalLocalIDmap, vector<vector<double>>& bondDamage){

    // PARTICLE PAIR DEBUG: Print function entry
    if (pi.globalID == 0) {
        cout << "CPU CALL computeForceDensityStates: pi_local=" << globalLocalIDmap.at(pi.globalID) << "(global=" << pi.globalID << ") -> pj_local=" << globalLocalIDmap.at(pj.globalID) << "(global=" << pj.globalID << ")" << endl;
    }

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
    
    // Since this is sequential, all particles are local
    int piIndex = globalLocalIDmap.at(pi.globalID);
    int pjIndex = 0;

    int neighbor_idx = 0;
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

        // PARTICLE PAIR DEBUG: Print shape tensor computation call
        if (pi.globalID == 0) {
            cout << "CPU PAIR PROCESSING: pi=" << pi.globalID << "(local=" << globalLocalIDmap.at(pi.globalID) << ") -> pj=" << pj.globalID << "(local=" << globalLocalIDmap.at(pj.globalID) << "), processing neighbor " << neighbor_idx << ": nb=" << nb->globalID << "(local=" << globalLocalIDmap.at(nb->globalID) << ")" << endl;
        }
        
        if (pi.globalID == 0) {
            cout << "CPU CALL computeShapeTensors: pi_local=" << globalLocalIDmap.at(pi.globalID) << "(global=" << pi.globalID << ") -> nb_local=" << globalLocalIDmap.at(nb->globalID) << "(global=" << nb->globalID << ")" << endl;
        }
        
        vector<matrix> shapeTensors = computeShapeTensors(ndim, n1, n2, dx, horizon, piNeighbors, pi, *nb);

        // PARTICLE PAIR DEBUG: Print shape tensor values
        if (pi.globalID == 0) {
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": nb_shape_tensor0=[" << scientific << setprecision(12) 
                 << shapeTensors[0].elements[0][0] << "," << shapeTensors[0].elements[0][1] << "," 
                 << shapeTensors[0].elements[1][0] << "," << shapeTensors[0].elements[1][1] << "]" << endl;
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": nb_shape_tensor1=[" << scientific << setprecision(12) 
                 << shapeTensors[1].elements[0][0] << "," << shapeTensors[1].elements[0][1] << "," 
                 << shapeTensors[1].elements[1][0] << "," << shapeTensors[1].elements[1][1] << "]" << endl;
        }

        // PARTICLE PAIR DEBUG: Print stress tensor computation call  
        if (pi.globalID == 0) {
            cout << "CPU CALL computeStressTensor: pi_local=" << globalLocalIDmap.at(pi.globalID) << "(global=" << pi.globalID << "), neighbor_idx=" << neighbor_idx << endl;
        }
        
        matrix stress = computeStressTensor(shapeTensors[0], shapeTensors[1], ndim, StiffnessTensor, bondDamage, piIndex, pjIndex);
        pjIndex += 1;

        // Debug output for detailed neighbor comparison with GPU
        if (pi.globalID == 0 && neighbor_idx < 5 && pj.globalID == 1) {
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": global_id=" << nb->globalID << endl;
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": pi_pos=[" << pi.initialPositions[0] << "," << pi.initialPositions[1] << "], nb_pos=[" << nb->initialPositions[0] << "," << nb->initialPositions[1] << "]" << endl;
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": bondINb=[" << bondINb[0] << "," << bondINb[1] << "], lengthNb=" << lengthNb << endl;
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": weight=" << weight << ", volume=" << nb->volume << endl;
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": shape_tensor0=[" << shapeTensors[0].elements[0][0] << "," << shapeTensors[0].elements[0][1] << "," << shapeTensors[0].elements[1][0] << "," << shapeTensors[0].elements[1][1] << "]" << endl;
            cout << "CPU DEBUG P0 NEIGHBOR " << neighbor_idx << ": stress=[" << stress.elements[0][0] << "," << stress.elements[0][1] << "," << stress.elements[1][0] << "," << stress.elements[1][1] << "]" << endl;
        }

        Tmatrix = Tmatrix.matrixAdd((stress.timeMatrix(shapeTensors[0].inverse2D())).timeScalar(weight * nb->volume));
        horizonVolume += nb->volume;
        neighbor_idx++;
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


// updateGhostParticlePositions function removed - not needed for sequential execution


void updatePositions(int ndim, vector<Particle>& localParticles, long double stepSize, long double massDensity, vector<vector<long double>>& velocity, vector<vector<long double>>& acce, vector<vector<long double>>& netF){
    for (int i = 0; i < localParticles.size(); ++i){
        for(int m = 0; m < ndim; ++m){
            acce[i][m] = netF[i][m] / massDensity;
            localParticles[i].currentPositions[m] += velocity[i][m] * stepSize + 0.5 * acce[i][m] * pow(stepSize, 2);
        }
    }
}


void computeVelocity(int ndim, double n1, double n2, double horizon, double dx, long double massDensity, matrix& StiffnessTensor, long double stepSize, vector<vector<long double>>& velocity, vector<vector<Particle*>>& Neighborslist,
                     vector<vector<long double>>& acce, vector<vector<long double>>& netF, vector<Particle>& localParticles, 
                     const unordered_map<int, int>& globalLocalIDmap, vector<vector<double>>& bondDamage){
    
    vector<long double> forceIJ, forceJI;

    for (int i = 0; i < localParticles.size(); ++i){

        Particle pi = localParticles[i];
        vector<Particle*>& piNeighbors = Neighborslist[i];
        vector<long double> acceNew(ndim, 0.0);
        
        //compute netForce
        vector<long double> netForce(ndim, 0.0);

        for (Particle* nb : piNeighbors){

            //build pjNeighbors - since sequential, all particles are local
            vector<Particle*> pjNeighbors;
            int localID = globalLocalIDmap.at(nb->globalID);
            pjNeighbors = Neighborslist[localID];

            // PARTICLE PAIR DEBUG: Print forceIJ computation call
            if (i == 0) {
                cout << "=== CPU VELOCITY LOOP: Processing pi=" << pi.globalID << "(local=" << i << ") -> pj=" << nb->globalID << "(local=" << globalLocalIDmap.at(nb->globalID) << ") ===" << endl;
            }
            
            if (i == 0) {
                cout << "CPU CALL forceIJ: pi_local=" << i << "(global=" << pi.globalID << ") -> pj_local=" << globalLocalIDmap.at(nb->globalID) << "(global=" << nb->globalID << ")" << endl;
            }
            
            forceIJ = computeForceDensityStates(ndim, n1, n2, horizon, StiffnessTensor, dx, piNeighbors, pi, *nb, globalLocalIDmap, bondDamage);
            
            // PARTICLE PAIR DEBUG: Print forceJI computation call
            if (i == 0) {
                cout << "CPU CALL forceJI: pj_local=" << globalLocalIDmap.at(nb->globalID) << "(global=" << nb->globalID << ") -> pi_local=" << i << "(global=" << pi.globalID << ")" << endl;
            }
            
            forceJI = computeForceDensityStates(ndim, n1, n2, horizon, StiffnessTensor, dx, pjNeighbors, *nb, pi, globalLocalIDmap, bondDamage);

            // Debug output for particle 0 (index 0)
            if (i == 0) {
                cout << "CPU DEBUG P0: pi=" << pi.globalID << ", pj=" << nb->globalID << endl;
                cout << "CPU DEBUG P0: forceIJ=[" << scientific << setprecision(6) << forceIJ[0] << "," << forceIJ[1] << "]" << endl;
                cout << "CPU DEBUG P0: forceJI=[" << scientific << setprecision(6) << forceJI[0] << "," << forceJI[1] << "]" << endl;
                cout << "CPU DEBUG P0: volume=" << scientific << setprecision(6) << nb->volume << endl;
                cout << "CPU DEBUG P0: force_contrib=[" << scientific << setprecision(6) << (forceIJ[0] - forceJI[0]) * nb->volume << "," << (forceIJ[1] - forceJI[1]) * nb->volume << "]" << endl;
            }

            pjNeighbors.clear();

            for (int j = 0; j < ndim; ++j) {
                netForce[j] += (forceIJ[j] - forceJI[j]) * nb->volume;
            }
        }

        netF[i] = netForce;

        // Debug output for particle 0 final results
        if (i == 0) {
            cout << "CPU DEBUG P0: final_netF=[" << scientific << setprecision(6) << netF[i][0] << "," << netF[i][1] << "]" << endl;
            cout << "CPU DEBUG P0: acceleration=[" << scientific << setprecision(6) << netF[i][0] / massDensity << "," << netF[i][1] / massDensity << "]" << endl;
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
    
        computeVelocity(ndim, n1, n2, horizon, dx, massDensity, StiffnessTensor, stepSize, velocity, Neighborslist, acce, netF,
                        allParticles, globalLocalIDmap, bondDamage);
        
        storeVelocity(ndim, velocity, "./output/velocity_step_0_after_seq_ref.txt");
    
        if (j % 5000 == 0) {
            outputLocalParticlesPositions(0, allParticles, ndim, j);
        }

        break; // Remove this line to run all time steps
    }

    return 0;
}
