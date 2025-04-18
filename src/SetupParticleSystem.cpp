#include <math.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>
#include "SetupParticleSystem.h"

using namespace std;

vector<Particle> readMeshFile(vector<long double>& boxlimit, long double& dx){
    ifstream file("2D_Hassa_test_mesh.dat");
    if (!file.is_open()) {
        cerr << "Error: Could not open the mesh file '2D_Hassa_test_mesh.dat'" << endl;
    }
    vector<Particle> globalParticles;
    string line;

    //skip the header line
    getline(file, line);

    //read line by line
    int pid = 0;
    while (std::getline(file, line)) {
        istringstream iss(line);
        string temp;
        Particle p;
        long double x, y, volume;

        // Read the next three tokens (centerX, centerY, area)
        if (iss >> x >> y >> volume) {
            //check the distance to creat cracks
            if (fabs(x) > 17.5 && fabs(y) < 2.0){
                continue;
            }
            else{
                vector<long double> positions = {x, y};
                p.globalID = pid++;
                p.damageStatus = 0.0;
                p.initialPositions = positions;
                p.currentPositions = positions;
                p.volume = volume;
                p.partitionID = 0;
                globalParticles.push_back(p);

            }

        }
    }
    file.close();

    dx = 1.0;
    boxlimit[0] = -34.5;
    boxlimit[1] = 34.5;
    boxlimit[2] = -29.5;
    boxlimit[3] = 29.5;

    return globalParticles;

}

vector<Particle> computeInitialPositions(int numParticlesRow, long double dx, int ndim, vector<long double>& boxlimit){

    vector<Particle> globalParticles;
    int numParticles = pow(numParticlesRow, ndim);

    vector<long double> boxMin(ndim, 0.0);
    for(int i = 0; i < ndim; ++i){
        boxMin[i] = -1.0 * floor(numParticlesRow / 2) * dx;
        boxlimit[2 * i] = -1.0 * floor(numParticlesRow / 2) * dx;
        boxlimit[2 * i + 1] = boxlimit[2 * i] + (numParticlesRow - 1) * dx;
    }

    for(int i = 0; i < numParticles; ++i){

        Particle p;
        vector<int> dimIndex(ndim);
        dimIndex[0] = i % numParticlesRow; //index for x coordinate
        if(ndim > 1){ for(int j = 1; j < ndim; ++j) {dimIndex[j] = i / pow(numParticlesRow, j); }}//index for y /and z coordinate

        vector<long double> coordinate(ndim);
        for(int j = 0; j < ndim; ++j){
            coordinate[j] = boxMin[j] + dimIndex[j] * dx;
        }

        p.initialPositions = coordinate;
        p.currentPositions = coordinate;
        p.globalID = i;
        p.partitionID = 0;
        p.volume = pow(dx, ndim);

        globalParticles.push_back(p);

    }

return globalParticles;

}

void findNeighbor(vector<Particle>& globalParticles, long double dx, long double horizon, int ndim){

    for (int i = 0; i < globalParticles.size(); ++i){
        for (int j = 0; j < globalParticles.size(); ++j){
            if(i != j){
                vector<long double> relativePositions(ndim);
                long double distance2 = 0.0;
                for(int k = 0; k < ndim; ++k) {
                    relativePositions[k] = globalParticles[j].initialPositions[k] - globalParticles[i].initialPositions[k];
                    distance2 += relativePositions[k] * relativePositions[k];
                }

                long double distance = sqrt(distance2);
                if(distance < 1.01 * dx * horizon) {
                    globalParticles[i].neighbors.push_back(j);
                }
            }

        }

    }

    for (auto& pi : globalParticles){
        for (auto pj : pi.neighbors) {
            vector<int> pjnb = globalParticles[pj].neighbors;
            pi.nneighbors.insert(pjnb.begin(), pjnb.end());
            pi.nneighbors.erase(pj);
        }

        for (auto pj : pi.neighbors) {
            pi.nneighbors.erase(pj);
        }

        pi.nneighbors.erase(pi.globalID);
    }

}

vector<long double> serializeParticle(const Particle& p) {
    vector<long double> buffer;
    buffer.push_back(static_cast<long double>(p.globalID));
    buffer.push_back(static_cast<long double>(p.partitionID));
    buffer.push_back(p.volume);
    buffer.push_back(static_cast<long double>(p.initialPositions.size()));
    buffer.insert(buffer.end(), p.initialPositions.begin(), p.initialPositions.end());
    buffer.push_back(static_cast<long double>(p.currentPositions.size()));
    buffer.insert(buffer.end(), p.currentPositions.begin(), p.currentPositions.end());
    buffer.push_back(static_cast<long double>(p.neighbors.size()));
    for (int neighbor : p.neighbors) {
        buffer.push_back(static_cast<long double>(neighbor));
    }
    buffer.push_back(static_cast<long double>(p.nneighbors.size()));
    for (int nneighbor : p.nneighbors) {
        buffer.push_back(static_cast<long double>(nneighbor));
    }


    return buffer;
}

vector<Particle> deserializeParticles(const vector<long double>& buffer) {
    vector<Particle> particles;
    size_t idx = 0;

    //cout << "rank " << rank << " buffer size " << buffer.size() << endl;

    while(idx < buffer.size()){
        Particle p;
        p.globalID = static_cast<int>(buffer[idx++]);
        p.partitionID = static_cast<int>(buffer[idx++]);
        p.volume = buffer[idx++];
        // Deserialize initialPositions
        size_t initialSize = static_cast<size_t>(buffer[idx++]);
        p.initialPositions.assign(buffer.begin() + idx, buffer.begin() + idx + initialSize);
        idx += initialSize;

        // Deserialize currentPositions
        size_t currentSize = static_cast<size_t>(buffer[idx++]);
        p.currentPositions.assign(buffer.begin() + idx, buffer.begin() + idx + currentSize);
        idx += currentSize;

        // Deserialize neighbors
        size_t neighborSize = static_cast<size_t>(buffer[idx++]);
        p.neighbors.resize(neighborSize);
        for (size_t i = 0; i < neighborSize; ++i) {
            p.neighbors[i] = static_cast<int>(buffer[idx++]);
        }

        // Deserialize nneighbors
        size_t nneighborSize = static_cast<size_t>(buffer[idx++]);
        for (size_t i = 0; i < nneighborSize; ++i) {
            p.nneighbors.insert(static_cast<int>(buffer[idx++]));
        }

        particles.push_back(p);

    }

    //cout << "rank " << rank << " particles size " << particles.size() << endl;

    return particles;
}

void distributeParticles(int rank, int size, const vector<Particle>& globalParticles, vector<Particle>& localParticles, MPI_Comm comm){

    //Prepare serialized buffers
    vector<vector<long double>> serializedParticles(size);
    if (rank == 0) {
        for (const auto& p : globalParticles) {
            vector<long double> flatParticle = serializeParticle(p);
            for (const auto& val : flatParticle){
              serializedParticles[p.partitionID].push_back(val);
            }

        }
    }

    vector<long double> sendBuffer, recvBuffer;
    vector<int> sendDispls(size, 0);
    vector<int> sendSizes(size, 0);

    MPI_Barrier(comm);

    if (rank == 0){
        for (int i = 0; i < size; ++i) {
            sendSizes[i] = serializedParticles[i].size();
            sendDispls[i] = sendBuffer.size();
            sendBuffer.insert(sendBuffer.end(), serializedParticles[i].begin(), serializedParticles[i].end());

        }
    }

    //Scatter serialized data sizes
    int recvSize = 0;
    MPI_Scatter(sendSizes.data(), 1, MPI_INT, &recvSize, 1, MPI_INT, 0, comm);


    //Scatter serialized data
    recvBuffer.resize(recvSize);
    MPI_Scatterv(sendBuffer.data(), sendSizes.data(), sendDispls.data(), MPI_LONG_DOUBLE,
                 recvBuffer.data(), recvSize, MPI_LONG_DOUBLE, 0, comm);

    //Deserialize and store particles locally
    localParticles = deserializeParticles(recvBuffer);

    // if (rank == 1){
    //     cout << "localParticles size " << localParticles.size() << endl;
    //     for (const auto& p : localParticles) {
    //         cout << "Rank " << rank << " Particle " << p.globalID
    //                   << " with partitionID " << p.partitionID << endl;
    //     }
    // }


}

void defineVtxdist(idx_t &numVertices, int size, vector<idx_t> &vtxdist, MPI_Comm comm){
    vtxdist[0] = 0;
    for (int i = 1; i <= size; ++i){
        if(i == size){
            vtxdist[i] = vtxdist[i - 1] + numVertices / size + numVertices % size;
        } else{
            vtxdist[i] = vtxdist[i - 1] + numVertices / size;

        }
    }
}

void buildGraph(const vector<Particle>& globalParticles, int size, vector<idx_t> &vtxdist, map<int, vector<idx_t>> &xadjMap, map<int, vector<idx_t>> &adjncyMap){
    for (int i = 0; i < size; ++i){
        xadjMap[i].push_back(0);
        for (int j = vtxdist[i]; j < vtxdist[i + 1]; ++j){
            const auto &p = globalParticles[j];
            adjncyMap[i].insert(adjncyMap[i].end(), p.neighbors.begin(), p.neighbors.end());
            adjncyMap[i].insert(adjncyMap[i].end(), p.nneighbors.begin(), p.nneighbors.end());
            xadjMap[i].push_back(adjncyMap[i].size());
        }
    }
}

void distributeGraph(const map<int, vector<idx_t>> &xadjMap,
                     const map<int, vector<idx_t>> &adjncyMap,
                     int size, int rank, vector<idx_t> &xadj, vector<idx_t> &adjncy,
                     MPI_Comm comm) {
    if (rank == 0) {

        xadj = xadjMap.at(0);
        adjncy = adjncyMap.at(0);

        // On rank 0: Send xadj and adjncy to corresponding ranks
        for (int targetRank = 1; targetRank < size; ++targetRank) {
            // Send xadj
            int xadjSize = xadjMap.at(targetRank).size();
            MPI_Send(&xadjSize, 1, MPI_INT, targetRank, 0, comm); // Send size
            MPI_Send(xadjMap.at(targetRank).data(), xadjSize, MPI_INT, targetRank, 1, comm); // Send data

            // Send adjncy
            int adjncySize = adjncyMap.at(targetRank).size();
            MPI_Send(&adjncySize, 1, MPI_INT, targetRank, 2, comm); // Send size
            MPI_Send(adjncyMap.at(targetRank).data(), adjncySize, MPI_INT, targetRank, 3, comm); // Send data
        }
    } else {
        // On non-zero ranks: Receive xadj and adjncy
        // Receive xadj
        int xadjSize;
        MPI_Recv(&xadjSize, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        xadj.resize(xadjSize);
        MPI_Recv(xadj.data(), xadjSize, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);

        // Receive adjncy
        int adjncySize;
        MPI_Recv(&adjncySize, 1, MPI_INT, 0, 2, comm, MPI_STATUS_IGNORE);
        adjncy.resize(adjncySize);
        MPI_Recv(adjncy.data(), adjncySize, MPI_INT, 0, 3, comm, MPI_STATUS_IGNORE);

        // Output received data for verification
        /*
        cout << "Rank " << rank << " received xadj: ";
        for (int val : xadj) cout << val << " ";
        cout << "\nRank " << rank << " received adjncy: ";
        for (int val : adjncy) cout << val << " ";
        cout << endl;
        */
    }
}


void partitionGraph(int rank, vector<idx_t> &vtxdist, vector<idx_t> &xadj,
    vector<idx_t> &adjncy, int numPartitions, idx_t *part, MPI_Comm comm){

    idx_t *vwgt = nullptr;   // No vertex weights
    idx_t *adjwgt = nullptr; // No edge weights
    idx_t numflag = 0;       // C-style indexing
    idx_t wgtflag = 0;       // No vertex or edge weights
    idx_t ncon = 1;          // One balancing constraint
    real_t tpwgts[numPartitions]; // Target weights
    real_t ubvec[1] = {1.05}; // Load imbalance tolerance
    idx_t options[3] = {0};   // Default options
    idx_t edgecut;            // Edge cut output


    // Set uniform target weights
    for (int i = 0; i < numPartitions; ++i) {
        tpwgts[i] = 1.0 / numPartitions;
    }

    ParMETIS_V3_PartKway(
       vtxdist.data(),
       xadj.data(),              // Adjacency offsets
       adjncy.data(),            // Adjacency list
       vwgt,
       adjwgt,
       &wgtflag,
       &numflag,   // Weights (optional)
       &ncon,
       &numPartitions,    // Constraints and partitions
       tpwgts,
       ubvec,
       options,   // Target weights and imbalance tolerance
       &edgecut,
       part,
       &comm);

}

void updatePartitionID(vector<Particle> &globalParticles, int rank, int size, idx_t *part, int localNumVertices, map<int, int>& globalPartitionIDmap, MPI_Comm comm){

    vector<idx_t> gatheredPart;
    vector<int> partCounts;
    vector<int> partDispls;

    //gather the receive counts from each processor
    if (rank == 0){
        partCounts.resize(size);
    }
    MPI_Gather(&localNumVertices, 1, MPI_INT,
                partCounts.data(), 1, MPI_INT,
                0, MPI_COMM_WORLD);
    //define the displacement for MPI_Gatherv of part
    if (rank == 0){
        partDispls.resize(size);
        partDispls[0] = 0;
        for (int i = 1; i < size; ++i){
            partDispls[i] = partDispls[i - 1] + partCounts[i - 1];

        }
        int totalrecvNum = accumulate(partCounts.begin(), partCounts.end(), 0);
        gatheredPart.resize(totalrecvNum);
    }

    MPI_Gatherv(part, localNumVertices, MPI_INT,
                gatheredPart.data(), partCounts.data(),
                partDispls.data(), MPI_INT, 0, comm);

    if (rank == 0){
        for (int i = 0; i < globalParticles.size();++i){
            globalParticles[i].partitionID = gatheredPart[i];
            globalPartitionIDmap[globalParticles[i].globalID] = globalParticles[i].partitionID;
            //cout << "Particle id " << globalParticles[i].globalID << " partitionID " << globalParticles[i].partitionID << endl;
        }
    }

    //Serilize the map for broadcast
    vector<pair<int, int>> flatmap;
    int sendsize;
    if (rank == 0){
        for (const auto& [key, value] : globalPartitionIDmap){
            flatmap.emplace_back(key, value);
        }
        sendsize = flatmap.size();
    }

    //broadcast sendsize
    MPI_Bcast(&sendsize, 1, MPI_INT, 0, comm);
    if (rank != 0) flatmap.resize(sendsize);

    //broadcast globalPartitionIDmap
    MPI_Bcast(flatmap.data(), 2 * sendsize, MPI_INT, 0, comm);

    // Deserialize the map on all processors
    globalPartitionIDmap.clear();
    for (const auto& [key, value] : flatmap){
        globalPartitionIDmap[key] = value;
    }

}

void createGhostParticles(const vector<Particle>& localParticles, vector<Particle>& ghostParticles, const unordered_map<int, int>& globalLocalIDmap,
                          const map<int, int>& globalPartitionIDmap, int rank, MPI_Comm comm){
    //define all the neighbor ranks
    set<int> neighborRanks;
    unordered_map<int, set<int>> particlesRequest;
    for (const auto& p : localParticles){

        for (int nb : p.neighbors){
            int nbRank = globalPartitionIDmap.at(nb);
            if (nbRank != rank){
                neighborRanks.insert(nbRank);
                particlesRequest[nbRank].insert(nb);
            }
        }

        for (int nnb : p.nneighbors){
            int nnbRank = globalPartitionIDmap.at(nnb);
            if (nnbRank != rank){
                neighborRanks.insert(nnbRank);
                particlesRequest[nnbRank].insert(nnb);
            }
        }
    }

    // Convert sets back to vectors for communication
    unordered_map<int, vector<int>> particlesRequestVec;
    for (const auto& [nbRank, particleSet] : particlesRequest) {
        particlesRequestVec[nbRank] = vector<int>(particleSet.begin(), particleSet.end());
    }

    //exchange particlesRequestVec
    unordered_map<int, vector<int>> particlesSend;
    vector<int> recvparticles;
    for (int nbrank : neighborRanks){
        vector<int> sendparticles = particlesRequestVec[nbrank];
        int sendsize  = sendparticles.size(), recvsize;
        // exchange send size
        MPI_Sendrecv(&sendsize, 1, MPI_INT, nbrank, 0,
                     &recvsize, 1, MPI_INT, nbrank, 0, comm, MPI_STATUS_IGNORE);

        //exchange particle IDs
        recvparticles.resize(recvsize);
        MPI_Sendrecv(sendparticles.data(), sendsize, MPI_INT, nbrank, 1,
                     recvparticles.data(), recvsize, MPI_INT, nbrank, 1, comm, MPI_STATUS_IGNORE);

        particlesSend[nbrank] = recvparticles;

    }

    vector<long double> recvParticlesData;
    for (int nbrank : neighborRanks){
        vector<long double> sendParticlesData;
        for (const auto& pid : particlesSend[nbrank]){
            Particle particle = localParticles[globalLocalIDmap.at(pid)];
            vector<long double> flatdata = serializeParticle(particle);
            sendParticlesData.insert(sendParticlesData.end(), flatdata.begin(), flatdata.end());
        }

        // exchange send size
        int sendPdataSize = sendParticlesData.size(), recvPdataSize;
        MPI_Sendrecv(&sendPdataSize, 1, MPI_INT, nbrank, 2,
                     &recvPdataSize, 1, MPI_INT, nbrank, 2, comm,  MPI_STATUS_IGNORE);

        // exchange particles data
        recvParticlesData.resize(recvPdataSize);
        MPI_Sendrecv(sendParticlesData.data(), sendPdataSize, MPI_LONG_DOUBLE, nbrank, 3,
                     recvParticlesData.data(), recvPdataSize, MPI_LONG_DOUBLE, nbrank, 3, comm, MPI_STATUS_IGNORE);

        //Deserialize and store ghost particles
        vector<Particle> particlesReceive = deserializeParticles(recvParticlesData);
        ghostParticles.insert(ghostParticles.end(), particlesReceive.begin(), particlesReceive.end());

    }

}


void defineParticleIDmap(const vector<Particle>& localParticles, unordered_map<int, int>& globalLocalIDmap){
    globalLocalIDmap.clear();
    int idx = 0;
    for (const auto& particle : localParticles){
        globalLocalIDmap[particle.globalID] = idx++;
    }
}


void defineBoundarySet(int ndim, long double dx, const vector<long double>& boxlimit, vector<set<int>>& boundarySet, const vector<Particle> localParticles){

    for (int k = 0; k < localParticles.size(); ++k){
        Particle p = localParticles[k];
        if (p.neighbors.size() <= 17){ //it is possible to be a boundary particle
            for (int i = 0; i < ndim; ++i){
                long double coord = p.initialPositions[i];
                if (coord < (boxlimit[2 * i] + dx / 2.0)){
                    boundarySet[2 * i].insert(k);
                }
                else if (coord > (boxlimit[2 * i + 1] - dx / 2.0)){
                    boundarySet[2 * i + 1].insert(k);
                }
            }
        }
    }
}


void outputParticles(int rank, const vector<Particle>& localParticles){
    ofstream outFile;
    outFile.open("./output/particle_info_rank_" + to_string(rank) + ".txt");
    for (const auto &p : localParticles) {
        outFile << "Particle ID: " << p.globalID << " Partition ID: " << p.partitionID <<  " Position: (" << p.initialPositions[0] << ", " << p.initialPositions[1] << ", " << p.initialPositions[2] << ")";
        outFile << " Neighbors: ";
        for (int neighborIndex : p.neighbors) {
            outFile << neighborIndex << " ";
        }
        outFile << endl;
        outFile << " nNeighbors: ";
        for (int nneighborIndex : p.nneighbors) {
            outFile << nneighborIndex << " ";
        }
        outFile << endl;
    }
    outFile.close();

}

void outputGatheredPositions(int rank, int size, int ndim, int step, const vector<Particle>& localParticles, MPI_Comm comm){

    vector<long double> flatGatheredPositions;

    vector<long double> flatLocalPositions;


    for (const auto& p : localParticles) {
        flatLocalPositions.push_back(p.globalID);
        flatLocalPositions.insert(flatLocalPositions.end(), p.initialPositions.begin(), p.initialPositions.end());
        flatLocalPositions.insert(flatLocalPositions.end(), p.currentPositions.begin(), p.currentPositions.end());
        flatLocalPositions.push_back(p.damageStatus);
    }


    //gather the receive counts from each processor
    vector<int> recvSizes;
    vector<int> gatherDispls;
    int localSize = flatLocalPositions.size();


    if (rank == 0) recvSizes.resize(size);

    MPI_Gather(&localSize, 1, MPI_INT,
                recvSizes.data(), 1, MPI_INT,
                0, comm);


    //define the displacement for MPI_Gatherv of part
    if (rank == 0){
        gatherDispls.resize(size);
        gatherDispls[0] = 0;
        for (int i = 1; i < size; ++i){
            gatherDispls[i] = gatherDispls[i - 1] + recvSizes[i - 1];

        }
        int totalrecvSizes = accumulate(recvSizes.begin(), recvSizes.end(), 0);
        flatGatheredPositions.resize(totalrecvSizes);
    }

    MPI_Gatherv(flatLocalPositions.data(), localSize, MPI_LONG_DOUBLE,
                flatGatheredPositions.data(), recvSizes.data(),
                gatherDispls.data(), MPI_LONG_DOUBLE, 0, comm);

    MPI_Barrier(comm);
    // deserialize the gathered positions data on the root processor
    if (rank == 0){
        vector<vector<long double>> gatheredPositions;
        int idx = 0;
        while (idx < flatGatheredPositions.size()) {
            vector<long double> positionsData;
            positionsData.push_back(flatGatheredPositions[idx++]); //globalID
            for (int j = 0; j < 2*ndim; ++j){
                positionsData.push_back(flatGatheredPositions[idx++]);
            }
            positionsData.push_back(flatGatheredPositions[idx++]); //damageStatus
            gatheredPositions.push_back(positionsData);
        }

        ofstream outFile;
        outFile.open("./output/rank_" + to_string(rank) + "_particle_positions_step_" + to_string(step) + ".txt");
        outFile << "ID, x, y, ux, uy, damage" << endl;
        for (int i = 0; i < gatheredPositions.size(); ++i){
            for (int j = 0; j < 1 + ndim; ++j){
                outFile << gatheredPositions[i][j] << ",";
            }
            outFile << gatheredPositions[i][ndim + 1] -  gatheredPositions[i][1]<< ",";
            outFile << gatheredPositions[i][ndim + 2] -  gatheredPositions[i][2]<< ",";
            outFile << gatheredPositions[i][ndim + 3];
            outFile << endl;
        }
        outFile.close();

    }

}

void outputGhostParticles(int rank, const vector<Particle>& ghostParticles){
    ofstream outFile;
    outFile.open("../ouput/Ghostparticles_info_rank_" + to_string(rank) + ".txt");
    for (const auto &p : ghostParticles) {
        outFile << "Particle ID: " << p.globalID << " partition ID: " << p.partitionID <<" Position: ";
        for (int i = 0; i < p.initialPositions.size(); ++i) outFile << p.initialPositions[i] << " ";
        outFile << endl;
        outFile << " Neighbors: ";
        for (int neighborIndex : p.neighbors) {
            outFile << neighborIndex << " ";
        }
        outFile << endl;
    }
    outFile.close();

}

void outputLocalParticlesPositions(int rank, const vector<Particle>& localParticles, int ndim, int step){
    ofstream outputFile;
    outputFile.open("./output/LocalParticlesPositions_rank_" + to_string(rank) + "step_" + to_string(step) + ".txt");
    outputFile << "ID, x, y, ux, uy " << endl;
    for (int i = 0; i < localParticles.size(); ++i){
        outputFile << localParticles[i].globalID << ",";
        for (int j = 0; j < ndim; ++j){
            outputFile << localParticles[i].initialPositions[j] << ",";
        }
        outputFile << localParticles[i].currentPositions[0] - localParticles[i].initialPositions[0] << ",";
        outputFile << localParticles[i].currentPositions[1] - localParticles[i].initialPositions[1] << endl;
    }
}

void outputBuildLocalNeighborList(int rank, const vector<vector<Particle*>>& Neighborslist){
    ofstream outputFile;
    outputFile.open("./output/LocalNeighborList_rank_" + to_string(rank) + ".txt");
    for (int i = 0; i < Neighborslist.size(); ++i){
        for (auto& nb : Neighborslist[i]){
            outputFile << nb->globalID << " ";
        }
        outputFile << endl;
    }

}

void outputGlobalLocalIDmap(int rank, const vector<Particle>& localParticles, const unordered_map<int, int>& globalLocalIDmap){
    
    ofstream outputFile;
    outputFile.open("./output/GlobalLocalIDmap_rank_" + to_string(rank) + ".txt");
    for (int i = 0; i < localParticles.size(); ++i){
        int Gid = localParticles[i].globalID;
        int Lid = globalLocalIDmap.at(Gid);
        outputFile << " Gid " << Gid << " --> " <<  Lid << endl;
    }
    outputFile << endl;

}
