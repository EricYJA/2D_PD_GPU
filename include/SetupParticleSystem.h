#pragma once
#include <math.h>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <parmetis.h>

using namespace std;

struct Particle {

  int globalID;
  int partitionID;
  long double volume;
  double damageStatus;
  vector<long double> initialPositions;
  vector<long double> currentPositions;
  vector<int> neighbors;
  set<int> nneighbors;

};

vector<Particle> computeInitialPositions(int numParticlesRow, long double dx, int ndim, vector<long double>& boxlimit);
vector<Particle> readMeshFile(vector<long double>& boxlimit, long double& dx);
vector<long double> serializeParticle(const Particle& p);
vector<Particle> deserializeParticles(const std::vector<long double>& buffer);
void defineVtxdist(idx_t &numVertices, int size, vector<idx_t> &vtxdist, MPI_Comm comm);
void buildGraph(const vector<Particle>& globalParticles, int size, vector<idx_t> &vtxdist,
        map<int, vector<idx_t>> &xadjMap, map<int, vector<idx_t>> &adjncyMap);
void distributeGraph(const map<int, vector<idx_t>> &xadjMap, const map<int, vector<idx_t>> &adjncyMap,
        int size, int rank, vector<idx_t> &xadj, vector<idx_t> &adjncy, MPI_Comm comm);
void partitionGraph(int rank, vector<idx_t> &vtxdist, vector<idx_t> &xadj,
        vector<idx_t> &adjncy, int numPartitions, idx_t *part, MPI_Comm comm);
void updatePartitionID(vector<Particle> &globalParticles, int rank, int size, idx_t *part, int localNumParticles, map<int, int>& globalPartitionIDmap, MPI_Comm comm);
void distributeParticles(int rank, int size, const vector<Particle>& globalParticles, vector<Particle>& localParticles, MPI_Comm comm);
void defineParticleIDmap(const vector<Particle>& localParticles, unordered_map<int, int>& globalLocalIDmap);
void createGhostParticles(const vector<Particle>& localParticles, vector<Particle>& ghostParticles, const unordered_map<int, int>& globalLocalIDmap, const map<int, int>& globalPartitionIDmap, int rank, MPI_Comm comm);
void findNeighbor(vector<Particle>& globalParticles, long double dx, long double horizon, int ndim);
void defineBoundarySet(int ndim, long double dx, const vector<long double>& boxlimit, vector<set<int>>& boundarySet, const vector<Particle> localParticles);
void outputParticles(int rank, const vector<Particle>& localParticles);
void outputGhostParticles(int rank, const vector<Particle>& ghostParticles);
void outputGatheredPositions(int rank, int size, int ndim, int step, const vector<Particle>& localParticles, MPI_Comm comm);
void outputLocalParticlesPositions(int rank, const vector<Particle>& localParticles, int ndim, int step);
void outputBuildLocalNeighborList(int rank, const vector<vector<Particle*>>& Neighborslist);
void outputGlobalLocalIDmap(int rank, const vector<Particle>& localParticles, const unordered_map<int, int>& globalLocalIDmap);