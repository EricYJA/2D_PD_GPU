#pragma once
#include <iostream>
#include <fstream>
#include <vector>

// outstream operator for std::vector
// only valid for types that can be streamed to std::ostream
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

// store the velocity vector as a file
void storeVelocity(int ndim, std::vector<std::vector<long double>>& particles, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    for (const auto& p : particles) {
        if (p.size() >= ndim) {
            for (int i = 0; i < ndim; ++i) {
                file << p[i] << (i < ndim - 1 ? " " : "");
            }
            file << std::endl;
        } else {
            std::cerr << "Skipping particle with insufficient data.\n";
        }
    }
    file.close();

    std::cout << "Current velocity data stored in " << filename << std::endl;
}

