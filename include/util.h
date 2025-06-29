#pragma once
#include <iostream>
#include <vector>

#define INVALID_INT -1

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