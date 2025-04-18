#include "matrix.h"
#include <stdexcept>

using namespace std;

matrix::matrix() {}

matrix::matrix(int ndim, const vector<long double>& values) : ndim(ndim), elements(ndim, vector<long double>(ndim)) {
    // if (values.size() != ndim * ndim) {
    //     throw invalid_argument("Values of matrix does not match size ndim * ndim.");
    // }
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            elements[i][j] = values[i * ndim + j];
        }
    }
}

matrix matrix::matrixAdd(const matrix& M){
    matrix result(ndim, vector<long double>(ndim*ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            result.elements[i][j] = elements[i][j] + M.elements[i][j];
        }
    }

    return result;
}

matrix matrix::matrixSub(const matrix& M){
    matrix result(ndim, vector<long double>(ndim*ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            result.elements[i][j] = elements[i][j] - M.elements[i][j];
        }
    }

    return result;
}

matrix matrix::timeScalar(long double constant) {
    matrix result(ndim, vector<long double>(ndim*ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            result.elements[i][j] = elements[i][j] * constant;
        }
    }

    return result;
}

vector<long double> matrix::timeVector(const vector<long double>& V) {
    vector<long double> result(ndim, 0.0);

    // if (V.size() != ndim) {
    //     throw invalid_argument("Error in timeVector: Input vector size must match ndim.");
    // }

    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            result[i] += elements[i][j] * V[j];
        }
    }

    return result;
}


matrix matrix::timeMatrix(const matrix& M) {
    // if (ndim != M.ndim) {
    //     throw invalid_argument("Matrix dimensions do not match for multiplication.");
    // }
    matrix result(ndim, vector<long double>(ndim*ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            for (int k = 0; k < ndim; ++k) {
                result.elements[i][j] += elements[i][k] * M.elements[k][j];
            }
        }
    }

  return result;
}


matrix matrix::transpose() {
    matrix result(ndim, vector<long double>(ndim*ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            result.elements[j][i] = elements[i][j];
        }
    }

    return result;
}


// matrix matrix::getMinorMatrix(int row, int col) {
//     matrix minorMatrix(ndim - 1, vector<long double>((ndim - 1) * (ndim - 1), 0.0));
//     int minorRow = 0, minorCol = 0;

//     for (int i = 0; i < ndim; ++i) {
//         if (i == row) continue;
//         minorCol = 0;
//         for (int j = 0; j < ndim; ++j) {
//             if (j == col) continue;
//             minorMatrix.elements[minorRow][minorCol] = elements[i][j];
//             ++minorCol;
//         }
//         minorRow++;
//     }

//     return minorMatrix;
// }

matrix matrix::inverse2D() {

    matrix inv(ndim, vector<long double>(ndim * ndim, 0.0));

    //compute determinant
    long double det = elements[0][0] * elements[1][1] - elements[0][1] * elements[1][0];

    inv.elements[0][0] =  elements[1][1] / det;
    inv.elements[0][1] = -elements[0][1] / det;
    inv.elements[1][0] = -elements[1][0] / det;
    inv.elements[1][1] =  elements[0][0] / det;

    return inv;
}


matrix matrix::inverse3D() {

    //compute determinant
    long double det = elements[0][0] * (elements[1][1] * elements[2][2] - elements[1][2] * elements[2][1])
                    - elements[0][1] * (elements[1][0] * elements[2][2] - elements[1][2] * elements[2][0])
                    + elements[0][2] * (elements[1][0] * elements[2][1] - elements[1][1] * elements[2][0]);                

    matrix inv(ndim, vector<long double>(ndim * ndim, 0.0));

    inv.elements[0][0] =   (this->elements[1][1] * this->elements[2][2] - this->elements[1][2] * this->elements[2][1]) / det;
    inv.elements[0][1] =  -(this->elements[0][1] * this->elements[2][2] - this->elements[0][2] * this->elements[2][1]) / det;
    inv.elements[0][2] =   (this->elements[0][1] * this->elements[1][2] - this->elements[0][2] * this->elements[1][1]) / det;

    inv.elements[1][0] =  -(this->elements[1][0] * this->elements[2][2] - this->elements[1][2] * this->elements[2][0]) / det;
    inv.elements[1][1] =   (this->elements[0][0] * this->elements[2][2] - this->elements[0][2] * this->elements[2][0]) / det;
    inv.elements[1][2] =  -(this->elements[0][0] * this->elements[1][2] - this->elements[0][2] * this->elements[1][0]) / det;

    inv.elements[2][0] =   (this->elements[1][0] * this->elements[2][1] - this->elements[1][1] * this->elements[2][0]) / det;
    inv.elements[2][1] =  -(this->elements[0][0] * this->elements[2][1] - this->elements[0][1] * this->elements[2][0]) / det;
    inv.elements[2][2] =   (this->elements[0][0] * this->elements[1][1] - this->elements[0][1] * this->elements[1][0]) / det;

    return inv;
    
}
