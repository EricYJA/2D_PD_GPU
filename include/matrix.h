#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

using namespace std;

class matrix
{

public:
    int ndim;
    vector<vector<long double>> elements;
    matrix();
    matrix(int ndim, const vector<long double>& values);
    matrix matrixAdd(const matrix& M);
    matrix matrixSub(const matrix& M);
    matrix timeScalar(long double constant);
    vector<long double> timeVector(const vector<long double>& V);
    matrix timeMatrix(const matrix& M);
    matrix transpose();
    //long double determinant();
    //matrix getMinorMatrix(int row, int col);
    matrix inverse2D();
    matrix inverse3D();

};

#endif
