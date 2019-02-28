﻿/*
BKSVD implemented by C++ based on https://github.com/cpmusco/bksvd
Author: ItakEjgo@SUSTech
*/

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <random>
#include <ctime>

using namespace std;
using namespace Eigen;

struct bksvd_output {
	MatrixXd U;	// an orthogonal matrix whose columns are approximate top k left singular vectors for A
	MatrixXd S; // a diagonal matrix whose entries are A's approximate top k singular values
	MatrixXd V; // an orthogonal matrix whose columns are approximate top k right singular vectors for A
};

bksvd_output bksvd(MatrixXd A, int k = 6, int iter = 3, int bsize = 6, bool center = false) {
	bksvd_output result;
	static default_random_engine e((unsigned int)time(0));
	static normal_distribution<double> m_n(0, 1);
	MatrixXd u = MatrixXd::Zero(1, A.cols());
	k = min(k, min(A.rows(), A.cols()));
	bsize = k;
	//	Calculate row mean if rows should be centered.
	if (center) {
		for (int i = 0; i < A.cols(); i++) {
			u(0, i) = A.col(i).mean();
		}
	}
	MatrixXd l = MatrixXd::Ones(A.rows(), 1);

	//	We want to iterate on the smaller dimension of A.
	int n, ind;
	if (A.rows() <= A.cols()) {
		n = A.rows();
		ind = 1;
	}
	else {
		n = A.cols();
		ind = 2;
	}
	bool tpose = false;
	if (ind == 1) {
		tpose = true;
		l = u.adjoint(); u.setOnes(1, A.rows());
		A = A.adjoint();
	}

	//	Allocate space for Krylov subspace.
	MatrixXd K = MatrixXd::Zero(A.cols(), bsize * iter);
	//	Random block initialization.
	MatrixXd block = MatrixXd::Zero(A.cols(), bsize).unaryExpr([](double dummy) {return m_n(e); });
	HouseholderQR<MatrixXd> qr;
	qr.compute(block);
	block = qr.householderQ();
	MatrixXd R = qr.matrixQR().triangularView<Upper>();	// [block, R] = qr(block, 0), But this is not simplified QR

	//	Preallocate space for temporary products.
	MatrixXd T = MatrixXd::Zero(A.cols(), bsize);

	//	Construct and orthonormalize Krlov Subspace.
	//	Orthogonalize at each step using economy size QR decomposition
	for (int i = 1; i <= iter; i++) {
		T = A * block - l * (u * block);
		block = A.adjoint() * T - u.adjoint() * (l.adjoint() * T);
		HouseholderQR<MatrixXd> tmp_qr;
		tmp_qr.compute(block);
		block = tmp_qr.householderQ();
		R = tmp_qr.matrixQR().triangularView<Upper>();
		K.middleCols((i - 1)*bsize, bsize) = block;
	}

	//	Rayleigh-Ritz postprocessing with economy size dense SVD.
	HouseholderQR<MatrixXd> tmp_qr;
	tmp_qr.compute(K);
	MatrixXd Q = tmp_qr.householderQ();
	R = tmp_qr.matrixQR().triangularView<Upper>();
	T = A * Q - l * (u * Q);
	JacobiSVD<MatrixXd>svd(T, ComputeFullU | ComputeFullV);
	MatrixXd Ut = svd.matrixU(), Vt = svd.matrixV();
	MatrixXd St = Ut.inverse() * T * Vt.transpose().inverse();
	result.S = St.block(0, 0, k, k);
	if (!tpose) {
		result.U = Ut.leftCols(k);
		result.V = Q * Vt.leftCols(k);
	}
	else {
		result.V = Ut.leftCols(k);
		result.U = Q * Vt.leftCols(k);
	}
	return result;
}


int main(){
	MatrixXd A = MatrixXd::Random(8, 6);
	bksvd_output result;
	result = bksvd(A);
	cout << "U = \n" << result.U << endl;
	cout << "S = \n" << result.S << endl;
	cout << "V = \n" << result.V << endl;
	cout << "Done." << endl;
}
