/*
BKSVD implementing by C++ based on https://github.com/cpmusco/bksvd
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
		l = u.transpose(); u.setOnes(1, A.rows());
		A = A.transpose();
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
		block = A.transpose() * T - u.transpose() * (l.transpose() * T);
		HouseholderQR<MatrixXd> tmp_qr;
		qr.compute(block);
		block = qr.householderQ();
		R = qr.matrixQR().triangularView<Upper>();
		
	}
	return result;
}


int main(){
	MatrixXd A(4, 3);
	A(0, 0) = 0; A(0, 1) = 1; A(0, 2) = 1;
	A(1, 0) = 2; A(1, 1) = 3; A(1, 2) = 2;
	A(2, 0) = 1; A(2, 1) = 3; A(2, 2) = 2;
	A(3, 0) = 4; A(3, 1) = 2; A(3, 2) = 2;
	bksvd(A);
	cout << "Done." << endl;
}
