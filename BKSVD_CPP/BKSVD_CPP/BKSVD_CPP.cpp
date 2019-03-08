/*
BKSVD implemented by C++ based on https://github.com/cpmusco/bksvd
Author: ItakEjgo@SUSTech
*/

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <random>
#include <ctime>

#define EIGEN_USE_MKL_ALL

using namespace std;
using namespace Eigen;

struct bksvd_output {
	MatrixXd U;		// an orthogonal matrix whose columns are approximate top k left singular vectors for A
	MatrixXd S;		// a diagonal matrix whose entries are A's approximate top k singular values
	MatrixXd V;		// an orthogonal matrix whose columns are approximate top k right singular vectors for A
};

//	function bksvd

bksvd_output bksvd(MatrixXd A, int k = 6, int iter = 3, int bsize = 6, bool center = false) {
	bksvd_output result;

	static default_random_engine e((unsigned int)time(0));
	static normal_distribution<double> m_n(0, 1);
	
	k = min(k, (int)min(A.rows(), A.cols()));
	bsize = k;

	if (k < 1 || iter < 1 || bsize < k) {
		cerr << "bksvd: BadInput, Oner or more inputs outside required range" << endl;
	}

	//	Calculate row mean if rows should be centered.
	MatrixXd u = MatrixXd::Zero(1, A.cols());
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
		A.transposeInPlace();
	}

	//	Allocate space for Krylov subspace.
	MatrixXd K = MatrixXd::Zero(A.cols(), bsize * iter);
	//	Random block initialization.
	MatrixXd block = MatrixXd::Zero(A.cols(), bsize).unaryExpr([](double dummy) {return m_n(e); });

	HouseholderQR<MatrixXd> qr;
	qr.compute(block);
	block = qr.householderQ() * MatrixXd::Identity(A.cols(), bsize);

	//	Preallocate space for temporary products.
	MatrixXd T = MatrixXd::Zero(A.cols(), bsize);

	//	Construct and orthonormalize Krlov Subspace.
	//	Orthogonalize at each step using economy size QR decomposition
	for (int i = 1; i <= iter; i++) {
		T = A * block - l * (u * block);
		block = A.transpose() * T - u.transpose() * (l.transpose() * T);
		HouseholderQR<MatrixXd> tmp_qr;
		tmp_qr.compute(block);
		block = tmp_qr.householderQ() * MatrixXd::Identity(block.rows(), block.cols());;
		K.middleCols((i - 1)*bsize, bsize) = block;
	}

	//	Rayleigh-Ritz postprocessing with economy size dense SVD.
	HouseholderQR<MatrixXd> tmp_qr;
	
	tmp_qr.compute(K);
	MatrixXd Q = tmp_qr.householderQ() * MatrixXd::Identity(K.rows(), K.cols());

	T = A * Q - l * (u * Q);
	
	clock_t st, ed;
	st = clock();
	BDCSVD<MatrixXd> svd(T, ComputeThinU | ComputeThinV);    //	For large scale Matrix
	//JacobiSVD<MatrixXd>svd(T, ComputeThinU | ComputeThinV); // For small scale Matrix
	ed = clock();
	printf("svd time is %.5f second(s)\n", (double)(ed - st) / CLOCKS_PER_SEC);
	MatrixXd Ut = svd.matrixU(), Vt = svd.matrixV();
	MatrixXd St = svd.singularValues().asDiagonal();
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

//	function sisvd

bksvd_output sisvd(MatrixXd A, int k = 6, int iter = 3, int bsize = 6, bool center = false) {
	bksvd_output result;

	static default_random_engine e((unsigned int)time(0));
	static normal_distribution<double> m_n(0, 1);
	
	k = min(k, (int)min(A.rows(), A.cols()));
	bsize = k;
	if (k < 1 || iter < 1 || bsize < k) {
		cerr << "sisvd:BadInput, one or more inputs outside required range" << endl;
	}

	//	Calculate row mean if rows should be centered.
	MatrixXd u = MatrixXd::Zero(1, A.cols());
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
		A.transposeInPlace();
	}

	//	Random block initialization.
	MatrixXd block = MatrixXd::Zero(A.cols(), bsize).unaryExpr([](double dummy) {return m_n(e); });

	HouseholderQR<MatrixXd> qr;
	qr.compute(block);
	block = qr.householderQ() * MatrixXd::Identity(A.cols(), bsize);

	//	Preallocate space for temporary products.
	MatrixXd T = MatrixXd::Zero(A.cols(), bsize);

	//	Run power iteration, orthogonalizing at each step using economy size QR.
	for (int i = 1; i <= iter; i++) {
		T = A * block - l * (u * block);
		block = A.transpose() * T - u.transpose() * (l.transpose() * T);
		HouseholderQR<MatrixXd> tmp_qr;
		tmp_qr.compute(block);
		block = tmp_qr.householderQ() * MatrixXd::Identity(block.rows(), block.cols());
	}

	//	Rayleigh-Ritz postprocessing with economy size dense SVD.
	T = A * block - l * (u * block);
	//JacobiSVD<MatrixXd>svd(T, ComputeFullU | ComputeFullV);
	BDCSVD<MatrixXd> svd(T, ComputeThinU | ComputeThinV);
	MatrixXd Ut = svd.matrixU(), Vt = svd.matrixV();
	MatrixXd St = svd.singularValues().asDiagonal();
	result.S = St.block(0, 0, k, k);
	if (!tpose) {
		result.U = Ut.leftCols(k);
		result.V = block * Vt.leftCols(k);
	}
	else {
		result.V = Ut.leftCols(k);
		result.U = block * Vt.leftCols(k);
	}
	return result;
}

int main(){
	static default_random_engine e((unsigned int)time(0));
	static normal_distribution<double> m_n(0, 1);
	/*
	ofstream fout;
	fout.open("Matrix_A.txt");
	MatrixXd A = MatrixXd::Zero(10000, 161).unaryExpr([](double dummy) {return m_n(e); });
	fout << A << endl;
	*/
	MatrixXd A = MatrixXd::Zero(10000, 161);
	ifstream fin; 
	double val;
	for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < 161; j++) {
			fin >> val;
			A(i, j) = val;
		}
	}
	
	bksvd_output result1, result2;
	clock_t start, end;
	start = clock();
	result1 = bksvd(A,10);
	end = clock();
	//cout << "bksvd U = \n" << result1.U << endl;
	//cout << "bksvd S = \n" << result1.S << endl;
	//cout << "bksvd V = \n" << result1.V << endl;
	//cout << "bksvd U * S * V = \n" << (result.U * result.S) * result.V << endl;
	printf("Time used = :%.5f second(s)\n", (double)(end - start) / CLOCKS_PER_SEC);
	
	start = clock();
	result2 = sisvd(A);
	end = clock();
	//cout << "sisvd U = \n" << result2.U << endl;
	//cout << "sisvd S = \n" << result2.S << endl;
	//cout << "sisvd V = \n" << result2.V << endl;
	//cout << "sisvd U * S * V = \n" << (result.U * result.S) * result.V << endl;
	printf("Time used = :%.5f second(s)\n", (double)(end - start) / CLOCKS_PER_SEC);

	return 0;
}
