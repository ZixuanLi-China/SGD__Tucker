#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <armadillo>
#include <string.h>
#include <omp.h>

using namespace std;
using namespace arma;

double learn_rate_a_init = 0.002;
double lambda_a = 0.05;

double init_mean = 0.5;
double init_variance = 0.1;

double learn_rate_a;

char* InputPath_train;
char* InputPath_test;

int order;
int *dimen;
int nnz_train, nnz_test;
double *value_train, *value_test;
int *index_train, *index_test;
double data_norm;

int core_count;
int *core_dimen;
int core_kernel;
double max_value = -100.0, min_value = 100.0;
mat* parameter_a;
mat* parameter_g;

double train_rmse, rmse_test;

inline double seconds() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

double frand(double x, double y) {
	return ((y - x) * ((double) rand() / RAND_MAX)) + x;
}

void Getting_Input() {

	dimen = (int *) malloc(sizeof(int) * order);
	for (int i = 0; i < order; i++) {
		dimen[i] = 0;
	}
	data_norm = 0.0;

	char tmp[1024];

	FILE *train_file_count = fopen(InputPath_train, "r");
	FILE *train_file = fopen(InputPath_train, "r");

	FILE *test_file_count = fopen(InputPath_test, "r");
	FILE *test_file = fopen(InputPath_test, "r");

	nnz_train = 0;
	nnz_test = 0;

	while (fgets(tmp, 1024, train_file_count)) {
		nnz_train++;
	}

	while (fgets(tmp, 1024, test_file_count)) {
		nnz_test++;
	}

	index_train = (int *) malloc(sizeof(int) * nnz_train * order);
	value_train = (double *) malloc(sizeof(double) * nnz_train);

	index_test = (int *) malloc(sizeof(int) * nnz_test * order);
	value_test = (double *) malloc(sizeof(double) * nnz_test);

	char* p;

	for (int i = 0; i < nnz_train; i++) {
		fgets(tmp, 1024, train_file);
		p = strtok(tmp, "\t");
		for (int j = 0; j < order; j++) {
			int int_temp = atoi(p);
			index_train[i * order + j] = int_temp - 1;
			p = strtok(NULL, "\t");
			if (int_temp > dimen[j]) {
				dimen[j] = int_temp;
			}
		}
		double double_temp = atof(p);
		if (double_temp > max_value) {
			max_value = double_temp;
		}
		if (double_temp < min_value) {
			min_value = double_temp;
		}
		value_train[i] = double_temp;
		data_norm += double_temp * double_temp;
	}
	data_norm = sqrt(data_norm / nnz_train);

	for (int i = 0; i < nnz_test; i++) {
		fgets(tmp, 1024, test_file);
		p = strtok(tmp, "\t");
		for (int j = 0; j < order; j++) {
			int int_temp = atoi(p);
			index_test[i * order + j] = int_temp - 1;
			p = strtok(NULL, "\t");
			if (int_temp > dimen[j]) {
				dimen[j] = int_temp;
			}
		}
		value_test[i] = atof(p);
		if (value_test[i] > max_value) {
			max_value = value_test[i];
		}
		if (value_test[i] < min_value) {
			min_value = value_test[i];
		}
	}
}

void Parameter_Initialization() {

	parameter_a = new mat[order];
	parameter_g = new mat[order];

	for (int i = 0; i < order; i++) {

		parameter_a[i] = sqrt(1.0 / core_dimen[i]) * 2.0
				* (init_variance * randn(dimen[i], core_dimen[i])
						+ init_mean);

		parameter_g[i] = pow(
						(sqrt((data_norm * data_norm / core_count)) / core_kernel),
						1.0 / order) * 2.0
						* (0.4+0.2*randu(core_dimen[i], core_kernel));
	}

}

void Update_Parameter_A() {
	mat g_temp;
	mat* g_list;
	g_list = new mat[order];

	for (int dim_index = 0; dim_index < order; dim_index++) { 
		g_list[dim_index] = zeros(core_count, 1);
		for (int r_index = 0; r_index < core_kernel; r_index++) { 
			g_temp = ones(1, 1);
			g_temp = kron(g_temp, parameter_g[dim_index].col(r_index));
			for (int dim_index_inner = 0; dim_index_inner < order;
					dim_index_inner++) {
				if (dim_index != dim_index_inner) {
					g_temp = kron(g_temp,
							parameter_g[dim_index_inner].col(r_index));
				}
			}
			g_list[dim_index] += g_temp;
		}
		g_list[dim_index].reshape(core_dimen[dim_index],
				core_count / core_dimen[dim_index]);
	}

#pragma omp parallel for
	for (int nnz_index = 0; nnz_index < nnz_train; nnz_index++) {
		mat s_temp;
		mat *gs_t;
		mat x_r;
		gs_t = new mat[order];
		for (int dim_index = 0; dim_index < order; dim_index++) {
			s_temp = ones(1, 1);
			for (int dim_index_inner = order - 1; dim_index_inner > -1;
					dim_index_inner--) {
				if (dim_index_inner != dim_index) {
					s_temp = kron(s_temp,
							parameter_a[dim_index_inner].row(
									index_train[nnz_index * order
											+ dim_index_inner]));
				}
			}

			gs_t[dim_index] = g_list[dim_index] * s_temp.t();
		}
		for (int dim_index = 0; dim_index < order; dim_index++) {
			x_r = mat(1, 1);
			x_r(0, 0) = value_train[nnz_index];
			parameter_a[dim_index].row(
					index_train[nnz_index * order + dim_index]) -= learn_rate_a
					* (-(x_r * gs_t[dim_index].t())
							+ parameter_a[dim_index].row(
									index_train[nnz_index * order + dim_index])
									* (gs_t[dim_index] * gs_t[dim_index].t())
							+ lambda_a
									* parameter_a[dim_index].row(
											index_train[nnz_index * order
													+ dim_index]));
		}
		delete[] gs_t;
	}
	delete[] g_list;
}

double Get_RMSE(int nnz_rmse, int* index_rmse, double* value_rmse) {
	mat h_temp;
	mat* q_list;
	mat q_temp;
	mat* ho_list;
	mat* hob_list;
	mat x_r;
	q_list = new mat[core_kernel];
	ho_list = new mat[core_kernel];
	hob_list = new mat[core_kernel];
	double mse = 0.0;

	for (int nnz_index = 0; nnz_index < nnz_rmse; nnz_index++) {
		h_temp = ones(1);
		for (int dim_index_inner = order - 1; dim_index_inner > -1;
				dim_index_inner--) {
			if (dim_index_inner != 0) {
				h_temp =
						kron(h_temp,
								parameter_a[dim_index_inner].row(
										index_rmse[nnz_index * order
												+ dim_index_inner]));
			}
		}
		h_temp = kron(h_temp,
				parameter_a[0].row(index_rmse[nnz_index * order]));
		h_temp.reshape(h_temp.n_cols / core_dimen[0], core_dimen[0]);

		for (int r_index = 0; r_index < core_kernel; r_index++) {
			q_temp = ones(1);
			for (int dim_index_inner = 0; dim_index_inner < order;
					dim_index_inner++) {
				if (dim_index_inner != 0) {
					q_temp = kron(q_temp,
							parameter_g[dim_index_inner].col(r_index));
				}
			}
			q_list[r_index] = q_temp.t();
			ho_list[r_index] = q_list[r_index] * h_temp;
			hob_list[r_index] = ho_list[r_index] * parameter_g[0].col(r_index);

		}

		x_r = mat(1, 1);
		x_r(0, 0) = value_rmse[nnz_index];
		for (int r_index = 0; r_index < core_kernel; r_index++) {
			x_r -= hob_list[r_index];
		}
		mse += x_r(0, 0) * x_r(0, 0);

	}
	delete[] q_list;
	delete[] ho_list;
	delete[] hob_list;
	return sqrt(mse / nnz_rmse);
}

double Get_MAE(int nnz_rmse, int* index_rmse, double* value_rmse) {
	mat h_temp;
	mat* q_list;
	mat q_temp;
	mat* ho_list;
	mat* hob_list;
	mat x_r;
	q_list = new mat[core_kernel];
	ho_list = new mat[core_kernel];
	hob_list = new mat[core_kernel];
	double mse = 0.0;

	for (int nnz_index = 0; nnz_index < nnz_rmse; nnz_index++) {
		h_temp = ones(1);
		for (int dim_index_inner = order - 1; dim_index_inner > -1;
				dim_index_inner--) {
			if (dim_index_inner != 0) {
				h_temp =
						kron(h_temp,
								parameter_a[dim_index_inner].row(
										index_rmse[nnz_index * order
												+ dim_index_inner]));
			}
		}
		h_temp = kron(h_temp,
				parameter_a[0].row(index_rmse[nnz_index * order]));
		h_temp.reshape(h_temp.n_cols / core_dimen[0], core_dimen[0]);

		for (int r_index = 0; r_index < core_kernel; r_index++) {
			q_temp = ones(1);
			for (int dim_index_inner = 0; dim_index_inner < order;
					dim_index_inner++) {
				if (dim_index_inner != 0) {
					q_temp = kron(q_temp,
							parameter_g[dim_index_inner].col(r_index));
				}
			}
			q_list[r_index] = q_temp.t();
			ho_list[r_index] = q_list[r_index] * h_temp;
			hob_list[r_index] = ho_list[r_index] * parameter_g[0].col(r_index);

		}

		x_r = mat(1, 1);
		x_r(0, 0) = value_rmse[nnz_index];
		for (int r_index = 0; r_index < core_kernel; r_index++) {
			x_r -= hob_list[r_index];
		}
		mse += abs(x_r(0, 0));

	}
	delete[] q_list;
	delete[] ho_list;
	delete[] hob_list;
	return mse / nnz_rmse;
}

int main(int argc, char* argv[]) {

	if (argc == 5 + atoi(argv[4])) {

		InputPath_train = argv[1];
		InputPath_test = argv[2];
		core_kernel = atoi(argv[3]);
		order = atoi(argv[4]);
		core_count = 1;
		core_dimen = (int *) malloc(sizeof(int) * order);
		for (int i = 0; i < order; i++) {
			core_dimen[i] = atoi(argv[5 + i]);
			core_count *= atoi(argv[5 + i]);
		}

	}
	Getting_Input();
	Parameter_Initialization();
	printf("max:\t%f\n", max_value);
	printf("min:\t%f\n", min_value);
	printf("nnz_train:\t%d\n", nnz_train);
	printf("nnz_test:\t%d\n", nnz_test);
	for (int i = 0; i < order; i++) {
		printf("order %d:\t%d\n", i + 1, dimen[i]);
	}
	printf("init:\t%f\t%f\t%f\t%f\n",
			Get_RMSE(nnz_train, index_train, value_train),
			Get_RMSE(nnz_test, index_test, value_test),
			Get_MAE(nnz_train, index_train, value_train),
			Get_MAE(nnz_test, index_test, value_test));
	double time_spend = 0.0, start_time, stop_time;
	printf("it\ttrain rmse\ttest rmse\ttrain mae\ttest mae\ttrain time\n");
	for (int i = 0; i < 100; i++) {
		learn_rate_a = learn_rate_a_init / (1 + 1.0 * pow(i, 1.5));
		start_time = seconds();
		Update_Parameter_A();
		stop_time = seconds();
		time_spend += stop_time - start_time;
		printf("%d\t%f\t%f\t%f\t%f\t%f\n", 
			i, 
			Get_RMSE(nnz_train, index_train, value_train), 
			Get_RMSE(nnz_test, index_test, value_test), 
			Get_MAE(nnz_train, index_train, value_train), 
			Get_MAE(nnz_test, index_test, value_test), 
			time_spend);
	}
	return 0;
}
