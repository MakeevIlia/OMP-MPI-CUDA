#include "mpi.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath> 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>

using namespace std;


#define Pi 3.141592653589793

struct Point {
	float x, y, z, value, oldvalue;
};

__device__
float uanalitical(float at, float Lx, float Ly, float Lz, float x, float y, float z, float t)
{
	return sin((x * Pi) / Lx) * sin((2 * y * Pi) / Ly) * sin((3 * z * Pi) / Lz) * cos(at * t);
}


float phi(float Lx, float Ly, float Lz, float x, float y, float z)
{
	return sin((x * Pi) / Lx) * sin((2 * y * Pi) / Ly) * sin((3 * z * Pi) / Lz);
}


void multi(int n, int& a1, int& a2, int& a3)
{
	a1 = 1;
	a2 = 1;
	a3 = 1;

	int k = 1;
	int div = 2;
	while (n > 1)
	{
		while (n % div == 0)
		{
			n = n / div;
			if (k % 3 == 1)
			{
				a1 *= div;
				k++;
			}
			else
				if (k % 3 == 2)
				{
					a2 *= div;
					k++;
				}
				else
					if (k % 3 == 0)
					{
						a3 *= div;
						k++;
					}
		}
		div++;
	}
}

__global__
void counterror(float* GPUError, int Nxl, int Nyl, int Nzl, Point* GCP, float tauh, float at, float Lx, float Ly, float Lz, float t)
{
	const int blocksize = 512; // Число нитей в блоке
	int thread = threadIdx.x; // Вычисление адреса нити
	float error = 0; // Максимум ошибки для этой нити
	float value;
	float x;
	float y;
	float z;
	Point current;
	int n = (Nxl + 2) * (Nyl + 2) * (Nzl + 2); // Общее число элементов в каждоми блоке

	for (int id = thread; id < n; id += blocksize) //Для каждой нити "шагаем" по всем блокам
	{
		// Так как в блоке хранятся не только значения самого блока, но и граничные значения, то координаты необходимо исправить, чтобы они лежали в диапазоне от 1 до (максимум по этой координате - 1)
		int k = int(thread / ((Nxl + 2) * (Nyl + 2)));
		int j = int((thread - k * (Nxl + 2) * (Nyl + 2)) / Nxl);
		int i = int(thread - k * (Nxl + 2) * (Nyl + 2) - (Nxl + 2) * j);

		k = 1 + k % (Nzl + 1);
		j = 1 + j % (Nyl + 1);
		i = 1 + i % (Nxl + 1);


		current = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i]; // Берем точку из схемы и считаем для нее максимум
		x = current.x;
		y = current.y;
		z = current.z;
		value = current.value;

		float diff = fabs(value - uanalitical(at, Lx, Ly, Lz, x, y, z, t));

		if (diff > error)
		{
			error = diff; //Подсчет максимальной ошибки для этой нити
		}
	}

	// Далее каждая нить запишет в свой адрес максимум, который она вычислила по всем блокам

	__shared__ float ErrorPerThread[blocksize];
	ErrorPerThread[thread] = error;
	__syncthreads();

	for (int i = blocksize / 2; i > 0; i /= 2)
	{
		if (thread < i)
		{
			if (ErrorPerThread[thread] > ErrorPerThread[thread + i])
			{
				ErrorPerThread[thread] = ErrorPerThread[thread];
			}
			else
			{
				ErrorPerThread[thread] = ErrorPerThread[thread + i];
			}
		}
		__syncthreads();
	}

	if (thread == 0)
	{
		*GPUError = ErrorPerThread[0];
	}
}

__global__
void Kernel1(int Nxl, int Nyl, int Nzl, Point* GCP, Point* GLS, Point* GRS, Point* GDS, Point* GUS, Point* GFS, Point* GBS, float tauh)
{
	int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
	int k = int(thread / ((Nxl + 2) * (Nyl + 2)));
	int j = int((thread - k * (Nxl + 2) * (Nyl + 2)) / Nxl);
	int i = int(thread - k * (Nxl + 2) * (Nyl + 2) - (Nxl + 2) * j);
	Point current = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i];
	float value;
	value = current.value;
	//printf("Thread = %10d, i = %10d, j = %10d, k = %10d, value =  %3.10f \n", thread, i, j, k, value );
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = 2 * value - current.oldvalue + tauh * (GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i - 1].value + GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i + 1].value + GCP[k * (Nxl + 2) * (Nyl + 2) + (j - 1) * (Nxl + 2) + i].value + GCP[k * (Nxl + 2) * (Nyl + 2) + (j + 1) * (Nxl + 2) + i].value + GCP[(k - 1) * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value + GCP[(k + 1) * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value - 6 * value);
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
	j = 1;
	GFS[k * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GFS[k * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GFS[k * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GFS[k * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GFS[k * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	j = Nyl;
	GBS[k * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GBS[k * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GBS[k * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GBS[k * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GBS[k * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	j = int((thread - k * (Nxl + 2) * (Nyl + 2)) / Nxl);
	i = 1;
	GLS[k * (Nyl + 2) + j].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GLS[k * (Nyl + 2) + j].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GLS[k * (Nyl + 2) + j].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GLS[k * (Nyl + 2) + j].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GLS[k * (Nyl + 2) + j].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	i = Nxl;
	GRS[k * (Nyl + 2) + j].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GRS[k * (Nyl + 2) + j].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GRS[k * (Nyl + 2) + j].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GRS[k * (Nyl + 2) + j].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GRS[k * (Nyl + 2) + j].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	i = int(thread - k * (Nxl + 2) * (Nyl + 2) - Nxl * j);
	k = 1;
	GDS[j * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GDS[j * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GDS[j * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GDS[j * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GDS[j * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	k = Nzl;
	GUS[j * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GUS[j * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GUS[j * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GUS[j * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GUS[j * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
}

__global__
void Kernel2(int Nxl, int Nyl, int Nzl, Point* GCP, Point* GLS, Point* GRS, Point* GDS, Point* GUS, Point* GFS, Point* GBS, Point* GLR, Point* GRR, Point* GDR, Point* GUR, Point* GFR, Point* GBR, float tauh)
{
	int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
	int k = int(thread / ((Nxl + 2) * (Nyl + 2)));
	int j = int((thread - k * (Nxl + 2) * (Nyl + 2)) / (Nxl + 2));
	int i = int(thread - k * (Nxl + 2) * (Nyl + 2) - (Nxl + 2) * j);

	k = 0;

	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = GDR[j * (Nxl + 2) + i].oldvalue;
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = GDR[j * (Nxl + 2) + i].value;


	k = Nzl + 1;

	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = GUR[j * (Nxl + 2) + i].oldvalue;
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = GUR[j * (Nxl + 2) + i].value;
	k = int(thread / ((Nxl + 2) * (Nyl + 2)));

	i = Nxl + 1;

	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = GRR[k * (Nyl + 2) + j].oldvalue;
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = GRR[k * (Nyl + 2) + j].value;

	i = 0;

	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = GLR[k * (Nyl + 2) + j].oldvalue;
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = GLR[k * (Nyl + 2) + j].value;
	i = int(thread - k * (Nxl + 2) * (Nyl + 2) - Nxl * j);

	j = Nyl + 1;

	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = GBR[k * (Nxl + 2) + i].oldvalue;
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = GBR[k * (Nxl + 2) + i].value;


	j = 0;

	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = GFR[k * (Nxl + 2) + i].oldvalue;
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = GFR[k * (Nxl + 2) + i].value;
	j = int((thread - k * (Nxl + 2) * (Nyl + 2)) / Nxl);


	Point current = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i];
	float value;
	value = current.value;
	//printf("Thread = %10d, i = %10d, j = %10d, k = %10d, value =  %3.10f \n", thread, i, j, k, value );
	__syncthreads();
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = 2 * value - current.oldvalue + tauh * (GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i - 1].value + GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i + 1].value + GCP[k * (Nxl + 2) * (Nyl + 2) + (j - 1) * (Nxl + 2) + i].value + GCP[k * (Nxl + 2) * (Nyl + 2) + (j + 1) * (Nxl + 2) + i].value + GCP[(k - 1) * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value + GCP[(k + 1) * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value - 6 * value);
	GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;

	j = 1;
	GFS[k * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GFS[k * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GFS[k * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GFS[k * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GFS[k * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	j = Nyl;
	GBS[k * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GBS[k * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GBS[k * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GBS[k * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GBS[k * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	j = int((thread - k * (Nxl + 2) * (Nyl + 2)) / Nxl);
	i = 1;
	GLS[k * (Nyl + 2) + j].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GLS[k * (Nyl + 2) + j].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GLS[k * (Nyl + 2) + j].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GLS[k * (Nyl + 2) + j].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GLS[k * (Nyl + 2) + j].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	i = Nxl;
	GRS[k * (Nyl + 2) + j].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GRS[k * (Nyl + 2) + j].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GRS[k * (Nyl + 2) + j].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GRS[k * (Nyl + 2) + j].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GRS[k * (Nyl + 2) + j].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	i = int(thread - k * (Nxl + 2) * (Nyl + 2) - Nxl * j);
	k = 1;
	GDS[j * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GDS[j * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GDS[j * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GDS[j * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GDS[j * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
	k = Nzl;
	GUS[j * (Nxl + 2) + i].x = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x;
	GUS[j * (Nxl + 2) + i].y = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y;
	GUS[j * (Nxl + 2) + i].z = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z;
	GUS[j * (Nxl + 2) + i].oldvalue = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue;
	GUS[j * (Nxl + 2) + i].value = GCP[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value;
}

void cpuf(int Nxl, int Nyl, int Nzl, Point* CenterPoints, float tauh)
{
	float value;
	for (int k = 1; k < Nzl + 1; k += 1)
	{
		for (int j = 1; j < Nyl + 1; j += 1)
		{
			for (int i = 1; i < Nxl + 1; i += 1)
			{
				Point current = CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i];
				value = current.value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i - 1].value + CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i + 1].value + CenterPoints[k * (Nxl + 2) * (Nyl + 2) + (j - 1) * (Nxl + 2) + i].value + CenterPoints[k * (Nxl + 2) * (Nyl + 2) + (j + 1) * (Nxl + 2) + i].value + CenterPoints[(k - 1) * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value + CenterPoints[(k + 1) * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value - 6 * value);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
			}
		}
	}
}

void Initvalues(int Nxl, int Nyl, int Nzl, Point* CenterPoints, int px, int py, int pz, float tauh2, float h, float Lx, float Ly, float Lz, int n)
{
	float x, y, z, value;
	int i, j, k;
	for (int k = 0; k < Nzl + 2; k += 1)
	{
		for (int j = 0; j < Nyl + 2; j += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				z = pz * Nzl * h + k * h - h;
				y = py * Nyl * h + j * h - h;
				x = px * Nxl * h + i * h - h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	float xmin, ymin, zmin, xmax, ymax, zmax;

	xmin = CenterPoints[0].x;
	xmax = CenterPoints[n - 1].x;
	ymin = CenterPoints[0].y;
	ymax = CenterPoints[n - 1].y;
	zmin = CenterPoints[0].z;
	zmax = CenterPoints[n - 1].z;
	if (ymin < 0)
	{
		j = 0;
		y = ymin;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = 0;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = 0;
			}
		}
	}
	else
	{
		j = 0;
		y = ymin;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	if (ymax > Lx)
	{
		j = Nyl + 1;
		y = ymax;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = 0;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = 0;
			}
		}
	}
	else
	{
		j = Nyl + 1;
		y = ymax;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	if (zmin < 0)
	{
		k = 0;
		z = zmin;
		for (int j = 0; j < Nyl + 2; j += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = 0;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = 0;
			}
		}
	}
	else
	{
		k = 0;
		z = zmin;
		for (int j = 0; j < Nyl + 2; j += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	if (zmax > Lz)
	{
		k = Nzl + 1;
		z = zmax;
		for (int j = 0; j < Nyl + 2; j += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = 0;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = 0;
			}
		}
	}
	else
	{
		k = Nzl + 1;
		z = zmax;
		for (int j = 0; j < Nyl + 2; j += 1)
		{
			for (int i = 0; i < Nxl + 2; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	if (xmin < 0)
	{
		i = 0;
		x = Lx;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int j = 0; j < Nyl + 2; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}
	else
	{
		i = 0;
		x = xmin;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int j = 0; j < Nyl + 2; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	if (xmax > Lx)
	{
		i = Nxl + 1;
		x = 0;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int j = 0; j < Nyl + 2; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}
	else
	{
		i = 0;
		x = xmax;
		for (int k = 0; k < Nzl + 2; k += 1)
		{
			for (int j = 0; j < Nyl + 2; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].x = x;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].y = y;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].z = z;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].oldvalue = value;
				CenterPoints[k * (Nxl + 2) * (Nyl + 2) + j * (Nxl + 2) + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}
}

void SendRecvValues(int Nxl, int Nyl, int Nzl, int px, int py, int pz, int rx, int ry, int rz, int Np, Point* DownSend, Point* UpSend, Point* LeftSend, Point* RightSend, Point* BehindSend, Point* FrontSend, Point* DownRecv, Point* UpRecv, Point* LeftRecv, Point* RightRecv, Point* BehindRecv, Point* FrontRecv)
{
	MPI_Status status;
	MPI_Datatype dt_point;
	MPI_Type_contiguous(5, MPI_FLOAT, &dt_point);
	MPI_Type_commit(&dt_point);
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int UpComp, DownComp, LeftComp, RightComp, FrontComp, BehindComp;

	UpComp = rx * ry * (pz + 1) + py * rx + px;
	DownComp = rx * ry * (pz - 1) + py * rx + px;

	LeftComp = rx * ry * pz + py * rx + px - 1;
	RightComp = rx * ry * pz + py * rx + px + 1;

	FrontComp = rx * ry * pz + (py - 1) * rx + px;
	BehindComp = rx * ry * pz + (py + 1) * rx + px;

	MPI_Request requests1[2], requests2[2], requests3[2], requests4[2], requests5[2], requests6[2];
	MPI_Status statuses1[2], statuses2[2], statuses3[2], statuses4[2], statuses5[2], statuses6[2];

	if ((0 <= UpComp) && (UpComp < Np))
	{
		MPI_Isend(&UpSend[0], (Nxl + 2) * (Nyl + 2), dt_point, UpComp, 1, MPI_COMM_WORLD, &requests1[0]);

		MPI_Irecv(&UpRecv[0], (Nxl + 2) * (Nyl + 2), dt_point, UpComp, 1, MPI_COMM_WORLD, &requests1[1]);
	}

	if ((0 <= DownComp) && (DownComp < Np))
	{
		MPI_Isend(&DownSend[0], (Nxl + 2) * (Nyl + 2), dt_point, DownComp, 1, MPI_COMM_WORLD, &requests2[0]);

		MPI_Irecv(&DownRecv[0], (Nxl + 2) * (Nyl + 2), dt_point, DownComp, 1, MPI_COMM_WORLD, &requests2[1]);
	}

	if ((0 <= LeftComp) && (LeftComp < Np))
	{
		MPI_Isend(&LeftSend[0], (Nyl + 2) * (Nzl + 2), dt_point, LeftComp, 1, MPI_COMM_WORLD, &requests3[0]);

		MPI_Irecv(&LeftRecv[0], (Nyl + 2) * (Nzl + 2), dt_point, LeftComp, 1, MPI_COMM_WORLD, &requests3[1]);
	}
	else
	{
		LeftComp += rx;
		MPI_Isend(&LeftSend[0], (Nyl + 2) * (Nzl + 2), dt_point, LeftComp, 1, MPI_COMM_WORLD, &requests3[0]);

		MPI_Irecv(&LeftRecv[0], (Nyl + 2) * (Nzl + 2), dt_point, LeftComp, 1, MPI_COMM_WORLD, &requests3[1]);
	}

	if ((0 <= RightComp) && (RightComp < Np))
	{
		MPI_Isend(&RightSend[0], (Nyl + 2) * (Nzl + 2), dt_point, RightComp, 1, MPI_COMM_WORLD, &requests4[0]);

		MPI_Irecv(&RightRecv[0], (Nyl + 2) * (Nzl + 2), dt_point, RightComp, 1, MPI_COMM_WORLD, &requests4[1]);
	}
	else
	{
		RightComp -= rx;
		MPI_Isend(&RightSend[0], (Nyl + 2) * (Nzl + 2), dt_point, RightComp, 1, MPI_COMM_WORLD, &requests4[0]);

		MPI_Irecv(&RightRecv[0], (Nyl + 2) * (Nzl + 2), dt_point, RightComp, 1, MPI_COMM_WORLD, &requests4[1]);
	}

	if ((0 <= FrontComp) && (FrontComp < Np))
	{
		MPI_Isend(&FrontSend[0], (Nxl + 2) * (Nzl + 2), dt_point, FrontComp, 1, MPI_COMM_WORLD, &requests5[0]);

		MPI_Irecv(&FrontRecv[0], (Nxl + 2) * (Nzl + 2), dt_point, FrontComp, 1, MPI_COMM_WORLD, &requests5[1]);
	}

	if ((0 <= BehindComp) && (BehindComp < Np))
	{
		MPI_Isend(&BehindSend[0], (Nxl + 2) * (Nzl + 2), dt_point, BehindComp, 1, MPI_COMM_WORLD, &requests6[0]);

		MPI_Irecv(&BehindRecv[0], (Nxl + 2) * (Nzl + 2), dt_point, BehindComp, 1, MPI_COMM_WORLD, &requests6[1]);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}


void show(Point* P, int n)
{
	for (int i = 0; i < n; i += 1)
	{
		cout << P[i].x << "  " << P[i].y << "  " << P[i].z << "  " << P[i].value << "  " << endl;
	}
}

int main(int argc, char* argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;

	float Lx = 1;
	float Ly = 1;
	float Lz = 1;
	float at = sqrt(1 / (Lx * Lx) + 4 / (Lx * Lx) + 9 / (Lz * Lz)) * Pi;
	float tau = 0.000001;
	int P = 512;
	int Nx = P;
	int Ny = P;
	int Nz = P;
	float h = Lx / (Nx - 1);
	float tauh = tau * tau / (h * h);
	float tauh2 = tau * tau / (2 * h * h);
	int Np = size;

	int rx, ry, rz;
	multi(Np, rx, ry, rz);

	int pz = int(rank / (rx * ry));
	int py = int((rank - pz * rx * ry) / rx);
	int px = int(rank - pz * rx * ry - rx * py);

	int Nxl = Nx / rx;
	int Nyl = Ny / ry;
	int Nzl = Nz / rz;

	int n = (Nxl + 2) * (Nyl + 2) * (Nzl + 2);

	Point* CenterPoints = new Point[n];
	Point* DownSend = new Point[(Nxl + 2) * (Nyl + 2)];
	Point* UpSend = new Point[(Nxl + 2) * (Nyl + 2)];
	Point* LeftSend = new Point[(Nyl + 2) * (Nzl + 2)];
	Point* RightSend = new Point[(Nyl + 2) * (Nzl + 2)];
	Point* FrontSend = new Point[(Nxl + 2) * (Nzl + 2)];
	Point* BehindSend = new Point[(Nxl + 2) * (Nzl + 2)];

	Point* DownRecv = new Point[(Nxl + 2) * (Nyl + 2)];
	Point* UpRecv = new Point[(Nxl + 2) * (Nyl + 2)];
	Point* LeftRecv = new Point[(Nyl + 2) * (Nzl + 2)];
	Point* RightRecv = new Point[(Nyl + 2) * (Nzl + 2)];
	Point* FrontRecv = new Point[(Nxl + 2) * (Nzl + 2)];
	Point* BehindRecv = new Point[(Nxl + 2) * (Nzl + 2)];

	Point* GCP;

	Point* GDS;
	Point* GUS;
	Point* GLS;
	Point* GRS;
	Point* GFS;
	Point* GBS;

	Point* GDR;
	Point* GUR;
	Point* GLR;
	Point* GRR;
	Point* GFR;
	Point* GBR;
	float* error;


	int gsize = n * 5 * sizeof(float);
	int gsizfb = (Nxl + 2) * (Nzl + 2) * 5 * sizeof(float);
	int gsizlr = (Nyl + 2) * (Nzl + 2) * 5 * sizeof(float);
	int gsizud = (Nxl + 2) * (Nyl + 2) * 5 * sizeof(float);


	const int blocksize = 512;
	int gridsize = (n - 1) / blocksize + 1;

	double start;
	if (rank == 0)
	{
		start = MPI_Wtime();
	}

	Initvalues(Nxl, Nyl, Nzl, CenterPoints, px, py, pz, tauh2, h, Lx, Ly, Lz, n);

	cudaMalloc((void**)&GCP, gsize);
	cudaMemcpy(GCP, CenterPoints, gsize, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&GDS, gsizud);
	cudaMalloc((void**)&GUS, gsizud);
	cudaMalloc((void**)&GLS, gsizlr);
	cudaMalloc((void**)&GRS, gsizlr);
	cudaMalloc((void**)&GFS, gsizfb);
	cudaMalloc((void**)&GBS, gsizfb);

	cudaMalloc((void**)&GDR, gsizud);
	cudaMalloc((void**)&GUR, gsizud);
	cudaMalloc((void**)&GLR, gsizlr);
	cudaMalloc((void**)&GRR, gsizlr);
	cudaMalloc((void**)&GFR, gsizfb);
	cudaMalloc((void**)&GBR, gsizfb);
	float errorCPU;
	float* errorGPU;
	cudaMalloc((void**)&errorGPU, sizeof(float));

	Kernel1 << < gridsize, blocksize >> > (Nxl, Nyl, Nzl, GCP, GLS, GRS, GDS, GUS, GFS, GBS, tauh);

	cudaMemcpy(LeftSend, GLS, gsizlr, cudaMemcpyDeviceToHost);
	cudaMemcpy(RightSend, GRS, gsizlr, cudaMemcpyDeviceToHost);
	cudaMemcpy(UpSend, GUS, gsizud, cudaMemcpyDeviceToHost);
	cudaMemcpy(DownSend, GDS, gsizud, cudaMemcpyDeviceToHost);
	cudaMemcpy(BehindSend, GBS, gsizfb, cudaMemcpyDeviceToHost);
	cudaMemcpy(FrontSend, GFS, gsizfb, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	SendRecvValues(Nxl, Nyl, Nzl, px, py, pz, rx, ry, rz, Np, DownSend, UpSend, LeftSend, RightSend, BehindSend, FrontSend, DownRecv, UpRecv, LeftRecv, RightRecv, BehindRecv, FrontRecv);

	float ALLERROR;

	for (int t1 = 1; t1 < 20; t1 += 1)
	{

		cudaMemcpy(GLR, LeftRecv, gsizlr, cudaMemcpyHostToDevice);
		cudaMemcpy(GRR, RightRecv, gsizlr, cudaMemcpyHostToDevice);
		cudaMemcpy(GUR, UpRecv, gsizud, cudaMemcpyHostToDevice);
		cudaMemcpy(GDR, DownRecv, gsizud, cudaMemcpyHostToDevice);
		cudaMemcpy(GBR, BehindRecv, gsizfb, cudaMemcpyHostToDevice);
		cudaMemcpy(GFR, FrontRecv, gsizfb, cudaMemcpyHostToDevice);

		Kernel2 << < gridsize, blocksize >> > (Nxl, Nyl, Nzl, GCP, GLS, GRS, GDS, GUS, GFS, GBS, GLR, GRR, GDR, GUR, GFR, GBR, tauh);

		counterror << < 1, blocksize >> > (errorGPU, Nxl, Nyl, Nzl, GCP, tauh, at, Lx, Ly, Lz, t1);

		cudaMemcpy(&errorCPU, errorGPU, sizeof(float), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		MPI_Reduce(&errorCPU, &ALLERROR, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

		if (rank == 0)
		{
			cout << "Слой номер " << t1 << ", ошибка: " << ALLERROR * t1 << endl;
		}

		cudaMemcpy(LeftSend, GLS, gsizlr, cudaMemcpyDeviceToHost);
		cudaMemcpy(RightSend, GRS, gsizlr, cudaMemcpyDeviceToHost);
		cudaMemcpy(UpSend, GUS, gsizud, cudaMemcpyDeviceToHost);
		cudaMemcpy(DownSend, GDS, gsizud, cudaMemcpyDeviceToHost);
		cudaMemcpy(BehindSend, GBS, gsizfb, cudaMemcpyDeviceToHost);
		cudaMemcpy(FrontSend, GFS, gsizfb, cudaMemcpyDeviceToHost);


		SendRecvValues(Nxl, Nyl, Nzl, px, py, pz, rx, ry, rz, Np, DownSend, UpSend, LeftSend, RightSend, BehindSend, FrontSend, DownRecv, UpRecv, LeftRecv, RightRecv, BehindRecv, FrontRecv);

	}
	if (rank == 0)
	{
		double end = MPI_Wtime();
		printf("Time = %3.10f", end - start);
	}



	//cudaMemcpy(CenterPoints, GCP, gsize, cudaMemcpyDeviceToHost);
	//show(CenterPoints, n);

	cudaFree(GDS);
	cudaFree(GCP);
	cudaFree(GBR);
	cudaFree(GBS);
	cudaFree(GDR);
	cudaFree(GFR);
	cudaFree(GFS);
	cudaFree(GLR);
	cudaFree(GLS);
	cudaFree(GRR);
	cudaFree(GRS);
	cudaFree(GUR);
	cudaFree(GUS);
	cudaFree(error);

	cudaProfilerStop();
	MPI_Finalize();
}
