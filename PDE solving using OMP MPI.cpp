#include "mpi.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath> 
using namespace std;


#define Pi 3.141592653589793

struct Point {
	double x, y, z, value, oldvalue;
};

double uanalitical(double at, double Lx, double Ly, double Lz, double x, double y, double z, double t)
{
	return sin((x * Pi) / Lx) * sin((2 * y * Pi) / Ly) * sin((3 * z * Pi) / Lz) * cos(at * t);
}

double phi(double Lx, double Ly, double Lz, double x, double y, double z)
{
	return sin((x * Pi) / Lx) * sin((2 * y * Pi) / Ly) * sin((3 * z * Pi) / Lz);
}

void multi(int n, int &a1, int &a2, int &a3)
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

void show(Point* P, int n)
{
	for (int i = 0; i < n; i += 1)
	{
		cout << P[i].x << "  " << P[i].y << "  " << P[i].z << "  " << P[i].oldvalue << "  " << P[i].value << "  " << endl;
	}
}

int main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;

	MPI_Datatype dt_point;
	MPI_Type_contiguous(5, MPI_DOUBLE, &dt_point);
	MPI_Type_commit(&dt_point);


	// Начальные условия
	double Lx = Pi;
	double Ly = Pi;
	double Lz = Pi;
	double at = sqrt(1 / (Lx*Lx) + 4 / (Lx*Lx) + 9 / (Lz * Lz)) * Pi;
	double tau = 0.000000001;

	int Np = size;
	int P = atoi(argv[1]);
	double t = 0;
	double dt = tau;

	// Число точек для всей сетки
	int Nx = P;
	int Ny = P;
	int Nz = P;

	// Число разбиений по каждой оси
	int rx, ry, rz;
	multi(Np, rx, ry, rz);

	// Координаты блока в системе координат блока
	int pz = int(rank / (rx * ry));
	int py = int((rank - pz * rx * ry) / rx);
	int px = int(rank - pz * rx * ry - rx * py);



	// Число точек для каждого процесса
	int Nxl = Nx / rx;
	int Nyl = Ny / ry;
	int Nzl = Nz / rz;

	// Длина шага во всех координатах
	double h = Lx / (Nx - 1);
	double tauh = tau * tau / (h * h);
	double tauh2 = tau * tau / (2 * h * h);

	// Массив точек для каждого процесса
	int n = Nxl * Nyl * Nzl;

	Point* CenterPoints = new Point[n];

	// Заполнение начальных значений

	double y;
	double x;
	double z;
	double value;
	double start;
	if (rank == 0)
	{
		start = MPI_Wtime();
	}	

	 
	for (int k = 0; k < Nzl; k += 1)
	{
		for (int j = 0; j < Nyl; j += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				z = pz * Nzl * h + k * h;
				y = py * Nyl * h + j * h;
				x = px * Nxl * h + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].x = x;
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].y = y;
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].z = z;
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	double xmin, ymin, zmin, xmax, ymax, zmax;

	xmin = CenterPoints[0].x;
	xmax = CenterPoints[n - 1].x;
	ymin = CenterPoints[0].y;
	ymax = CenterPoints[n - 1].y;
	zmin = CenterPoints[0].z;
	zmax = CenterPoints[n - 1].z;

	Point* FrontPoints = new Point[Nxl * Nzl];
	Point* BehindPoints = new Point[Nxl * Nzl];
	Point* LeftPoints = new Point[Nyl * Nzl];
	Point* RightPoints = new Point[Nyl * Nzl];
	Point* UpPoints = new Point[Nxl * Nyl];
	Point* DownPoints = new Point[Nxl * Nyl];

	// Слой точек перед блоком FrontPoints

	if (ymin == 0)
	{
		y = ymin - h;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				FrontPoints[k * Nxl + i].x = x;
				FrontPoints[k * Nxl + i].y = y;
				FrontPoints[k * Nxl + i].z = z;
				FrontPoints[k * Nxl + i].oldvalue = 0;
				FrontPoints[k * Nxl + i].value = 0;
			}
		}
	}
	else
	{
		y = ymin - h;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				FrontPoints[k * Nxl + i].x = x;
				FrontPoints[k * Nxl + i].y = y;
				FrontPoints[k * Nxl + i].z = z;
				FrontPoints[k * Nxl + i].oldvalue = value;
				FrontPoints[k * Nxl + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	// Слой точек за блоком BehindPoints

	if (ymax == Ly)
	{
		y = ymax + h;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				BehindPoints[k * Nxl + i].x = x;
				BehindPoints[k * Nxl + i].y = y;
				BehindPoints[k * Nxl + i].z = z;
				BehindPoints[k * Nxl + i].oldvalue = 0;
				BehindPoints[k * Nxl + i].value = 0;
			}
		}
	}
	else
	{
		y = ymax + h;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				z = zmin + k * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				BehindPoints[k * Nxl + i].x = x;
				BehindPoints[k * Nxl + i].y = y;
				BehindPoints[k * Nxl + i].z = z;
				BehindPoints[k * Nxl + i].oldvalue = value;
				BehindPoints[k * Nxl + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	// Слой точек слева от блока LeftPoints

	if (xmin == 0)
	{
		x = Lx;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int j = 0; j < Nyl; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				LeftPoints[k * Nyl + j].x = x;
				LeftPoints[k * Nyl + j].y = y;
				LeftPoints[k * Nyl + j].z = z;
				LeftPoints[k * Nyl + j].oldvalue = value;
				LeftPoints[k * Nyl + j].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}
	else
	{
		x = xmin - h;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int j = 0; j < Nyl; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				LeftPoints[k * Nyl + j].x = x;
				LeftPoints[k * Nyl + j].y = y;
				LeftPoints[k * Nyl + j].z = z;
				LeftPoints[k * Nyl + j].oldvalue = value;
				LeftPoints[k * Nyl + j].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	// Слой точек справа от блока RightPoints
	if (xmax == Lx)
	{
		x = 0;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int j = 0; j < Nyl; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				RightPoints[k * Nyl + j].x = x;
				RightPoints[k * Nyl + j].y = y;
				RightPoints[k * Nyl + j].z = z;
				RightPoints[k * Nyl + j].oldvalue = value;
				RightPoints[k * Nyl + j].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}
	else
	{
		x = xmax + h;
		 
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int j = 0; j < Nyl; j += 1)
			{
				z = zmin + k * h;
				y = ymin + j * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				RightPoints[k * Nyl + j].x = x;
				RightPoints[k * Nyl + j].y = y;
				RightPoints[k * Nyl + j].z = z;
				RightPoints[k * Nyl + j].oldvalue = value;
				RightPoints[k * Nyl + j].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	// Слой точек снизу от блока DownPoints

	if (zmin == 0)
	{
		z = zmin - h;
		 
		for (int j = 0; j < Nyl; j += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				DownPoints[j * Nxl + i].x = x;
				DownPoints[j * Nxl + i].y = y;
				DownPoints[j * Nxl + i].z = z;
				DownPoints[j * Nxl + i].oldvalue = 0;
				DownPoints[j * Nxl + i].value = 0;
			}
		}
	}
	else
	{
		z = zmin - h;
		 
		for (int j = 0; j < Nyl; j += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				DownPoints[j * Nxl + i].x = x;
				DownPoints[j * Nxl + i].y = y;
				DownPoints[j * Nxl + i].z = z;
				DownPoints[j * Nxl + i].oldvalue = value;
				DownPoints[j * Nxl + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	// Слой точек сверху от блока UpPoints

	if (zmax == Lz)
	{
		z = zmax + h;
		 
		for (int j = 0; j < Nyl; j += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				UpPoints[j * Nxl + i].x = x;
				UpPoints[j * Nxl + i].y = y;
				UpPoints[j * Nxl + i].z = z;
				UpPoints[j * Nxl + i].oldvalue = 0;
				UpPoints[j * Nxl + i].value = 0;
			}
		}
	}
	else
	{
		z = zmax + h;
		 
		for (int j = 0; j < Nyl; j += 1)
		{
			for (int i = 0; i < Nxl; i += 1)
			{
				y = ymin + j * h;
				x = xmin + i * h;
				value = phi(Lx, Ly, Lz, x, y, z);
				UpPoints[j * Nxl + i].x = x;
				UpPoints[j * Nxl + i].y = y;
				UpPoints[j * Nxl + i].z = z;
				UpPoints[j * Nxl + i].oldvalue = value;
				UpPoints[j * Nxl + i].value = value + tauh2 * (phi(Lx, Ly, Lz, x - h, y, z) + phi(Lx, Ly, Lz, x + h, y, z) + phi(Lx, Ly, Lz, x, y - h, z) + phi(Lx, Ly, Lz, x, y + h, z) + phi(Lx, Ly, Lz, x, y, z - h) + phi(Lx, Ly, Lz, x, y, z + h) - 6 * value);
			}
		}
	}

	// Слои для отправки

	Point* DownSend = new Point[Nxl * Nyl];
	Point* UpSend = new Point[Nxl * Nyl];
	Point* LeftSend = new Point[Nyl * Nzl];
	Point* RightSend = new Point[Nyl * Nzl];
	Point* FrontSend = new Point[Nxl * Nzl];
	Point* BehindSend = new Point[Nxl * Nzl];

	double du;
	for (int t1 = 0; t1 < 20; t1 += 1)
	{
		t += dt;

		// Вычисление следующего временного слоя во внутренних точках

		for (int k = 1; k < Nzl - 1; k += 1)
		{
			for (int j = 1; j < Nyl - 1; j += 1)
			{
				for (int i = 1; i < Nxl - 1; i += 1)
				{
					Point current = CenterPoints[k * Nxl * Nyl + j * Nxl + i];
					value = current.value;
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
			}
		}



		// Вычисление следующего временного слоя в ближайшем слое
		int j = 0;
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int i = 0; i < Nxl - 1; i += 1)
			{
				Point current = CenterPoints[k * Nxl * Nyl + j * Nxl + i];
				value = current.value;
				if ((i == 0) && (k == 0))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if ((i == 0) && (k == Nzl - 1))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (i == 0)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (k == 0)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (k == Nzl - 1)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].oldvalue + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
			}
		}

		// Вычисление следующего временного слоя в правом слое
		int i = Nxl - 1;
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int j = 0; j < Nyl - 1; j += 1)
			{
				Point current = CenterPoints[k * Nxl * Nyl + j * Nxl + i];
				value = current.value;
				if ((j == 0) && (k == 0))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + RightPoints[k * Nyl + j].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if ((j == 0) && (k == Nzl - 1))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + RightPoints[k * Nyl + j].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (j == 0)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + RightPoints[k * Nyl + j].value + FrontPoints[k * Nxl + i].value + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (k == 0)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].value + RightPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (k == Nzl - 1)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].value + RightPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + RightPoints[k * Nyl + i].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
			}
		}


		// Вычисление следующего временного слоя в дальнем слое
		j = Nyl - 1;
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int i = 1; i < Nxl; i += 1)
			{
				Point current = CenterPoints[k * Nxl * Nyl + j * Nxl + i];
				value = current.value;
				if ((i == Nxl - 1) && (k == 0))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + RightPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if ((i == Nxl - 1) && (k == Nzl - 1))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + RightPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (i == Nxl - 1)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + RightPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (k == 0)
				{
					if (i == 1)
					{
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].value + BehindPoints[k * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
					}
					else
					{
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].value + BehindPoints[k * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
					}
				}
				else if (k == Nzl - 1)
				{
					if (i == 1)
					{
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].value + BehindPoints[k * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
					}
					else
					{
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].value + BehindPoints[k * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
					}
				}
				else
				{
					if (i == 1)
					{
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
					}
					else
					{
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
						CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
					}
				}
			}
		}

		// Вычисление следующего временного слоя в левом слое
		i = 0;
		for (int k = 0; k < Nzl; k += 1)
		{
			for (int j = 1; j < Nyl; j += 1)
			{
				Point current = CenterPoints[k * Nxl * Nyl + j * Nxl + i];
				value = current.value;
				if ((j == Nyl - 1) && (k == 0))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].oldvalue + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if ((j == Nyl - 1) && (k == Nzl - 1))
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].oldvalue + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (j == Nyl - 1)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].oldvalue + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + BehindPoints[k * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (k == 0)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else if (k == Nzl - 1)
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
				else
				{
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (LeftPoints[k * Nyl + j].value + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].oldvalue + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].value - 6 * value);
					CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
				}
			}
		}


		// Вычисление следующего временного слоя в нижнем слое
		int k = 0;
		for (int j = 1; j < Nyl - 1; j += 1)
		{
			for (int i = 1; i < Nxl - 1; i += 1)
			{
				Point current = CenterPoints[k * Nxl * Nyl + j * Nxl + i];
				value = current.value;
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + DownPoints[j * Nxl + i].value + CenterPoints[(k + 1) * Nxl * Nyl + j * Nxl + i].oldvalue - 6 * value);
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
			}
		}


		// Вычисление следующего временного слоя в верхнем слое
		k = Nzl - 1;
		for (int j = 1; j < Nyl - 1; j += 1)
		{
			for (int i = 1; i < Nxl - 1; i += 1)
			{
				Point current = CenterPoints[k * Nxl * Nyl + j * Nxl + i];
				value = current.value;
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].value = 2 * value - current.oldvalue + tauh * (CenterPoints[k * Nxl * Nyl + j * Nxl + i - 1].oldvalue + CenterPoints[k * Nxl * Nyl + j * Nxl + i + 1].value + CenterPoints[k * Nxl * Nyl + (j - 1) * Nxl + i].oldvalue + CenterPoints[k * Nxl * Nyl + (j + 1) * Nxl + i].value + CenterPoints[(k - 1) * Nxl * Nyl + j * Nxl + i].oldvalue + UpPoints[j * Nxl + i].value - 6 * value);
				CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue = value;
			}
		}

		// Подготовка слоев к отправке
		// Подготовка переднего слоя

		y = ymin;
		j = 0;

		for (int k = 0; k < Nzl; k += 1)
		{
			z = zmin + k * h;
			for (int i = 0; i < Nxl; i += 1)
			{
				x = xmin + i * h;
				FrontSend[k * Nxl + i].x = x;
				FrontSend[k * Nxl + i].y = y;
				FrontSend[k * Nxl + i].z = z;
				FrontSend[k * Nxl + i].oldvalue = CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue;
				FrontSend[k * Nxl + i].value = CenterPoints[k * Nxl * Nyl + j * Nxl + i].value;
			}
		}

		// Подготовка заднего слоя

		j = Nyl - 1;
		y = ymax;

		for (int k = 0; k < Nzl; k += 1)
		{
			z = zmin + k * h;
			for (int i = 0; i < Nxl; i += 1)
			{
				x = xmin + i * h;
				BehindSend[k * Nxl + i].x = x;
				BehindSend[k * Nxl + i].y = y;
				BehindSend[k * Nxl + i].z = z;
				BehindSend[k * Nxl + i].oldvalue = CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue;
				BehindSend[k * Nxl + i].value = CenterPoints[k * Nxl * Nyl + j * Nxl + i].value;
			}
		}

		// Подготовка левого слоя

		x = xmin;
		i = 0;

		for (int k = 0; k < Nzl; k += 1)
		{
			z = zmin + k * h;
			for (int j = 0; j < Nyl; j += 1)
			{
				y = ymin + j * h;
				LeftSend[k * Nyl + j].x = x;
				LeftSend[k * Nyl + j].y = y;
				LeftSend[k * Nyl + j].z = z;
				LeftSend[k * Nyl + j].oldvalue = CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue;
				LeftSend[k * Nyl + j].value = CenterPoints[k * Nxl * Nyl + j * Nxl + i].value;
			}
		}

		// Подготовка правого слоя

		x = xmax;
		i = Nxl - 1;

		for (int k = 0; k < Nzl; k += 1)
		{
			z = zmin + k * h;
			for (int j = 0; j < Nyl; j += 1)
			{
				y = ymin + j * h;
				RightSend[k * Nyl + j].x = x;
				RightSend[k * Nyl + j].y = y;
				RightSend[k * Nyl + j].z = z;
				RightSend[k * Nyl + j].oldvalue = CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue;
				RightSend[k * Nyl + j].value = CenterPoints[k * Nxl * Nyl + j * Nxl + i].value;
			}
		}

		// Подготовка нижнего слоя

		z = zmin;
		k = 0;

		for (int j = 0; j < Nyl; j += 1)
		{
			y = ymin + j * h;
			for (int i = 0; i < Nxl; i += 1)
			{
				x = xmin + i * h;
				DownSend[j * Nxl + i].x = x;
				DownSend[j * Nxl + i].y = y;
				DownSend[j * Nxl + i].z = z;
				DownSend[j * Nxl + i].oldvalue = CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue;
				DownSend[j * Nxl + i].value = CenterPoints[k * Nxl * Nyl + j * Nxl + i].value;
			}
		}

		// Подготовка верхнего слоя

		z = zmax;
		k = Nzl - 1;

		for (int j = 0; j < Nyl; j += 1)
		{
			y = ymin + j * h;
			for (int i = 0; i < Nxl; i += 1)
			{
				x = xmin + i * h;
				UpSend[j * Nxl + i].x = x;
				UpSend[j * Nxl + i].y = y;
				UpSend[j * Nxl + i].z = z;
				UpSend[j * Nxl + i].oldvalue = CenterPoints[k * Nxl * Nyl + j * Nxl + i].oldvalue;
				UpSend[j * Nxl + i].value = CenterPoints[k * Nxl * Nyl + j * Nxl + i].value;
			}
		}

		int UpComp, DownComp, LeftComp, RightComp, FrontComp, BehindComp;

		UpComp = rx * ry * (pz + 1) + py * rx + px;
		DownComp = rx * ry * (pz - 1) + py * rx + px;

		LeftComp = rx * ry * pz + py * rx + px - 1;
		RightComp = rx * ry * pz + py * rx + px + 1;

		FrontComp = rx * ry * pz + (py - 1)* rx + px;
		BehindComp = rx * ry * pz + (py + 1) * rx + px;



		if ((0 <= UpComp) && (UpComp < Np))
		{
			MPI_Send(UpSend, Nxl * Nyl, dt_point, UpComp, 1, MPI_COMM_WORLD);

			MPI_Recv(UpPoints, Nxl * Nyl, dt_point, UpComp, 1, MPI_COMM_WORLD, &status);
		}


		if ((0 <= DownComp) && (DownComp < Np))
		{
			MPI_Recv(DownPoints, Nxl * Nyl, dt_point, DownComp, 1, MPI_COMM_WORLD, &status);

			MPI_Send(DownSend, Nxl * Nyl, dt_point, DownComp, 1, MPI_COMM_WORLD);
		}


		if ((0 <= LeftComp) && (LeftComp < Np))
		{
			MPI_Send(LeftSend, Nyl * Nzl, dt_point, LeftComp, 1, MPI_COMM_WORLD);

			MPI_Recv(LeftPoints, Nyl * Nzl, dt_point, LeftComp, 1, MPI_COMM_WORLD, &status);
		}


		if ((0 <= RightComp) && (RightComp < Np))
		{
			MPI_Recv(RightPoints, Nyl * Nzl, dt_point, RightComp, 1, MPI_COMM_WORLD, &status);

			MPI_Send(RightSend, Nyl * Nzl, dt_point, RightComp, 1, MPI_COMM_WORLD);
		}


		if ((0 <= FrontComp) && (FrontComp < Np))
		{
			MPI_Send(FrontSend, Nxl * Nzl, dt_point, FrontComp, 1, MPI_COMM_WORLD);

			MPI_Recv(FrontPoints, Nxl * Nzl, dt_point, FrontComp, 1, MPI_COMM_WORLD, &status);
		}


		if ((0 <= BehindComp) && (BehindComp < Np))
		{
			MPI_Recv(BehindPoints, Nxl * Nzl, dt_point, BehindComp, 1, MPI_COMM_WORLD, &status);

			MPI_Send(BehindSend, Nxl * Nzl, dt_point, BehindComp, 1, MPI_COMM_WORLD);
		}
	}



	if (rank == 0)
	{
		double end = MPI_Wtime();
		printf("Time = %3.10f", end - start);
	}
	MPI_Finalize();
}