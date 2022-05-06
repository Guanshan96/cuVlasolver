#include <string>
#include <fstream>
#include <iostream>

#include "tinyxml2.h"
#include "vlasov1d.h"

using namespace tinyxml2;

#define SINT sizeof(int)
#define SDOU sizeof(double)

/// <summary>
/// Print matrix in column-major style
/// </summary>
/// <param name="A">: Matrix</param>
/// <param name="n">: Number of rows</param>
/// <param name="m">: Number of columns</param>
void printmatrix(double* A, int n, int m)
{
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			std::cout << A[i + j * n] << " ";
		}
		std::cout << "\n";
	}
}

/// <summary>
/// User API which reads the configuration of solver
/// and constructs a structure that contains them
/// </summary>
/// <param name="filename">: path of config file</param>
/// <returns></returns>
Particle loadplasma(std::string filename)
{
	XMLDocument doc;

	doc.LoadFile(filename.c_str());

	XMLElement* specie = doc.RootElement()->FirstChildElement("Specie");
	
	int nspecies = 0;

	while (specie != NULL)
	{
		nspecies++;

		specie = specie->NextSiblingElement();
	}

	specie = doc.RootElement()->FirstChildElement("Specie");

	Particle particles(nspecies);

	double* charge = new double[nspecies];
	double* cmratio = new double[nspecies];
	double* density = new double[nspecies];
	bool* isSave = new bool[nspecies];

	for (size_t i = 0; i < nspecies; i++)
	{
		if (0 == strcmp(specie->FindAttribute("save")->Value(), "true"))
			isSave[i] = true;
		else
			isSave[i] = false;

		charge[i] = std::stod(specie->FirstChildElement("charge")->GetText());
		cmratio[i] = std::stod(specie->FirstChildElement("cmratio")->GetText());
		density[i] = std::stod(specie->FirstChildElement("density")->GetText());

		specie = specie->NextSiblingElement();
	}

	particles.setsaveflag(isSave);
	particles.setcharge(charge);
	particles.setcmratio(cmratio);
	particles.setdensity(density);

	delete[] charge;  charge = NULL;
	delete[] cmratio; cmratio = NULL;
	delete[] density; density = NULL;
	delete[] isSave; isSave = NULL;
	return particles;
}

/// <summary>
/// User API which reads the initial condition of particle system
/// and constructs a structure that contains them
/// </summary>
/// <param name="filename">: path of config file</param>
/// <param name="nspecies">: number of particle specie</param>
/// <returns></returns>
Solver loadsolver(std::string filename, int nspecies)
{
	XMLDocument doc;

	doc.LoadFile(filename.c_str());

	Solver solver(nspecies);

	XMLElement* grid = doc.RootElement()->FirstChildElement("Grid");

	double xmax = std::stod(grid->FirstChildElement("xmax")->GetText());
	int xngrids = std::stoi(grid->FirstChildElement("xngrids")->GetText());

	solver.setxgrid(xmax, xngrids);

	XMLElement* temp = grid->FirstChildElement("velGrid");

	double* vmin = new double[nspecies];
	double* vmax = new double[nspecies];
	int* vngrids = new int[nspecies];

	for (size_t i = 0; i < nspecies; i++)
	{
		vmin[i] = std::stod(temp->FirstChildElement("vmin")->GetText());
		vmax[i] = std::stod(temp->FirstChildElement("vmax")->GetText());
		vngrids[i] = std::stoi(temp->FirstChildElement("vngrids")->GetText());

		temp = temp->NextSiblingElement();
	}

	solver.setvmin(vmin);
	solver.setvmax(vmax);
	solver.setvngrids(vngrids);

	temp = doc.RootElement()->FirstChildElement("Temporal");

	solver.settemporal(std::stod(temp->FirstChildElement("tstep")->GetText()),
		std::stoi(temp->FirstChildElement("ntsteps")->GetText()));

	grid = doc.RootElement()->FirstChildElement("Config");
	temp = grid->FirstChildElement("diagnostics")->FirstChildElement("path");

	solver.path = temp->GetText();

	temp = temp->NextSiblingElement();

	int nvars = 0;

	while (temp != NULL)
	{
		nvars++;

		temp = temp->NextSiblingElement();
	}

	solver.initdiagvars(nvars);
	
	temp = grid->FirstChildElement("diagnostics")->FirstChildElement("path");
	temp = temp->NextSiblingElement();

	nvars = 0;
	while (temp != NULL)
	{
		std::string name = temp->FirstChildElement("name")->GetText();
		int rate = std::stoi(temp->FirstChildElement("rate")->GetText());
		temp = temp->NextSiblingElement();

		solver.setdiagvars(name, rate, nvars);
		nvars++;
	}

	solver.nvars = nvars;

	delete[] vmin;
	delete[] vmax;
	delete[] vngrids;

	return solver;
}

/// <summary>
/// User API used to print initial conditions on console
/// </summary>
/// <param name="particle">: structure which contains initial particle system</param>
void printplasma(Particle particle)
{

}

/// <summary>
/// User API used to print solver config on console
/// </summary>
/// <param name="solver">: structure which contains solver configurations</param>
void printsolver(Solver solver)
{

}

/// <summary>
/// User API which used to save data as a binary .dat file.
/// Note the old file will be overwritten if filename are same.
/// </summary>
/// <param name="filename">: full path of file</param>
/// <param name="data">: pointer to data on host memory</param>
/// <param name="shape">: shape of data cubic</param>
/// <param name="ndim">: number of dimensions</param>
void save(std::string filename, double *data, int *shape, int ndim)
{
	/*
	* 
	*	STRUCTURE OF FILE (number at left side is position in byte):
	*	0: number of dimensions
	*	4: size of first dimension
	*	...
	*	4n: size of nth dimension
	*	4n+4: total length of the data structure (in number of elements)
	*	4n+8 to eof: data (e.g. double)
	* 
	*/

	std::ofstream outFile;
	outFile.open(filename, std::ios::out | std::ios::binary);

	int length = 1;

	outFile.write((char*)&ndim, sizeof(int));

	for (size_t i = 0; i < ndim; length=length*shape[i++])
	{
		outFile.write((char*)&shape[i], sizeof(int));
	}

	outFile.write((char*)&length, sizeof(int));

	for (size_t i = 0; i < length; i++)
	{
		outFile.write((char*)&data[i], sizeof(double));
	}

	outFile.close();
}

/// <summary>
/// User API which used to save a piece of data into
/// a file stream.
/// </summary>
/// <param name="stream">: Write-only stream to a file on hard disk</param>
/// <param name="data">: Data array</param>
/// <param name="length">: Total length (number of elements) of data array</param>
/// <param name="isCuda">: Indicates if the data pointer is on v-RAM</param>
void save(std::ofstream& stream, double* data, int length, char isCuda)
{
	double* data_temp = new double[length];
	switch (isCuda)
	{
	case 'y':

		cudaMemcpy(data_temp, data, sizeof(double) * length, cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < length; i++)
		{
			stream.write((char*)&data_temp[i], sizeof(double));
		}
		break;
	case 'n':
		for (size_t i = 0; i < length; i++)
		{
			stream.write((char*)&data[i], sizeof(double));
		}
		break;
	default:
		break;
	}

	delete[] data_temp;
}

/// <summary>
/// User API which used to save size of data (header) into
/// a file stream.
/// </summary>
/// <param name="stream">: Write-only stream to a file on hard disk</param>
/// <param name="shape">: Shape of data array</param>
/// <param name="ndim">: Dimension of data array</param>
void save(std::ofstream& stream, int* shape, int ndim)
{
	int length = 1;

	stream.write((char*)&ndim, sizeof(int));

	for (size_t i = 0; i < ndim; length = length * shape[i++])
	{
		stream.write((char*)&shape[i], sizeof(int));
	}

	stream.write((char*)&length, sizeof(int));
}


Memorystream::Memorystream()
{
	filename = "";
	bufferlength = 0;

	offset = 0;

	buffer = NULL;
	first = NULL;
}

Memorystream::Memorystream(std::string filename, int bufferlength)
{
	this->filename = filename;
	this->bufferlength = bufferlength;

	buffer = new char[bufferlength];

	offset = 0;

	first = buffer;
}

void Memorystream::save(int* shape, int ndim)
{
	int length = 1;

	memcpy_s(buffer, SINT, (char*)&ndim, SINT);

	buffer += 4;

	memcpy_s(buffer, SINT, (char*)shape, SINT);
	buffer += 4 * ndim;

	for (size_t i = 0; i < ndim; length = length * shape[i++]) {

	}

	memcpy_s(buffer, SINT, (char*)&length, SINT);
	buffer += 4;
}

void Memorystream::save(double* data, int length, char isCuda, char isLift)
{
	cudaError_t err;

	switch (isCuda)
	{
	case 'y':

		err = cudaMemcpy(buffer + offset, data, SDOU * length, cudaMemcpyDeviceToHost);

		break;
	case 'n':
		
		memcpy_s(buffer + offset, SDOU * length, (char*)data, SDOU * length);

		break;
	default:
		break;
	}

	if (isLift == 'y')
		buffer += SDOU * length;
}

void Memorystream::save()
{
	std::ofstream outFile;
	outFile.open(filename, std::ios::out | std::ios::binary);

	outFile.write(buffer, bufferlength*sizeof(char));

	outFile.close();
}

void Memorystream::moveto(int position)
{
	buffer = (char*)first + position;
}

void Memorystream::lift(int liftoffset)
{
	buffer += liftoffset;
}

void Memorystream::setoffset(int offset)
{
	this->offset = offset;
}

void Memorystream::dispose()
{
	delete[] buffer;
}