#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include "field.c"
#include <time.h>
//#include <omp.h>

// Exemplo de compilação do programa
// mpicc -g -Wall -o mpi_prop_working.bin mpi_prop_working.c

// Exemplo de execução do programa
// mpiexec -n 2 mpi_prop_working.bin VEL_210x427x427.ad 427 427 210 20 20 20 0.003 3720 11840 500 8660 3DSEGEAGE.su 544 2286908 
// mpiexec -n <comm_sz> mpi_separa_tiro.bin <velocityModel_file> <Xi> <Yi> <Zi> <dX> <dY> <dZ> <dt> <traces_file.su> <ntssMax> <ntt>

//Resolucao do modelo de velocidades é o dobro da resolucao do sismograma
//dX = 20, dY = 20, dz = 20

//Menor valor de dt = 0,001115573 - Teste 1
//Menor valor de dt = 0,002576306 - Teste 2

//Limites do mapeamento
//gx       3720 - 11840
//gy       500 - 8660

//Verificar variáveis interpoladas e não interp
//Verificar pq primeiro retroP é 0
//Melhorar funções

//Fazer para Multiplos tiros

//Melhorar Propagação
	//Fazer bordas não reflexivas
	//Melhorar Restrição espacial

#define TH 	240 	// Trace Header bytes
#define ncx 	5 	// number of coefficients of the finite difference in x: order/2 + 1
#define ncy 	5 	// number of coefficients of the finite difference in y: order/2 + 1
#define ncz 	5 	// number of coefficients of the finite difference in z: order/2 + 1
#define freq 	25.0 	// source frequency

typedef struct { // 240-byte Trace Header + Data
    int tracl; // trace sequence number within line
    int tracr; // trace sequence number within reel
    int fldr; // field record number
    int tracf; // trace number within field record
    int ep; // energy source point number
    int cdp; // CDP ensemble number
    int cdpt; // trace number within CDP ensemble
    short trid; // trace identification code
    short nvs; // number of vertically summed traces
    short nhs; // number of horizontally summed traces
    short duse; // data use
    int offset; // distance from source point to receiver group
    int gelev; // receiver group elevation from sea level
    int selev; // source elevation from sea level
    int sdepth; // source depth
    int gdel; // datum elevation at receiver group
    int sdel; // datum elevation at source
    int swdep; // water depth at source
    int gwdep; // water depth at receiver group
    short scalel; // scale factor for previous 7 entries
    short scalco; // scale factor for next 4 entries
    int sx; // X source coordinate
    int sy; // Y source coordinate
    int gx; // X group coordinate
    int gy; // Y group coordinate
    short counit; // coordinate units code
    short wevel; // weathering velocity
    short swevel; // subweathering velocity
    short sut; // uphole time at source
    short gut; // uphole time at receiver group
    short sstat; // source static correction
    short gstat; // group static correction
    short tstat; // total static applied
    short laga; // lag time A
    short lagb; // lag time B
    short delrt; // delay recording time
    short muts; // mute time--start
    short mute; // mute time--end
    unsigned short ns; // number of samples in this trace
    unsigned short dt; // sample interval
    short gain; // gain type of field instruments code
    short igc; // instrument gain constant
    short igi; // instrument early or initial gain
    short corr; // correlated
    short sfs; // sweep frequency at start
    short sfe; // sweep frequency at end
    short slen; // sweep length in ms
    short styp; // sweep type code
    short stas; // sweep trace length at start in ms
    short stae; // sweep trace length at end in ms
    short tatyp; // taper type
    short afilf; // alias filter frequency if used
    short afils; // alias filter slope
    short nofilf; // notch filter frequency if used
    short nofils; // notch filter slope
    short lcf; // low cut frequency if used
    short hcf; // high cut frequncy if used
    short lcs; // low cut slope
    short hcs; // high cut slope
    short year; // year data recorded
    short day; // day of year
    short hour; // hour of day
    short minute; // minute of hour
    short sec; // second of minute
    short timbas; // time basis code
    short trwf; // trace weighting factor
    short grnors; // geophone group number of roll switch position one
    short grnofr; // geophone group number of trace one within original field record
    short grnlof; // geophone group number of last trace within original field record
    short gaps; // gap size
    short otrav; // overtravel taper code
    int x1; // X coordinate of ensemble (CDP) position of this trace
    int y1; // Y coordinate of ensemble (CDP) position of this trace
    int x2; // For 3-D poststack data, this field should be used for the in-line number
    int y2; // For 3-D poststack data, this field should be used for the cross-line number
    float spn; // Shotpoint number
    short sspn; //Scalar to be applied to the shotpoint number
    short tvmu; // Trace value measurement unit
    int tcm; // Transduction Constant Mantissa
    short tce; // Transduction Constant Expoente
    short tu; // Transduction Units
    short dti; // Device/Trace Identifier
    short sth; // Scalar to be applied to times specified in Trace Header
    short sto; // Source Type/Orientation
    short sed[3]; // Source Energy Direction with respect to the source orientation
    int smm; // Source Measurement Mantissa
    short sme; // Source Measurement Expoente
    short smu; // Source Measurement Unit
    short unass[4]; // Unassigned
    float * data; // Data
} SuTrace;


//Function declarations
void usage(char* errorMessage, int rank);
void printFile(SuTrace* traces, unsigned long localTraceNumber, unsigned int TD, float* velocity_model_data,  unsigned long vModelSize, int my_rank);
void propagation(float *vel, unsigned int localXi, unsigned int localYi, int Yi, int Zi, int sZi, float dx, float dy, float dz, float *tr, unsigned short Ti, float dt, int *trc, int Ntr, int wri, char *wffn, int bw, int my_rank, unsigned int shotNumber, short pMode, double divisor, double sDivisor, double sDivY, double sDivZ, float *localImage);
void invBytes(void * valor, int nBytes);
float source(float t);
float minDs(float dx, float dy, float dz);
float maxDs(float dx, float dy, float dz);
void readVelData(float *vel, float *vMin, float *vMax, float *sdt, float *rdt, unsigned short *sTi, double *divisor, float *sdX, float *sdY, float *sdZ, float *rds, double *sDivisor, unsigned short ns, int Xi, int Yi, int Zi, unsigned long Xbi, unsigned long Ybi, unsigned long Zbi, const char* velFileName, unsigned int bw, float dX, float dY, float dZ, float dt, int *sXi, int *sYi, int *sZi, double *sDivY, double *sDivZ);
void getComShotTraces(FILE* suFile, SuTrace* traces, float* tracesData, int* tracesC, float* tr, int* trc, unsigned long* traceNumber, unsigned long* localTraceNumber, unsigned int* localXi, unsigned int* localYi, unsigned int* localGxMinC, unsigned int* localGyMinC, unsigned int gxMin, unsigned int gyMin, unsigned long ntt, unsigned short ns, unsigned int TD, float dX, float dY, float dZ, float dt, float sdX, float sdY, float sdZ);

int main(int argc, char* argv[]) {
    
    //Variable declarations
    //SuTrace* localTraces;		//Local Traces Data		
    //float* vModelData;			//Local Velocity Model Data
    //unsigned long localTraceNumber;  	//Local Trace Number
    unsigned long traceNumber;       	//Global Trace Number counter 
    unsigned int  shotNumber;        	//Number of shots counter
    int sx; 				//Position x of shot source
    int sy; 				//Position y of shot source
    int sz; 				//Position z of shot source
    unsigned short ns;			//Number of Samples
    unsigned long ntssMax;		//Maximum Number of Traces of the same shot
    unsigned int shotLimit;		//Number of shots analyzed
    unsigned int localShotNumberLimit;	//Number of shots analyzed for each process
    unsigned long ntt;			//Total Number of traces in SU file
    unsigned int TD;			//Size of trace's data (in Bytes)
    unsigned int i;			//Iterator
    //FILE *vModelFile;			//File containing velocity Model data
    int my_rank;                         //Rank of process
    int comm_sz;                         //Number of processes
    //MPI_Comm comm; 			//Processes' Communicator
    //MPI_Status status;			//Status of MPI communication
    int Xi;				//Velocity Model length (X axis)
    int Yi;				//Velocity Model width  (Y axis)
    int Zi;				//Velocity Model depth  (Z axis)
    float dX;				//Velocity Model length step (X axis)
    float dY;				//Velocity Model width step  (Y axis)
    float dZ;				//Velocity Model depth step  (Z axis)
    float dt;				//Propagation wave time step (T axis)
    float rdt;				//Propagation wave time step (T axis) restricted
    float sdt;				//Propagation wave time step (T axis) corrected
    //float *str;                          //Traces data corrected
    unsigned short sTi;			//Number of Samples corrected
    unsigned int gxMin;			//Distance of the first hidrophone in X axis (in meters)
    //unsigned int gxMax;			//Distance of the last  hidrophone in X axis (in meters)
    unsigned int gyMin;			//Distance of the first hidrophone in Y axis (in meters)
    //unsigned int gyMax;			//Distance of the last  hidrophone in Y axis (in meters)
    unsigned long Xbi;			//Length of Velocity Model with border (X axis)
    unsigned long Ybi; 			//Width of  Velocity Model with border (Y axis)
    unsigned long Zbi;			//Depth of  Velocity Model with border (Z axis) 
    //unsigned long indb;			//Index counter of Velocity Model with border
    //unsigned long ind;			//Index counter of Velocity Model
    //int xi;				//Index counter of Velocity Model length (Xi) 
    //int xbi;				//Index counter of Velocity Model with border length (Xi)
    //int yi;				//Index counter of Velocity Model width  (Yi)
    //int ybi; 				//Index counter of Velocity Model with border width  (Yi)
    //int zi;				//Index counter of Velocity Model depth  (Zi)
    //int zbi;				//Index counter of Velocity Model with border depth  (Xi)
    unsigned int bw;			//Border Width applied to Velocity Model
    float *vel;				//Velocity Model with border
    int curSx;				//Position x of shot source of the current trace
    int curSy;				//Position y of shot source of the current trace
    //unsigned int curGx;			//Position x of hidrophone of the current trace
    //unsigned int curGy;			//Position y of hidrophone of the current trace
    unsigned long curTraceNumber;        //Local Trace Number Counter
    //unsigned int  dest;		//Destination process index
    FILE *suFile;			//SU File
    SuTrace* traces;			//SU Traces
    //float* iTracesData;			//SU Traces interpolated
    float *tr;                           //Traces data
    int *trc;				//Traces data coordinate (row, column, depth)
    //int ti; 				//Index counter of propagation wave time
    int Ntr;				//Number of Traces in Propagation
    unsigned int localXi; 		//Velocity Model length (X axis) used in the current shot
    unsigned int localYi; 		//Velocity Model width  (Y axis) used in the current shot
    //unsigned int localGxMin; 		//Distance of the first hidrophone in X axis (in meters) of the current shot
    //unsigned int localGxMax;		//Distance of the last  hidrophone in X axis (in meters) of the current shot
    //unsigned int localGyMin;	 	//Distance of the first hidrophone in Y axis (in meters) of the current shot
    //unsigned int localGyMax;		//Distance of the last  hidrophone in Y axis (in meters) of the current shot
    unsigned int localGxMinC;		//Coordinate of the first hidrophone in X axis (in rows) of the current shot
    unsigned int localGyMinC;		//Coordinate of the first hidrophone in Y axis (in columns) of the current shot
    char fileName [20];			//Name of a file written by the program
    float vMin;				//Minimum velocity in velocity model
    float vMax;                          //Maximum velocity in velocity model
    double divisor;			//Divisor applied to dt accomplish time restriction
    /*unsigned int fIndex;			//Floor Index used in Interpolation
    unsigned int cIndex;			//Ceil Index used in Interpolation
    float fCoef;				//Coeficient of floor portion in Interpolation
    float cCoef;				//Coeficient of ceil portion in Interpolation*/
    int rest;
    int localCeilShotNumber;
    unsigned int localFirstShotNumber; 
    float* tracesData;
    int* tracesC;
    float sdX, sdY, sdZ, rds; 
    double sDivisor, sDivY, sDivZ;
    int sXi, sYi, sZi;

    float* localImage;
    float* image;
    
    
    //MPI start
    /*MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);
    */
    my_rank = 0;
    comm_sz = 1;
   
    clock_t start = clock(), diff;

    // Check command line arguments
    if (argc != 16) 
    {
        usage(argv[0], my_rank); 
    }
    
    // Get arguments from function call in command line
    Xi = atoi(argv[2]);
    Yi = atoi(argv[3]);
    Zi = atoi(argv[4]);
    dX = atof(argv[5]);
    dY = atof(argv[6]);
    dZ = atof(argv[7]);
    dt = atof(argv[8]);   
    gxMin = atoi(argv[9]);
    //gxMax = atoi(argv[10]);
    gyMin = atoi(argv[11]);
    //gyMax = atoi(argv[12]);
    ntssMax = atol(argv[14]);
    ntt = atol(argv[15]);
        
    //Variables Initialization
    //localTraceNumber = 0; 
    traceNumber = 0;       
    shotNumber = 0;
    curSx = 0;		
    curSy = 0;			
    curTraceNumber = 0;  
    divisor = 1;
    vMin = FLT_MAX;
    vMax = 0;
    shotLimit = 1;
    bw = 50;
    Xbi = Xi + 2*bw;
    Ybi = Yi + 2*bw;
    Zbi = Zi + 2*bw;
    
    
    // cria imagem local zerada
    //localImage = (uint8_t*) calloc(Xi*Yi*Zi, sizeof(uint8_t));
    
    //Allocate velocity model with border
    vel = (float *) malloc(Xbi*Ybi*Zbi*sizeof(float));
    if (vel == NULL) {
        printf("Memory allocation failed: vel.\n");
        return 0;
    }  
    
    //Open SU file
    suFile = fopen(argv[13], "rb");
    if (suFile == NULL)
    {
        fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[9]);
        exit(0);
    }
    
    
    //Get number of samples (ns)	   
    fseek(suFile, 114, SEEK_CUR);
    if (fread(&ns, 1, sizeof(unsigned short), suFile) != sizeof(unsigned short)) {
        printf("getSuTrace failed!\n");
        exit(0);
    }
    fseek(suFile, -116, SEEK_CUR);
    
    //Invert ns bytes (Big Endian -> Little Endian)
    invBytes(&ns, sizeof(unsigned short));
    
    
    //printf("My_rank: %d\tRealizando expansão do modelo de velocidades\n", my_rank);
    

    
    
    //Read Vel Data optimize and apply borders
    readVelData(vel, &vMin, &vMax, &sdt, &rdt, &sTi, &divisor, &sdX, &sdY, &sdZ, &rds, &sDivisor, ns, Xi, Yi, Zi, Xbi, Ybi, Zbi, argv[1], bw, dX, dY, dZ, dt, &sXi, &sYi, &sZi, &sDivY, &sDivZ);
        

    //Test output readVelData

    //Output Velocity Model files
    FILE *output_file;
    sprintf(fileName, "vel.ad");
    output_file = fopen(fileName, "wb");
    
    if (output_file == NULL)
    {
        fprintf(stderr, "Erro ao abrir arquivo output\n");
        exit(0);
    }
    
    
    fwrite(vel, 1, Xbi*Ybi*Zbi*sizeof(float), output_file);
    fflush(output_file);
    
    fclose(output_file);

    printf("My_rank: %d\tvMin: %.3f\tvMax: %.3f\tsdt: %.4f\trdt: %.4f\tsTi: %d\tdivisor: %.3f\n", my_rank, vMin, vMax, sdt, rdt, sTi, divisor);
    printf("My_rank: %d\trds: %.4f\tsXi: %d\tsYi: %d\tsZi: %d\tsdX: %.4f\tsdY: %.4f\tsdZ: %.4f\tsDivisor: %.2f\n", my_rank, rds, sXi, sYi, sZi, sdX, sdY, sdZ, sDivisor);

    
    
    
    
    
    
    /***** End of Get Velocity Model *****/
    
    
    
    /*****  Get Traces *****/  
    
    //Calculate Trace Data Bytes
    TD = ns*sizeof(float);
    
    //Allocate Image		   
    //image = (uint8_t*) calloc(Xi*Yi*Zi, sizeof(uint8_t));
    
    
    
    //shotLimit = 11; 0, 3, 6, 9
    //resto = 2;
    //comm_sz = 4;
    
    rest = shotLimit % comm_sz;
    localCeilShotNumber = (int) ceil((double)shotLimit/(double)comm_sz);
    localFirstShotNumber = my_rank*localCeilShotNumber + 2000;
    shotNumber = 0;
    
    
    //Read other local Shots
    while(traceNumber < ntt && shotNumber < localFirstShotNumber)
    {
        
        //Get Next Sdepth, Sx, Sy                    
        fseek(suFile, 72, SEEK_CUR);
        if (fread(&curSx, 1, sizeof(int), suFile) != sizeof(int)) {
            printf("getSuTrace failed!\n");
            exit(0);
        }
        if (fread(&curSy, 1, sizeof(int), suFile) != sizeof(int)) {
            printf("getSuTrace failed!\n");
            exit(0);
        }
        fseek(suFile, -80, SEEK_CUR);
        
        
        //Invert bytes (Big Endian -> Little Endian)
        invBytes(&curSx, sizeof(int));
        invBytes(&curSy, sizeof(int));	
        
        
        //Initialize variables of the current shot
        sx = curSx;
        sy = curSy;
        
        while(sx == curSx && sy == curSy && traceNumber < ntt)	//Get traces from the same shot
        {
            //Jump Trace
            fseek(suFile, TH+TD, SEEK_CUR);
            
            //Increment variables
            traceNumber++;
            
            //Get Next Sx and Sy   
            if(traceNumber < ntt)
            {
                fseek(suFile, 72, SEEK_CUR);
                if (fread(&curSx, 1, sizeof(int), suFile) != sizeof(int)) {
                    printf("getSuTrace failed!\n");
                    exit(0);
                }
                if (fread(&curSy, 1, sizeof(int), suFile) != sizeof(int)) {
                    printf("getSuTrace failed!\n");
                    exit(0);
                }
                fseek(suFile, -80, SEEK_CUR);
                
                invBytes(&curSx, sizeof(int));
                invBytes(&curSy, sizeof(int));
            }
        }
        shotNumber++;
    }
    
    //Calculate local ShotNumberLimit
    if(rest != 0 && shotLimit >= comm_sz)
    {
        if(my_rank > rest)
        {
            localShotNumberLimit = shotNumber+rest; 
        }
        else
        {
            localShotNumberLimit = shotNumber+localCeilShotNumber;	   	
        }
    }
    else if(shotLimit >= comm_sz)
    {
        localShotNumberLimit = shotNumber+localCeilShotNumber;
    }
    else
    {
        if(my_rank < rest)
        {
            localShotNumberLimit = shotNumber+localCeilShotNumber;
        }
        else
        {
            localShotNumberLimit = shotNumber;
        }
    }	
    
    printf("My_rank: %u\tNumberShot: %u\tLocalShotNumberLimit: %u\tTraceNumber: %lu\n", my_rank, shotNumber, localShotNumberLimit, traceNumber);
    
    
    //Allocate Traces
    traces = (SuTrace*) malloc(ntssMax*sizeof(SuTrace));
    for (i = 0; i < ntssMax; i++)
    {
        traces[i].data = (float *) malloc(TD);
    }

    Ntr = 1;  //Number of traces (sources)

    //Allocate traces coordinates and samples of traces
    trc = (int *) malloc(Ntr*3*sizeof(int));
    tr = (float *) malloc(Ntr*ns*sizeof(float));
    tracesC = (int *) malloc(ntssMax*3*sizeof(int)); 
    tracesData = (float *) malloc(ntssMax*ns*sizeof(float));
    if(my_rank == 0)
    {
	image = (float *) malloc(sXi*sYi*sZi*sizeof(float));
    }
	

    //Read local shots
    while(traceNumber < ntt && shotNumber < localShotNumberLimit)
    {             
        //Get Commom Shot Traces
        getComShotTraces(suFile, traces, tracesData, tracesC, tr, trc, &traceNumber, &curTraceNumber, &localXi, &localYi, &localGxMinC, &localGyMinC, gxMin, gyMin, ntt, ns, TD, dX, dY, dZ, dt, sdX, sdY, sdZ);        
        
        //printf("My_rank: %u\tNumberShot: %u\tLocalXi: %u\tLocalYi: %u\tlocalGxMinC = %u\tlocalGyMinC = %u\t numTraco: %lu\n", my_rank, shotNumber, localXi, localYi, localGxMinC, localGyMinC, curTraceNumber);

        printf("My_rank: %u\tNumberShot: %u\tLocalXi: %u\tLocalYi: %u\tZi: %d\t numTraco: %lu\n", my_rank, shotNumber, localXi, localYi, Zi, curTraceNumber);


        //printf("Fonte:\tTrc[0]: %d\tTrc[1]: %d\t Trc[2]: %d\n", trc[0], trc[1], trc[2]);
        //printf("Geofone1:\tTrc[0]: %d\tTrc[1]: %d\t Trc[2]: %d\n", tracesC[0], tracesC[1], tracesC[2]);
        //printf("Geofone-1:\tTrc[0]: %d\tTrc[1]: %d\t Trc[2]: %d\n", tracesC[0+(curTraceNumber-1)*3], tracesC[1+(curTraceNumber-1)*3], tracesC[2+(curTraceNumber-1)*3]);

        //Output Traces Data
    	FILE *output_file;
   	sprintf(fileName, "tracos.bin");
   	output_file = fopen(fileName, "wb");
    
   	if (output_file == NULL)
  	{
	         fprintf(stderr, "Erro ao abrir arquivo output\n");
        	 exit(0);
   	}
    
    
	fwrite(tracesData, 1, curTraceNumber*ns*sizeof(float), output_file);
    	fflush(output_file);
        fclose(output_file);
 
        printf("Escreveu tracos");         
            
        
        //Update propagation filename
        sprintf(fileName, "diretaTest_%d.bin", my_rank);                     
        
	localImage = (float *) malloc(localXi*localYi*sZi*sizeof(float));
 
        propagation( &(vel[localGxMinC*Ybi*Zbi + localGyMinC*Zbi]), localXi, localYi, Yi, Zi, sZi, sdX, sdY, sdZ, tr, sTi, sdt, trc, Ntr, 10, fileName, bw, my_rank, shotNumber, 1, divisor, sDivisor, sDivY, sDivZ, localImage);
        
	//propagation( &(vel[localGxMinC*Ybi*Zbi + localGyMinC*Zbi]), localXi, localYi, Yi, Zi, sZi, sdX, sdY, sdZ, tracesData, sTi, sdt, tracesC, curTraceNumber, 1, fileName, bw, my_rank, shotNumber, -1, divisor, sDivisor, sDivY, sDivZ, localImage); 


        free(localImage);
 
        shotNumber++;//Update number of shot

    }
    
    printf("\nMy_rank: %d\tCalculo dos traços concluído com sucesso!\n", my_rank);

    fclose(suFile);
    free(image);
    free(trc);
    free(tr);
    free(tracesData);
    free(tracesC);
    for (i = 0; i < ntssMax; i++)
    {
    	free(traces[i].data);
    }
    free(traces);
    //Free pointers
    free(vel);

    /*****  End of Get Traces *****/
    
    
    
    // adicionar a imagem local
    //MPI_Reduce(localImage, image, Xi*Yi*Zi, MPI_UINT8_T, MPI_SUM, 0, comm);
    
    
    //Print data to File
    //printFile(traces, localTraceNumber, TD, vel, Xbi*Ybi*Zbi, my_rank);
    
   
    
    /*
     *   free(localImage);
     *   free(image);
     *   free(u_i); 
     *   free(u_r);
     */

    diff = clock() - start;
    unsigned long sec = diff / CLOCKS_PER_SEC;

    printf("Time taken %lu seconds", sec);  
    //End of MPI Processes
    MPI_Finalize();
    
    return 0;
} 



void usage(char prog_name[], int my_rank) {
    if(my_rank == 0)
    {
        fprintf(stderr, "usage: %s ", prog_name); 
        fprintf(stderr, "<velocityModel_file> <Xi> <Yi> <Zi> <dX> <dY> <dZ> <dt> <traces_file.su> <ntssMax> <ntt> \n\n");
    }
    MPI_Finalize();
    exit(0);
} 


void invBytes(void * valor, int nBytes) { // Inverte ordem dos bytes
    
    int i;
    uint8_t swap, bytes[nBytes];
    
    memcpy(bytes, valor, nBytes);
    
    for (i = 0; i < nBytes/2; i++) 
    {
        swap = bytes[i];
        bytes[i] = bytes[nBytes-i-1];
        bytes[nBytes-i-1] = swap;
    }
    
    memcpy(valor, bytes, nBytes);
    
}


float minDs(float dx, float dy, float dz)
{
    if(dx <= dy)
    {
        if(dx <= dz)
        {
            return dx;
        }
        else
        {
            return dz;
        }		
    }
    else
    {
        if(dy <= dz)
        {
            return dy;
        }
        else
        {
            return dz;
        }
    }
}


float maxDs(float dx, float dy, float dz)
{
    if(dx >= dy)
    {
        if(dx >= dz)
        {
            return dx;
        }
        else
        {
            return dz;
        }		
    }
    else
    {
        if(dy >= dz)
        {
            return dy;
        }
        else
        {
            return dz;
        }
    }
}


float source(float t) { // Função da fonte
    return (1 - 2*pow(M_PI*freq*t,2))*exp(-pow(M_PI*freq*t,2));
}

void SwapPointers(float **pa, float **pb) {
    float *pc;
    pc  = *pa;
    *pa = *pb;
    *pb = pc;
}

/* 
 * "propagation" makes the forward of the acoustic wave
 * vm: velocity model data
 * Xi, Yi, Zi: velocity model dimensions n3, n2, n1
 * dx, dy, dz: spatial resolution
 * tr: source/seismic traces data
 * Ti: number of samples per data source/seismic trace
 * dt: temporal resolution
 * trc: coordinates of source/seismic traces
 * Ntr: number of source/seismic traces
 * wri: write the current wavefield in a binary file at each wri timesteps. wri=0 means do not write
 * wffn: wavefield record file name
 * bw: border width
 */
void propagation(float *vel, unsigned int localXi, unsigned int localYi, int Yi, int Zi, int sZi, float dx, float dy, float dz, float *tr, unsigned short Ti, float dt, int *trc, int Ntr, int wri, char *wffn, int bw, int my_rank, unsigned int shotNumber, short pMode, double divisor, double sDivisor, double sDivY, double sDivZ, float *localImage) {
    int xi, yi, zi, xbi, ybi, zbi, ti, c, ntr, fIndex, cIndex, xIndex, yIndex, zIndex; // auxiliary variables
    unsigned long localXbi, localYbi, Ybi, Zbi, sZbi, indb, ivb;
    float fdx, fdy, fdz, fCoef, cCoef; // auxiliary variables
    u_t u; // wavefield
    char fileName [40];
    float *image, *direta, *borderX, *borderY, *borderZ;
    FILE *wff;
    //FILE *wff2; // Wavefield record file
    //float cx[ncx] = {-2.847222222, 1.6, -0.2, 0.025396825, -0.001785714}; // coefficients of the finite difference in x
    //float cy[ncy] = {-2.847222222, 1.6, -0.2, 0.025396825, -0.001785714}; // coefficients of the finite difference in y
    //float cz[ncz] = {-2.847222222, 1.6, -0.2, 0.025396825, -0.001785714}; // coefficients of the finite difference in z
    
    float cx[ncx] = {-205.0/72.0, 1.6, -0.2, 8.0/315.0, -1.0/560.0}; // coefficients of the finite difference in x
    float cy[ncy] = {-205.0/72.0, 1.6, -0.2, 8.0/315.0, -1.0/560.0}; // coefficients of the finite difference in y
    float cz[ncz] = {-205.0/72.0, 1.6, -0.2, 8.0/315.0, -1.0/560.0}; // coefficients of the finite difference in z
    
    //vel: Xbi*Ybi*Zbi
    //u: localXbi*localYbi*sZbi
    //localImage: localXi*localYi*sZi
   
 
    // Auxiliary variables initialization
    localXbi = localXi + 2*bw;
    localYbi = localYi + 2*bw;
    Ybi = Yi + 2*bw;
    Zbi = Zi + 2*bw;
    sZbi = sZi + 2*bw;
    // Memory allocation
    u.pn = (float *) calloc(localXbi*localYbi*sZbi,sizeof(float));
    if (u.pn == NULL) {
        printf("Memory allocation failed: u.pn.\n");
        return;      
    }
    u.cur = (float *) calloc(localXbi*localYbi*sZbi,sizeof(float));
    if (u.cur == NULL) {
        printf("Memory allocation failed: u.cur.\n");
        return;      
    }
    direta = (float *) calloc(localXi*localYi*sZi,sizeof(float));
    if (direta == NULL) {
        printf("Memory allocation failed: direta.\n");
        return;
    }    
    image = (float *) calloc(localXi*localYi*sZi,sizeof(float));
    if (image == NULL) {
        printf("Memory allocation failed: image.\n");
        return;
    }
    borderX = (float *) malloc(localXbi*sizeof(float));
    if (borderX == NULL) {
        printf("Memory allocation failed: borderX.\n");
        return;
    }
    borderY = (float *) malloc(localYbi*sizeof(float));
    if (borderY == NULL) {
        printf("Memory allocation failed: borderY.\n");
        return;
    }
    borderZ = (float *) malloc(sZbi*sizeof(float));
    if (borderZ == NULL) {
        printf("Memory allocation failed: borderZ.\n");
        return;
    }


    //Making border to become non-reflexive
    for (xbi = 0; xbi <= localXbi/2; xbi++) {
   	if(xbi < bw)
	{
		borderX[xbi] = ((float)xbi)/bw;
		borderX[localXbi-1-xbi] = ((float)xbi)/bw;
		//borderX[xbi] = sin(((float) M_PI*xbi)/(2*bw));
		//borderX[localXbi-1-xbi] = sin(((float) M_PI*xbi)/(2*bw));

	}
	else
	{
		borderX[xbi] = 1.0;
		borderX[localXbi-1-xbi] = 1.0;
	}
    }
    for (xbi = 0; xbi < localXbi; xbi++) {
		printf("Border[%d]: %.2f\t localXbi-1-xbi: %lu\t ceil(localXbi/2): %.2f\t localXbi: %lu\n", xbi, borderX[xbi], localXbi-1-xbi, ceil(localXbi/2), localXbi);
    }

    for (ybi = 0; ybi <= localYbi/2; ybi++) {
   	if(ybi < bw)
	{
		borderY[ybi] = ((float)ybi)/bw;
		borderY[localYbi-1-ybi] = ((float)ybi)/bw;
                //borderY[ybi] =  sin(((float) M_PI*ybi)/(2*bw));
		//borderY[localYbi-1-ybi] = sin(((float) M_PI*ybi)/(2*bw));
	}
	else
	{
		borderY[ybi] = 1.0;
		borderY[localYbi-1-ybi] = 1.0;
	}
    }

    for (zbi = 0; zbi <= sZbi/2; zbi++) {
   	if(zbi < bw)
	{
		borderZ[zbi] = ((float)zbi)/bw;
		borderZ[sZbi-1-zbi] = ((float)zbi)/bw;

		//borderZ[zbi] = sin(((float) M_PI*zbi)/(2*bw));
		//borderZ[sZbi-1-zbi] = sin(((float) M_PI*zbi)/(2*bw));
	}
	else
	{
		borderZ[zbi] = 1.0;
		borderZ[sZbi-1-zbi] = 1.0;
	}
    }     


    printf("u Size: %lu\n", localXbi*localYbi*sZbi);
    printf("Rank: %d\tComputando tiro = %u\n", my_rank, shotNumber);
    printf("Ti = %d\n", Ti);
    fflush(stdout);
    
    int inc, sign;
    int firstI, lastI;    
	//int size;
    if(pMode >= 0)
    {//Propagation
	firstI = 0;
	//lastI = 300;
 	lastI = Ti;
	sign = 1;
	inc = 1;
	wff = fopen(wffn,"wb"); // Wafield record file

	// Check the opening file
    	if (!wff) {
	   printf("Failed opening file.\n");
	   return;
	}
    }
    else
    {//Backpropagation
 	firstI = Ti-1;
	lastI = 1;
	sign = -1;
 	inc = -1;

	//wff = fopen(wffn,"rb"); // Wafield record file
        //fseek(wff, (firstI+1)*localXi*localYi*Zi*sizeof(float), SEEK_SET);
	
	//fseek(wff, 0, SEEK_END);
	//size = ftell(wff);
	//printf("Ate o final wff tem %d bytes\n", size);
    }
 
    
    // Propagate
    for (ti = firstI; sign*ti < lastI; ti += inc) {
    //for (ti = 0; ti < Ti; ti++) {
	//#pragma omp parallel for num_threads(4)
        for (xbi = ncx-1; xbi < localXbi-ncx+1; xbi++) {
            xIndex = round(xbi/sDivisor);
            
            for (ybi = ncy-1; ybi < localYbi-ncy+1; ybi++) {
                yIndex = round(ybi/sDivY);
                
                for (zbi = ncz-1; zbi < sZbi-ncz+1; zbi++) {
                    indb = xbi*localYbi*sZbi + ybi*sZbi + zbi;
                    
                    fdx = cx[0]*u.cur[indb];
                    fdy = cy[0]*u.cur[indb];
                    fdz = cz[0]*u.cur[indb];
                    
                    for (c = 1; c < ncx; c++) {
                        fdx += cx[c]*(u.cur[indb + c*localYbi*Zbi] + u.cur[indb - c*localYbi*Zbi]);
                    }
                    for (c = 1; c < ncy; c++) {
                        fdy += cy[c]*(u.cur[indb + c*Zbi] + u.cur[indb - c*Zbi]);
                    }
                    for (c = 1; c < ncz; c++) {
                        fdz += cz[c]*(u.cur[indb + c] + u.cur[indb - c]);
                    }
                    
                    fdx *= 1/(dx*dx);
                    fdy *= 1/(dy*dy);
                    fdz *= 1/(dz*dz);

                    zIndex = round(zbi/sDivZ);
                    ivb = xIndex*Ybi*Zbi + yIndex*Zbi + zIndex;   //Alterar ivb (arredondar)
			 	
		    /*if(xbi == 68)
		    {
			printf("xbi = %d  xIndex = %d  localXbi-ncx+1 = %lu\tybi = %d  yIndex = %d  localYbi-ncy+1 = %lu\tzbi = %d  zIndex = %d  sZbi-ncz+1 = %lu\n", xbi, xIndex, localXbi-ncx+1, ybi, yIndex, localYbi-ncy+1, zbi, zIndex, sZbi-ncz+1);
		    }*/
		    u.pn[indb] = 2*u.cur[indb] - u.pn[indb] + vel[ivb]*(fdx + fdy + fdz);

		    /*if(zIndex < bw && zIndex > sZbi - bw)	
		    {
                    	u.pn[indb] = 2*u.cur[indb] - u.pn[indb] + 0*(fdx + fdy + fdz);
		    }		    
		    else
 		    if(zIndex < 80)	
		    {
                    	u.pn[indb] = 2*u.cur[indb] - u.pn[indb] + 5*(fdx + fdy + fdz);
		    }
		    else	
		    {
                    	u.pn[indb] = 2*u.cur[indb] - u.pn[indb] + 30*(fdx + fdy + fdz);
		    }*/

		    //u.pn[indb] = borderX[xbi]*borderY[ybi]*borderZ[zbi]*u.pn[indb];
		    /*if(xbi == 68)
		    {
		    	printf("xbi = %d  xIndex = %d  localXbi-ncx+1 = %lu\tybi = %d  yIndex = %d  localYbi-ncy+1 = %lu\tzbi = %d  zIndex = %d  sZbi-ncz+1 = %lu\n", xbi, xIndex, localXbi-ncx+1, ybi, yIndex, localYbi-ncy+1, zbi, zIndex, sZbi-ncz+1);
		    }*/
		    //xbi = 68  xIndex = 23  localXbi-ncx+1 = 139	ybi = 155  yIndex = 52  localYbi-ncy+1 = 313	zbi = 425  zIndex = 142  sZbi-ncz+1 = 724
		}
            }
	//printf("xbi = %d  xIndex = %d  localXbi-ncx+1 = %lu\tybi = %d  yIndex = %d  localYbi-ncy+1 = %lu\tzbi = %d  zIndex = %d  sZbi-ncz+1 = %lu\n", xbi, xIndex, localXbi-ncx+1, ybi, yIndex, localYbi-ncy+1, zbi, zIndex, sZbi-ncz+1);


        }
        //printf("Rank: %d\tTi: %d\tti = %d\n", my_rank, Ti, ti);
        
        //printf("-1\n");	
        // Source/seismic traces
        // Corrigir trc, ivb
        // Verificar indices
  
        for (ntr = 0; ntr < Ntr; ntr++) 
	{
	        //printf("0\n");	

        	indb = (trc[ntr*3]+bw)*localYbi*sZbi + (trc[ntr*3+1]+bw)*sZbi + trc[ntr*3+2]+bw;
                ivb = (trc[ntr*3]+bw)*Ybi*Zbi + (trc[ntr*3+1]+bw)*Zbi + (trc[ntr*3+2]+bw); //Alterar ivb (arredondar) 

			//Calculate Indexes and coefficients of interpolation
        		fIndex = (int) floor((double) ti / divisor);
	                cIndex = (int) ceil((double) ti / divisor);
        	        fCoef = ((float) (ti-fIndex*divisor)) / divisor ; //ti = 5 fI:1 cI:2
                	cCoef = 1.0 - fCoef ;
			//printf("1\n");	

                	u.pn[indb] -= vel[ivb] * (fCoef*tr[ntr*Ti+fIndex]+cCoef*tr[ntr*Ti+cIndex]);
				//printf("2\n");	

                printf("Rank: %d\tti = %d\tntr = %d\tTi = %d\tpMode = %d\tu.pn[%lu] = %.3f\n", my_rank, ti, ntr, Ti, pMode, indb, u.pn[indb]);
		//Investigar pq u.pn é o mesmo para todos ntr
	}// */
    

	//printf("Rank: %d\tti = %d\tu.pn[%lu] = %.3f\n", my_rank, ti, indb, u.pn[indb]);

	SwapPointers(&u.pn, &u.cur);

	// Write wavefield in a file
	if (wri != 0) {
		if ((ti+1)%wri == 0 || ti == 10) {
		    if(pMode >= 0)
		    {		
			sprintf(fileName, "pDireta2/diretaTest2_%d.bin", ti);                     
			FILE* wff2 = fopen(fileName,"wb"); // Wafield record file

			// Check the opening file
			if (!wff2) {
			   printf("Failed opening file.\n");
			   return;
			}
			
		  	for (xbi = bw; xbi < localXbi-bw; xbi++) {
			   for (ybi = bw; ybi < localYbi-bw; ybi++) {
			    	if (fwrite(&u.cur[xbi*localYbi*sZbi + ybi*sZbi + bw], sizeof(float), sZi, wff2) != sZi) {
			/*for (xbi = 0; xbi < localXbi; xbi++) {
			   for (ybi = 0; ybi < localYbi; ybi++) {
				if (fwrite(&u.cur[xbi*localYbi*sZbi + ybi*sZbi], sizeof(float), sZbi, wff2) != sZbi) {*/
					printf("Failed writing wavefield file\n");
					return;
			    	}	
			   }	
		    	}
			fclose(wff2);
		    }
		    else
		    {
			
			//char fileName [40];

			wff = fopen(wffn,"rb"); // Wafield record file
			    

			// Check the opening file
			if (!wff) {
			   printf("Failed opening file.\n");
			   return;
			}//*/

			//fseek(wff, -localXi*localYi*Zi*sizeof(float), SEEK_CUR);
			fseek(wff, ti*localXi*localYi*sZi*sizeof(float), SEEK_SET);
			int a = 0;
			a = fread(direta, sizeof(float), localXi*localYi*sZi, wff);
			if(a != localXi*localYi*sZi) {			   
	                   printf("Read %s file failed! Leu %d bytes e deveria ler %d bytes\n", wffn, a, localXi*localYi*sZi);
			   exit(0);
                	}
			//printf("Leu e Passou!\n");

			//Errado verificar indices	
		    	for (xi = 0; xi < localXi; xi++) {
			   for (yi = 0; yi < localYi; yi++) {				
                		for(zi = 0; zi < sZi; zi++)
				{

				   image[xi*localYi*sZi + yi*sZi + zi] += direta[xi*localYi*sZi + yi*sZi + zi] * u.cur[(xi+bw)*localYi*sZi + (yi+bw)*sZi + zi+bw];
				}
			   }
			}

			fclose(wff);				    
	     	   }
	     }
	}
    }

    sprintf(fileName, "imageTest_%d.bin", shotNumber);
    FILE *imageCond = fopen(fileName,"wb");

    if (fwrite(image, sizeof(float),localXi*localYi*Zi, imageCond) != localXi*localYi*Zi) {
	printf("Failed writing image file\n");
	return;
    }
   
    // Free memory
    free(u.pn);
    free(u.cur);
    free(image);
    free(direta);
    if(pMode >= 0)
    {
    	fclose(wff);           		  
    }
    printf("Fechou arquivo"); 
    fclose(imageCond);
}



void readVelData(float *vel, float *vMin, float *vMax, float *sdt, float *rdt, unsigned short *sTi, double *divisor, float *sdX, float *sdY, float *sdZ, float *rds, double *sDivisor, unsigned short ns, int Xi, int Yi, int Zi, unsigned long Xbi, unsigned long Ybi, unsigned long Zbi, const char* velFileName, unsigned int bw, float dX, float dY, float dZ, float dt, int *sXi, int *sYi, int *sZi, double *sDivY, double *sDivZ)

{
    //Declare indexes
    int xi, yi, zi, xbi, ybi, zbi;
    unsigned long ind, indb;

    //Allocate velocity model data
    float* vModelData = (float*) malloc(Xi*Yi*Zi*sizeof(float));
    if (vModelData == NULL) {fputs ("Memory error",stderr); exit (2);}
    
    //Open velocity model file
    FILE* vModelFile = fopen(velFileName, "rb");
    if (vModelFile == NULL)
    {
        fprintf(stderr, "Erro ao abrir arquivo %s\n", velFileName);
        exit(0);
    }
    
    // Read velocity model file and puts to vModelData
    if (fread(vModelData, 1, Xi*Yi*Zi*sizeof(float), vModelFile) != Xi*Yi*Zi*sizeof(float))
    {
        fputs ("Reading error",stderr);
        exit (3);
    }
            
    //Close velocity model file
    fclose(vModelFile);

    
    //Get Max velocity
    for (xi = 0; xi < Xi; xi++) {
        for (yi = 0; yi < Yi; yi++) {
            for (zi = 0; zi < Zi; zi++) {	
                
                ind = xi*Yi*Zi + yi*Zi + zi;
                if(*vMin > vModelData[ind])
                {
                    *vMin = vModelData[ind];
                }
                if(*vMax < vModelData[ind])
                {
                    *vMax = vModelData[ind];
                }
                
            }
        }
    }

    //Test Spatial Restriction
    *rds = *vMin/(4.0);
    if(*rds < maxDs(dX,dY,dZ))
    {
	*sDivisor = ceil(dX/(*rds)); //Consider dX = dY = dZ
	*sDivY = ceil(dY/(*rds));
	*sDivZ = ceil(dZ/(*rds));
    }
    else
    {
	*sDivisor = 1.0;
	*sDivY = 1.0;
	*sDivZ = 1.0;
    }

    *sXi = *sDivisor*(Xi-1)+1;
    *sYi = *sDivY*(Yi-1)+1;
    *sZi = *sDivZ*(Zi-1)+1;
    *sdX = dX/(*sDivisor);
    *sdY = dY/(*sDivY);
    *sdZ = dZ/(*sDivZ);
    
    
    //Test Time Restriction
    *rdt = 1.0/((*vMax)*sqrt(1.0/pow(*sdX,2) + 1.0/pow(*sdY,2) + 1.0/pow(*sdZ,2)));    
    if(dt > *rdt)
    {
	*divisor = ceil(dt/(*rdt));
    }
    else
    {
	*divisor = 1.0;
    }
    
    *sTi = *divisor*(ns-1) + 1;
    *sdt = dt/(*divisor);


    /*printf("sdt = %.3f\tsTi = %d\tdX = %.3f\tdY = %.3f\tdZ = %.3f\tdt = %.3f\tns = %d\n", *sdt, *sTi, dX, dY, dZ, dt, ns); 
    printf("vMin = %.3f\n", *vMin); 
    printf("vMax = %.3f\n", *vMax);  */

    
    // Optimization
    for (xi = 0; xi < Xi; xi++) {
        for (yi = 0; yi < Yi; yi++) {
            for (zi = 0; zi < Zi; zi++) {

                ind = xi*Yi*Zi + yi*Zi + zi;
                indb = (bw+xi)*Ybi*Zbi + (bw+yi)*Zbi + bw+zi;
                vel[indb] = (*sdt)*(*sdt)*vModelData[ind]*vModelData[ind];

            }
        }
    }
    
    //Free memory space allocated to vModelData
    free(vModelData);

    
    
    // Expansion of the velocity model for the borders  
    for (xbi = bw; xbi < bw+Xi; xbi++) { // copying top and bottom faces -> z
        for (ybi = bw; ybi < bw+Yi; ybi++) {
            for (zbi = 0; zbi < bw; zbi++) {
                vel[xbi*Ybi*Zbi + ybi*Zbi + zbi] = vel[xbi*Ybi*Zbi + ybi*Zbi + bw];
                vel[xbi*Ybi*Zbi + ybi*Zbi + Zbi-1-zbi] = vel[xbi*Ybi*Zbi + ybi*Zbi + Zbi-1-bw];
            }
        }
    }  
    for (xbi = bw; xbi < bw+Xi; xbi++) { // copying left and right faces -> y
        for (zbi = 0; zbi < Zbi; zbi++) {
            for (ybi = 0; ybi < bw; ybi++) {
                vel[xbi*Ybi*Zbi + ybi*Zbi + zbi] = vel[xbi*Ybi*Zbi + bw*Zbi + zbi];
                vel[xbi*Ybi*Zbi + (Ybi-1-ybi)*Zbi + zbi] = vel[xbi*Ybi*Zbi + (Ybi-1-bw)*Zbi + zbi];
            }
        }
    }
    for (ybi = 0; ybi < Ybi; ybi++) { // copying front and back faces -> x
        for (zbi = 0; zbi < Zbi; zbi++) {
            for (xbi = 0; xbi < bw; xbi++) {
                vel[xbi*Ybi*Zbi + ybi*Zbi + zbi] = vel[bw*Ybi*Zbi + ybi*Zbi + zbi];
                vel[(Xbi-1-xbi)*Ybi*Zbi + ybi*Zbi + zbi] = vel[(Xbi-1-bw)*Ybi*Zbi + ybi*Zbi + zbi];
            }
        }
    }    
} 
       
void getComShotTraces(FILE* suFile, SuTrace* traces, float* tracesData, int* tracesC, float* tr, int* trc, unsigned long* traceNumber, unsigned long* localTraceNumber, unsigned int* localXi, unsigned int* localYi, unsigned int* localGxMinC, unsigned int* localGyMinC, unsigned int gxMin, unsigned int gyMin, unsigned long ntt, unsigned short ns, unsigned int TD, float dX, float dY, float dZ, float dt, float sdX, float sdY, float sdZ)
{
    int curSx, curSy, sx, sy, sz, ti; 
    unsigned int localGxMin, localGxMax, localGyMin, localGyMax, curGx, curGy;
    unsigned long curTraceNumber;
    
    //Get Next Sdepth, Sx, Sy                    
    fseek(suFile, 48, SEEK_CUR);
    if (fread(&sz, 1, sizeof(int), suFile) != sizeof(int)) {
        printf("getSuTrace failed!\n");
        exit(0);
    }
    fseek(suFile, 20, SEEK_CUR);
    if (fread(&curSx, 1, sizeof(int), suFile) != sizeof(int)) {
        printf("getSuTrace failed!\n");
        exit(0);
    }
    if (fread(&curSy, 1, sizeof(int), suFile) != sizeof(int)) {
        printf("getSuTrace failed!\n");
        exit(0);
    }
    fseek(suFile, -80, SEEK_CUR);


    //Invert bytes (Big Endian -> Little Endian)
    invBytes(&curSx, sizeof(int));
    invBytes(&curSy, sizeof(int));	
    invBytes(&sz, sizeof(int));	

    
    //Initialize variables of the current shot
    sx = curSx;
    sy = curSy;
    curTraceNumber = 0;
    localGxMax = localGyMax = 0;
    localGxMin = localGyMin = INT_MAX;
    
    while(sx == curSx && sy == curSy && *traceNumber < ntt)	//Get traces from the same shot
    {
        //Get Trace
        if (fread(&(traces[curTraceNumber]), 1, TH, suFile) != TH) {
            printf("getSuTrace failed!\n");
            exit(0);
        }
        if (fread(&(tracesData[curTraceNumber*ns]), 1, TD, suFile) != TD) {
            printf("getSuTrace failed!\n");
            exit(0);
        }
        
	//Invert data bytes 
        for (ti = 0; ti < ns; ti++) {
		invBytes(&(tracesData[curTraceNumber*ns+ti]), sizeof(float));     
    	}
        
        //Get Current Gx and Gy
        curGx = traces[curTraceNumber].gx;
        curGy = traces[curTraceNumber].gy;
        invBytes(&curGx, sizeof(int));
        invBytes(&curGy, sizeof(int));

        
        //Get max/min Gx and Gy
        if(curGx > localGxMax)
        {
            localGxMax = curGx;
        }	
        if(curGx < localGxMin)
        {
            localGxMin = curGx;
        }
        
        if(curGy > localGyMax)
        {
            localGyMax = curGy;
        }
        if(curGy < localGyMin)
        {
            localGyMin = curGy;
        }
        

	
        //Increment variables
        (*traceNumber)++;
        curTraceNumber++;
        

        tracesC[curTraceNumber*3]   = (curGx-localGxMin)/sdX;
	tracesC[curTraceNumber*3+1] = (curGy-localGyMin)/sdY;
	tracesC[curTraceNumber*3+2] = 0;            

        //Get Next Sx and Sy   
        if(*traceNumber < ntt)
        {
            fseek(suFile, 72, SEEK_CUR);
            if (fread(&curSx, 1, sizeof(int), suFile) != sizeof(int)) {
                printf("getSuTrace failed!\n");
                exit(0);
            }
            if (fread(&curSy, 1, sizeof(int), suFile) != sizeof(int)) {
                printf("getSuTrace failed!\n");
                exit(0);
            }
            fseek(suFile, -80, SEEK_CUR);
            
            invBytes(&curSx, sizeof(int));
            invBytes(&curSy, sizeof(int));
        }
    }

    
    *localTraceNumber = curTraceNumber;
    
    for (ti = 0; ti < ns; ti++) {
        tr[ti] = source(ti*dt);
        //printf("tr[%d] = %.3f\n", ti, tr[ti]);
    }
    
    //Converting sx, sy and sx to coordinates
    trc[0] = (sx-localGxMin)/sdX;
    trc[1] = (sy-localGyMin)/sdY;
    trc[2] = sz/dZ;            
   
     
   
    //Calculate local bounds and local velocity model size
    *localGxMinC = (localGxMin-gxMin)/dX;
    *localGyMinC = (localGyMin-gyMin)/dY;
    *localXi = (localGxMax-localGxMin)/sdX + 1;
    *localYi = (localGyMax-localGyMin)/sdY + 1;
}
    
    
    
void printFile(SuTrace* traces, unsigned long localTraceNumber, unsigned int TD, float* velocity_model_data, unsigned long vModelSize, int my_rank)
{
    //Output Su Files
    char fileName [20];
    sprintf(fileName, "output_%d.su", my_rank);
    FILE *outputSu_file;
    outputSu_file = fopen(fileName, "wb");
    if (outputSu_file == NULL)
    {
        fprintf(stderr, "Erro ao abrir arquivo outputSU\n");
        exit(0);
    }
    
    int i;
    
    for(i = 0; i < localTraceNumber; i++)
    {
        fwrite(&(traces[i]), 1, TH, outputSu_file);
        fflush(outputSu_file);
        fwrite(traces[i].data, 1, TD, outputSu_file);
        fflush(outputSu_file);
    }
    
    fclose(outputSu_file);
    
    printf("My_rank: %d\tArquivo .su exportado\n", my_rank);
    
    //Output Velocity Model files
    FILE *output_file;
    sprintf(fileName, "vModel_%d.ad", my_rank);
    output_file = fopen(fileName, "wb");
    
    if (output_file == NULL)
    {
        fprintf(stderr, "Erro ao abrir arquivo output\n");
        exit(0);
    }  
    
    
    fwrite(velocity_model_data, 1, vModelSize, output_file);
    fflush(output_file);
    
    fclose(output_file);
    printf("My_rank: %d\tArquivo de modelo de velocidades exportado\n", my_rank);
} 





