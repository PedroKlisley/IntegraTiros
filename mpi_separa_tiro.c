#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include "field.c"

// Exemplo de compilação do programa
// mpicc -g -Wall -o mpi_separa_tiro.bin mpi_separa_tiro.c

// Exemplo de execução do programa
// mpiexec -n 2 mpi_separa_tiro.bin VEL_210x427x427.ad 427 427 210 20 20 20 0.003 3720 11840 500 8660 3DSEGEAGE.su 544 2286908 
// mpiexec -n <comm_sz> mpi_separa_tiro.bin <velocityModel_file> <Xi> <Yi> <Zi> <dX> <dY> <dZ> <dt> <traces_file.su> <ntssMax> <ntt>

//Resolucao do modelo de velocidades é o dobro da resolucao do sismograma
//dX = 20, dY = 20, dz = 20

//Menor valor de dt = 0,001115573 - Teste 1
//Menor valor de dt = 0,002576306 - Teste 2

//Limites do mapeamento
//gx       3720 - 11840
//gy       500 - 8660

#define TH 	240 	// Trace Header bytes
#define ncx 	5 	// number of coefficients of the finite difference in x: order/2 + 1
#define ncy 	5 	// number of coefficients of the finite difference in y: order/2 + 1
#define ncz 	5 	// number of coefficients of the finite difference in z: order/2 + 1
#define freq 	12.0 	// source frequency

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
void propagation(float *vel, unsigned int localXi, unsigned int localYi, int Yi, int Zi, float dx, float dy, float dz, float *tr, unsigned short Ti, float dt, int *trc, int Ntr, int wri, char *wffn, int bw, int my_rank, unsigned int shotNumber);
void backpropagation(uint8_t* u_r, SuTrace* localTraces, float* vModelData, int Xi, int Yi, int Zi, float dX, float dY, float dZ, int dt, int gx, int gy);
void imageCondition(uint8_t* image, uint8_t* u_i, uint8_t* u_r, int Xi, int Yi, int Zi);
void invBytes(void * valor, int nBytes);
float source(float t);
float minDs(float dx, float dy, float dz);
float maxDs(float dx, float dy, float dz);

int main(int argc, char* argv[]) {

   //Variable declarations
   SuTrace* localTraces;		//Local Traces Data		
   float* vModelData;			//Local Velocity Model Data
   unsigned long localTraceNumber;  	//Local Trace Number
   unsigned long traceNumber;       	//Global Trace Number counter 
   unsigned int  shotNumber;        	//Number of shots counter
   int sx; 				//Position x of shot source
   int sy; 				//Position y of shot source
   int sz; 				//Position z of shot source
   unsigned short ns;			//Number of Samples
   unsigned long ntssMax;		//Maximum Number of Traces of the same shot
   unsigned int shotLimit;		//Number of shots analyzed
   unsigned long ntt;			//Total Number of traces in SU file
   unsigned int TD;			//Size of trace's data (in Bytes)
   unsigned int i;			//Iterator
   FILE *vModelFile;			//File containing velocity Model data
   int my_rank;                         //Rank of process
   int comm_sz;                         //Number of processes
   MPI_Comm comm; 			//Processes' Communicator
   MPI_Status status;			//Status of MPI communication
   int Xi;				//Velocity Model length (X axis)
   int Yi;				//Velocity Model width  (Y axis)
   int Zi;				//Velocity Model depth  (Z axis)
   float dX;				//Velocity Model length step (X axis)
   float dY;				//Velocity Model width step  (Y axis)
   float dZ;				//Velocity Model depth step  (Z axis)
   float dt;				//Propagation wave time step (T axis)
   unsigned int gxMin;			//Distance of the first hidrophone in X axis (in meters)
   unsigned int gxMax;			//Distance of the last  hidrophone in X axis (in meters)
   unsigned int gyMin;			//Distance of the first hidrophone in Y axis (in meters)
   unsigned int gyMax;			//Distance of the last  hidrophone in Y axis (in meters)
   unsigned long Xbi;			//Length of Velocity Model with border (X axis)
   unsigned long Ybi; 			//Width of  Velocity Model with border (Y axis)
   unsigned long Zbi;			//Depth of  Velocity Model with border (Z axis) 
   unsigned long indb;			//Index counter of Velocity Model with border
   unsigned long ind;			//Index counter of Velocity Model
   int xi;				//Index counter of Velocity Model length (Xi) 
   int xbi;				//Index counter of Velocity Model with border length (Xi)
   int yi;				//Index counter of Velocity Model width  (Yi)
   int ybi; 				//Index counter of Velocity Model with border width  (Yi)
   int zi;				//Index counter of Velocity Model depth  (Zi)
   int zbi;				//Index counter of Velocity Model with border depth  (Xi)
   unsigned int bw;			//Border Width applied to Velocity Model
   float *vel;				//Velocity Model with border
   int curSx;				//Position x of shot source of the current trace
   int curSy;				//Position y of shot source of the current trace
   unsigned int curGx;			//Position x of hidrophone of the current trace
   unsigned int curGy;			//Position y of hidrophone of the current trace
   unsigned long curTraceNumber;        //Local Trace Number Counter
   unsigned int  dest;			//Destination process index
   FILE *suFile;			//SU File
   SuTrace* traces;			//SU Traces
   float *tr;                           //Traces data
   int *trc;				//Traces data coordinate (row, column, depth)
   int ti; 				//Index counter of propagation wave time
   int Ntr;				//Number of Traces in Propagation
   unsigned int localXi; 		//Velocity Model length (X axis) used in the current shot
   unsigned int localYi; 		//Velocity Model width  (Y axis) used in the current shot
   unsigned int localGxMinC;		//Coordinate of the first hidrophone in X axis (in rows) of the current shot
   unsigned int localGyMinC;		//Coordinate of the first hidrophone in Y axis (in columns) of the current shot
   char fileName [20];			//Name of a file written by the program
   float vMin;				//Minimum velocity in velocity model
   float vMax;                          //Maximum velocity in velocity model


   //uint8_t* localImage;
   //uint8_t* image;


   //MPI start
   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);


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
   gxMax = atoi(argv[10]);
   gyMin = atoi(argv[11]);
   gyMax = atoi(argv[12]);
   ntssMax = atol(argv[14]);
   ntt = atol(argv[15]);


   //Variables Initialization
   localTraceNumber = 0; 
   traceNumber = 0;       
   shotNumber = 0;
   curSx = 0;		
   curSy = 0;			
   curTraceNumber = 0;  
   vMin = FLT_MAX;
   vMax = 0;
   shotLimit = 1;
   bw = 50;
   Xbi = Xi + 2*bw;
   Ybi = Yi + 2*bw;
   Zbi = Zi + 2*bw;
   
  
   // cria imagem local zerada
   //localImage = (uint8_t*) calloc(Xi*Yi*Zi, sizeof(uint8_t));


   //Allocate velocity model data
   vModelData = (float*) malloc(Xi*Yi*Zi*sizeof(float));
   if (vModelData == NULL) {fputs ("Memory error",stderr); exit (2);}


   if(my_rank == 0)
   {
	 //Open velocity model file
	 vModelFile = fopen(argv[1], "rb");
         if (vModelFile == NULL)
         {
                fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[1]);
                exit(0);
         }

         // Read velocity model file and puts to vModelData
         if (fread(vModelData, 1, Xi*Yi*Zi*sizeof(float), vModelFile) != Xi*Yi*Zi*sizeof(float))
         {
                fputs ("Reading error",stderr);
                exit (3);
         }

	 printf("My_rank: %d\t Leu modelo de velocidades\n", my_rank);
   
         //Close velocity model file
         fclose(vModelFile);
   }


  //Allocate velocity model with border
  vel = (float *) malloc(Xbi*Ybi*Zbi*sizeof(float));
  if (vel == NULL) {
    printf("Memory allocation failed: vel.\n");
    return 0;
  }  

  
  if(my_rank == 0)
  {

	  printf("My_rank: %d\tRealizando expansão do modelo de velocidades\n", my_rank);

	  // Optimization
	  for (xi = 0; xi < Xi; xi++) {
	    for (yi = 0; yi < Yi; yi++) {
	      for (zi = 0; zi < Zi; zi++) {
		ind = xi*Yi*Zi + yi*Zi + zi;
		indb = (bw+xi)*Ybi*Zbi + (bw+yi)*Zbi + bw+zi;
		vel[indb] = dt*dt*vModelData[ind]*vModelData[ind];
		if(vMin > vModelData[ind])
		{
			vMin = vModelData[ind];
		}
		if(vMax < vModelData[ind])
                {
                        vMax = vModelData[ind];
                }
	      }
	    }
	  }

	  printf("vMin = %.3f\n", vMin); 
	  printf("vMax = %.3f\n", vMax);   
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

   //Free memory space allocated to vModelData
   free(vModelData);


   //Broadcast velocity model with border and min and max velocities
   MPI_Bcast(vel, Xbi*Ybi*Zbi, MPI_FLOAT, 0, comm);
   MPI_Bcast(&vMin, 1, MPI_FLOAT, 0, comm);
   MPI_Bcast(&vMax, 1, MPI_FLOAT, 0, comm);
   printf("My_rank: %d\tBroadcast do modelo de velocidades feito\n", my_rank);


   /***** End of Get Velocity Model *****/



   /*****  Get Traces *****/  
   if(my_rank == 0)
   {
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


	   //Broadcast number of samples (ns)
	   MPI_Bcast(&ns, 1, MPI_SHORT, 0, comm);	   
	   TD = ns*sizeof(float);

	   //printf("NS: %d\t TD: %d\tSizeof(short): %lu\n", ns, TD, sizeof(short));


	   //Allocate Traces
           traces = (SuTrace*) malloc(ntssMax*sizeof(SuTrace));
           for (i = 0; i < ntssMax; i++)
           {
	           traces[i].data = (float *) malloc(TD);
           }


	   //Allocate Image		   
	   //image = (uint8_t*) calloc(Xi*Yi*Zi, sizeof(uint8_t));


           while(traceNumber < ntt && shotNumber < shotLimit)
           {
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

		//Compute destination process
		dest = (shotNumber % (comm_sz-1)) + 1; 


		//Send shot source (sx,sy)
                MPI_Ssend(&curSx, 1, MPI_INT, dest, 0, comm);
                MPI_Ssend(&curSy, 1, MPI_INT, dest, 0, comm);		
                MPI_Ssend(&sz, 1, MPI_INT, dest, 0, comm);

		printf("Rank: %d\tNumberShot: %u\tenviou coordenada da fonte\n", my_rank, shotNumber);

		//Initialize variables of the current shot
                sx = curSx;
                sy = curSy;
                curTraceNumber = 0;
		gxMax = gyMax = 0;
		gxMin = gyMin = INT_MAX;

                while(sx == curSx && sy == curSy && traceNumber < ntt)	//Get traces from the same shot
                {
		        //Get Trace
                        if (fread(&(traces[curTraceNumber]), 1, TH, suFile) != TH) {
                                printf("getSuTrace failed!\n");
                                exit(0);
                        }
                        if (fread(traces[curTraceNumber].data, 1, TD, suFile) != TD) {
                                printf("getSuTrace failed!\n");
                                exit(0);
                        }


			//Get Current Gx and Gy
			curGx = traces[curTraceNumber].gx;
			curGy = traces[curTraceNumber].gy;
			invBytes(&curGx, sizeof(int));
			invBytes(&curGy, sizeof(int));

			//printf("TraceNumber = %lu\t\tgxMin = %u\tgxMax = %u\tgyMin = %u\tgyMax = %u\n", traceNumber, gxMin, gxMax, gyMin, gyMax);
	
			//Get max/min Gx and Gy
			if(curGx > gxMax)
			{
				gxMax = curGx;
			}	
			if(curGx < gxMin)
			{
				gxMin = curGx;
			}

			if(curGy > gyMax)
                        {
                                gyMax = curGy;
                        }
			if(curGy < gyMin)
                        {
                                gyMin = curGy;
                        }

                        //Increment variables
                        traceNumber++;
                        curTraceNumber++;


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
			//printf("Calculando\tTotalTraceNumber: %lu\tTraceNumber: %lu\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", ntt, traceNumber, curSx, curSy, sx, sy);

                }

		printf("gxMin = %u\tgxMax = %u\tgyMin = %u\tgyMax = %u\n", gxMin, gxMax, gyMin, gyMax);

		
		//Send hidrophone bounds of the shot
                MPI_Ssend(&gxMin, 1, MPI_INT, dest, 0, comm);
                MPI_Ssend(&gxMax, 1, MPI_INT, dest, 0, comm);
                MPI_Ssend(&gyMin, 1, MPI_INT, dest, 0, comm);
                MPI_Ssend(&gyMax, 1, MPI_INT, dest, 0, comm);


                //Send common shot traces number
                MPI_Ssend(&curTraceNumber, 1, MPI_LONG, dest, 0, comm);
		
		
		//Send Traces
                for (i = 0; i < curTraceNumber; i++)
                {
                	MPI_Ssend( &(traces[i]), TH, MPI_UINT8_T, dest, 0, comm);
                        MPI_Ssend( traces[i].data, ns, MPI_FLOAT, dest, 0, comm);
                        //printf("%d enviou traço %u\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", my_rank, i, traces[i].sx, traces[i].sy, sx, sy);
                }


		//Broadcast total trace number count	
		MPI_Bcast(&traceNumber, 1, MPI_LONG, 0, comm);		


                printf("Destination: %u\tNumberShot: %u\tLocalTraceNumber: %lu\tTraceNumber: %lu\n", dest, shotNumber, curTraceNumber, traceNumber);
		shotNumber++;
        }

        printf("\nMy_rank: %d\tDistribuição dos traços concluída com sucesso!\n", my_rank);
	free(traces);
        fclose(suFile);

   }
   else
   {
			
	//Receive number of samples (ns)
	MPI_Bcast(&ns, 1, MPI_SHORT, 0, comm);       
	TD = ns*sizeof(float);
	

	while(traceNumber < ntt && shotNumber < shotLimit)
	{

		if(my_rank != (shotNumber % (comm_sz-1)) + 1)
		{	
			//Receive total trace number count
			MPI_Bcast(&traceNumber, 1, MPI_LONG, 0, comm);
		}
		else
		{
			unsigned int localGxMin, localGxMax, localGyMin, localGyMax;

			//Receive shot source
			MPI_Recv(&sx, 1, MPI_INT, 0, 0, comm, &status);
			MPI_Recv(&sy, 1, MPI_INT, 0, 0, comm, &status);
			MPI_Recv(&sz, 1, MPI_INT, 0, 0, comm, &status);


			//Receive hidrophone bounds of the shot
                        MPI_Recv(&localGxMin, 1, MPI_INT, 0, 0, comm, &status);
			MPI_Recv(&localGxMax, 1, MPI_INT, 0, 0, comm, &status);
                        MPI_Recv(&localGyMin, 1, MPI_INT, 0, 0, comm, &status);
			MPI_Recv(&localGyMax, 1, MPI_INT, 0, 0, comm, &status);


			//Receive common shot traces number and shot source
                        MPI_Recv(&localTraceNumber, 1, MPI_LONG, 0, 0, comm, &status);


			//Allocate Local Traces
			localTraces = (SuTrace*) malloc(localTraceNumber*sizeof(SuTrace));
			for (i = 0; i < localTraceNumber; i++)
			{
				localTraces[i].data = (float *) malloc(TD);
			}

			//Receive Traces
			for (i = 0; i < localTraceNumber; i++)
			{
				MPI_Recv( &(localTraces[i]), TH, MPI_UINT8_T, 0, 0, comm, &status);
				MPI_Recv( localTraces[i].data, ns, MPI_FLOAT, 0, 0, comm, &status);
				//printf("%d recebeu traço %u\n", my_rank, i);
			}

		
			//Receive current shot traces number
			MPI_Bcast(&traceNumber, 1, MPI_LONG, 0, comm);


			printf("My_rank: %u\tNumberShot: %u\tLocalTraceNumber: %lu\tTraceNumber: %lu\n", my_rank, shotNumber, localTraceNumber, traceNumber);
			printf("\nMy_rank: %d\tDistribuição dos traços concluída com sucesso!\n", my_rank);
			fflush(stdout);


			Ntr = 1;  //Number of traces (sources)

			//Allocate traces coordinates and samples of traces
                        trc = (int *) malloc(Ntr*3*sizeof(int));
                        tr = (float *) malloc(Ntr*ns*sizeof(float));

                        for (ti = 0; ti < ns; ti++) {
                                tr[ti] = source(ti*dt);
				//printf("tr[%d] = %.3f\n", ti, tr[ti]);
                        }

			//Converting sx, sy and sx to coordinates
                        trc[0] = (sx-gxMin)/dX;
                        trc[1] = (sy-gyMin)/dY;
                        trc[2] = sz/dZ;


                        sprintf(fileName, "direta_%d.bin", shotNumber);

			//Calculate local bounds and local velocity model size
			localGxMinC = (localGxMin-gxMin)/dX;
			localGyMinC = (localGyMin-gyMin)/dY;
			localXi = (localGxMax-localGxMin)/dX + 1;
			localYi = (localGyMax-localGyMin)/dY + 1;


			printf("My_rank: %u\tNumberShot: %u\tLocalXbi: %u\tLocalYbi: %u\tlocalGxMinC = %u\tlocalGyMinC = %u\n", my_rank, shotNumber, localXi, localYi, localGxMinC, localGyMinC);
			fflush(stdout);

			printf("trc[0] = %d\ttrc[1] = %d\ttrc[2] = %d\n", trc[0], trc[1], trc[2]);

			
			//Test Propagation Restrictions
			/*printf("maxDs tem que ser menor que %.6f\n", vMin/(4.0*freq));
			if(maxDs(dX,dY,dZ) <= vMin/(4.0*freq))
			{
				printf("Passou no teste espacial\n");
			}
			else
			{
				printf("Nao passou no teste espacial\n");
				exit(0);
			}

			printf("dt tem que ser menor que %.6f\t MinDs = %.3f\t maxDs = %.3f\n", minDs(dX,dY,dZ)/(4.0*vMax), minDs(dX,dY,dZ), maxDs(dX,dY,dZ));
			if(dt <= minDs(dX,dY,dZ)/(4.0*vMax))
                        {
                                printf("Passou no teste temporal 1\n");
                        }
                        else
                        {
                                printf("Nao passou no teste temporal 1\n");
                                exit(0);
                        }

			printf("dt tem que ser menor que %.6f\n",  1/(vMax*sqrt(1/pow(dX,2) + 1/pow(dY,2) + 1/pow(dZ,2))));
                        if(dt <= 1/(vMax*sqrt(1/pow(dX,2) + 1/pow(dY,2) + 1/pow(dZ,2))))
                        {
                                printf("Passou no teste temporal 2\n");
                        }
                        else
                        {
                                printf("Nao passou no teste temporal 2\n");
                                exit(0);
                        }*/

			propagation( &(vel[localGxMinC*Ybi*Zbi + localGyMinC*Zbi]), localXi, localYi, Yi, Zi, dX, dY, dZ, tr, ns, dt, trc, Ntr, 50, fileName, bw, my_rank, shotNumber);

			// assinatura retropropagacao
			//backpropagation(u_r, localTraces, vModelData, Xi, Yi, Zi, dX, dY, dZ, dt, gx, gy);


			// assinatura correlacao cruzada
			//imageCondition(localImage, u_i, u_r, Xi, Yi, Zi);

		}
		shotNumber++;//Update number of shot
		
		
	}

	//printf("\nMy_rank: %d\tDistribuição dos traços concluída com sucesso!\n", my_rank);
	// recebe info propagacao
	// assinatura propagacao
	// recebe tracos
	// assinatura retropropagacao
	// assinatura correlacao cruzada
	// adicionar a imagem local
}
    /*****  End of Get Traces *****/



   // adicionar a imagem local
   //MPI_Reduce(localImage, image, Xi*Yi*Zi, MPI_UINT8_T, MPI_SUM, 0, comm);

 
   //Print data to File
   printFile(localTraces, localTraceNumber, TD, vel, Xbi*Ybi*Zbi, my_rank);


   //Free pointers
   if(localTraceNumber > 0)
   {
	printf("Thread: %d\t ficou com %lu traços\tSx: %d\t Sy: %d\n", my_rank, localTraceNumber, sx, sy);
   	free(localTraces);
   }
   free(vel);

/*
   free(localImage);
   free(image);
   free(u_i); 
   free(u_r);
*/

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
  return (1 - 2*M_PI*pow(M_PI*freq*t,2))*exp(-M_PI*pow(M_PI*freq*t,2));
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
void propagation(float *vel, unsigned int localXi, unsigned int localYi, int Yi, int Zi, float dx, float dy, float dz, float *tr, unsigned short Ti, float dt, int *trc, int Ntr, int wri, char *wffn, int bw, int my_rank, unsigned int shotNumber) {
  int xbi, ybi, zbi, ti, c, ntr; // auxiliary variables
  unsigned long localXbi, localYbi, Ybi, Zbi, indb, ivb;
  float fdx, fdy, fdz; // auxiliary variables
  u_t u; // wavefield
  char fileName [40];
  FILE *wff2; // Wavefield record file
  //float cx[ncx] = {-2.847222222, 1.6, -0.2, 0.025396825, -0.001785714}; // coefficients of the finite difference in x
  //float cy[ncy] = {-2.847222222, 1.6, -0.2, 0.025396825, -0.001785714}; // coefficients of the finite difference in y
  //float cz[ncz] = {-2.847222222, 1.6, -0.2, 0.025396825, -0.001785714}; // coefficients of the finite difference in z

  float cx[ncx] = {-205.0/72.0, 1.6, -0.2, 8.0/315.0, -1/560.0}; // coefficients of the finite difference in x
  float cy[ncy] = {-205.0/72.0, 1.6, -0.2, 8.0/315.0, -1/560.0}; // coefficients of the finite difference in y
  float cz[ncz] = {-205.0/72.0, 1.6, -0.2, 8.0/315.0, -1/560.0}; // coefficients of the finite difference in z

  FILE *wff = fopen(wffn,"wb"); // Wafield record file
  
  // Check the opening file
  if (!wff) {
    printf("Failed opening file.\n");
    return;
  }
  
  // Auxiliary variables initialization
  localXbi = localXi + 2*bw;
  localYbi = localYi + 2*bw;
  Ybi = Yi + 2*bw;
  Zbi = Zi + 2*bw;
  
  // Memory allocation
  u.pn = (float *) calloc(localXbi*localYbi*Zbi,sizeof(float));
  if (u.pn == NULL) {
    printf("Memory allocation failed: u.pn.\n");
    return;      
  }
  u.cur = (float *) calloc(localXbi*localYbi*Zbi,sizeof(float));
  if (u.cur == NULL) {
    printf("Memory allocation failed: u.cur.\n");
    return;      
  }

  printf("u Size: %lu\n", localXbi*localYbi*Zbi);
  printf("Rank: %d\tComputando tiro = %u\n", my_rank, shotNumber);
  fflush(stdout);
 
  // Propagation
  for (ti = 0; ti < Ti; ti++) {
    for (xbi = ncx-1; xbi < localXbi-ncx; xbi++) {
      for (ybi = ncy-1; ybi < localYbi-ncy; ybi++) {
        for (zbi = ncz-1; zbi < Zbi-ncz; zbi++) {
          indb = xbi*localYbi*Zbi + ybi*Zbi + zbi;

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
          
	  ivb = xbi*Ybi*Zbi + ybi*Zbi + zbi;
          u.pn[indb] = 2*u.cur[indb] - u.pn[indb] + vel[ivb]*(fdx + fdy + fdz);	 
	  if(xbi == localXbi/2 && ybi == localYbi/2 && zbi == Zbi/2 && ti < 20)
	  {
		//printf("Ti: %d\tu.cur[%lu] = %.3f\tu.pn[%lu] = %.3f\n", ti, indb, u.cur[indb], indb, u.pn[indb]);
	  }
        }
      }
    }
    //printf("Ti: %d\tu.cur[%lu] = %.3f\tu.pn[%lu] = %.3f\n", ti, indb, u.cur[indb], indb, u.pn[indb]);
    // Source/seismic traces
    // Corrigir trc, ivb
    // Verificar indices
    for (ntr = 0; ntr < Ntr; ntr++) {
      indb = (trc[ntr*3]+bw)*localYbi*Zbi + (trc[ntr*3+1]+bw)*Zbi + trc[ntr*3+2]+bw;
      ivb = (trc[ntr*3]+bw)*Ybi*Zbi + (trc[ntr*3+1]+bw)*Zbi + (trc[ntr*3+2]+bw);
      u.pn[indb] -= vel[ivb]*tr[ntr*Ti + ti];
      //printf("Ti: %d\tu.cur[%lu] = %.3f\tu.pn[%lu] = %.3f\n", ti, indb, u.cur[indb], indb, u.pn[indb]);
    }

    SwapPointers(&u.pn, &u.cur);

    // Write wavefield in a file
    if (wri != 0) {
      if ((ti+26)%wri == 0) {
        sprintf(fileName, "./pDireta/direta_%d_%d.bin", shotNumber,ti);	
	*wff2 = fopen(fileName,"wb"); 
	
        for (xbi = bw; xbi < localXbi-bw; xbi++) {
          for (ybi = bw; ybi < localYbi-bw; ybi++) {
	    //printf("u atual: %.3f\n", u.cur[xbi*localYbi*Zbi + ybi*Zbi + bw]); 
            if (fwrite(&u.cur[xbi*localYbi*Zbi + ybi*Zbi + bw], sizeof(float), Zi, wff2) != Zi) {
              printf("Failed writing wavefield file\n");
              return;
            }
          }
        }
	fclose(wff2);
      }
    }
  }
  
  // Free memory
  free(u.pn);
  free(u.cur);
  fclose(wff);
}

void backpropagation(uint8_t* u_r, SuTrace* localTraces, float* vModelData, int Xi, int Yi, int Zi, float dX, float dY, float dZ, int dt, int gx, int gy)
{

}


void imageCondition(uint8_t* image, uint8_t* u_i, uint8_t* u_r, int Xi, int Yi, int Zi)
{

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



