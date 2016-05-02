#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>


// Exemplo de compilação do programa
// mpicc -g -Wall -o mpi_separa_tiro.bin mpi_separa_tiro.c

// Exemplo de execução do programa
// mpiexec -n 4 mpi_separa_tiro.bin VEL_210x427x427.ad 210 427 427 1 1 1 1 3DSEGEAGE.su 544 2286908 

#define TH 	240 	// Trace Header bytes

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
void printFile(SuTrace* traces, unsigned long localTraceNumber, unsigned int TD, uint8_t* velocity_model_data,  unsigned long vModelSize, int my_rank);
void propagation(uint8_t* u_i, uint8_t* vModelData, int Xi, int Yi, int Zi, int dX, int dY, int dZ, int dt, int sx, int sy);
void backpropagation(uint8_t* u_r, SuTrace* localTraces, uint8_t* vModelData, int Xi, int Yi, int Zi, int dX, int dY, int dZ, int dt, int gx, int gy);
void imageCondition(uint8_t* image, uint8_t* u_i, uint8_t* u_r, int Xi, int Yi, int Zi);
void invBytes(void * valor, int nBytes);

 
//Global variables declaration
unsigned int ns;
unsigned long ntssMax; 
unsigned int shotLimit; 
unsigned long ntt;
unsigned int TD;


int main(int argc, char* argv[]) {

   //Variable declarations
   SuTrace* localTraces;		//Local Traces Data		
   uint8_t* vModelData;			//Local Velocity Model Data
   unsigned long localTraceNumber = 0;  //Local Trace Number
   unsigned long traceNumber = 0;       //Global Trace Number counter 
   unsigned int  numberShot = 0;        //Number of shots counter
   unsigned long vModelSize;		//Size of Velocity Model Data (in Bytes)
   int sx; 				//Position x of shot source
   int sy; 				//Position y of shot source
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
   int Xi;
   int Yi;
   int Zi;
   int dX;
   int dY;
   int dZ;
   int dt;
   int gx = 1;
   int gy = 1;
   uint8_t* localImage;
   uint8_t* image;
   uint8_t* u_i; 
   uint8_t* u_r;

   //MPI start
   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);


   // Check command line arguments
   if (argc != 12) 
   {
	usage(argv[0], my_rank); 
   }

 
   // Get arguments from function call in command line
   Xi = atoi(argv[2]);
   Yi = atoi(argv[3]);
   Zi = atoi(argv[4]);
   dX = atoi(argv[5]);
   dY = atoi(argv[6]);
   dZ = atoi(argv[7]);
   dt = atoi(argv[8]);
   ntssMax = atol(argv[10]);
   ntt = atol(argv[11]);
   shotLimit = 11;


   // cria imagem local zerada
   localImage = (uint8_t*) malloc(Xi*Yi*Zi*sizeof(uint8_t));
   for (i = 0; i < Xi*Yi*Zi; i++)
   {
   	localImage[i] = 0;
   }


   /***** Get Velocity Model *****/
   if(my_rank == 0)
   {
           //Open Velocity Model File
           vModelFile = fopen(argv[1], "rb");
           if (vModelFile == NULL)
           {
                fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[1]);
                exit(0);
           }

           //Get size of velocity model file
           fseek(vModelFile, 0, SEEK_END);
           vModelSize = ftell(vModelFile);
           fseek(vModelFile, 0, SEEK_SET);
	
           printf("My_rank: %d\tAbriu arquivo e pegou tamanho\n", my_rank);
   }

   //Broadcast size of velocity model file
   MPI_Bcast(&vModelSize, 1, MPI_LONG, 0, comm);

   printf("My_rank: %d\tBroadcast do tamanho de velocidades feito\n", my_rank);

   //Allocate velocity model data
   vModelData = (uint8_t*) malloc((vModelSize)*sizeof(uint8_t));
   if (vModelData == NULL) {fputs ("Memory error",stderr); exit (2);}

   printf("My_rank: %d\t Alocou velocidades feito\n", my_rank);

   if(my_rank == 0)
   {
         // Read velocity model file and puts to vModelData
         if (fread(vModelData, 1, vModelSize, vModelFile) != vModelSize)
         {
                fputs ("Reading error",stderr);
                exit (3);
         }

	 printf("My_rank: %d\t Leu velocidades feito\n", my_rank);
         //Close velocity model file
         fclose(vModelFile);

   }

   //Broadcast velocity model data
   MPI_Bcast(vModelData, vModelSize, MPI_UINT8_T, 0, comm);

   printf("My_rank: %d\tBroadcast do modelo de velocidades feito\n", my_rank);

   /***** End of Get Velocity Model *****/


   /*****  Get Traces *****/  
   if(my_rank == 0)
   {
	   //Declare and initialize local variables
           int curSx = 0;			//Position x of shot source of the current trace
	   int curSy = 0;			//Position y of shot source of the current trace
	   unsigned long curTraceNumber = 0;    //Local Trace Number Counter
           unsigned int  dest;			//Destination process index
	   FILE *suFile;			//SU File
	   SuTrace* traces;			//SU Traces

	           
           //Open SU file
           suFile = fopen(argv[9], "rb");
           if (suFile == NULL)
           {
                fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[9]);
                exit(0);
           }

	   
	   //Get number of samples (ns)
	   //Valor de ns no arquivo = 28930
	   
	   fseek(suFile, 114, SEEK_CUR);
           if (fread(&ns, 1, sizeof(unsigned short), suFile) != sizeof(unsigned short)) {
                   printf("getSuTrace failed!\n");
                   exit(0);
           }
	   fseek(suFile, -116, SEEK_CUR);
	   
	   
	   invBytes(&ns, sizeof(unsigned short));//Invert bytes
	   //ns = 625;


	   //Broadcast number of samples (ns)
	   MPI_Bcast(&ns, 1, MPI_SHORT, 0, comm);	   
	   TD = ns*sizeof(float);

	   printf("NS: %d\t TD: %d\tSizeof(short): %lu\n", ns, TD, sizeof(short));


	   //Allocate Traces
           traces = (SuTrace*) malloc(ntssMax*sizeof(SuTrace));
           for (i = 0; i < ntssMax; i++)
           {
	           traces[i].data = (float *) malloc(TD);
           }


	   //Allocate Image		   
	   image = (uint8_t*) malloc(Xi*Yi*Zi*sizeof(uint8_t));
   	   for (i = 0; i < Xi*Yi*Zi; i++)
   	   {
   	   	image[i] = 0;
   	   }


           while(traceNumber < ntt && numberShot < shotLimit)
           {
	        //Get Next Sx and Sy                    
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

		//Invert bytes
		invBytes(&curSx, sizeof(int));
		invBytes(&curSy, sizeof(int));	

		dest = (numberShot % (comm_sz-1)) + 1; //Compute destination process


		//Send shot source (sx,sy)
                MPI_Send(&curSx, 1, MPI_INT, dest, 0, comm);
                MPI_Send(&curSy, 1, MPI_INT, dest, 0, comm);		


		//Initialize variables
                sx = curSx;
                sy = curSy;
                curTraceNumber = 0;



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


                //Send common shot traces number and shot source
                MPI_Send(&curTraceNumber, 1, MPI_LONG, dest, 0, comm);
                //printf("%d enviou num traços locais %lu\n", my_rank, curTraceNumber);	
		
		
		//Send Traces
                for (i = 0; i < curTraceNumber; i++)
                {
                	MPI_Send( &(traces[i]), TH, MPI_UINT8_T, dest, 0, comm);
                        MPI_Send( traces[i].data, ns, MPI_FLOAT, dest, 0, comm);
                        //printf("%d enviou traço %u\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", my_rank, i, traces[i].sx, traces[i].sy, sx, sy);
                }


		//Broadcast total trace number count	
		MPI_Bcast(&traceNumber, 1, MPI_LONG, 0, comm);		


                printf("Destination: %u\tNumberShot: %u\tLocalTraceNumber: %lu\tTraceNumber: %lu\n", dest, numberShot, curTraceNumber, traceNumber);
		numberShot++;
        }

        printf("\nMy_rank: %d\tDistribuição dos traços concluída com sucesso!\n", my_rank);
	free(traces);
        fclose(suFile);

   }
   else
   {
	MPI_Status status;	//Status of MPI communication
			

	//Allocate Wave values
	u_i = (uint8_t*) malloc(Xi*Yi*Zi*sizeof(uint8_t));
	u_r = (uint8_t*) malloc(Xi*Yi*Zi*sizeof(uint8_t));
	

	//Broadcast number of samples (ns)
	MPI_Bcast(&ns, 1, MPI_SHORT, 0, comm);       
	TD = ns*sizeof(float);
	

	while(traceNumber < ntt && numberShot < shotLimit)
	{

		
		if(my_rank != (numberShot % (comm_sz-1)) + 1)
		{
			MPI_Bcast(&traceNumber, 1, MPI_LONG, 0, comm);
		}
		else
		{
			//Receive shot source
			MPI_Recv(&sx, 1, MPI_INT, 0, 0, comm, &status);
			MPI_Recv(&sy, 1, MPI_INT, 0, 0, comm, &status);


			// assinatura propagacao
			//Se propagacao for muito custoso deve estar depois de receber traços pois processo 0 ficará esperando
			propagation(u_i, vModelData, Xi, Yi, Zi, dX, dY, dZ, dt, sx, sy);


			//Receive common shot traces number and shot source
			MPI_Recv(&localTraceNumber, 1, MPI_LONG, 0, 0, comm, &status);
			//printf("%d recebeu num traços locais %lu\n", my_rank, localTraceNumber);


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

		
			// recebe numero total de tracos
			MPI_Bcast(&traceNumber, 1, MPI_LONG, 0, comm);


			printf("My_rank: %u\tNumberShot: %u\tLocalTraceNumber: %lu\tTraceNumber: %lu\n", my_rank, numberShot, localTraceNumber, traceNumber);


			// assinatura retropropagacao
			backpropagation(u_r, localTraces, vModelData, Xi, Yi, Zi, dX, dY, dZ, dt, gx, gy);


			// assinatura correlacao cruzada
			imageCondition(localImage, u_i, u_r, Xi, Yi, Zi);

		}
		numberShot++;
		
		
	}

	printf("\nMy_rank: %d\tDistribuição dos traços concluída com sucesso!\n", my_rank);
	// recebe info propagacao
	// assinatura propagacao
	// recebe tracos
	// assinatura retropropagacao
	// assinatura correlacao cruzada
	// adicionar a imagem local
}
    /*****  End of Get Traces *****/



   // adicionar a imagem local
   MPI_Reduce(localImage, image, Xi*Yi*Zi, MPI_UINT8_T, MPI_SUM, 0, comm);

 
   //Print data to File
   //printFile(localTraces, localTraceNumber, TD, vModelData, vModelSize, my_rank);


   //Free pointers
   if(localTraceNumber > 0)
   {
	invBytes(&localTraces[0].sx, sizeof(int));
	invBytes(&localTraces[localTraceNumber-1].sy, sizeof(int));
	printf("Thread: %d\t ficou com %lu traços\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", my_rank, localTraceNumber, localTraces[0].sx, localTraces[localTraceNumber-1].sy, sx, sy);
   	free(localTraces);
   }
   
   if(vModelSize > 0)
   {
   	free(vModelData);
   }

   //End of MPI Processes
   MPI_Finalize();

   return 0;
} 



void usage(char prog_name[], int my_rank) {
   if(my_rank == 0)
   {
   	fprintf(stderr, "usage: %s ", prog_name); 
   	fprintf(stderr, "<velocityModel_file> <Xi> <Yi> <Zi> <dX> <dY> <dZ> <dt> <traces_file.su> <ntssMax> <ntt> \n\n"); //Dimensoes do modelo de velocidade: Xi, Yi, Zi, DeltaX, Deltay DeltaZ
	//ns: Lê do arquivo
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


void propagation(uint8_t* u_i, uint8_t* vModelData, int Xi, int Yi, int Zi, int dX, int dY, int dZ, int dt, int sx, int sy)
{

}


void backpropagation(uint8_t* u_r, SuTrace* localTraces, uint8_t* vModelData, int Xi, int Yi, int Zi, int dX, int dY, int dZ, int dt, int gx, int gy)
{

}


void imageCondition(uint8_t* image, uint8_t* u_i, uint8_t* u_r, int Xi, int Yi, int Zi)
{

}


void printFile(SuTrace* traces, unsigned long localTraceNumber, unsigned int TD, uint8_t* velocity_model_data, unsigned long vModelSize, int my_rank)
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



