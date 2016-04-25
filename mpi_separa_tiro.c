#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>

//Tiro 1 - 0 enviou traço 111	SxReal: 336527360	SyReal: -66912256 	Sx: 336527360 	Sy: -66912256
//Tiro 2 - 0 enviou traço 127	SxReal: 336527360	SyReal: 1275330560	Sx: 336527360	Sy: 1275330560

//Melhorar variáveis e prints
//Organizar e comentar código

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
void getInput(char* argv[], unsigned int* nSQtt, unsigned long* nTMax, unsigned int* shotLimit, unsigned long int* nTQtt, unsigned int* TD, int my_rank, int comm_sz, MPI_Comm comm);
void getTraces(char* argv[], SuTrace** traces_data_pp, unsigned long* localTraceNumber, unsigned int* sx_p, unsigned int* sy_p, int my_rank, int comm_sz, MPI_Comm comm);
void getVModel(char* argv[], uint8_t** velocity_model_data_pp, unsigned long* vModelSize, int my_rank, int comm_sz, MPI_Comm comm);
void printFile(SuTrace* traces, unsigned long localTraceNumber, uint8_t* velocity_model_data, unsigned long vModelSize, int my_rank);

//Global variables declaration
unsigned int nSQtt;
unsigned long nTMax; 
unsigned int shotLimit; 
unsigned long int nTQtt;
unsigned int TD;


int main(int argc, char* argv[]) {

   //Variable declarations
   int comm_sz, my_rank;
   SuTrace* traces;
   uint8_t *velocity_model_data;
   unsigned long localTraceNumber, vModelSize;
   unsigned int sx, sy;
   MPI_Comm comm;

   //MPI start
   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   // Check command line arguments
   if (argc != 4) 
   {
	usage(argv[0], my_rank); 
   }
 
   getInput(argv, &nSQtt, &nTMax, &shotLimit, &nTQtt, &TD, my_rank, comm_sz, comm);
   getTraces(argv, &traces, &localTraceNumber, &sx, &sy, my_rank, comm_sz, comm);
   getVModel(argv, &velocity_model_data, &vModelSize, my_rank, comm_sz, comm);
   printFile(traces, localTraceNumber, velocity_model_data, vModelSize, my_rank);

   if(localTraceNumber > 0)
   {
	printf("Thread: %d\t ficou com %lu traços\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", my_rank, localTraceNumber, traces[0].sx, traces[localTraceNumber-1].sy, sx, sy);
   	free(traces);
   }
   
   if(vModelSize > 0)
   {
   	free(velocity_model_data);
   }

   MPI_Finalize();

   return 0;
} 



void usage(char prog_name[], int my_rank) {
   if(my_rank == 0)
   {
   	fprintf(stderr, "usage: %s ", prog_name); 
   	fprintf(stderr, "<traces_file.su> <velocityModel_file> <defInput>\n\n");
   }
   MPI_Finalize();
   exit(0);
} 



void getInput(char* argv[], unsigned int* nSQtt, unsigned long* nTMax, unsigned int* shotLimit, unsigned long int* nTQtt, unsigned int* TD, int my_rank, int comm_sz, MPI_Comm comm)
{
        if(my_rank == 0)
        {
                int defInput = atoi(argv[3]);
                if(defInput == 0)
                {
                        printf("Which is the number of samples in a trace?\n");
                        scanf("%u", nSQtt);
                        printf("Which is the maximum number of traces from the same shot?\n");
                        scanf("%lu", nTMax);
                        printf("How many shots do you want to extract?\n");
                        scanf("%u", shotLimit);
                        printf("Which is the number of traces in the file?\n");
                        scanf("%lu", nTQtt);
                }
                else
                {
                        *nSQtt = 625;
                        *nTMax = 544;
                        *shotLimit = comm_sz-1;
                        *nTQtt = 2286908;
                }
         
                *TD = *nSQtt * sizeof(float);
        }

        MPI_Bcast(nSQtt, 1, MPI_INT, 0, comm);
        MPI_Bcast(nTMax, 1, MPI_LONG, 0, comm);
        MPI_Bcast(shotLimit, 1, MPI_INT, 0, comm);
        MPI_Bcast(nTQtt, 1, MPI_LONG, 0, comm);
	MPI_Bcast(TD, 1, MPI_INT, 0, comm);
}


void getNextSxSy(FILE *suFile, int *nextSx, int *nextSy)
{
  fseek(suFile, 72, SEEK_CUR);
   
  if (fread(nextSx, 1, sizeof(int), suFile) != sizeof(int)) {
    printf("getSuTrace failed!\n");
    return;
  }
  
  if (fread(nextSy, 1, sizeof(int), suFile) != sizeof(int)) {
    printf("getSuTrace failed!\n");
    return;
  }
   
  fseek(suFile, -80, SEEK_CUR);
}


void getSuTrace(FILE *suFile, SuTrace* trace, int curTraceIndex)
{

  if (fread(&(trace[curTraceIndex]), 1, TH, suFile) != TH) {
    printf("getSuTrace failed!\n");
    return;
  }

  if (fread(trace[curTraceIndex].data, 1, sizeof(float)*nSQtt, suFile) != sizeof(float)*nSQtt) {
    printf("getSuTrace failed!\n");
    return;
  }

}

/*
void Build_mpi_type(SuTrace* st, MPI_Datatype*  input_mpi_t_p  */	/* out *//*) {

   int array_of_blocklengths[2] = {TH, 625};
   MPI_Datatype array_of_types[2] = {MPI_CHAR, MPI_FLOAT};
   MPI_Aint st_addr, data_addr;
   MPI_Aint array_of_displacements[2] = {0};
   MPI_Get_address(st, &st_addr);
   MPI_Get_address(st->data, &data_addr);
   array_of_displacements[1] = data_addr - st_addr; 
   MPI_Type_create_struct(2, array_of_blocklengths, 
         array_of_displacements, array_of_types,
         input_mpi_t_p);
   MPI_Type_commit(input_mpi_t_p);
} */ /* Build_mpi_type */


void getTraces(
      char*    		argv[]        		/* in  */,
      SuTrace**  	localTraces_pp		/* out */,
      unsigned long*	localTraceNumber	/* out */,
      unsigned int*	sx_p			/* out */,
      unsigned int*	sy_p			/* out */,
      int 		my_rank			/* in  */,
      int		comm_sz			/* in  */,
      MPI_Comm 		comm		       	/* in  */) {

   unsigned int dest = 0;
   //MPI_Datatype segytrace_t;
   *localTraceNumber = 0;

   if(my_rank == 0)
   {
           //Open files 
	   FILE *traces_SU_p = fopen(argv[1], "rb");
	   if (traces_SU_p == NULL) 
	   {
	   	fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[1]);
		exit(0);
	   }
	  
	   //Get Traces
	   int i;
	   SuTrace* traces;
	   traces = (SuTrace*) malloc(nTMax*sizeof(SuTrace));

	   for (i = 0; i < nTMax; i++) 
	   {
   		 traces[i].data = (float *) malloc(TD);
	   }	

	   int curSx = 0, curSy = 0;
	   unsigned long traceNumber = 0, curTraceNumber = 0, prev2 = 0; // prevTraceNumber = 0;
           unsigned int  numberShot = 0;//, dest;
           

	   while(traceNumber < nTQtt && numberShot < shotLimit)
	   {
		traces = (SuTrace*) malloc(nTMax*sizeof(SuTrace));

	        for (i = 0; i < nTMax; i++)
        	{
        		traces[i].data = (float *) malloc(TD);
	        }

		curTraceNumber = 0;
		prev2 = curTraceNumber;
		getNextSxSy(traces_SU_p, &curSx, &curSy);
           	traces[0].sx = curSx;
           	traces[0].sy = curSy;

        	while(traces[prev2].sx == curSx && traces[prev2].sy == curSy && traceNumber < nTQtt)
		{	
			getSuTrace(traces_SU_p, traces, curTraceNumber);
			getNextSxSy(traces_SU_p, &curSx, &curSy);
			traceNumber++;
			prev2 = curTraceNumber;
			curTraceNumber++;
	        }

		*sx_p = traces[prev2].sx;
		*sy_p = traces[prev2].sy;
		numberShot++;
       		dest = numberShot % comm_sz;
		MPI_Send(&dest, 1, MPI_INT, dest, 0, comm);
		//Send Traces
		if(dest != 0)
		{
	        	MPI_Send(&curTraceNumber, 1, MPI_LONG, dest, 0, comm);
		        MPI_Send(sx_p, 1, MPI_INT, dest, 0, comm);
			MPI_Send(sy_p, 1, MPI_INT, dest, 0, comm);	
			printf("%d enviou num traços locais %lu\n", my_rank, curTraceNumber);

			for (i = 0; i < curTraceNumber; i++)
        		{
				MPI_Send( &(traces[i]), TH, MPI_CHAR, dest, 0, comm);			
				MPI_Send( traces[i].data, nSQtt, MPI_FLOAT, dest, 0, comm);
				printf("%d enviou traço %u\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", my_rank, i, traces[i].sx, traces[i].sy, *sx_p, *sy_p);
			}	
		}
		else
		{

			*localTraceNumber = curTraceNumber;
			*localTraces_pp = (SuTrace*) malloc(*localTraceNumber*sizeof(SuTrace));
			for (i = 0; i < *localTraceNumber; i++)
           		{
                 		(*localTraces_pp)[i].data = (float *) malloc(TD);
           		}
			
			for (i = 0; i < *localTraceNumber; i++)
        		{
				memcpy( &((*localTraces_pp)[i]), &(traces[i]), TH);
				memcpy( (*localTraces_pp)[i].data, traces[i].data, TD);
                		printf("%d recebeu traço %d\n", my_rank, i);
        		}				
		}

		printf("Destination: %u\tNumberShot: %u\tLocalTraceNumber: %lu\tTraceNumber: %lu\n", dest, numberShot, curTraceNumber, traceNumber);
		//prevTraceNumber = traceNumber;
		free(traces);
	}

	printf("\nDistribuição dos traços concluída com sucesso!\n\n");
        //MPI_Type_free(&segytrace_t);
	fclose(traces_SU_p);
       
   }	
   else 
   {
        MPI_Recv(&dest, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);	
	/*if(dest == my_rank)
	{*/
		//Receive Traces
	   	MPI_Status status;
		long i;
	
		MPI_Recv(localTraceNumber, 1, MPI_LONG, 0, 0, comm, &status);
		MPI_Recv(sx_p, 1, MPI_INT, 0, 0, comm, &status);
		MPI_Recv(sy_p, 1, MPI_INT, 0, 0, comm, &status);
		printf("%d recebeu num traços locais %lu\n", my_rank, *localTraceNumber);
		*localTraces_pp = (SuTrace*) malloc(*localTraceNumber*sizeof(SuTrace));

        	for (i = 0; i < *localTraceNumber; i++)
        	{
	        	(*localTraces_pp)[i].data = (float *) malloc(TD);
		}

		printf("nSQTT: %u\t nTMax: %lu\t sL: %u\t nTQtt: %lu\t TD: %u\n", nSQtt, nTMax, shotLimit, nTQtt, TD);
		printf("TD: %d\t localTData[0]: %.2f\n", TD, (*localTraces_pp)[111].data[0]);
		for (i = 0; i < *localTraceNumber; i++)
       		{
			MPI_Recv( &((*localTraces_pp)[i]), TH, MPI_CHAR, 0, 0, comm, &status);
			MPI_Recv( (*localTraces_pp)[i].data, nSQtt, MPI_FLOAT, 0, 0, comm, &status);
			printf("%d recebeu traço %lu\n", my_rank, i);
		}
	//}
    }	 
}

void getVModel(
      char*             argv[]                  /* in  */,
      uint8_t**         velocity_model_data_pp  /* out */,
      unsigned long*    vModelSize              /* out */,
      int               my_rank                 /* in  */,
      int               comm_sz                 /* in  */,
      MPI_Comm          comm                    /* in  */) 
{

   FILE *velocity_model_file_p;
   //*vModelSize = 0;

   //Broadcast Velocity Model
   if(my_rank == 0)
   {
	   velocity_model_file_p = fopen(argv[2], "rb");
           if (velocity_model_file_p == NULL)
           {
                fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[2]);
                exit(0);
           }

           fseek(velocity_model_file_p, 0, SEEK_END);
           *vModelSize = ftell(velocity_model_file_p);
           fseek(velocity_model_file_p, 0, SEEK_SET);
   }
   
   MPI_Bcast(vModelSize, 1, MPI_LONG, 0, comm);

   *velocity_model_data_pp = (uint8_t*) malloc((*vModelSize)*sizeof(uint8_t));
   if (*velocity_model_data_pp == NULL) {fputs ("Memory error",stderr); exit (2);}

   if(my_rank == 0)
   {
	   size_t result;
           result = fread(*velocity_model_data_pp, 1, *vModelSize, velocity_model_file_p);
           if (result != *vModelSize) {fputs ("Reading error",stderr); exit (3);}

	   fclose(velocity_model_file_p);

   }

   MPI_Bcast(*velocity_model_data_pp, *vModelSize, MPI_CHAR, 0, comm);	
 
   printf("My_rank: %d\tBroadcast do modelo de velocidades feito\n", my_rank);   
}


void printFile(SuTrace* traces, unsigned long localTraceNumber, uint8_t* velocity_model_data, unsigned long vModelSize, int my_rank)
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



