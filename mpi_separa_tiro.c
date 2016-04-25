#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
//#include <string>
#include <mpi.h>
#include <limits.h>

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
void getInput(char* argv[], unsigned int* ns, unsigned long* ntssMax, unsigned int* shotLimit, unsigned long int* ntt, unsigned int* TD, int my_rank, int comm_sz, MPI_Comm comm);
void getTraces(char* argv[], SuTrace** traces_data_pp, unsigned long* localTraceNumber, int* sx_p, int* sy_p, int my_rank, int comm_sz, MPI_Comm comm);
void getVModel(char* argv[], uint8_t** velocity_model_data_pp, unsigned long* vModelSize, int my_rank, int comm_sz, MPI_Comm comm);
void printFile(SuTrace* traces, unsigned long localTraceNumber, uint8_t* velocity_model_data, unsigned long vModelSize, int my_rank);

//Global variables declaration
unsigned int ns;
unsigned long ntssMax; 
unsigned int shotLimit; 
unsigned long ntt;
unsigned int TD;


int main(int argc, char* argv[]) {

   //Variable declarations
   int comm_sz, my_rank;
   SuTrace* localTraces;
   uint8_t *velocity_model_data;
   unsigned long localTraceNumber, vModelSize;
   int sx, sy;
   MPI_Comm comm;

   //MPI start
   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   // Check command line arguments
   if (argc != 6) 
   {
	usage(argv[0], my_rank); 
   }
 
   // Get arguments from function call in command line
   ns = atoi(argv[3]);
   ntssMax = atol(argv[4]);
   ntt = atol(argv[5]);
   shotLimit = comm_sz-1;
   TD = ns*sizeof(float);

   if(my_rank == 1)
   {
	printf("NS: %u\n", ns);
   }

   //getInput(argv, &ns, &ntssMax, &shotLimit, &ntt, &TD, my_rank, comm_sz, comm);
   getTraces(argv, &localTraces, &localTraceNumber, &sx, &sy, my_rank, comm_sz, comm);
   getVModel(argv, &velocity_model_data, &vModelSize, my_rank, comm_sz, comm);
   printFile(localTraces, localTraceNumber, velocity_model_data, vModelSize, my_rank);

   if(localTraceNumber > 0)
   {
	printf("Thread: %d\t ficou com %lu traços\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", my_rank, localTraceNumber, localTraces[0].sx, localTraces[localTraceNumber-1].sy, sx, sy);
   	free(localTraces);
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
   	fprintf(stderr, "<traces_file.su> <velocityModel_file> <ns> <ntssMax> <ntt> \n\n");
   }
   MPI_Finalize();
   exit(0);
} 

/*
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

  if (fread(trace[curTraceIndex].data, 1, TD, suFile) != TD) {
    printf("getSuTrace failed!\n");
    return;
  }

}
*/

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
      int*		sx_p			/* out */,
      int*		sy_p			/* out */,
      int 		my_rank			/* in  */,
      int		comm_sz			/* in  */,
      MPI_Comm 		comm		       	/* in  */) {

   unsigned int dest = 0;
   //MPI_Datatype segytrace_t;
   *localTraceNumber = 0;

   if(my_rank == 0)
   {
           //Open files 
	   FILE *suFile = fopen(argv[1], "rb");
	   if (suFile == NULL) 
	   {
	   	fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[1]);
		exit(0);
	   }
	  
	   //Get Traces
	   int i;
	   SuTrace* traces;
	   traces = (SuTrace*) malloc(ntssMax*sizeof(SuTrace));

	   for (i = 0; i < ntssMax; i++) 
	   {
   		 traces[i].data = (float *) malloc(TD);
	   }	

	   int curSx = 0, curSy = 0;
	   unsigned long traceNumber = 0, curTraceNumber = 0;
           unsigned int  numberShot = 0;//, dest;
           

	   while(traceNumber < ntt && numberShot < shotLimit)
	   {
		traces = (SuTrace*) malloc(ntssMax*sizeof(SuTrace));

	        for (i = 0; i < ntssMax; i++)
        	{
        		traces[i].data = (float *) malloc(TD);
	        }

	
		//Get Next Sx and Sy                    
                fseek(suFile, 72, SEEK_CUR);

                if (fread(&curSx, 1, sizeof(int), suFile) != sizeof(int)) {
                        printf("getSuTrace failed!\n");
                        return;
                }

                if (fread(&curSy, 1, sizeof(int), suFile) != sizeof(int)) {
                        printf("getSuTrace failed!\n");
	                return;
                }

                fseek(suFile, -80, SEEK_CUR);

                *sx_p = curSx;
           	*sy_p = curSy;
	        curTraceNumber = 0;	

        	while(*sx_p == curSx && *sy_p == curSy && traceNumber < ntt)
		{
			//Get Trace
			if (fread(&(traces[curTraceNumber]), 1, TH, suFile) != TH) {
   				printf("getSuTrace failed!\n");
				return;
  			}

  			if (fread(traces[curTraceNumber].data, 1, TD, suFile) != TD) {
			    	printf("getSuTrace failed!\n");
    				return;
  			}
	

			//Increment variables
                        traceNumber++;
                        curTraceNumber++;

                        if(traceNumber < ntt)
			{
				//Get Next Sx and Sy			
				fseek(suFile, 72, SEEK_CUR);

	  			if (fread(&curSx, 1, sizeof(int), suFile) != sizeof(int)) {
    					printf("getSuTrace failed!\n");
    					return;
  				}

	  			if (fread(&curSy, 1, sizeof(int), suFile) != sizeof(int)) {
   					printf("getSuTrace failed!\n");
   	 				return;
  				}

	  			fseek(suFile, -80, SEEK_CUR);		
			}
	        }

       		dest = (numberShot % (comm_sz-1)) + 1;
	        numberShot++;	
		
		//Send Traces
	        	MPI_Send(&curTraceNumber, 1, MPI_LONG, dest, 0, comm);
		        MPI_Send(sx_p, 1, MPI_INT, dest, 0, comm);
			MPI_Send(sy_p, 1, MPI_INT, dest, 0, comm);	
			printf("%d enviou num traços locais %lu\n", my_rank, curTraceNumber);

			for (i = 0; i < curTraceNumber; i++)
        		{
				MPI_Send( &(traces[i]), TH, MPI_CHAR, dest, 0, comm);			
				MPI_Send( traces[i].data, ns, MPI_FLOAT, dest, 0, comm);
				printf("%d enviou traço %u\tSxReal: %d\tSyReal: %d\t Sx: %d\t Sy: %d\n", my_rank, i, traces[i].sx, traces[i].sy, *sx_p, *sy_p);
			}	

		printf("Destination: %u\tNumberShot: %u\tLocalTraceNumber: %lu\tTraceNumber: %lu\n", dest, numberShot, curTraceNumber, traceNumber);
		free(traces);
	}

	printf("\nDistribuição dos traços concluída com sucesso!\n\n");
        //MPI_Type_free(&segytrace_t);
	fclose(suFile);
       
   }	
   else 
   {
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

		printf("nSQTT: %u\t ntssMax: %lu\t sL: %u\t ntt: %lu\t TD: %u\n", ns, ntssMax, shotLimit, ntt, TD);
		printf("TD: %d\t localTData[0]: %.2f\n", TD, (*localTraces_pp)[111].data[0]);
		for (i = 0; i < *localTraceNumber; i++)
       		{
			MPI_Recv( &((*localTraces_pp)[i]), TH, MPI_CHAR, 0, 0, comm, &status);
			MPI_Recv( (*localTraces_pp)[i].data, ns, MPI_FLOAT, 0, 0, comm, &status);
			printf("%d recebeu traço %lu\n", my_rank, i);
		}
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



