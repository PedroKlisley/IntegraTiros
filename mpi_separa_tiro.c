#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>


#define TH 	240 	// Trace Header bytes
#define TD 	2500 	// Trace Data bytes
#define QTFloat 625	// Trace Data Floats
#define TQtt	2286908	// Trace Quantity

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
} SegyTrace;

void Usage(char* errorMessage);
void Check_for_error(int local_ok, char fname[], char message[],
      MPI_Comm comm);
void Get_args(char* argv[], FILE** traces_SU_pp, SegyTrace** traces_data_pp, FILE** velocity_model_file_pp, uint8_t** velocity_model_data_pp, int my_rank, MPI_Comm comm);


/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   int comm_sz, my_rank;
   FILE *traces_SU, *velocity_model_file;
   SegyTrace* traces_data;
   uint8_t *velocity_model_data;
   MPI_Comm comm;

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   /* Check and get command line args */
   if (argc != 3) Usage(argv[0]); 
   Get_args(argv, &traces_SU, &traces_data, &velocity_model_file, &velocity_model_data, my_rank, comm);

   free(traces_data);
   free(velocity_model_data);

   MPI_Finalize();

   return 0;
}  /* main */

/*-------------------------------------------------------------------
 * Function:  Check_for_error
 * Purpose:   Check whether any process has found an error.  If so,
 *            print message and terminate all processes.  Otherwise,
 *            continue execution.
 * In args:   local_ok:  1 if calling process has found an error, 0
 *               otherwise
 *            fname:     name of function calling Check_for_error
 *            message:   message to print if there's an error
 *            comm:      communicator containing processes calling
 *                       Check_for_error:  should be MPI_COMM_WORLD.
 *
 * Note:
 *    The communicator containing the processes calling Check_for_error
 *    should be MPI_COMM_WORLD.
 */
void Check_for_error(
      int       local_ok   /* in */, 
      char      fname[]    /* in */,
      char      message[]  /* in */, 
      MPI_Comm  comm       /* in */) {
   int ok;

   MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
   if (ok == 0) {
      int my_rank;
      MPI_Comm_rank(comm, &my_rank);
      if (my_rank == 0) {
         fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, 
               message);
         fflush(stderr);
      }
      MPI_Finalize();
      exit(-1);
   }
}  /* Check_for_error */


void Usage(char prog_name[]) {
   fprintf(stderr, "usage: %s ", prog_name); 
   fprintf(stderr, "<traces_file.su> <veloityModel_file>\n");
   exit(0);
} 

void getSegyTrace(SegyTrace* st, long tn, FILE *fsegy) { // Lê tn-ésimo traço (com Zi amostras) 0 <= tn < Ntr
  fseek(fsegy, tn*(TH+TD), SEEK_SET);

    
  if (fread(st, 1, TH, fsegy) != TH) {
    printf("getSegyTrace %li failed!\n", tn);
    return;
  }
  
  if (fread(st->data, 1, TD, fsegy) != TD) {
    printf("getSegyTrace %li failed!\n", tn);
    return;
  }
  
}

void Build_mpi_type(SegyTrace* st, MPI_Datatype*  input_mpi_t_p  	/* out */) {

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
}  /* Build_mpi_type */


void TestHexDump(int my_rank)
{
	/*if(my_rank == 0)
   {
           
           int v = 1;
           int w = 1037;
           int x = 512;
           int y = 257;
           int z = 2;

           FILE *fh = fopen ("TesteBin", "wb");
           if (fh != NULL) {
                fwrite (&v, sizeof (v), 1, fh);
                fwrite (&w, sizeof (w), 1, fh);
                fwrite (&x, sizeof (x), 1, fh);
                fwrite (&y, sizeof (y), 1, fh);
                fwrite (&z, sizeof (z), 1, fh);
                fclose (fh);
           }
           
   }  */
}

void Get_args(
      char*    		argv[]        		/* in  */,
      FILE**   		traces_SU_pp   		/* out */,
      SegyTrace**       traces_data_pp		/* out */,
      FILE**   		velocity_model_file_pp  /* out */,
      uint8_t**		velocity_model_data_pp  /* out */,
      int 		my_rank			/* in  */,
      MPI_Comm 		comm		       	/* in  */) {

   long size, traceNumberLocal;
   MPI_Datatype segytrace_t;

   if(my_rank == 0)
   {
           //Open files 
	   *traces_SU_pp = fopen(argv[1], "rb");
	   if (*traces_SU_pp == NULL) 
	   {
	   	fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[1]);
		exit(0);
	   }
	  
           /*
	   fseek(*traces_SU_pp, 0, SEEK_SET);
	   *traces_data_pp = (SegyTrace*) malloc(TQtt*sizeof(SegyTrace));

	   if (fread(*traces_data_pp, 1, 20, *traces_SU_pp) != 20) {
   		printf("getSegyTrace %li failed!\n", 0);
    		return;
  	   }

	   freopen(NULL, "wb", stdout);
           fwrite(*traces_data_pp, 1, 16, stdout);
	   */
	
	   *velocity_model_file_pp = fopen(argv[2], "rb");
	   if (*velocity_model_file_pp == NULL)
	   {
	   	fprintf(stderr, "Erro ao abrir arquivo %s\n", argv[2]);
		exit(0);
	   }

	   fseek(*velocity_model_file_pp, 0, SEEK_END);
	   size = ftell(*velocity_model_file_pp);
	   fseek(*velocity_model_file_pp, 0, SEEK_SET);
	   
           
	
	   //Get Traces
	   int i;
	   *traces_data_pp = (SegyTrace*) malloc(TQtt*sizeof(SegyTrace));
	   for (i = 0; i < TQtt; i++) 
	   {
   		 (*traces_data_pp)[i].data = (float *) malloc(TD);
	   }	
	   int curSx = 0, curSy = 0;
           long traceNumber = 0;
           uint8_t numberShot = 0;
           

           getSegyTrace(*traces_data_pp, traceNumber, *traces_SU_pp);
           curSx = (*traces_data_pp)[traceNumber].sx;
	   curSy = (*traces_data_pp)[traceNumber].sy;
	   traceNumber++;
	
	  /* 
	   freopen(NULL, "wb", stdout);
           fwrite(*traces_data_pp, 1, TH, stdout); 
           //fwrite(&a, 1, 8, stdout);
	   fwrite((*traces_data_pp)[0].data, 1, TD, stdout);
	   */

	   
        while(numberShot == 0)
	{
	   	getSegyTrace(&((*traces_data_pp)[traceNumber]), traceNumber, *traces_SU_pp);
		if((*traces_data_pp)[traceNumber].sx != curSx || (*traces_data_pp)[traceNumber].sy != curSy)
		{
			numberShot = 1;
			curSx = (*traces_data_pp)[traceNumber].sx;
           		curSy = (*traces_data_pp)[traceNumber].sy;
		}
		traceNumber++;
        }
     	   
	long traceNumber1 = traceNumber - 1;
        while(numberShot == 1)
        {
                getSegyTrace(&((*traces_data_pp)[traceNumber]), traceNumber, *traces_SU_pp);
                if((*traces_data_pp)[traceNumber].sx != curSx || (*traces_data_pp)[traceNumber].sy != curSy)
                {
                        numberShot = 2;
                        curSx = (*traces_data_pp)[traceNumber].sx;
                        curSy = (*traces_data_pp)[traceNumber].sy;
                }
        	traceNumber++;
        }

	long traceNumber2 = traceNumber - traceNumber1 - 1;
	
 	
        
	//Send Traces
        MPI_Send(&traceNumber2, 1, MPI_LONG, 1, 0, MPI_COMM_WORLD);
	//Build_mpi_type(*traces_data_pp,&segytrace_t);
	//MPI_Send(*traces_data_pp, 1, segytrace_t, 1, 0, MPI_COMM_WORLD);
	printf("Trace1: %ld\tTrace2: %ld\n", traceNumber1, traceNumber2);
	//freopen(NULL, "wb", stdout);
	//fwrite(*traces_data_pp, 1, 16, stdout);			
	//fwrite(*traces_SU_pp, 1, 16, stdout);
	for (i = 0; i < traceNumber2; i++)
        {
		MPI_Send( &((*traces_data_pp)[traceNumber1+i]), TH, MPI_CHAR, 1, 0, MPI_COMM_WORLD);			
		MPI_Send( (*traces_data_pp)[traceNumber1+i].data, QTFloat, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
	}
	
	//MPI_Send(*traces_data_pp, traceNumber1*(TH+TD), MPI_CHAR, 1, 0, MPI_COMM_WORLD);	   
	//MPI_Send(&((*traces_data_pp)[traceNumber1]), traceNumber2, segytrace_t, 1, 0, MPI_COMM_WORLD);
	traceNumberLocal = traceNumber1;        
	
   }	

   
   if(my_rank == 1)
   {
	
	//Receive Traces
   	MPI_Status status;
	long traceNumber2;
	int i;
	MPI_Recv(&traceNumber2, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, &status);
	*traces_data_pp = (SegyTrace*) malloc(traceNumber2*sizeof(SegyTrace));
        for (i = 0; i < traceNumber2; i++)
        {
        	(*traces_data_pp)[i].data = (float *) malloc(TD);
        }
	for (i = 0; i < traceNumber2; i++)
        {
		MPI_Recv( &((*traces_data_pp)[i]), TH, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv((*traces_data_pp)[i].data, QTFloat, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
	}
        //MPI_Recv(*traces_data_pp, traceNumber2*(TH+TD), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
	//MPI_Recv(*traces_data_pp, traceNumber2, segytrace_t, 0, 0, MPI_COMM_WORLD, &status);
  	traceNumberLocal = traceNumber2;
	
    }
   

   //Broadcast Velocity Model
   MPI_Bcast(&size, 1, MPI_LONG, 0, comm);	 
   *velocity_model_data_pp = (uint8_t*) malloc(size*sizeof(uint8_t));
   if (*velocity_model_data_pp == NULL) {fputs ("Memory error",stderr); exit (2);}
    
   if(my_rank == 0)
   {
          size_t result; 
	  result = fread(*velocity_model_data_pp, 1, size, *velocity_model_file_pp);
          if (result != size) {fputs ("Reading error",stderr); exit (3);}
   }
   
   MPI_Bcast(*velocity_model_data_pp, size, MPI_CHAR, 0, comm);	

   
   //Output Su Files
   char suFileName [20];
   sprintf(suFileName, "output_%d.su", my_rank);

   FILE *outputSu_file;
   outputSu_file = fopen(suFileName, "wb");
   if (outputSu_file == NULL)
   {
        fprintf(stderr, "Erro ao abrir arquivo outputSU\n");
  	exit(0);
   }
   int i;
   for(i = 0; i < traceNumberLocal; i++)
   {
	fwrite(&((*traces_data_pp)[i]), 1, TH, outputSu_file);
	fflush(outputSu_file);
	fwrite((*traces_data_pp)[i].data, 1, TD, outputSu_file);
	fflush(outputSu_file);
   }
   fclose(outputSu_file);

  
   //Output different files
   //Test
   if(my_rank == 1)
   {
       	FILE *output_file;
   	output_file = fopen("outputVel", "wb");
   	if (output_file == NULL)
	{
	   	fprintf(stderr, "Erro ao abrir arquivo output\n");
		exit(0);
   	}  
        fwrite(*velocity_model_data_pp, 1, size, output_file);
   	fflush(output_file);
   	fclose(output_file);
   }
  
   if(my_rank == 0)
   {
	//MPI_Type_free(&segytrace_t);
   	fclose(*traces_SU_pp);
   	fclose(*velocity_model_file_pp);
   }
   
}  /* Get_args */



