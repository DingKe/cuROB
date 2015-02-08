#include <stdio.h>
#include <stdlib.h>

#include "curand.h"
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

#define EXCLUDE
#include "cuROB.h"
#undef EXCLUDE

#pragma comment(lib,"cublas.lib")

#define CHECK_CUDA_ERROR()                                                                                \
{                                                                                                         \
	cudaError_t err = cudaGetLastError();                                                                   \
	if (err != cudaSuccess)                                                                                 \
  {                                                                                                       \
  printf("%s.%s.%d: Error %d - %s.\n", __FILE__, __FUNCTION__, __LINE__, err, cudaGetErrorString(err)); \
  \
  }                                                                                                       \
}                                                                                                         \

static int a = 1111;
int irand()
{
	a =  ( a * 1103515245 + 12345 ) & 0x7fffffff;
	return a;
}
double drand()
{
	return (double)(irand()+1.0)/(0x7fffffff+1.0);
}


/*
** Always be 32 for the current CUDA-enabled GPUs
*/
#define WARP_SIZE 32
/*
** BLOCK_SIZE_EVALUATION should be equal to WARP_SIZE in the implementation
*/
#define BLOCK_SIZE_EVALUATION WARP_SIZE

#define MAX_COMPOSITION_NUM (5)
#define SUBCOMPONENT_NUM (3)


/*
** Miscellaneous
*/
#define SWAP(x,y) { int tem = (y); (y) = (x); (x) = tem;}
#define DIVUP(x,y) ( ((x)+(y)-1)/(y) )

#define PI (3.1415926535897932384626433832795029)
#define E (2.7182818284590452353602874713526625)

#define BIAS (100)
#define OPTIMA_RANGE (40)

/*
** Data for composition functions
*/
static int COMPOSITION_NUMS[] = {5,3,3,5,5,5,3,3};

static double bias[COMPOSITION_FUNC_NUM][MAX_COMPOSITION_NUM] = {
	{0,100,200,300,400},{0,100,200},{0,100,200},{0,100,200,300,400},{0,100,200,300,400},{0,100,200,300,400},{0,100,200},{0,100,200}
};
static double lamda[COMPOSITION_FUNC_NUM][MAX_COMPOSITION_NUM] = {
	{1,1e-6,1e-26,1e-6,1e-6},{1,1,1},{0.25,1,1e-7},{0.25,1,1e-7,2.5,10},{10,10,2.5,2.5,1e-6},{2.5,10,2.5,5e-4,1e-6},{1,1,1},{1,1,1}
};
static double w[MAX_COMPOSITION_NUM*MAX_CONCURRENCY];
static double sigma[COMPOSITION_FUNC_NUM][MAX_COMPOSITION_NUM] = {
	{10,20,30,40,50},{20,20,20},{10,30,50},{10,10,10,10,10},{10,10,10,20,20},{10,20,30,40,50},{10,30,50},{10,30,50}
};

static float biasf[COMPOSITION_FUNC_NUM][MAX_COMPOSITION_NUM] = {
	{0,100,200,300,400},{0,100,200},{0,100,200},{0,100,200,300,400},{0,100,200,300,400},{0,100,200,300,400},{0,100,200},{0,100,200}
};
static float lamdaf[COMPOSITION_FUNC_NUM][MAX_COMPOSITION_NUM] = {
	{1,1e-6f,1e-26f,1e-6f,1e-6f},{1,1,1},{0.25f,1,1e-7f},{0.25f,1,1e-7f,2.5f,10},{10,10,2.5,2.5,1e-6f},{2.5f,10,2.5f,5e-4f,1e-6f},{1,1,1},{1,1,1}
};
static float wf[MAX_COMPOSITION_NUM*MAX_CONCURRENCY];
static float sigmaf[COMPOSITION_FUNC_NUM][MAX_COMPOSITION_NUM] = {
	{10,20,30,40,50},{20,20,20},{10,30,50},{10,10,10,10,10},{10,10,10,20,20},{10,20,30,40,50},{10,30,50},{10,30,50}
};

/*
** Pre-allocated Memory Space
*/
static double *o, *m, *optima, *bufferA, *bufferB, *bufferC, *omega, *devx, *devy;
static float *of, *mf, *optimaf, *bufferAf, *bufferBf, *bufferCf, *omegaf, *devxf, *devyf;


/*
** structure for CUBLAS
*/
static cublasHandle_t handle;

static bool initialized = false;

#ifdef __cplusplus
extern "C" {
#endif

	void curob_initialize()
	{	
		if( initialized )
		{
			return;
		}

		cublasCreate(&handle);

		cudaMalloc(&o, sizeof(double)*BASIC_FUNC_NUM*DIM);
		cudaMalloc(&of, sizeof(float)*BASIC_FUNC_NUM*DIM);
		cudaMalloc(&m,sizeof(double)*BASIC_FUNC_NUM*DIM*DIM);
		cudaMalloc(&mf,sizeof(float)*BASIC_FUNC_NUM*DIM*DIM);

		cudaMalloc(&optima,sizeof(double)*COMPOSITION_FUNC_NUM*MAX_COMPOSITION_NUM*DIM);
		cudaMalloc(&optimaf,sizeof(float)*COMPOSITION_FUNC_NUM*MAX_COMPOSITION_NUM*DIM);

		cudaMalloc(&omega,sizeof(double)*MAX_CONCURRENCY*MAX_COMPOSITION_NUM);
		cudaMalloc(&omegaf,sizeof(float)*MAX_CONCURRENCY*MAX_COMPOSITION_NUM);

		cudaMalloc(&bufferA,sizeof(double)*DIM*MAX_CONCURRENCY);
		cudaMalloc(&bufferAf,sizeof(float)*DIM*MAX_CONCURRENCY);
		cudaMalloc(&bufferB,sizeof(double)*DIM*MAX_CONCURRENCY);
		cudaMalloc(&bufferBf,sizeof(float)*DIM*MAX_CONCURRENCY);
		cudaMalloc(&bufferC,sizeof(double)*MAX_CONCURRENCY*MAX_COMPOSITION_NUM);
		cudaMalloc(&bufferCf,sizeof(float)*MAX_CONCURRENCY*MAX_COMPOSITION_NUM);

		cudaMalloc(&devx,sizeof(double)*DIM*MAX_CONCURRENCY);
		cudaMalloc(&devxf,sizeof(float)*DIM*MAX_CONCURRENCY);

		cudaMalloc(&devy,sizeof(double)*MAX_CONCURRENCY);
		cudaMalloc(&devyf,sizeof(float)*MAX_CONCURRENCY);


		CHECK_CUDA_ERROR();	


		double *h_o = (double*)malloc(sizeof(double)*BASIC_FUNC_NUM*DIM);
		float *h_of = (float*)malloc(sizeof(float)*BASIC_FUNC_NUM*DIM);
		double *h_m = (double*)malloc(sizeof(double)*BASIC_FUNC_NUM*DIM*DIM);
		float *h_mf = (float*)malloc(sizeof(float)*BASIC_FUNC_NUM*DIM*DIM);

		int i, j, k, n;

		/* Generate shifting data randomly */
		for( i = 0; i < BASIC_FUNC_NUM; i++ )
		{
			for( j = 0; j < DIM; j++)
			{			
				h_o[i*DIM+j] = (drand()-0.5)*2*OPTIMA_RANGE;
				h_of[i*DIM+j] = (float)h_o[i*DIM+j];			
			}
		}
		cudaMemcpy(o,h_o,sizeof(double)*BASIC_FUNC_NUM*DIM,cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR();
		cudaMemcpy(of,h_of,sizeof(float)*BASIC_FUNC_NUM*DIM,cudaMemcpyHostToDevice);	
		CHECK_CUDA_ERROR();

		/* Generate rotation data randomly */
		double x1, x2;
		double prod, tem[DIM], * mptr;

		for ( n = 0; n < BASIC_FUNC_NUM; n++ )
		{
			int x[DIM];
			int y[DIM];
			j = 0;

			if( n+1 == HYBRID1 || n+1 == HYBRID2 )
			{
				for( i = 0; i < DIM; i++)
				{
					if ( i < ceil(0.3*DIM) )	
						x[i] = y[i] = 0;
					else if ( i < ceil(0.3*DIM)*2 )
						x[i] = y[i] = 1;
					else
						x[i] = y[i] = 2;
				}
			}
			else if ( n+1 == HYBRID3 || n+1 == HYBRID4 )
			{
				for ( i = 0; i < DIM; i++ )
				{
					if ( i < ceil(0.2*DIM) )	
						x[i] = y[i] = 0;
					else if ( i < ceil(0.2*DIM)*2 )
						x[i] = y[i] = 1;
					else if ( i < ceil(0.3*DIM)+ceil(0.2*DIM)*2 )
						x[i] = y[i] = 2;
					else
						x[i] = y[i] = 3;
				}
			}
			else if( n+1 == HYBRID5 || n+1 == HYBRID6 )
			{
				for ( i = 0; i < DIM; i++)
				{
					if ( i < ceil(0.1*DIM) )	
						x[i] = y[i] = 0;
					else if ( i < ceil(0.2*DIM)+ceil(0.1*DIM) )
						x[i] = y[i] = 1;
					else if ( i < ceil(0.2*DIM)*2+ceil(0.1*DIM) )
						x[i] = y[i] = 2;
					else if ( i < ceil(0.2*DIM)*3+ceil(0.1*DIM) )
						x[i] = y[i] = 3;
					else
						x[i] = y[i] = 4;
				}	
			} else // non-hybrid functions
			{
				for ( i = 0; i < SUBCOMPONENT_NUM; i++ )
				{
					for ( j = i*DIM/SUBCOMPONENT_NUM; j < (i+1)*DIM/SUBCOMPONENT_NUM; j++ )
					{
						x[j] = i;
						y[j] = i;
					}			
				}
			}

			/* Permutation */
			for ( i = 0; i < DIM; i++ )
			{
				for ( j = 0; j < DIM; j++ )
				{
					if ( drand() > 0.5 )
					{
						SWAP(x[i],x[j]);
					}
					if ( drand() > 0.5 && (n < HYBRID1 || n > HYBRID6) )
					{
						SWAP(y[i],y[j]);
					}
				}
			}

			mptr = &h_m[n*DIM*DIM];		
			for ( i = 0; i < DIM; i++ )
			{
				for ( j = 0; j < DIM; j++ )
				{
					if ( x[i] == y[j] )
					{
						/*  Box-Muller for normal distribution */				
						x1 = drand();				
						x2 = drand();;				
						mptr[i*DIM+j] = sqrt(-2.0*log(x1))*cos(2*PI*x2);
					}
					else
					{
						mptr[i*DIM+j] = 0;
					}
				}
			}		 
		}

		for ( n = 0; n < BASIC_FUNC_NUM; n++ )
		{
			mptr = &h_m[n*DIM*DIM];
			/* Gram-Schmidt Orthonormalization */
			for ( i = 0; i < DIM; i++)
			{
				/* Orthogonalize */
				for ( k = 0; k < DIM; k++ )
				{
					tem[k] = mptr[i*DIM+k];
				}
				for ( j = 0; j < i; j++ )
				{		
					/* Inner product */
					prod = 0;
					for ( k = 0; k < DIM; k++ )
					{
						prod += mptr[i*DIM+k]*mptr[j*DIM+k];
					}

					for ( k = 0; k < DIM; k++ )
					{
						tem[k] -= prod*mptr[j*DIM+k];
					}
				}

				/* Normalize */
				prod = 0;
				for( k = 0; k < DIM; k++)
				{
					prod += tem[k]*tem[k];
				}	
				prod = sqrt(prod);
				for( k = 0; k < DIM; k++)
				{
					mptr[i*DIM+k] = tem[k]/prod;
				}
			}
		}

		for ( n = 0; n < BASIC_FUNC_NUM; n++ )
		{		
			for( i = 0; i < DIM; i++)
			{
				for( j = 0; j < DIM; j++)
				{
					h_mf[n*DIM*DIM+i*DIM+j] = (float)h_m[n*DIM*DIM+i*DIM+j];
				}
			}
		}	

		cudaMemcpy(m,h_m,sizeof(double)*BASIC_FUNC_NUM*DIM*DIM,cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR();
		cudaMemcpy(mf,h_mf,sizeof(float)*BASIC_FUNC_NUM*DIM*DIM,cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR();

		double *h_optima = (double*)malloc(sizeof(double)*COMPOSITION_FUNC_NUM*MAX_COMPOSITION_NUM*DIM);
		float *h_optimaf = (float*)malloc(sizeof(float)*COMPOSITION_FUNC_NUM*MAX_COMPOSITION_NUM*DIM);

		for( n = 0; n < COMPOSITION_FUNC_NUM; n++ )
		{		
			double * A = h_optima+n*MAX_COMPOSITION_NUM*DIM;
			float * Af = h_optimaf+n*MAX_COMPOSITION_NUM*DIM;
			for( i = 0; i < MAX_COMPOSITION_NUM; i++ )
			{
				for( j = 0; j < DIM; j++)
				{
					A[i*DIM+j] = (drand()-0.5)*2*OPTIMA_RANGE;
				}
			}
			/* set 3rd to 0s */
			for( j = 0; j < DIM; j++)
			{
				A[2*DIM+j] = 0;
			}

			for( i = 0; i < MAX_COMPOSITION_NUM; i++ )
			{
				for( j = 0; j < DIM; j++)
				{
					Af[i*DIM+j] = (float)A[i*DIM+j];
				}
			}
		}

		cudaMemcpy( optima, h_optima, sizeof(double)*COMPOSITION_FUNC_NUM*MAX_COMPOSITION_NUM*DIM, cudaMemcpyHostToDevice );
		CHECK_CUDA_ERROR();
		cudaMemcpy( optimaf, h_optimaf, sizeof(float)*COMPOSITION_FUNC_NUM*MAX_COMPOSITION_NUM*DIM, cudaMemcpyHostToDevice );
		CHECK_CUDA_ERROR();


		//#define VERIFY	
#ifdef VERIFY

		/* Shift Vectors */
		for( n = 0; n < BASIC_FUNC_NUM; n++ )
		{		
			double * A = h_o+n*DIM;		
			printf("No. %d:\n",n+1);		
			for( j = 0; j < DIM; j++)
			{
				printf("%.20f\t", A[j]);
			}
			printf("\n");		
		}


		for( n = 0; n < COMPOSITION_FUNC_NUM; n++ )
		{		
			double * A = h_optima+n*MAX_COMPOSITION_NUM*DIM;		
			printf("No. %d:\n",n+BASIC_FUNC_NUM+1);
			for( i = 0; i < COMPOSITION_NUMS[n]; i++ )
			{
				printf("Composition %d\n",i+1);
				for( j = 0; j < DIM; j++)
				{
					printf("%.20f\t", A[i*DIM+j]);
				}
				printf("\n");
			}
		}


		/* rotation matrice */
		for( n = 0; n < BASIC_FUNC_NUM; n++)
		{
			printf("No. %d:\n",n+1);
			double* A = &h_m[n*DIM*DIM];			
			for( i = 0; i < DIM; i++)
			{
				for( j = 0; j < DIM; j++)
				{
					printf("%.20f\t",A[i*DIM+j]);
				}
				printf("\n");
			}
		}

		/* verify orthogoalism and normalism */
		for( n = 0; n < BASIC_FUNC_NUM; n++)
		{
			printf("No. %d:\n",n+1);
			double* A = &h_m[n*DIM*DIM];			
			for( i = 0; i < DIM; i++)
			{
				for( j = 0; j < DIM; j++)
				{
					/* inner product*/
					double product = 0;
					for( k = 0; k < DIM; k++)
					{
						product += A[i*DIM+k]*A[j*DIM+k];
					}
					printf("%.3f\t",product);
				}
				printf("\n");
			}		
		}	

#endif

		cudaDeviceSynchronize();

		free(h_o);
		free(h_of);
		free(h_m);
		free(h_mf);

		free(h_optima);
		free(h_optimaf);


		initialized = true;
		return;
	}

	void curob_dispose()
	{
		cublasDestroy(handle);

		cudaFree(o);
		cudaFree(of);
		cudaFree(m);
		cudaFree(mf);
		cudaFree(optima);
		cudaFree(optimaf);

		cudaFree(bufferA);
		cudaFree(bufferAf);
		cudaFree(bufferB);	
		cudaFree(bufferBf);	
		cudaFree(bufferC);
		cudaFree(bufferCf);

		cudaFree(omega);
		cudaFree(omegaf);

		cudaFree(devx);	
		cudaFree(devxf);
		cudaFree(devy);
		cudaFree(devyf);

		CHECK_CUDA_ERROR();

		initialized = false;
	}

	int fids[] = {
		SPHERE, ELLIPSOID, ELLIPTIC, DISCUS, CIGAR, POWERS, SHARPV,
		STEP, WEIERSTRASS, GRIEWANK, RASTRIGIN_U, RASTRIGIN, SCHAFFERSF7, GRIE_ROSEN,
		ROSENBROCK, SCHWEFEL_U, SCHWEFEL, KATSUURA,  LUNACEK,  ACKLEY, HAPPYCAT, HGBAT, SCHAFFERSF6,
		HYBRID1, HYBRID2, HYBRID3, HYBRID4, HYBRID5, HYBRID6,
		COMPOSITION1, COMPOSITION2, COMPOSITION3, COMPOSITION4, COMPOSITION5, COMPOSITION6, COMPOSITION7, COMPOSITION8
	};

#ifdef __cplusplus
}
#endif


/*
** Single Precison Float
*/
#define float_t float
#define SUFFIX(name) name ## f
#define gemm cublasSgemm
#define scal cublasSscal
#define nrm2 cublasSnrm2
#define axpy cublasSaxpy
#include "cuROB_kernels.inc"


#undef float_t
#undef SUFFIX
#undef gemm
#undef scal
#undef nrm2
#undef axpy


/*
** Double Precision Float
*/
#define float_t double
#define SUFFIX(name) name
#define gemm cublasDgemm
#define scal cublasDscal
#define nrm2 cublasDnrm2
#define axpy cublasDaxpy
#include "cuROB_kernels.inc"




