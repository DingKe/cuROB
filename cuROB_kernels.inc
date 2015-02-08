/*
** Elementary Functions
*/
__device__ float_t SUFFIX( basic_rosenbrock ) ( float_t x1, float_t x2 )
{
	float_t rs;
	float_t t;

	t = x1*x1 - x2;
	rs = t*t*100;

	t = x1-1;
	rs += t*t;

	return rs;
}

__device__ float_t SUFFIX( basic_griewank ) ( float_t x )
{	
	return x*x/4000  - cos(x) + 1;	
}

__device__ float_t SUFFIX( basic_schaffersf6 ) (float_t x1, float_t x2)
{
	float_t t1, t2;
	t1 = x1*x1 + x2*x2;
	t2 = sin(sqrt(t1));
	t1 = 1 + (float_t)0.001*t1;

	t2 *= t2;	
	t1 *= t1;

	return (t2-(float_t)0.5)/t1 + (float_t)0.5;
}

__device__ float_t SUFFIX( basic_modified_schwefel ) ( float_t z )
{	
	float_t rs;
	float_t tem;

	z += SUFFIX(420.9687462275036);

	if( z < -500 )
	{
		tem = fmod(-z,SUFFIX(500.)) - SUFFIX(500.);
		rs = tem*sin(sqrt(-tem)) - (z+500)*(z+500)/(10000*DIM);
	}
	else if( z > 500)
	{
		tem = 500 - fmod(z, SUFFIX(500.));
		rs = tem*sin(sqrt(tem)) - (z-500)*(z-500)/(10000*DIM);
	}
	else
	{
		rs = z*sin(sqrt(fabs(z)));
	}
	return SUFFIX(418.9829) - rs;
}

__device__ float_t SUFFIX( basic_rastrigin ) ( float_t z )
{
	float_t rs;
	rs = z*z - 10*cospi(2*z) + 10;	
	return rs;
}

#define KMAX 20
__device__ float_t SUFFIX( basic_weierstrass ) ( float_t z )
{
	z += (float_t)0.5;
	int k;
	float_t rs = 0;
#pragma unroll
	for( k = 0; k <= KMAX; k++ )
	{		
		rs +=  cospi( SUFFIX(2.0)*pow(SUFFIX(3.0),k)*z ) / SUFFIX(exp2)((float_t)k);
	}
	return rs + SUFFIX(1.999999046325684);
}

__device__ float_t SUFFIX( basic_schaffersf7 ) (float_t z)
{
	float_t s;
	
	s = SUFFIX(50.0)*pow(z,SUFFIX(0.2));
	s = sin(s);
	s *= s;	

	s = sqrt(z)*(s+1);
	s *= s;

	return s;
}

// Shuffle operations need Kepler devices or newer
#if __CUDA_ARCH__ >= 300

__device__
double __shfl_down( double var, unsigned int srcLane, int width=32 ) {
  int2 a = *reinterpret_cast<int2*>( &var );
  a.x = __shfl_down( a.x, srcLane, width );
  a.y = __shfl_down( a.y, srcLane, width );
  return *reinterpret_cast<double*>(&a);
}
__device__ float_t reduceSum( float_t value )
{
	for ( unsigned int offset = WARP_SIZE/2; offset > 0; offset >>= 1 )
		value += __shfl_down( value, offset );
	return value;
}
__device__ float_t reduceProduct( float_t value )
{
	for ( unsigned int offset = WARP_SIZE/2; offset > 0; offset >>= 1 )
		value *= __shfl_down( value, offset );
	return value;
}

__global__ void SUFFIX( sphere_using_shuffle ) ( float_t *ys,  float_t *xs )
{		
	xs += DIM*blockIdx.x;
	
	float_t sum = 0;
	float_t z;
	for ( int i = threadIdx.x; i < DIM; i += WARP_SIZE )
	{		
		z = xs[i];
		sum += z*z;
	}
	sum = reduceSum(sum);
	if ( threadIdx.x == 0 )
	{
		ys[blockIdx.x] = sum;		
	}	
}

#endif

/* 
** Unimodal Functions
*/
__global__ void SUFFIX( sphere ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	xs = &xs[bid*DIM];
		
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += z*z;
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if ( tid == 0 )
	{
		ys[bid] = partialSum[0];		
	}	
}

__global__ void SUFFIX( discus ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += z*z;
	}
	partialSum[tid] = sum;
	__syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if ( tid == 0 )
	{
		ys[bid] = partialSum[0] + SUFFIX(999999.0)*x[0]*x[0]; // 10^6-1
	}	
} 

__global__ void SUFFIX( cigar ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
		
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += z*z*1000000;
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if ( tid == 0 )
	{
		z = x[0];
		ys[bid] = partialSum[0] - 999999*z*z; // 10^6-1
	}	
} 

__global__ void SUFFIX( elliptic ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
		
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{	
		z = x[i];
		sum += z*z*SUFFIX(exp10)( i*6/( DIM-SUFFIX(1.0) ) ); // z*z*powf( 10.0f, i/(DIM-1.0f) );
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if( tid == 0)
	{
		ys[bid] = partialSum[0];		
	}	
}

__global__ void SUFFIX( ellipsoid ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += z*z*(i+1);
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if( tid == 0)
	{
		ys[bid] = partialSum[0];		
	}	
}

__global__ void SUFFIX( step ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		z = floor ( z + SUFFIX(0.5) );
		sum += z*z;
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if ( tid == 0 )
	{
		ys[bid] = partialSum[0];		
	}	
}

__global__ void SUFFIX( powers ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
		
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];
		
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += pow( fabs(z), SUFFIX(2.0)+(float_t)(i*4)/(DIM-1) );
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if( tid == 0)
	{
		ys[bid] = sqrt(partialSum[0]);		
	}	
}

__global__ void SUFFIX( sharpv ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];
		
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid+1; i < DIM-1; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += z*z;
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if ( tid == 0)
	{
		sum = partialSum[0];
		z = x[0];
		z *= z;
		ys[bid] = z+100*sqrt(sum-z);		
	}	
}

/*
** Multimodal Functions
*/
__global__ void SUFFIX( rosenbrock ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	for ( int i = tid; i < DIM-1; i += BLOCK_SIZE_EVALUATION )
	{		
		sum += SUFFIX( basic_rosenbrock ) ( x[i], x[i+1] );		
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if ( tid == 0 )
	{
		ys[bid] = partialSum[0];		
	}	
}

__global__ void SUFFIX( ackley ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum1 = 0;
	float_t sum2 = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum1 += z*z;
		sum2 += cospi(2*z);
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
		
		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}

	if( tid == 0)
	{
		sum1 = partialSum1[0];
		sum2 =  partialSum2[0];
		ys[bid] = 20*exp(-SUFFIX(0.2)*sqrt(sum1/DIM)) - exp( sum2/DIM ) + 20 + E; //exp( SUFFIX(1) );	
	}	
}

__global__ void SUFFIX( weierstrass ) ( float_t * ys, float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += SUFFIX(basic_weierstrass) (z);
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if( tid == 0)
	{
		ys[bid] = partialSum[0];
	}		
}

__global__ void SUFFIX( schaffersf7 ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
		
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];	

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;	
	float_t tem;
	float_t z;
	for ( int i = tid; i < DIM-1; i += BLOCK_SIZE_EVALUATION )
	{		
		z = sqrt( x[i]*x[i]+x[i+1]*x[i+1] );
		
		tem = pow(z,SUFFIX(0.2));
		tem = sin(tem*50);
		tem *= tem;		
		tem++;
				
		sum += tem*sqrt(z);		
	}
	partialSum[tid] = sum;	
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];			
	}

	if ( tid == 0 )
	{		
		sum = partialSum[0];
		sum /= DIM-1;
		ys[bid] = sum*sum;
	}	
}

__global__ void SUFFIX( griewank ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION]; 
	volatile __shared__ float_t partialProduct[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];	

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t product = 1;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += z*z;
		product *= cos(z/sqrt(i+SUFFIX(1.0)));
	}
	partialSum[tid] = sum;
	partialProduct[tid] = product;
	// __syncthreads();

	// reduction.	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
		
		partialProduct[tid] *= partialProduct[tid+16];
		partialProduct[tid] *= partialProduct[tid+8];
		partialProduct[tid] *= partialProduct[tid+4];
		partialProduct[tid] *= partialProduct[tid+2];
		partialProduct[tid] *= partialProduct[tid+1];
	}

	if( tid == 0)
	{
		ys[bid] = partialSum[0]/4000 - partialProduct[0] + 1;	
	}	
}

__global__ void SUFFIX( rastrigin ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;	
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += SUFFIX( basic_rastrigin )(z);		
	}
	partialSum[tid] = sum;	
	// __syncthreads();

	// reduction.	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if( tid == 0)
	{
		ys[bid] =  partialSum[0];	
	}	
}

__global__ void SUFFIX( lunacek ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum3[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	xs = &xs[bid*DIM];

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum1, sum2, sum3, z;
	sum1 = sum2 = sum3 = 0;	
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum1 += (z-SUFFIX(2.5))*(z-SUFFIX(2.5));
		sum1 += (z+SUFFIX(2.5))*(z+SUFFIX(2.5));
		sum3 += cospi(2*(z-SUFFIX(2.5)));		
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	partialSum3[tid] = sum3;	
	__syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];

		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];

		partialSum3[tid] += partialSum3[tid+16];
		partialSum3[tid] += partialSum3[tid+8];
		partialSum3[tid] += partialSum3[tid+4];
		partialSum3[tid] += partialSum3[tid+2];
		partialSum3[tid] += partialSum3[tid+1];
	}

	if ( tid == 0)
	{
		sum1 = partialSum1[0];
		sum2 = partialSum2[0];
		sum2 = SUFFIX(0.9)*sum2 + DIM;
		sum3 = partialSum3[0];
		
		sum1 = min(sum1,sum2);
		ys[bid] =  sum1 - 10*sum3 + 10*DIM;	
	}	
}

__global__ void SUFFIX( happycat) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];

	int tid = threadIdx.x;	
	int bid = blockIdx.x;	

	xs = &xs[bid*DIM];	
	
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum1, sum2, z;
	sum1 = sum2 = 0;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum1 += z*z;
		sum2 += z;
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];

		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}

	if ( tid == 0 )
	{
		sum1 = partialSum1[0];
		sum2 = partialSum2[0];
		ys[bid] = pow( fabs(sum1-DIM), SUFFIX(0.25) ) + ( sum1/2  +sum2 )/DIM + SUFFIX(0.5);		
	}	
}

__global__ void SUFFIX( hgbat ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;
		
	xs = &xs[bid*DIM];	
			
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum1, sum2, z;	
	sum1 = sum2 = 0;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum1 += z*z;
		sum2 += z;
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];

		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}

	if ( tid == 0 )
	{
		sum1 = partialSum1[0];
		sum2 = partialSum2[0];
		ys[bid] = sqrt( fabs( sum1*sum1-sum2*sum2 ) ) + (sum1/2+sum2)/DIM + SUFFIX(0.5);		
	}	
}

__global__ void SUFFIX( katsuura ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialProduct[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;	
	int bid = blockIdx.x;	

	xs = &xs[bid*DIM];		

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t factor = 10/pow((float_t)DIM,SUFFIX(1.2));
	float_t product = 1;
	float_t z, sum, shift, two;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum = 0;
		for ( int j = 1; j <= 32; j++ )
		{
			two = exp2((float_t)j) ;
			shift = two * z;
			sum += fabs( shift - round(shift) )/two;
		}
		sum = sum *(i+1) + 1;
		sum = pow( sum, factor );
		product *= sum;
	}
	partialProduct[tid] = product;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialProduct[tid] *= partialProduct[tid+16];
		partialProduct[tid] *= partialProduct[tid+8];
		partialProduct[tid] *= partialProduct[tid+4];
		partialProduct[tid] *= partialProduct[tid+2];
		partialProduct[tid] *= partialProduct[tid+1];
	}

	if ( tid == 0 )
	{
		ys[bid] = 10 * ( partialProduct[0] - 1) / (DIM*DIM);		
	}	
}

__global__ void SUFFIX( modified_schwefel ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
		
	int tid = threadIdx.x;
	int bid = blockIdx.x;	

	xs = &xs[bid*DIM];	

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	float_t z;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		z = x[i];
		sum += SUFFIX( basic_modified_schwefel )(z);
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if( tid == 0)
	{
		ys[bid] = partialSum[0];		
	}	
}

/*
** Expanded Functions
*/
__global__ void SUFFIX( expanded_griewank_rosenbrock ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
		
	int tid = threadIdx.x;	
	int bid = blockIdx.x;	

	xs = &xs[bid*DIM];	
	
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;	
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{					
		sum += SUFFIX(basic_griewank)( SUFFIX(basic_rosenbrock)(x[i], x[(i+1)%DIM]) );
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if ( tid == 0 )
	{
		ys[bid] = partialSum[0];		
	}	
}

__global__ void SUFFIX( expanded_schaffersf6 ) (  float_t *ys,  float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;	

	xs = &xs[bid*DIM];	

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();	
	
	float_t sum = 0;
	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		sum += SUFFIX( basic_schaffersf6 ) ( x[i], x[(i+1)%DIM] );				
	}
	partialSum[tid] = sum;
	// __syncthreads();

	// reduction.	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}

	if( tid == 0)
	{
		ys[bid] = partialSum[0];		
	}	
}

/*
** Hybrid Functions
*/

__global__ void SUFFIX( hybrid1 )( float_t *ys, float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;
		
	xs = &xs[bid*DIM];	

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();

	float_t rs = 0;
	float_t sum, z;
	int stride, dim;

	/* Modified Schwefel */
	sum = 0;
	stride = 0;
	dim = (int)ceil(0.3*DIM);	
	for ( int i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum += SUFFIX( basic_modified_schwefel ) (z);
	}
	partialSum[tid] = sum;
	// __syncthreads();
		
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if( tid == 0)
	{
		rs += partialSum[0];
	}

	/* Rastrigin */
	sum = 0;
	stride += dim;
	dim = (int)ceil(0.3*DIM);	
	for ( int i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum += SUFFIX( basic_rastrigin ) (z);
	}
	partialSum[tid] = sum;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if( tid == 0)
	{
		rs += partialSum[0];
	}

	/* Elliptic */
	sum = 0;
	stride += dim;
	dim = DIM - stride;	
	for ( int i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum += z*z*SUFFIX(exp10)( (i*6)/( dim - SUFFIX(1.0) ) );
	}
	partialSum[tid] = sum;
	// __syncthreads();	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if( tid == 0)
	{
		rs += partialSum[0];
		ys[bid] = rs;
	}
}

__global__ void SUFFIX( hybrid2 )( float_t *ys, float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];
	
	int tid = threadIdx.x;
	int bid = blockIdx.x;	

	xs = &xs[bid*DIM];	

	for ( int i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();

	float_t rs = 0;
	float_t sum1, sum2, z;
	int stride, dim;

	/* Cigar */
	sum1 = 0;
	stride = 0;
	dim = (int)ceil(0.3*DIM);	
	for ( int i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += 1000000*z*z;
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	

	if( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if( tid == 0)
	{
		rs = partialSum1[0];
		rs -= 999999*x[0]*x[0];
	}

	/* HGBat */
	sum1 = sum2 = 0;
	stride += dim;
	dim = (int)ceil(0.3*DIM);	
	for ( int i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += z*z;
		sum2 += z;
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();	

	if( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];

		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}
	if( tid == 0)
	{		
		sum1 = partialSum1[0];
		sum2 = partialSum2[0];
		rs += sqrt( fabs( sum1*sum1-sum2*sum2 ) ) + (sum1/2+sum2)/dim + SUFFIX(0.5);		
	}

	/* Rastrigin */
	sum1 = 0;
	stride += dim;	
	dim = DIM - stride;	
	for ( int i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += SUFFIX( basic_rastrigin ) (z);
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0)
	{
		rs += partialSum1[0];
		ys[bid] = rs;
	}
}

__global__ void SUFFIX( hybrid3 )( float_t *ys, float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialProduct[BLOCK_SIZE_EVALUATION];

	int i, tid, bid, stride, dim;
	float_t sum,product, z, w, rs;	

	tid = threadIdx.x;
	bid = blockIdx.x;
		
	xs = &xs[bid*DIM];	

	for ( i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();

	rs = 0;

	/* Griewank */
	sum = 0;
	product = 1;
	stride = 0;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum += z*z;
		product *= cos(z/sqrt(i+SUFFIX(1.0)));
	}
	partialSum[tid] = sum;
	partialProduct[tid] = product;
	// __syncthreads();

	if( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
		
		partialProduct[tid] *= partialProduct[tid+16];
		partialProduct[tid] *= partialProduct[tid+8];
		partialProduct[tid] *= partialProduct[tid+4];
		partialProduct[tid] *= partialProduct[tid+2];
		partialProduct[tid] *= partialProduct[tid+1];
	}

	if ( tid == 0 )
	{
		rs += partialSum[0]/4000 - partialProduct[0] + 1;	
	}
	
	/* Weierstrass */
	sum = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum += SUFFIX(basic_weierstrass)(z);		
	}
	partialSum[tid] = sum;	
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if ( tid == 0 )
	{		
		rs += partialSum[0];		
	}

	/* Rosenbrock */
	sum = 0;
	stride += dim;
	dim = (int)ceil(0.3*DIM);	
	for ( i = tid; i < dim-1; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];
		w = x[i+1+stride]; 
		sum += SUFFIX(basic_rosenbrock) (z,w);		
	}
	partialSum[tid] = sum;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if ( tid == 0 )
	{		
		rs += partialSum[0];	
	}

	/* Expanded Scaffer¡¯s F6 */
	sum = 0;
	stride += dim;
	dim = DIM - stride;	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];	
		w = x[(i+1)%dim+stride];
		sum += SUFFIX( basic_schaffersf6 ) ( z, w );
	}
	partialSum[tid] = sum;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum[tid] += partialSum[tid+16];
		partialSum[tid] += partialSum[tid+8];
		partialSum[tid] += partialSum[tid+4];
		partialSum[tid] += partialSum[tid+2];
		partialSum[tid] += partialSum[tid+1];
	}
	if ( tid == 0 )
	{
		rs += partialSum[0];
		ys[bid] = rs;
	}
}

__global__ void SUFFIX( hybrid4 )( float_t *ys, float_t *xs)
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];

	int i, tid, bid, stride, dim;
	float_t sum1, sum2, z, w, rs;	

	tid = threadIdx.x;
	bid = blockIdx.x;
		
	xs = &xs[bid*DIM];	

	for ( i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();

	rs = 0;

	/* HGBat */
	sum1 = sum2 = 0;
	stride = 0;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += z*z;
		sum2 += z;
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];

		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}
	if ( tid == 0 )
	{		
		sum1 = partialSum1[0];
		sum2 = partialSum2[0];
		rs += sqrt( fabs( sum1*sum1-sum2*sum2 ) ) + (sum1/2+sum2)/dim + SUFFIX(0.5);		
	}
		
	/* Discus */
	sum1 = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += z*z;		
	}
	partialSum1[tid] = sum1;	
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{		
		rs += partialSum1[0] + SUFFIX(999999.0)*x[0+stride]*x[0+stride];		
	}

	/* Expanded Griewank¡¯s plus Rosenbrock */
	sum1 = 0;
	stride += dim;
	dim = (int)ceil(0.3*DIM);	
	for ( i = 0; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];
		w = x[(i+1)%dim+stride]; 
		sum1 += SUFFIX(basic_griewank) ( SUFFIX(basic_rosenbrock) (z,w) );		
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{		
		rs += partialSum1[0];	
	}

	/* Rastrigin */
	sum1 = 0;
	stride += dim;
	dim = DIM - stride;	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];			
		sum1 += SUFFIX( basic_rastrigin ) (z);
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{
		rs += partialSum1[0];
		ys[bid] = rs;
	}
}

__global__ void SUFFIX( hybrid5 )( float_t *ys, float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];

	int i, tid, bid, stride, dim;
	float_t sum1, sum2, z, w, rs;	

	tid = threadIdx.x;
	bid = blockIdx.x;
		
	xs = &xs[bid*DIM];	

	for ( i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();

	rs = 0;

	/* Expanded Scaffer¡¯s F6 */
	sum1 = 0;
	stride = 0;
	dim = (int)ceil(0.1*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];	
		w = x[(i+1)%dim+stride];
		sum1 += SUFFIX( basic_schaffersf6 ) (z,w);
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{
		rs += partialSum1[0];		
	}
	
	/* HGBat */
	sum1 = sum2 = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += z*z;
		sum2 += z;
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];

		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}
	if ( tid == 0 )
	{		
		sum1 = partialSum1[0];
		sum2 = partialSum2[0];
		rs += sqrt( fabs( sum1*sum1-sum2*sum2 ) ) + (sum1/2+sum2)/dim + SUFFIX(0.5);		
	}

	/* Rosenbrock */
	sum1 = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim-1; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];
		w = x[i+1+stride]; 
		sum1 += SUFFIX(basic_rosenbrock) (z,w);		
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{		
		rs += partialSum1[0];	
	}
	
	/* Modified Schwefel */
	sum1 = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += SUFFIX( basic_modified_schwefel ) (z);
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{
		rs += partialSum1[0];
	}

	/* Elliptic */
	sum1 = 0;
	stride += dim;
	dim = DIM - stride;	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += z*z*SUFFIX(exp10)( (6*i)/( dim - SUFFIX(1.0) ) );
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{
		rs += partialSum1[0];
		ys[bid] = rs;
	}	
}

__global__ void SUFFIX( hybrid6 )( float_t *ys, float_t *xs )
{
	__shared__ float_t x[DIM];
	volatile __shared__ float_t partialSum1[BLOCK_SIZE_EVALUATION];
	volatile __shared__ float_t partialSum2[BLOCK_SIZE_EVALUATION];

	int i, j, tid, bid, stride, dim;
	float_t sum1, sum2, z, w, rs;	

	tid = threadIdx.x;
	bid = blockIdx.x;
		
	xs = &xs[bid*DIM];	

	for ( i = tid; i < DIM; i += BLOCK_SIZE_EVALUATION )
	{		
		x[i] = xs[i];		
	}	
	// __syncthreads();

	rs = 0;

	/* Katsuura */
	sum1 = 1;
	stride = 0;
	dim = (int)ceil(0.1*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];

		sum2 = (float_t)0;
		for ( j = 1; j <= 32; j++ )
		{
			float_t two = exp2((float_t)j);
			w = two * z;
			sum2 += fabs( w - round(w) )/ two;
		}
		sum2 = sum2 *(i+1) + 1;
		sum2 = pow( sum2, 10/SUFFIX(pow)((float_t)DIM, SUFFIX(1.2)) );
		sum1 *= sum2;
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] *= partialSum1[tid+16];
		partialSum1[tid] *= partialSum1[tid+8];
		partialSum1[tid] *= partialSum1[tid+4];
		partialSum1[tid] *= partialSum1[tid+2];
		partialSum1[tid] *= partialSum1[tid+1];
	}
	if ( tid == 0 )
	{
		rs += 10 * (partialSum1[0] - 1) /(DIM*DIM);		
	}
	
	/* HappyCat */
	sum1 = sum2 = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += z*z;
		sum2 += z;
	}
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];

		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}
	if ( tid == 0 )
	{		
		sum1 = partialSum1[0];
		sum2 = partialSum2[0];
		rs +=  pow( fabs(sum1-dim),SUFFIX(0.25) ) +( sum1/2+sum2 )/dim + SUFFIX(0.5);		
	}

	/* Expanded Griewank's plus Rosenbrock */
	sum1 = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];
		w = x[(i+1)%dim+stride]; 
		sum1 += SUFFIX(basic_griewank)( SUFFIX(basic_rosenbrock) (z,w) );		
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{		
		rs += partialSum1[0];	
	}
	
	/* Modified Schwefel */
	sum1 = 0;
	stride += dim;
	dim = (int)ceil(0.2*DIM);	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += SUFFIX( basic_modified_schwefel ) (z);
	}
	partialSum1[tid] = sum1;
	// __syncthreads();	
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
	}
	if ( tid == 0 )
	{
		rs += partialSum1[0];
	}

	/* Ackley */
	sum1 = sum2 = 0;
	stride += dim;
	dim = DIM - stride;	
	for ( i = tid; i < dim; i += BLOCK_SIZE_EVALUATION )
	{
		z = x[i+stride];					
		sum1 += z*z;
		sum2 += cospi(2*z);
	}	
	partialSum1[tid] = sum1;
	partialSum2[tid] = sum2;
	// __syncthreads();		
	if ( tid < WARP_SIZE/2 )
	{		
		partialSum1[tid] += partialSum1[tid+16];
		partialSum1[tid] += partialSum1[tid+8];
		partialSum1[tid] += partialSum1[tid+4];
		partialSum1[tid] += partialSum1[tid+2];
		partialSum1[tid] += partialSum1[tid+1];
		
		partialSum2[tid] += partialSum2[tid+16];
		partialSum2[tid] += partialSum2[tid+8];
		partialSum2[tid] += partialSum2[tid+4];
		partialSum2[tid] += partialSum2[tid+2];
		partialSum2[tid] += partialSum2[tid+1];
	}
	if ( tid == 0 )
	{
		rs += -SUFFIX(20.0)*exp(-SUFFIX(0.2)*sqrt(partialSum1[0]/dim)) - exp( partialSum2[0]/dim ) + 20 + (float_t)E;
		ys[bid] = rs;
	}	
}


/*
** Composition Functions
*/


/*
** Helper functions
*/
__global__ void SUFFIX( addScalar ) ( float_t *out, float_t *in, float_t scalar, int n )
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;	
	if ( tid < n )
	{				
		out[tid] = in[tid]+scalar;
	}
}

__global__ void SUFFIX( addVector ) ( float_t* matrix_out, float_t *matrix_in, float_t *vector, int len, int n )
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float_t x, y;
	if ( tid < len*n )
	{
		x = matrix_in[tid];
		y = vector[tid%len];		
		matrix_out[tid] = x + y;
	}
}

__global__ void SUFFIX( subtractVector ) ( float_t* matrix_out, float_t *matrix_in, float_t *vector, int len, int n )
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float_t x, y;
	if ( tid < len*n )
	{
		x = matrix_in[tid];
		y = vector[tid%len];		
		matrix_out[tid] = x - y;
	}
}

__global__ void SUFFIX( multiplyScalar ) ( float_t *out, float_t * in, float_t alpha, int n )
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;	
	if ( tid < n )
	{
		float_t x = in[tid];
		out[tid] = x*alpha;
	}
}

__global__ void SUFFIX( weightedRowwiseSum ) ( float_t* matrix, float_t* rows, int len, int n, float_t* rs )
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if ( tid < n )
	{
		float_t sum = 0;
		for ( int i = 0; i < len; i++ )
		{
			sum += matrix[len*tid+i]*rows[len*tid+i];
		}
		rs[tid] = sum;
	}
}

__global__ void SUFFIX( rowwiseSum ) ( float_t* rs, float_t* matrix, int width, int height  )
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if ( tid < height )
	{
		float_t sum = 0;
		for ( int i = 0; i < width; i++ )
		{
			sum += matrix[width*tid+i];
		}
		rs[tid] = sum;
	}
}

__global__ void SUFFIX( weightedColwiseSum ) ( float_t* matrix, float_t* cols, int width, int height, int pitch, float_t* rs )
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if ( tid < width )
	{
		float_t sum = 0;
		for ( int i = 0; i < height; i++ )
		{
			sum += matrix[i*pitch+tid]*cols[i*pitch+tid];
		}
		rs[tid] = sum;
	}
}

void SUFFIX( evaluate_basic_func ) ( float_t *y, float_t *x, int n, int id, float_t * opt = NULL, int rotated = 1 )
{
	dim3 block(WARP_SIZE);
	dim3 grid(DIVUP(n*DIM,block.x));
	
	/* Shift Optima */	
	if ( opt == NULL )
	{
		SUFFIX(subtractVector) <<<grid, block>>>( SUFFIX(bufferA), x, SUFFIX(o)+(id-1)*DIM, DIM, n );
	}else
	{		
		SUFFIX(subtractVector) <<<grid, block>>>( SUFFIX(bufferA), x, opt, DIM, n );
	}		

	/* Scaling */
	switch(id)
	{
	case ROSENBROCK: /* Rosenbrock */		
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(0.02048), n*DIM );
		break;
	case WEIERSTRASS: /* Weierstrass */		
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(0.005), n*DIM );
		break;		
	case GRIEWANK: /* Griewank */		
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(0.6), n*DIM );
		break;
	case RASTRIGIN_U: case RASTRIGIN: /* Rastrigin */		
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(0.0512), n*DIM );
		break;
	case SCHWEFEL_U: case SCHWEFEL: /* Modified Schwefel */		
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(10.0), n*DIM );
		break;
	case KATSUURA: /* Katsuura */
	case HAPPYCAT: /* HappyCat */
	case HGBAT: /* HGBat */
	case GRIE_ROSEN: /* Expanded Griewank's plus Rosenbrock */			
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(0.05), n*DIM );
		break;
	case POWERS: /* Different Powers */
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(0.01), n*DIM );
		break;
	case LUNACEK: /* Lunacek bi-Rastrigin */
		SUFFIX( multiplyScalar )<<<grid, block>>>( SUFFIX(bufferA), SUFFIX(bufferA), SUFFIX(0.1), n*DIM );
		break;
	}	

	/* Rotate */	
	float_t alpha=1, beta=0;
	switch(id)
	{
	case RASTRIGIN_U: /* Shifted Rastrigin */
	case SCHWEFEL_U: /* Shifted Modified Schwefel */	
		cudaMemcpy( SUFFIX(bufferB), SUFFIX(bufferA), sizeof(float_t)*DIM*n, cudaMemcpyDeviceToDevice );		
		break;
	default:	
		if(rotated)
		{
			gemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, n, DIM, &alpha, SUFFIX(m)+(id-1)*DIM*DIM, DIM, SUFFIX(bufferA), DIM, &beta, SUFFIX(bufferB), DIM );
		}else
		{
			cudaMemcpy( SUFFIX(bufferB), SUFFIX(bufferA), sizeof(float_t)*DIM*n, cudaMemcpyDeviceToDevice );
		}		
		break;
	}

	/* Extra Shift */
	switch(id)
	{
	case ROSENBROCK: /* Rosenbrok */
	case GRIE_ROSEN:		
		SUFFIX( addScalar )<<<grid, block>>>( SUFFIX(bufferB), SUFFIX(bufferB), 1, n*DIM );		
		break;
	case HAPPYCAT:
	case HGBAT:		
		SUFFIX( addScalar )<<<grid, block>>>( SUFFIX(bufferB), SUFFIX(bufferB), -1, n*DIM );		
		break;
	case LUNACEK:		
		SUFFIX( addScalar )<<<grid, block>>>( SUFFIX(bufferB), SUFFIX(bufferB), SUFFIX(2.5), n*DIM );
		break;
	}

	/* evaluate */
	grid.x = n;
	block.x = BLOCK_SIZE_EVALUATION;
	switch(id)
	{		
	case SPHERE:
		SUFFIX( sphere ) <<<grid,block >>> ( y, SUFFIX(bufferB) );		
		break;	
	case ELLIPTIC:
		SUFFIX( elliptic ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case ELLIPSOID:
		SUFFIX( ellipsoid )  <<<grid,block >>> ( y, SUFFIX(bufferB) );
		break;
	case POWERS:
		SUFFIX( powers )  <<<grid,block >>> ( y, SUFFIX(bufferB) );
		break;
	case SHARPV:
		SUFFIX( sharpv ) <<<grid,block >>> ( y, SUFFIX(bufferB) );
		break;
	case STEP:
		SUFFIX( step )  <<<grid,block >>> ( y, SUFFIX(bufferB) );
		break;
	case DISCUS: 
		SUFFIX( discus )  <<<grid,block >>> ( y, SUFFIX(bufferB) );	
		break;
	case CIGAR: 
		SUFFIX( cigar )  <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case ROSENBROCK: 
		SUFFIX( rosenbrock ) <<<grid,block >>> ( y, SUFFIX(bufferB) );
		break;
	case ACKLEY: 
		SUFFIX( ackley ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case WEIERSTRASS: 
		SUFFIX( weierstrass ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case GRIEWANK: 
		SUFFIX( griewank ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case SCHAFFERSF7:
		SUFFIX( schaffersf7 ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case RASTRIGIN_U: 
	case RASTRIGIN:
		SUFFIX( rastrigin ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;	
	case LUNACEK:
		SUFFIX( lunacek ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;	
	case SCHWEFEL_U:
	case SCHWEFEL:
		SUFFIX( modified_schwefel ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;	
	case KATSUURA: 
		SUFFIX( katsuura ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HAPPYCAT: 
		SUFFIX( happycat ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HGBAT: 
		SUFFIX( hgbat ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case GRIE_ROSEN: 
		SUFFIX( expanded_griewank_rosenbrock ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case SCHAFFERSF6: 
		SUFFIX( expanded_schaffersf6 ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HYBRID1: 
		SUFFIX( hybrid1 ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HYBRID2: 
		SUFFIX( hybrid2 ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HYBRID3: 
		SUFFIX( hybrid3 ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HYBRID4: 
		SUFFIX( hybrid4 )  <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HYBRID5: 
		SUFFIX( hybrid5 ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	case HYBRID6: 
		SUFFIX( hybrid6 ) <<<grid,block >>>( y, SUFFIX(bufferB) );
		break;
	}		
}


__global__ void SUFFIX( rowwiseSquareError )( float_t * matrix_out, float_t* matrix_in, float_t* row, int width, int height )
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if ( tid < width*height )
	{
	   float_t val = matrix_in[tid];
	   val -= row[tid%width];	   
	   matrix_out[tid] = val*val;
	}
}

void SUFFIX( evaluate_composition_func ) ( float_t *y, float_t *x, int n, int cid )
{
	int i, k, flag, idx;		
	float_t *rs, sum;	
	float_t *o = SUFFIX(optima)+(cid-1)*MAX_COMPOSITION_NUM*DIM;

	/* Calculate weights */
	for ( i = 0; i < COMPOSITION_NUMS[cid-1]; i++ )
	{
	    dim3 block(WARP_SIZE);
		dim3 grid(DIVUP(n*DIM, block.x));
		SUFFIX(rowwiseSquareError)<<<grid,block>>>( SUFFIX(bufferA), x, o+i*DIM, DIM, n  );
				
	    block.x = WARP_SIZE;
		grid.x = DIVUP(n, block.x);
        SUFFIX( rowwiseSum )<<<grid,block>>>( SUFFIX(bufferC)+i*MAX_CONCURRENCY, SUFFIX(bufferA), DIM, n );		
			
	}
	cudaMemcpy( SUFFIX(w), SUFFIX(bufferC), sizeof(float_t)*COMPOSITION_NUMS[cid-1]*MAX_CONCURRENCY, cudaMemcpyDeviceToHost );
		
	 for ( k = 0; k < n; k++ )
	{
	   flag = 0; idx = -1;
	   sum = 0;
	   for ( i = 0; i < COMPOSITION_NUMS[cid-1]; i++ )
	   {
	         float_t val =  SUFFIX(w)[i*MAX_CONCURRENCY+k];			 
			 if ( val == 0 )
			 {
			     flag = 1; idx = i;
				 break;
			 } else
			 {
			      val = SUFFIX(exp)(-val/(2*DIM*SUFFIX(sigma)[cid-1][i]*SUFFIX(sigma)[cid-1][i]))/ SUFFIX(sqrt)(val);
				  sum += val;
				  SUFFIX(w)[i*MAX_CONCURRENCY+k] = val;
			 }
	   }
	   if ( flag == 1 )
	   {
			for ( i = 0; i < COMPOSITION_NUMS[cid-1]; i++ )
		   {
				 if ( idx == i )
				 {
					   SUFFIX(w)[i*MAX_CONCURRENCY+k] = 1;
				 } else
				 {
					   SUFFIX(w)[i*MAX_CONCURRENCY+k] = 0;
				 }
		   }
	   } else
	   {
			for ( i = 0; i < COMPOSITION_NUMS[cid-1]; i++ )
		   {
				 
				  SUFFIX(w)[i*MAX_CONCURRENCY+k] /= sum;				
		   }	
	   }
	}	
	cudaMemcpy( SUFFIX(omega), SUFFIX(w), sizeof(float_t)*MAX_CONCURRENCY*COMPOSITION_NUMS[cid-1], cudaMemcpyHostToDevice );
	CHECK_CUDA_ERROR() ;	 

	dim3 block(WARP_SIZE);
	dim3 grid(DIVUP(n,WARP_SIZE));
	
	int fid;
	if( cid == 1 ) /* Composition 1 */
	{
		/* No.1: Rotated Rosenbrock */
		idx = 0;
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;	
		fid = ROSENBROCK;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );
		

		/* No.2: Elliptic */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = ELLIPTIC;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM, 0 );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Rotated Cigar */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = CIGAR;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.4: Rotated Discus */
		idx = 3;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = DISCUS;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.5: Rotated Elliptic */
		idx = 4;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = ELLIPTIC;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

	} else if( cid == 2 ) /* Composition 2 */
	{
		/* No.1: Schwefel */
		idx = 0;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = SCHWEFEL;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM, 0 );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.2: Rotated Rastrigin */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = RASTRIGIN;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Rotated HGBat */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = HGBAT;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );		

	} else if( cid == 3 ) /* Composition 3 */
	{
		/* No.1: Rotated Schwefel */
		idx = 0;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = SCHWEFEL;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.2: Rotated Rastrigin */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = RASTRIGIN;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Rotated Elliptic */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = ELLIPTIC;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );
		
	} else if( cid == 4 ) /* Composition 4 */
	{
		/* No.1: Rotated Schwefel */
		idx = 0;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = SCHWEFEL;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.2: Rotated HappyCat */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = HAPPYCAT;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Rotated Elliptic */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = ELLIPTIC;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.4: Rotated Weierstrass */
		idx = 3;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = WEIERSTRASS;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.5: Rotated Griewank */
		idx = 4;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = GRIEWANK;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );		

	} else if( cid == 5 ) /* Composition 5 */
	{
		/* No.1: Rotated HGBat */
		idx = 0;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = HGBAT;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.2: Rotated Rastrigin */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = RASTRIGIN;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Rotated Schwefel */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = SCHWEFEL;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.4: Rotated Weierstrass */
		idx = 3;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = WEIERSTRASS;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.5: Rotated Elliptic */
		idx = 4;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = ELLIPTIC;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );
		

	} else if( cid == 6 ) /* Composition 6 */
	{
		/* No.1: Rotated Expanded Griewank plus Rosenbrock */
		idx = 0;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = GRIE_ROSEN;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );


		/* No.2: Rotated HappyCat */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = HAPPYCAT;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Rotated Schwefel */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = SCHWEFEL;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );


		/* No.4: Rotated Expanded Schaffer's F6 */
		idx = 3;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = SCHAFFERSF6;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );
		

		/* No.5: Rotated Elliptic */
		idx = 4;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = ELLIPTIC;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );
		
	} else if( cid == 7 ) /* Composition 7 */
	{
		/* No.1: Hybrid 1 */
		idx = 0;
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;		
		fid = HYBRID1;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.2: Hybrid 2 */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;			
		fid = HYBRID2;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Hybrid 3 */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;				
		fid = HYBRID3;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );
		
	} else if( cid == 8) /* Composition 8 */
	{
		/* No.1: Hybrid 4 */
		idx = 0;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;				
		fid = HYBRID4;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.2: Hybrid 5 */
		idx = 1;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;				
		fid = HYBRID5;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );

		/* No.3: Hybrid 6 */
		idx = 2;		
		rs = SUFFIX(bufferC) + idx*MAX_CONCURRENCY;					
		fid = HYBRID6;
		
		SUFFIX( evaluate_basic_func ) ( rs, x, n, fid, o+idx*DIM );
		SUFFIX(multiplyScalar) <<<grid,block>>>( rs, rs, SUFFIX(lamda)[cid-1][idx], n );		
		SUFFIX(addScalar) <<<grid,block>>>( rs, rs, SUFFIX(bias)[cid-1][idx], n );	
	}
	
	SUFFIX( weightedColwiseSum )<<<grid, block>>>( SUFFIX(bufferC), SUFFIX(omega), n, COMPOSITION_NUMS[cid-1], MAX_CONCURRENCY, y );	
	cudaDeviceSynchronize();
}

#ifdef __cplusplus 
extern "C" 
#endif
 void SUFFIX( evaluate_func ) ( float_t *y, float_t *x, int n, int id )
{
	if ( !initialized )
	{
		printf( "initialize() not called yet.\n" );
		exit(1);
	}
	if ( n > MAX_CONCURRENCY )
	{
		printf( "At most MAX_CONCURRENCY = %d vectors can be evaluated at once.\n", MAX_CONCURRENCY );
		exit(1);
	}		
	
	if ( id > BASIC_FUNC_NUM ) 
	{
		SUFFIX( evaluate_composition_func )( y, x, n, id - BASIC_FUNC_NUM ) ;
	} 
	else
	{
		 SUFFIX( evaluate_basic_func ) ( y, x, n, id );
	}
	
	SUFFIX(addScalar) <<<DIVUP(n,WARP_SIZE), WARP_SIZE>>>( y, y, BIAS, n );
	cudaDeviceSynchronize();	
}


#ifdef __cplusplus 
extern "C" 
#endif
void SUFFIX( h_evaluate_func ) ( float_t *y, float_t *x, int n, int id )
{	
	cudaMemcpy( SUFFIX(devx), x, sizeof(float_t)*n*DIM, cudaMemcpyHostToDevice );	
	SUFFIX( evaluate_func ) ( SUFFIX(devy), SUFFIX(devx), n, id );	
	cudaMemcpy( y, SUFFIX(devy), sizeof(float_t)*n, cudaMemcpyDeviceToHost );	
}