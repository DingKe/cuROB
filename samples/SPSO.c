#include "stdlib.h"
#include "string.h"
#include "time.h"

#include "KISS.h"
#include "cuROB.h"


#define E 2.7182818284590
#define PI 3.141592653

#define UBOUND (100)
#define LBOUND (-100)

#define W (0.721347520444482f)
#define C1 (1.193147180559945f)
#define C2 (1.193147180559945f)

#define POP (50)
#define MAX_ITERATION (10000)

#define INDEX(X,Y) ((X)*DIM+(Y))


// Standard PSO, refer to Daniel Bratton and James Kennedy, Defining a Standard for Particle Swarm Optimization, 2007
double SPSO( double * x, int id )
{
	double *positions, *fitnesses, *velocities,
		*p_positions, *p_fitnesses,
		*l_positions, *l_fitnesses;	

	int i,j,iter;
	int chosen;
	double val;

	curob_initialize();

	//allocate memory
	positions = (double *)calloc(POP,sizeof(double)*DIM);
	velocities = (double *)calloc(POP,sizeof(double)*DIM);
	p_positions = (double *)calloc(POP,sizeof(double)*DIM);
	l_positions = (double *)calloc(POP,sizeof(double)*DIM);
	fitnesses = (double *)calloc(POP,sizeof(double));
	p_fitnesses = (double *)calloc(POP,sizeof(double));
	l_fitnesses = (double *)calloc(POP,sizeof(double));

	seed_rand( (ulong)clock() ); 

	for(i = 0; i < POP; i++)
	{		
		for (j = 0; j < DIM; j++)
		{
			positions[INDEX(i,j)] = alea(LBOUND,UBOUND);
			velocities[INDEX(i,j)] = (alea(LBOUND,UBOUND) - positions[INDEX(i,j)])/2;           
		}  		
	}
	memcpy(p_positions,positions,DIM*POP);
	memcpy(l_positions,positions,DIM*POP);

	h_evaluate_func(fitnesses,positions, POP, id); 

	memcpy(p_fitnesses,fitnesses,sizeof(double)*POP);
	memcpy(l_fitnesses,fitnesses,sizeof(double)*POP);


	for( iter = 0; iter < MAX_ITERATION; iter++ )
	{
		// update local best		
		for( i = 0; i < POP; i ++)
		{
			chosen = i; val = l_fitnesses[i];
			if( val > fitnesses[(i-1+POP)%POP] )
			{
				chosen = (i-1+POP)%POP;	
				val = fitnesses[(i-1+POP)%POP];
			}
			if( val >  fitnesses[(i+1)%POP] )
			{
				chosen = (i+1)%POP;
				val = fitnesses[(i+1)%POP];				
			}
			if( chosen != i)
			{
				l_fitnesses[i] = val;
				memcpy(&l_positions[INDEX(i,0)], &positions[INDEX(chosen,0)], sizeof(double)*DIM);
			}
		}

		// update velocity and position
		for( i = 0; i < POP; i += 1 )
		{
			for( j = 0; j < DIM;j += 1)
			{
				velocities[INDEX(i,j)] = W*velocities[INDEX(i,j)] + alea(0,C1)*(p_positions[INDEX(i,j)] - positions[INDEX(i,j)]) + alea(0,C2)*(l_positions[INDEX(i,j)] - positions[INDEX(i,j)]);
				positions[INDEX(i,j)] += velocities[INDEX(i,j)];

				if(positions[INDEX(i,j)] < LBOUND)
				{
					positions[INDEX(i,j)] = LBOUND+alea(0,0.2*(UBOUND-LBOUND));
					velocities[INDEX(i,j)] = 0;					
				}else if(positions[INDEX(i,j)] > UBOUND)
				{
					positions[INDEX(i,j)] = UBOUND-alea(0,0.2*(UBOUND-LBOUND));
					velocities[INDEX(i,j)] = 0;					
				}
			}
		}

		h_evaluate_func( fitnesses, positions, POP, id ); 		

		// update private best
		for( i = 0; i < POP; i += 1)
		{
			if( p_fitnesses[i] > fitnesses[i] )
			{			
				p_fitnesses[i] = fitnesses[i];
				memcpy( &p_positions[INDEX(i,0)], &positions[INDEX(i,0)], DIM*sizeof(double) );
			}					
		}
	}

	chosen = 0; val = p_fitnesses[0];
	for( j = 0; j < POP;j++ )
	{
		if( val > p_fitnesses[j] )
		{
			val = p_fitnesses[j];
			chosen = j;
		}
	}

	if (x != NULL)
	{
		memcpy( x, &p_positions[INDEX(chosen,0)], sizeof(double)*DIM );
	}

	//free memory

	free(positions);
	free(fitnesses);
	free(velocities);

	free(p_positions);
	free(p_fitnesses);

	free(l_positions);
	free(l_fitnesses);

	curob_dispose();

	return val;

} //end PSO