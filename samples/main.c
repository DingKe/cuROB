#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "string.h"

#include "cuROB.h"

double SPSO( double * x,int id );

int main()
{
	double rs;	
	double x[DIM];	
	int i,j;

	int id = 1; 

	for ( i = 0; i < 37; i++ )
	{
		id = fids[i];
		rs = SPSO( x, id );
		printf( "Function ID %d\n rs: %.10f\n", i, rs );
		for( j = 0; j < DIM; j++ )
		{
			printf( "D%d: %.3f\t", j, x[j] );
		}
		printf( "\n" ); 
	}
	system( "pause" );	
}
