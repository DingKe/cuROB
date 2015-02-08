#include "math.h"
#include "stdlib.h"

#include "KISS.h"


static ulong kiss_x;
static  ulong kiss_y;
static  ulong kiss_z;
static ulong kiss_w;
static ulong kiss_carry;
static ulong kiss_k;
static ulong kiss_m;


// random number (uniform distribution) in [a, b]
double alea (double a, double b) 
{
	double r;
	r=a+(double)basic_rand()*(b-a)/BASIC_RAND_MAX;
	return r; 
}

// integer random number in [a, b]
int alea_integer (int a, int b) 
{				
	int ir;
	double r;

	r = alea (0, 1);
	ir = (int) (a + r * (b + 1 - a));

	if (ir > b)	ir = b;

	return ir;  
}

// Box-Muller Transformation 
double alea_normal (double mean, double std_dev) 
{ 	
	double x1, x2, w, y1;  	

	do  
	{
		x1 = 2.0 * alea (0, 1) - 1.0;
		x2 = 2.0 * alea (0, 1) - 1.0;
		w = x1 * x1 + x2 * x2;     
	}
	while (w >= 1.0);

	w = sqrt (-2.0 * log (w) / w);
	y1 = x1 * w;	
	if( alea(0,1) < 0.5 ) y1=-y1; 
	y1 = y1 * std_dev + mean;
	return y1;  
}

void seed_rand(ulong seed) 
{
	kiss_x = seed | 1;
	kiss_y = seed | 2;
	kiss_z = seed | 4;
	kiss_w = seed | 8;
	kiss_carry = 0;

}

ulong basic_rand() 
{
	kiss_x = kiss_x * 69069 + 1;
	kiss_y ^= kiss_y << 13;
	kiss_y ^= kiss_y >> 17;
	kiss_y ^= kiss_y << 5;
	kiss_k = (kiss_z >> 2) + (kiss_w >> 3) + (kiss_carry >> 2);
	kiss_m = kiss_w + kiss_w + kiss_z + kiss_carry;
	kiss_z = kiss_w;
	kiss_w = kiss_m;
	kiss_carry = kiss_k >> 30;

	return kiss_x + kiss_y + kiss_w;

}
