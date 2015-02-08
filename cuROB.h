#ifndef __CUROB_H__
#define __CUROB_H__

/* Dimension of benchmark functions */
#define DIM (96) 

/* At most MAX_CONCURRENCY vectors are supported to evaluate one time. */
#define MAX_CONCURRENCY (100)


#ifdef __cplusplus
extern "C" {
#endif

void curob_initialize(); /*  Should be called before evaluate_func(f)  */
void curob_dispose();    /*  Should be called after evaluate_func(f)   */

/*
** Interface to the benchmark functions.
** dev_y: 1 by n vector, **DEVICE** pointer
** dev_x: n by DIM matrix, **DEVICE** pointer
** n: number of vectors to be evaluated
** id: function to be evaluated
*/
void evaluate_func( double *dev_y, double *dev_x, int n, int id ); 
void evaluate_funcf( float *dev_y, float *dev_x, int n, int id ); 

/*
** Interface to the benchmark functions.
** host_y: 1 by n vector, **HOST** pointer
** host_x: n by DIM matrix, **HOST** pointer
** n: number of vectors to be evaluated
** id: function to be evaluated
*/
void h_evaluate_func( double *host_y, double *host_x, int n, int id );
void h_evaluate_funcf( float *host_y, float *host_x, int n, int id ); 


#define UNIMODAL_BASE 0
#define SPHERE (UNIMODAL_BASE+1)
#define ELLIPSOID (UNIMODAL_BASE+2)
#define ELLIPTIC (UNIMODAL_BASE+3)
#define DISCUS (UNIMODAL_BASE+4)
#define CIGAR (UNIMODAL_BASE+5)
#define POWERS (UNIMODAL_BASE+6)
#define SHARPV (UNIMODAL_BASE+7)

#define MULTIMODAL_BASE SHARPV

#define STEP (MULTIMODAL_BASE+1)
#define WEIERSTRASS (MULTIMODAL_BASE+2)
#define GRIEWANK (MULTIMODAL_BASE+3)
#define RASTRIGIN_U (MULTIMODAL_BASE+4)
#define RASTRIGIN (MULTIMODAL_BASE+5)
#define SCHAFFERSF7 (MULTIMODAL_BASE+6)
#define GRIE_ROSEN (MULTIMODAL_BASE+7)
#define ROSENBROCK (MULTIMODAL_BASE+8)
#define SCHWEFEL_U (MULTIMODAL_BASE+9)
#define SCHWEFEL (MULTIMODAL_BASE+10)
#define KATSUURA (MULTIMODAL_BASE+11)
#define LUNACEK (MULTIMODAL_BASE+12)
#define ACKLEY (MULTIMODAL_BASE+13)
#define HAPPYCAT (MULTIMODAL_BASE+14)
#define HGBAT (MULTIMODAL_BASE+15)
#define SCHAFFERSF6 (MULTIMODAL_BASE+16)

#define HYBRID_BASE SCHAFFERSF6
#define HYBRID1 (HYBRID_BASE+1)
#define HYBRID2 (HYBRID_BASE+2)
#define HYBRID3 (HYBRID_BASE+3)
#define HYBRID4 (HYBRID_BASE+4)
#define HYBRID5 (HYBRID_BASE+5)
#define HYBRID6 (HYBRID_BASE+6)


#define COMPOSITION_BASE HYBRID6
#define COMPOSITION1 (COMPOSITION_BASE+1)
#define COMPOSITION2 (COMPOSITION_BASE+2)
#define COMPOSITION3 (COMPOSITION_BASE+3)
#define COMPOSITION4 (COMPOSITION_BASE+4)
#define COMPOSITION5 (COMPOSITION_BASE+5)
#define COMPOSITION6 (COMPOSITION_BASE+6)
#define COMPOSITION7 (COMPOSITION_BASE+7)
#define COMPOSITION8 (COMPOSITION_BASE+8)


#define MAX_FUNC_NUM (COMPOSITION8)
#define BASIC_FUNC_NUM (COMPOSITION_BASE)
#define COMPOSITION_FUNC_NUM  (MAX_FUNC_NUM-BASIC_FUNC_NUM)


#ifndef EXCLUDE
extern int fids[MAX_FUNC_NUM];
#endif

#ifdef __cplusplus
}
#endif

/*  _________________________________________
** | No |      Function      |      ID      |
** ------------------------------------------
** | 00 |      Sphere        |    SPHERE    |
** | 01 |      Ellipsoid     |   ELLIPSOID  |
** | 02 |      Elliptic      |   ELLIPTIC   |
** | 03 |      Discus        |    DISCUS    |
** | 04 |      Cigar         |     CIGAR    |
** | 05 |  Different Powers  |    POWERS    |
** | 06 |    Sharp Valley    |    SHARPV    |
** ------------------------------------------
** | 07 |      Step          |     STEP     |
** | 08 |     Weierstras     |  WEIERSTRASS |
** | 09 |      Griewank      |    GRIEWANK  |
** | 10 |      Rastrigin     |  RASTRIGIN_U |
** |    |     (Unrotated)    |              |
** | 11 |      Rastrigin     |   RASTRIGIN  |
** | 12 |    Schaffers F7    |  SCHAFFERSF7 |
** | 13 |	   Expanded      |  GRIE_ROSEN  |
** |    | Griewank Rosenbrock|              |
** | 14 |      Rosenbrock    |  ROSENBROCK  |
** | 15 |  Modified Schwefel |  SCHWEFEL_U  |
** |    |     (Unrotated)    |              |
** | 16 |  Modified Schwefel |   SCHWEFEL   |
** | 17 |      Katsuura      |   KATSUURA   |
** | 18 |       Lunacek      |   LUNACEK    |
** | 19 |       Ackley       |    ACKLEY    |
** | 20 |	    HappyCat 	 |   HAPPYCAT   |
** | 21 |	     HGBat		 |     HGBAT    |
** | 22 |	   Expanded      |  SCAFFERSF6  |
** |    |    Schaffer's F6   |              |
** ------------------------------------------
** | 23 |	   Hybrid 1		 |    Hybrid1   |
** | 24 |	   Hybrid 2		 |    Hybrid2   |
** | 25 |	   Hybrid 3		 |    Hybrid3   |
** | 26 |	   Hybrid 4		 |    Hybrid4   |
** | 27 |	   Hybrid 5		 |    Hybrid5   |
** | 28 |	   Hybrid 6		 |    Hybrid6   |
** ------------------------------------------
** | 29 |	 Composition 1	 | COMPOSITION1 |
** | 30 |	 Composition 2	 | COMPOSITION2 |
** | 31 |	 Composition 3	 | COMPOSITION3 |
** | 32 |	 Composition 4	 | COMPOSITION4 |
** | 33 |	 Composition 5	 | COMPOSITION5 |
** | 34 |	 Composition 6	 | COMPOSITION6 |
** | 35 |	 Composition 7	 | COMPOSITION7 |
** | 36 |	 Composition 8	 | COMPOSITION8 |
   ------------------------------------------
*/

#endif // __CUROB_H__