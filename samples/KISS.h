#ifndef __KISS_H__
#define __KISS_H__

#define ulong unsigned long

#ifdef __cplusplus
extern "C" {
#endif

#define BASIC_RAND_MAX ((unsigned long) 4294967295)

ulong	basic_rand();
void	seed_rand(ulong seed);

double alea (double a, double b);
int alea_integer (int a, int b);
double alea_normal (double mean, double stdev);

#ifdef __cplusplus
}
#endif

#endif // __KISS_H__