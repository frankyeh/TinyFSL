#if !defined(__CPROB_H)
#define __CPROB_H


/*
 *   This file was automatically generated by version 1.7 of cextract.
 *   Manual editing not recommended.
 *
 *   Created: Wed Mar 29 17:50:31 1995
 */

namespace CPROB {

double bdtrc ( int k, int n, double p );
double bdtr ( int k, int n, double p );
double bdtri ( int k, int n, double y );
double btdtr ( double a, double b, double x );
double chdtrc ( double df, double x );
double chdtr ( double df, double x );
double chdtri ( double df, double y );
int drand ( double *a );
double fdtrc ( int ia, int ib, double x );
double fdtr ( int ia, int ib, double x );
double fdtri ( int ia, int ib, double y );
double gamma ( double x );
double true_gamma ( double x );
double lgam ( double x );
double gdtr ( double a, double b, double x );
double gdtrc ( double a, double b, double x );
double igamc ( double a, double x );
double igam ( double a, double x );
double igami ( double a, double y0 );
double incbet ( double aa, double bb, double xx );
double incbi ( double aa, double bb, double yy0 );
int mtherr (const char *name, int code );
int mtherr_quiet (const char *name, int code );
int mtherr_default (const char *name, int code );
int set_mtherr(int (*fn)(const char *, int));
double nbdtrc ( int k, int n, double p );
double nbdtr ( int k, int n, double p );
double nbdtri ( int k, int n, double p );
double ndtr ( double a );
double erfc ( double a );
double erf ( double x );
double ndtri ( double y0 );
double pdtrc ( int k, double m );
double pdtr ( int k, double m );
double pdtri ( int k, double y );
void sdrand ( int seed1, int seed2, int seed3);
double stdtr ( int k, double t );
double stdtri ( int k, double p );
double log1p ( double x );
double expm1 ( double x );
double expx2 (double x, int sign);
double cos1m ( double x );
double polevl ( double x, double *P, int n );
double p1evl ( double x, double *P, int n );

}
#endif
