
// Intel SSE SIMD optimized low-level linear algebra operations for matrix and vector classes in col-major order

////////////////////////////////////////////////////////////
//
//
// (C) Alvaro Salmador 2011. All rights reserved.
// 
// $Id$
//
////////////////////////////////////////


#ifndef _SIMD_SSE_H__
#define _SIMD_SSE_H__

#include "SIMD_Base.h"
//#include "sse_math.h"

#ifndef _MSC_VER
// CPUID instruction takes no parameters as CPUID implicitly uses the EAX register.
// The EAX register should be loaded with a value specifying what information to return
inline void __cpuid(int* e, int code) {
	 __asm__ volatile("xchgl %%ebx, %k1;cpuid; xchgl %%ebx,%k1" \
					  : "=a" (e[0]), "=&r" (e[1]), "=c" (e[2]), "=d" (e[3]) : "a" (code));//, "c" (id));
}
#endif

inline void printx(const __m128& xm, const char* msg=0);

#define PREFETCH_DIST8	32 
#define PREFETCH_DIST	32 
#define CLS_F			16 //Cache line size in floats (64 bytes/4)



class SIMD : public SIMD_Base
{
#ifdef _MSC_VER
	#pragma optimize( "wgt", on )
	#pragma inline_depth( 255 )
#endif

public:
	static inline bool	Init();

	// Operations with floats
	static inline void	Copyf(const int N, const float* x, float* r);
	static inline void	Mulf (const int N, const float* x, const float k, float* r, bool bDiv=false);
	static inline void	Divf (const int N, const float* x, const float k, float* r);
	static inline void	Add  (const int N, const float* x, const float* y, float* r);
	static inline void	Sub  (const int N, const float* x, const float* y, float* r);
	static inline void	Mul  (const int N, const float* x, const float* y, float* r);
	static inline void	PadZf(const int N, float* r);
	static inline void	trans(const int _m, const int _n, const float* x, float* r);

	static inline void	Logvf	(const int N, const float* x, const float v, float* r); // log(x+1) (nwo +v)
	static inline void	AddMulf (const int N, const float* x, const float add, const float mul, float* r);
	static inline float	Sumf	(const int N, float* x, bool bAvg=false);
	static inline float	Sum2f	(const int N, float* x, bool bAvg=false);
	static inline float SubSum2f(const int N, float* x, const float k, bool bAvg=false);

	// Matrix product
	static inline void	MulMat	(const int _m, const int _n, const int n2, 
								 const float* m1, const float* m2, float* r);
	// Dot product (vector)
	static inline float	MulVec(int _n, const float* v1, const float* v2, const bool bSqrt=false);

	// Operations with complex numbers <float>
	static inline void	MulCf(const int N, const float* x, const float k, float* r);
	static inline void	DivCf(const int N, const float* x, const float k, float* r);
	static inline void	MulC (const int N, const float* x, const float* y, float* r);
	static inline void	AddC (const int N, const float* x, const float* y, float* r);
	static inline void	SubC (const int N, const float* x, const float* y, float* r);
	static inline void	DivC (const int N, const float* x, const float* y, float* r);
	static inline void	MagC (const int N, const float* x, float* r);

	// Important warning: vector and matrix classes that use this methods MUST be padded with zeros
	//	because some methods will overrun to avoid dealing with odd sizes.
	// Their allocators should align and pad appropriately for optimization
	
protected:
	// Constants for broadcast ops with shuffles
	static const int _BROAD3 = _MM_SHUFFLE(3,3,3,3);
	static const int _BROAD2 = _MM_SHUFFLE(2,2,2,2);
	static const int _BROAD1 = _MM_SHUFFLE(1,1,1,1);
	static const int _BROAD0 = _MM_SHUFFLE(0,0,0,0);
};


inline bool SIMD::Init()
{
NANOBEGIN
	SIMD_Base::Init();

	int e[4];
	__cpuid(e, 1);
	if (bits(e[3], 26)) SSE2 = true;
	else throw face::MatrixException("SIMD::Init: SSE2 not supported, cannot continue program execution");
	if (bits(e[2], 0)) SSE3 = true;
	else throw face::MatrixException("SIMD::Init: SSE3 not supported, cannot continue program execution");
	if (bits(e[2], 3)) MWAIT = true;
	if (bits(e[2], 9)) SSSE3 = true;
	if (bits(e[2], 19)) SSE41 = true;
	if (bits(e[2], 20)) SSE42 = true;
	
	__cpuid(e, 0x80000006); 
	CL2 = bits(e[2], 16, 31); // L2 cache size (per core), in kbytes
	CLS = bits(e[2], 0, 7); // Cache line size -- FIXME is this only valid for Intel, not AMD?

	// TODO: loop through tlb/cache for intel/amd to get L1 cache size
	CL1 = 32768; // default, assumed, L1 cache size
	// http://www.flounder.com/cpuid_explorer2.htm#CPUID%282%29 
	// https://en.wikipedia.org/wiki/CPUID#EAX.3D80000005h:_L1_Cache_and_TLB_Identifiers

NANOEND

	/* debug output..
	__cpuid(e, 0);
	printf("identification: \"%.4s%.4s%.4s\"\n", (char *)&e[1], (char *)&e[3], (char *)&e[2]);
	printf("cpu information:\n");
	__cpuid(e, 1);
	printf(" family %d model %d stepping %d efamily %d emodel %d\n",
					bits(e[0], 8, 11), bits(e[0], 4, 7), bits(e[0], 0, 3), bits(e[0], 20, 27), bits(e[0], 16, 19));
	printf(" brand %d cflush sz %d*8 nproc %d apicid %d\n",
					bits(e[1], 0, 7), bits(e[1], 8, 15), bits(e[1], 16, 23), bits(e[1], 24, 31));
	printf("L2 cache size: %d KB; cache line size: %d bytes\n", CL2, CLS);/**/

	if (!SSE2 || !SSE3) return false;
	else return true;
}


//#ifdef _MSC_VER
// #ifdef _WIN64 -- used to have inline assembly for GNU and VS - rep movsd... or movdqa/prefetch... 
// but a good memcpy usually takes care of all, dynamically too 
inline void	SIMD::Copyf(const int N, const float* x, float* r)
{
	// inline asm is not available on VS for x64
	if (N == 0) return;
	memcpy(r, x, N*sizeof(float));
}


inline void SIMD::PadZf(const int N, float* r)
{
	__m128 x0 = _mm_setzero_ps();
	int ofs = (((unsigned int)r)&15)>>2; //in case it's not aligned
	// notice -ofs and +=4: arrays passed should be padded!
	for(int i=-ofs; i<N; i+=4)
		_mm_stream_ps(r+i, x0);
}


// Matrix transposition
inline void SIMD::trans(const int _m, const int _n, const float* __restrict x, float* __restrict r)
{
	if ((_m|_n)&15) {
		for(int i=0; i<_m; ++i) 
		{
			const int in=i*_n;
			for(int j=0; j<_n; ++j)
				r[in+j] = x[j*_m + i];
		}
	} else {
		// 16x16 floats blocks (16f=1 cache line by the way)
		for(int i=0; i<_m; i+=16) 
		{
			float* __restrict xi = const_cast<float* __restrict>(x+i);
			for(int j=0; j<_n; j+=16)
			{
				float* __restrict ri = &r[i*_n+j];

				for (int ii=0; ii<16; ++ii) 
				{
					const int ii_n = ii*_n;
					//for (int jj=0; jj<16; ++jj) 
				 	//	ri[ii_n+jj] = xi[jj*_m+ii];
					//for (int jj=0; jj<16; jj+=4) 
					{
						float* __restrict xii = xi+ii;//+jj*_m;
						__m128 x0 = _mm_setr_ps(*xii, *(xii+_m), *(xii+(_m<<1)), *(xii+(_m<<1)+_m));
						xii += _m<<2;
						__m128 x1 = _mm_setr_ps(*xii, *(xii+_m), *(xii+(_m<<1)), *(xii+(_m<<1)+_m));
						xii += _m<<2;
						__m128 x2 = _mm_setr_ps(*xii, *(xii+_m), *(xii+(_m<<1)), *(xii+(_m<<1)+_m));
						xii += _m<<2;
						__m128 x3 = _mm_setr_ps(*xii, *(xii+_m), *(xii+(_m<<1)), *(xii+(_m<<1)+_m));

						_mm_store_ps(ri+ii_n    , x0);
						_mm_store_ps(ri+ii_n + 4, x1);
						_mm_store_ps(ri+ii_n + 8, x2);
						_mm_store_ps(ri+ii_n + 12, x3);
					}
				}
				
				xi += _m<<4;
			}
		}
	}
}

// Matrix multiplication
inline void SIMD::MulMat(const int _m, const int _n, const int n2, 
				  const float* __restrict m1, const float* __restrict m2, float* __restrict r)
{
	const int _mn2=_m*n2, _mn=_m*_m, _nn2=_n*n2;
	const bool n2aligned = (n2&3)==0;
	const bool naligned = (_n&3)==0;
	const bool maligned = (_m&3)==0;

	const int smax = (SIMD::CL1>>6)*640/512; // ~512 for 32kb L1 cache
	if (maligned && naligned && n2aligned && _m<=smax && _n<=smax && n2<=smax)
	{
		// This method is best for matrices that aren't too big (see smax ^) and are properly aligned.
		// We'll use other methods for bigger matrices further below.
		// m, n, n2 multiples of 4:
		for(int j=0; j<n2; j+=4)	
		{
			const float* __restrict m2_jn=m2+j*_n;
			for(int i=0; i<_m; i+=4)	// i,j define a 4x4 window in the result matrix
			{
				const float*	__restrict m1_i=m1+i;
				float*	__restrict r_i=r+i;
				__m128 a0, a1, a2, a3;
				a0 = _mm_setzero_ps();
				a1 = _mm_setzero_ps();
				a2 = _mm_setzero_ps();
				a3 = _mm_setzero_ps();
				__m128 x0, x1, x2, x3;
			
				// Here we traverse the input matrices, accumulating results for the 4x4 window we are dealing with now
				for(int k=0; k<_n; k+=4)  // _n es == m._m
				{
					x0 = _mm_load_ps(m1_i +  k*_m); // m1(i,k), col-major order
					x1 = _mm_load_ps(m1_i + (k+1)*_m); 
					x2 = _mm_load_ps(m1_i + (k+2)*_m); 
					x3 = _mm_load_ps(m1_i + (k+3)*_m); 

					__m128 yy;
					yy = _mm_load_ps(m2_jn + 0*_n	+ k); // m2(k,j)  // y0
					a0 = _mm_add_ps(a0, _mm_mul_ps(x0, _mm_shuffle_ps(yy, yy, _BROAD0)) );
					a0 = _mm_add_ps(a0, _mm_mul_ps(x1, _mm_shuffle_ps(yy, yy, _BROAD1)) );
					a0 = _mm_add_ps(a0, _mm_mul_ps(x2, _mm_shuffle_ps(yy, yy, _BROAD2)) );
					a0 = _mm_add_ps(a0, _mm_mul_ps(x3, _mm_shuffle_ps(yy, yy, _BROAD3)) );
					
					yy = _mm_load_ps(m2_jn + 1*_n	+ k);
					a1 = _mm_add_ps(a1, _mm_mul_ps(x0, _mm_shuffle_ps(yy, yy, _BROAD0)) );
					a1 = _mm_add_ps(a1, _mm_mul_ps(x1, _mm_shuffle_ps(yy, yy, _BROAD1)) );
					a1 = _mm_add_ps(a1, _mm_mul_ps(x2, _mm_shuffle_ps(yy, yy, _BROAD2)) );
					a1 = _mm_add_ps(a1, _mm_mul_ps(x3, _mm_shuffle_ps(yy, yy, _BROAD3)) );

					yy = _mm_load_ps(m2_jn + 2*_n	+ k);
					a2 = _mm_add_ps(a2, _mm_mul_ps(x0, _mm_shuffle_ps(yy, yy, _BROAD0)) );
					a2 = _mm_add_ps(a2, _mm_mul_ps(x1, _mm_shuffle_ps(yy, yy, _BROAD1)) );
					a2 = _mm_add_ps(a2, _mm_mul_ps(x2, _mm_shuffle_ps(yy, yy, _BROAD2)) );
					a2 = _mm_add_ps(a2, _mm_mul_ps(x3, _mm_shuffle_ps(yy, yy, _BROAD3)) );

					yy = _mm_load_ps(m2_jn + 3*_n	+ k);
					a3 = _mm_add_ps(a3, _mm_mul_ps(x0, _mm_shuffle_ps(yy, yy, _BROAD0)) );
					a3 = _mm_add_ps(a3, _mm_mul_ps(x1, _mm_shuffle_ps(yy, yy, _BROAD1)) );
					a3 = _mm_add_ps(a3, _mm_mul_ps(x2, _mm_shuffle_ps(yy, yy, _BROAD2)) );
					a3 = _mm_add_ps(a3, _mm_mul_ps(x3, _mm_shuffle_ps(yy, yy, _BROAD3)) );
				}
				_mm_store_ps(r_i+(j+0)*_m, a0);
				_mm_store_ps(r_i+(j+1)*_m, a1); 
				_mm_store_ps(r_i+(j+2)*_m, a2); 
				_mm_store_ps(r_i+(j+3)*_m, a3);
			} 
		}

		return; //RETURN
	}

	const __m128	z0 = _mm_setzero_ps();
	for(int i=0; i<_mn2; i+=4)
		_mm_store_ps(r+i, z0);

	// Fall back to less optimized method for unaligned matrices
	if (!naligned || (_m&0xf)!=0)  //sizes not multiples of 4,16
	{
		float* __restrict r_jm  = r;
		float* __restrict m2_jn = const_cast<float* __restrict>(m2);
		for(int j=0; j<n2; ++j)
		{
			float* __restrict m1_km = const_cast<float* __restrict>(m1);
			for(int k=0; k<_n; ++k)
			{
				//const float m2jnk = m2_jn[k];
				const __m128 c0 = _mm_load1_ps(m2_jn+k); // const m2jnk
				for(int i=0; i<_m; i+=4) 
				{
					//r_jm[i] += m1_km[i] * m2jnk;
					__m128 x0,x1;
					if (maligned) {
						x0 = _mm_load_ps(m1_km+i);
						x1 = _mm_load_ps(r_jm+i);
					} else {
						x0 = _mm_loadu_ps(m1_km+i);
						x1 = _mm_loadu_ps(r_jm+i);
					}
					__m128 x2 = _mm_mul_ps(c0, x0);
					__m128 x3 = _mm_add_ps(x2,x1);
					if (maligned)
						_mm_store_ps(r_jm+i, x3);
					else
						_mm_storeu_ps(r_jm+i, x3);
				}
				m1_km += _m;
			}
			r_jm  += _m;
			m2_jn += _n;
		}

		return;  //RETURN
	}

	// Optimized method for bigger matrices for which our first algo would fill up the L1 cache
	//  (as long as they satisfy alignment requirements, otherwise the second algo would be used)
	for(int j=0; j<n2; ++j)	// n2 es == m._n
	{
		const float* __restrict m2_jn = m2+j*_n;
		float* __restrict r_jm = r+j*_m;
		for(int k=0; k<_n; ++k)  // _n es == m._m
		{
			const float* __restrict m1_km = m1+k*_m;
			const __m128 c0 = _mm_load1_ps(m2_jn+k); // const m2jnk
			//__m128	x0 = _mm_setzero_ps();
			for(int i=0; i<_m; i+=16)	
			{
				__m128 x1, x2, x3, x4, x5, x6; 
				__m128 x1b, x2b, x3b, x4b, x5b, x6b; 
				x2 = _mm_load_ps(m1_km + i);
				x4 = _mm_load_ps(m1_km + i + 4);
				x2b = _mm_load_ps(m1_km + i + 8);
				x4b = _mm_load_ps(m1_km + i + 12);

				//x0 += x1*x2;
				_mm_prefetch((char const*)(m1_km+i+64), _MM_HINT_NTA);
				x5 = _mm_mul_ps(c0, x2);
				x6 = _mm_mul_ps(c0, x4);
				x5b = _mm_mul_ps(c0, x2b);
				x6b = _mm_mul_ps(c0, x4b);
				//x0 += x3*x4;
				x1 = _mm_load_ps(r_jm + i);
				x3 = _mm_load_ps(r_jm + i + 4);
				x1b = _mm_load_ps(r_jm + i + 8);
				x3b = _mm_load_ps(r_jm + i + 12);

				_mm_prefetch((char const*)(r_jm+i+64), _MM_HINT_NTA);
				x1 = _mm_add_ps(x1, x5);
				x3 = _mm_add_ps(x3, x6);
				x1b = _mm_add_ps(x1b, x5b);
				x3b = _mm_add_ps(x3b, x6b);
				
				_mm_store_ps(r_jm + i, x1);
				_mm_store_ps(r_jm + i + 4, x3);
				_mm_store_ps(r_jm + i + 8, x1b);
				_mm_store_ps(r_jm + i + 12, x3b);
			}
		}
	}

	/*for (i = 0; i < m; i += 2)
	{
		for (j = 0; j < n2; j += 2)
		{
			acc00 = acc01 = acc10 = acc11 = 0;
			for (k = 0; k < n; k++)
			{
				acc00 += B[k][j + 0] * A[i + 0][k];
				acc01 += B[k][j + 1] * A[i + 0][k];
				acc10 += B[k][j + 0] * A[i + 1][k];
				acc11 += B[k][j + 1] * A[i + 1][k];
			}
			C[i + 0][j + 0] = acc00;
			C[i + 0][j + 1] = acc01;
			C[i + 1][j + 0] = acc10;
			C[i + 1][j + 1] = acc11;
		}
	} /**/
}



inline float SIMD::MulVec(int _n, const float* v1, const float* v2, const bool bSqrt) //FIXME CHECK
{
	__m128	x0 = _mm_setzero_ps();
	for(; _n>=0; _n-=8)
	{
		__m128 x1 = _mm_load_ps(v1);
		__m128 x2 = _mm_load_ps(v2);
		__m128 x3 = _mm_load_ps(v1 + 4);
		__m128 x4 = _mm_load_ps(v2 + 4);

		// x0 += x1*x2
		_mm_prefetch((char const*)(v1+PREFETCH_DIST8), _MM_HINT_NTA);
		__m128 x5 = _mm_mul_ps(x1, x2);
		x0 = _mm_add_ps(x0, x5);
		// x0 += x3*x4
		_mm_prefetch((char const*)(v2+PREFETCH_DIST8), _MM_HINT_NTA);
		x5 = _mm_mul_ps(x3, x4);
		x0 = _mm_add_ps(x0, x5);

		// Important: vector must have been zero-padded when allocated!
		v1 += 8;
		v2 += 8;
	}
	__m128 x1 = _mm_hadd_ps(x0,x0); //hadd all 4 values
	__m128 x2 = _mm_hadd_ps(x1,x1);
	if (bSqrt)
		x2 = _mm_sqrt_ss(x2);
	// store result
	float r;
	_mm_store_ss(&r, x2);
	return r;
}


inline void SIMD::Mulf(const int N, const float* x, const float k, float* r, bool bDiv)
{
	__m128 x0 = _mm_load1_ps(&k);
	if (bDiv) x0 = _mm_rcp_ps(x0);
	for(int i=0; i<N; i+=16)
	{
		__m128 x1;
		__m128 x2 = _mm_load_ps(x+i);
		__m128 x3 = _mm_load_ps(x+i+4);
		__m128 x4 = _mm_load_ps(x+i+8);
		__m128 x5 = _mm_load_ps(x+i+12);

		x1 = _mm_mul_ps(x2, x0);
		_mm_prefetch((char const*)(x+i+PREFETCH_DIST), _MM_HINT_NTA);
		x2 = _mm_mul_ps(x3, x0);
		x3 = _mm_mul_ps(x4, x0);
		x4 = _mm_mul_ps(x5, x0);

		_mm_store_ps(r+i,   x1);
		_mm_store_ps(r+i+4, x2);
		_mm_store_ps(r+i+8, x3);
		_mm_store_ps(r+i+12,x4);
	}
}

inline void SIMD::Divf(const int N, const float* x, const float k, float* r)
{
	SIMD::Mulf(N, x, k, r, true);
}


inline void SIMD::Add(const int N, const float* x, const float* y, float* r)
{
	for(int i=0; i<N; i+=16)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x1 = _mm_load_ps(x+i+4);
		__m128 x2 = _mm_load_ps(x+i+8);
		__m128 x3 = _mm_load_ps(x+i+12);

		__m128 x4 = _mm_load_ps(y+i);
		__m128 x5 = _mm_load_ps(y+i+4);
		__m128 x6 = _mm_load_ps(y+i+8);
		__m128 x7 = _mm_load_ps(y+i+12);

		x0 = _mm_add_ps(x0, x4);
		x4 = _mm_add_ps(x1, x5);
		x1 = _mm_add_ps(x2, x6);
		x2 = _mm_add_ps(x3, x7);

		_mm_store_ps(r+i, x0);
		_mm_store_ps(r+i+4, x4);
		_mm_store_ps(r+i+8, x1);
		_mm_store_ps(r+i+12, x2);
	}
/*	for(int i=0; i<N; i+=4)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x1 = _mm_load_ps(y+i);
		__m128 x2 = _mm_add_ps(x0, x1);
		_mm_store_ps(r+i, x2);
		//_mm_stream_ps(r+i, x2);
	}/**/

}

inline void SIMD::Sub(const int N, const float* x, const float* y, float* r)
{
	for(int i=0; i<N; i+=16)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x1 = _mm_load_ps(x+i+4);
		__m128 x2 = _mm_load_ps(x+i+8);
		__m128 x3 = _mm_load_ps(x+i+12);

		__m128 x4 = _mm_load_ps(y+i);
		__m128 x5 = _mm_load_ps(y+i+4);
		__m128 x6 = _mm_load_ps(y+i+8);
		__m128 x7 = _mm_load_ps(y+i+12);

		x0 = _mm_sub_ps(x0, x4);
		x4 = _mm_sub_ps(x1, x5);
		x1 = _mm_sub_ps(x2, x6);
		x2 = _mm_sub_ps(x3, x7);

		_mm_store_ps(r+i, x0);
		_mm_store_ps(r+i+4, x4);
		_mm_store_ps(r+i+8, x1);
		_mm_store_ps(r+i+12, x2);
	}
/*	for(int i=0; i<N; i+=4)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x1 = _mm_load_ps(y+i);
		__m128 x2 = _mm_sub_ps(x0, x1);
		_mm_store_ps(r+i, x2);
		//_mm_stream_ps(r+i, x2);
	}/**/
}

inline void	SIMD::Mul(const int N, const float* x, const float* y, float* r)
{
	for(int i=0; i<N; i+=16)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x1 = _mm_load_ps(x+i+4);
		__m128 x2 = _mm_load_ps(x+i+8);
		__m128 x3 = _mm_load_ps(x+i+12);

		__m128 x4 = _mm_load_ps(y+i);
		__m128 x5 = _mm_load_ps(y+i+4);
		__m128 x6 = _mm_load_ps(y+i+8);
		__m128 x7 = _mm_load_ps(y+i+12);

		x0 = _mm_mul_ps(x0, x4);
		x4 = _mm_mul_ps(x1, x5);
		x1 = _mm_mul_ps(x2, x6);
		x2 = _mm_mul_ps(x3, x7);

		_mm_store_ps(r+i, x0);
		_mm_store_ps(r+i+4, x4);
		_mm_store_ps(r+i+8, x1);
		_mm_store_ps(r+i+12, x2);
	}
/*	for(int i=0; i<N; i+=4)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x1 = _mm_load_ps(y+i);
		__m128 x2 = _mm_mul_ps(x0, x1);
		_mm_store_ps(r+i, x2);
		//_mm_stream_ps(r+i, x2);
	}/**/
}


inline void SIMD::MulCf(const int N, const float* x, const float k, float* r)
{
	SIMD::Mulf(N<<1, x, k, r);
}

inline void SIMD::DivCf(const int N, const float* x, const float k, float* r)
{
	SIMD::Mulf(N<<1, x, k, r, true);
}

// Complex by complex multiplication
inline void SIMD::MulC(const int N, const float* x, const float* y, float* r)
{
	const int _2N = N<<1;
	for(int i=0; i<_2N; i+=4)
	{
		//	r._matrix[i] = _matrix[i] * m._matrix[i];

		//__m128 x1 = _mm_load1_ps(x+i);
		//_mm_unpackhi_ps(a,b) // a b => a2 b2 a3 b3
		//_mm_unpacklo_ps(a,b) // a b => a0 b0 a1 b1
		//_mm_movelh_ps(a,b) // a0 a1 b0 b1
		//_mm_movehl_ps(a,b) // b2 b3 a2 a3

		__m128 x0 = _mm_setr_ps(*(x+i), *(x+i), x[i+2], x[i+2]);
		__m128 x1 = _mm_load_ps(y+i);
		//x0 es x1r x1r x2r x2r
		//x1 es y1r y1i y2r y2i

		__m128 x2 = _mm_mul_ps(x1, x0);
		//x2 es x1r*y1r x1r*y1i x2r*y2r x2r*y2i

		x0 = _mm_setr_ps(x[i+1], x[i+1], x[i+3], x[i+3]);
		//x0 es x1i x1i x2i x2i

		__m128 x3 = _mm_mul_ps(x0, x1);
		// x3 es x1i*y1r x1i*y1i x2i*y2r x2i*y2i

		//shuffle x3
		__m128 x4 = _mm_shuffle_ps(x3, x3, _MM_SHUFFLE(2,3,0,1));
		// x4 es x1i*y1i x1i*y1r x2i*y2i x2i*y2r

		//addsub
		__m128 x5 = _mm_addsub_ps(x2, x4);
		// x5 es (x1r*y1r-x1i*y1i) (x1r*y1i+x1i*y1r) (x2r*y2r-x2i*y2i) (x2r*y2i+x2i*y2r)

		//store
		//s _mm_store_ps(r+i, x5);
		_mm_stream_ps(r+i, x5);
	}
}

inline void SIMD::AddC(const int N, const float* x, const float* y, float* r)
{
	SIMD::Add(N<<1, x, y, r);
}

inline void SIMD::SubC(const int N, const float* x, const float* y, float* r)
{
	SIMD::Sub(N<<1, x, y, r);
}

// Complex division
inline void SIMD::DivC(const int N, const float* x, const float* y, float* r)
{
	//	Expected result: complex division (x1/y1)r (x1/y1)i (x2/y2)r (x2/y2)i
	//
	// (x1r*y1r+x1i*y1i) (x1i*y1r-x1r*y1i) (x2r*y2r+x2i*y2i) (x2i*y2r-x2r*y2i)
	//  ---------------   --------------    ---------------   ---------------
	//    (y1r2+y1i2)      (y1r2+y1i2)        (y2r2+y2i2)      (y2r2+y2i2)
	
	const int _2N = N<<1;
	for(int i=0; i<_2N; i+=4)
	{
		// x0 = x1r x1i x2r x2i
		__m128 x0 = _mm_load_ps(x+i);
		
		// x2 = y1r y1i y2r y2i
		__m128 x2 = _mm_load_ps(y+i);
		// x2 = y1r y2r y1i y2i
		x2 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(3,1,2,0));
		
		// x1 = y1r y1r y2r y2r
		__m128 x1 = _mm_unpacklo_ps(x2,x2);
		//_mm_unpackhi_ps(a,b) // a b => a2 b2 a3 b3
		//_mm_unpacklo_ps(a,b) // a b => a0 b0 a1 b1

		// x3 = x1r*y1r x1i*y1r x2r*y2r x2i*y2r
		__m128 x3 = _mm_mul_ps(x0,x1);

		// shuffle x0 => x1i x1r x2i x2r
		__m128 x0b = _mm_shuffle_ps(x0, x0, _MM_SHUFFLE(2,3,0,1));
		// x1 = y1i y1i y2i y2i
		__m128 x1b = _mm_unpackhi_ps(x2,x2);
		// x4 = x1i*y1i x1r*y1i x2i*y2i x2r*y2i
		__m128 x4 = _mm_mul_ps(x0b,x1b);
		
		// x2 = addsub(x3,x4)
		x2 = _mm_addsub_ps(x3,x4);

		// x0 = y1r2+y1i2 y1r2+y1i2 y2r2+y2i2 y2r2+y2i2
		// (x0 = x1*x1 + x1b*x1b)
		x3 = _mm_mul_ps(x1,x1);
		x4 = _mm_mul_ps(x1b,x1b);
		// (x0 = x3 + x4)
		x0 = _mm_add_ps(x3,x4);

		// x5 = div(x2,x0)
		__m128 x5 = _mm_div_ps(x2,x0);
		
		//s _mm_store_ps(r+i, x5);
		_mm_stream_ps(r+i, x5);
	}
}

inline void SIMD::MagC(const int N, const float* x, float* r)
{
	// calcula la magnitud de cada complex en el array x (sqrt(r^2+i^2))
	// en x vienen N complex; en r se guardan N floats
	//for(int i=0; i<N; ++i) // <-equivalente sin opt
	//	r[i] = sqrtf(x[ i<<1   ] * x[ i<<1   ] +  
	//		 	     x[(i<<1)+1] * x[(i<<1)+1]);
	const int _2N = N<<1;
	for(int i=0; i<_2N; i+=8)
	{
		//load
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x2 = _mm_load_ps(x+i+4);
		//mul
		__m128 x1 = _mm_mul_ps(x0,x0);
		__m128 x3 = _mm_mul_ps(x2,x2);

		//hadd x4 = x0_1r^2+x0_1i^2 x0_2r^2+x0_2i^2 x2_1r^2+x2_1i^2 x2_2r^2+x2_2i^2
		__m128 x4 = _mm_hadd_ps(x1,x3);
		//sqrt
		__m128 x5 = _mm_sqrt_ps(x4);
		//store
		_mm_store_ps(r+(i>>1), x5);	//almacenamos 8 floats de golpe
		//s _mm_store_ps(r+(i>>1), x5);
		//_mm_stream_ps(r+(i>>1), x5);
	}
}


inline void SIMD::Logvf(const int N, const float* x, const float v, float* r)
{
	assert(0); //log_ps is in simd math header files

/*	__m128 c1 = _mm_set1_ps(v); //1.0f
	for(int i=0; i<N; i+=4)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x1 = _mm_add_ps(x0, c1);
		x0 = log_ps(x1);
		_mm_store_ps(r+i, x0);
		//_mm_stream_ps(r+i, x0);
	}*/
}



inline void SIMD::AddMulf(const int N, const float* x, const float add, const float mul, float* r)
{
	__m128 x1 = _mm_load1_ps(&add);
	__m128 x3 = _mm_load1_ps(&mul);
	for(int i=0; i<N; i+=4)
	{
		__m128 x0 = _mm_load_ps(x+i);
		__m128 x2 = _mm_add_ps(x0, x1);
		__m128 x4 = _mm_mul_ps(x2, x3);
		_mm_store_ps(r+i, x4);
		//_mm_stream_ps(r+i, x4);
	}
}

inline float SIMD::Sumf(const int N, float* x, bool bAvg)
{
	__m128 x0 = _mm_setzero_ps();
	//fill the extra padded space with zeros to avoid reading garbage when we overrun in the last read
	if ((N&3)!=0) _mm_storeu_ps(x+N, x0); 
	for(int i=0; i<N; i+=4)
	{
		__m128 x1 = _mm_load_ps(x+i);
		x0 = _mm_add_ps(x0, x1);
	}
	if (bAvg) {
		const float ndiv=(float)N;
		__m128 xdiv = _mm_load1_ps(&ndiv);
		x0 = _mm_div_ps(x0, xdiv);
	}
	__m128 x1 = _mm_hadd_ps(x0, x0);
	x0 = _mm_hadd_ps(x1, x1); //hadd all 4 values
	// store result
	float r;
	_mm_store_ss(&r, x0);
	return r;
}

inline float SIMD::Sum2f(const int N, float* x, bool bAvg)
{
	__m128 x0 = _mm_setzero_ps();
	//fill the extra buffer space with zeros to avoid reading garbage when we overrun in the last read
	if ((N&3)!=0) _mm_storeu_ps(x+N, x0); 
	for(int i=0; i<N; i+=4)
	{
		__m128 x1 = _mm_load_ps(x+i);
		__m128 x2 = _mm_mul_ps(x1, x1);
		x0 = _mm_add_ps(x0, x2);
	}
	if (bAvg) {
		const float ndiv=(float)N;
		__m128 xdiv = _mm_load1_ps(&ndiv);
		x0 = _mm_div_ps(x0, xdiv);
	}
	__m128 x1 = _mm_hadd_ps(x0, x0);
	x0 = _mm_hadd_ps(x1, x1); //hadd all 4 values
	// store result
	float r;
	_mm_store_ss(&r, x0);
	return r;
}

inline float SIMD::SubSum2f(const int N, float* x, const float k, bool bAvg)
{
	__m128 x0 = _mm_setzero_ps();
	__m128 xsub = _mm_load1_ps(&k);
	//fill the extra buffer space with zeros to avoid reading garbage when we overrun in the last read
	if ((N&3)!=0) _mm_storeu_ps(x+N, x0); 
	for(int i=0; i<N; i+=4)
	{
		__m128 x1 = _mm_load_ps(x+i);
		__m128 x2 = _mm_sub_ps(x1, xsub);
		__m128 x3 = _mm_mul_ps(x2, x2);
		x0 = _mm_add_ps(x0, x3);
	}
	if (bAvg) {
		const float ndiv=(float)N;
		__m128 xdiv = _mm_load1_ps(&ndiv);
		x0 = _mm_div_ps(x0, xdiv);
	}
	__m128 x1 = _mm_hadd_ps(x0, x0);
	x0 = _mm_hadd_ps(x1, x1); //hadd all 4 values
	// store result
	float r;
	_mm_store_ss(&r, x0);
	return r;
}

inline void printx(const __m128& xm, const char* msg)
{
	float d[4];
	_mm_storeu_ps(d, xm);
	printf("%s d0..3= %f %f %f %f\n", (msg!=0?msg:""), d[0], d[1], d[2], d[3]);
}


/////////////////////////////////////


#endif
