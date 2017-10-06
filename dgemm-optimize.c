#include <mmintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// void dgemm( int m, int n, float *A, float *C )
// {
// 	for( int k = 0; k < n; k++ ) 
//       for( int j = 0; j < m; j++ ) 
//   		for( int i = 0; i < m; i++ )
// 	C[i+j*m] += A[i+k*m] * A[j+k*m];
// }



// void dgemm( int strip, int size, float *A, float *C )
// {
// // FILL-IN 
//         for( int j = 0; j < strip; j++ ){
//     		for( int k = 0; k < size; k++ ){
// 				int unrollOps = strip/8;
// 				int kStrip =  k*strip;
// 				int jStrip = j*strip;
// 				float AjkStrip =  A[j + kStrip];
// 				int i8jStrip, i8kStrip;
// 				for( int i = 0; i < unrollOps; i++ ) {
// 					i8jStrip = i*8 + jStrip;
// 					i8kStrip = i*8 + kStrip;
// 					C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
// 					C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
// 					C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
// 					C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
// 					C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
// 					C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
// 					C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
// 					C[i8jStrip] += A[i8kStrip] * AjkStrip;
// 				}
//             for(int i = unrollOps*8; i < strip; i++){
//                 C[i + jStrip] += A[i + kStrip] * AjkStrip;
//             }
//         }
//     }
// }

void dgemm( int strip, int size, float *A, float *C )
{
	//float * AjkStrip_arr =  (float*)malloc(sizeof(float)*4);
	int unrollOps, unrollOps2, kStrip, kStrip2, jStrip, i8jStrip, i8kStrip;
	float AjkStrip;
	__m128 c_small1, c_small2, c_small3, c_small4, v_small1, v_small2, v_small3, v_small4, sum1, sum2, sum3, sum4;
	for( int j = 0; j < strip; j++ ){
	    for( int k = 0; k < size; k++ ){
            // unrollOps = strip/8;
            // kStrip =  k*strip;
            // jStrip = j*strip;
            // AjkStrip =  A[j + kStrip];
			//for (int q = 0; q < 4; q++)
			// float AjkStrip_arr[] = {AjkStrip, AjkStrip, AjkStrip, AjkStrip};
			// float AjkStrip_arr[] __attribute__((__aligned__(16))) = {AjkStrip, AjkStrip, AjkStrip, AjkStrip};
			// __m128 single = _mm_loadu_ps(AjkStrip_arr);
    //         for( int i = 0; i < unrollOps; i++ ) {
	// 			i8kStrip = i*8 + kStrip;
    //             // __m128 c_small1_pre = _mm_loadu_ps((A+i8kStrip));
	// 			__m128 c_small1 = _mm_loadu_ps((A+i8kStrip));
	// 			i8jStrip = i*8 + jStrip;	//sickest edit, saving mucho cycles
    //             // __m128 c_small2_pre = _mm_loadu_ps((A+i8kStrip+4));
	// 			__m128 c_small2 = _mm_loadu_ps((A+i8kStrip+4));
	// 			// __m128 single = _mm_set1_ps(AjkStrip);
	// 			// __m128 single = _mm_loadu_ps(AjkStrip_arr);
	// 			c_small1 *= AjkStrip;
	// 			c_small2 *= AjkStrip;
	// 			// __m128 c_small1 = _mm_mul_ps(c_small1_pre, single);
	// 			// __m128 c_small2 = _mm_mul_ps(c_small2_pre, single); 

	// 			//FOR PRINTING THE VECTORS
	// 			// float *val = (float*) & c_small1;
	// 			// printf("Vectorized: %f, %f, %f, %f, ", val[0], val[1], val[2], val[3]);
	// 			// val = (float*) & c_small2;
	// 			// printf("%f, %f, %f, %f\n", val[0], val[1], val[2], val[3]);
	// 			// printf("Not-Vec:");
	// 			// for (int q = 0; q < 8; q++)
	// 			// {
	// 			// 	//PRINTING ELEMENT
	// 			// 	//printf(" %f, ", A[i8kStrip+q] * AjkStrip);
	// 			// 	//PRINTING SUM
	// 			// 	printf("%i: %f ", q, C[i8jStrip+q] + A[i8kStrip++] * AjkStrip);
	// 			// }		
	// 			// printf("\n");

	// 			__m128 v_small1 = _mm_loadu_ps(C+i8jStrip);
	// 			__m128 v_small2 = _mm_loadu_ps(C+i8jStrip+4);
	// 			__m128 sum1 = _mm_add_ps(v_small1, c_small1);
	// 			__m128 sum2 = _mm_add_ps(v_small2, c_small2);
	// 			// v_small1 = _mm_add_ps(c_small1, v_small1);
	// 			// v_small1 = _mm_add_ps(c_small2, v_small2);
	// 			// float *val1 = (float*) & c_small1;
	// 			// float *val2 = (float*) & c_small2;
	// 			// float *val1 = (float*) & sum1;
	// 			// float *val2 = (float*) & sum2;
	// 			// memcpy(C+i8jStrip, val1, 4*sizeof(float));
	// 			// memcpy(C+4+i8jStrip, val2, 4*sizeof(float));
    //             // printf("start: ");
	// 			// for (int q = 0; i < 4; i++)
    //             // {
    //             //     printf("%f, ", val1[q]);
    //             //     C[i8jStrip+q] = val1[q];
    //             // }	
	// 			// for (int q = 0; i < 8; i++)
    //             // {
    //             //     printf("%f, ", val2[q]);
    //             //     C[i8jStrip+4+q] = val2[q];
    //             // }
    //             // printf("\n");

	// 			_mm_storeu_ps(C+i8jStrip, sum1);
	// 			_mm_storeu_ps(C+i8jStrip+4, sum2);
	// 			// _mm_storeu_ps(C+i8jStrip, v_small1);
	// 			// _mm_storeu_ps(C+i8jStrip+4, v_small2);

	// 			//FOR PRINTING SUMS
	// 			// printf("Vectorized: ");
	// 			// for (int q = 0; q < 8; q++)
	// 			// {
	// 			// 	//printf(" %f, ", A[i8kStrip++] * AjkStrip);
	// 			// 	printf("%i: %f ", q, C[i8jStrip+q]);
	// 			// }
	// 			// printf("\n\n");

	// 			//ANTIQUE STRATEGY
	// 			/*
    //             C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
	// 			printf("Not-Vec: %f, ", A[i8kStrip++] * AjkStrip);
    //             C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
    //             C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
    //             C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
    //             C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
    //             C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
    //             C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
    //             C[i8jStrip] += A[i8kStrip] * AjkStrip;*/
    //         }
    //         for(int i = unrollOps*8; i < strip; i++){
    //             C[i + jStrip] += A[i + kStrip] * AjkStrip;
    //         }



		    unrollOps = strip/12;
			unrollOps = strip/16;
            kStrip =  k*strip;
            jStrip = j*strip;
            AjkStrip =  A[j + kStrip];
        
			for( int i = 0; i < unrollOps; i++ ) {
				// i8kStrip = i*12 + kStrip;
				i8kStrip = i*16 + kStrip;
                
				c_small1 = _mm_loadu_ps((A+i8kStrip));
				c_small2 = _mm_loadu_ps(A+i8kStrip+4);
				c_small3 = _mm_loadu_ps(A+i8kStrip+8);
				c_small4 = _mm_loadu_ps(A+i8kStrip+12);

				// i8jStrip = i*12 + jStrip;	//sickest edit, saving mucho cycles
				i8jStrip = i*16 + jStrip;	//sickest edit, saving mucho cycles
                
				c_small1 *= AjkStrip;
				c_small2 *= AjkStrip;
				c_small3 *= AjkStrip;
				c_small4 *= AjkStrip;
				
				v_small1 = _mm_loadu_ps(C+i8jStrip);
				v_small2 = _mm_loadu_ps(C+i8jStrip+4);
				v_small3 = _mm_loadu_ps(C+i8jStrip+8);
				v_small4 = _mm_loadu_ps(C+i8jStrip+12);

				sum1 = _mm_add_ps(v_small1, c_small1);
				sum2 = _mm_add_ps(v_small2, c_small2);
				sum3 = _mm_add_ps(v_small3, c_small3);
				sum4 = _mm_add_ps(v_small4, c_small4);
				
				_mm_storeu_ps(C+i8jStrip, sum1);
				_mm_storeu_ps(C+i8jStrip+4, sum2);
				_mm_storeu_ps(C+i8jStrip+8, sum3);
				_mm_storeu_ps(C+i8jStrip+12, sum4);
            }
			// unrollOps = 0;		//take out
			// kStrip = k*strip;	//take out
			// jStrip = j*strip;	//take out
			// AjkStrip =  A[j + kStrip];	//take out
			// unrollOps2 = (strip-(unrollOps*16))/4;
			// kStrip2 = kStrip + (unrollOps*16);
			// for (int i = 0; i < unrollOps2; i++)
			// {
			// 	i8kStrip = i*4 + kStrip2;
			// 	c_small1 = _mm_loadu_ps(A+i8kStrip);
			// 	i8jStrip = i*4 + jStrip;
			// 	c_small1 *= AjkStrip;
			// 	v_small1 = _mm_loadu_ps(C+i8jStrip);
			// 	sum1 = _mm_add_ps(v_small1, c_small1);
			// 	_mm_storeu_ps(C+i8jStrip, sum1);
			// }

            // for(int i = unrollOps*12; i < strip; i++){
			// for (int i = unrollOps*16; i < strip; i++){
			for (int i = unrollOps*16 + unrollOps2 * 4; i < strip; i++){
			// for (int i = unrollOps2*4; i < strip; i++){
                C[i + jStrip] += A[i + kStrip] * AjkStrip;
            }
        }
    }
}


// #include <mmintrin.h>
// #include <x86intrin.h>
 
// void dgemm( int strip, int size, float *A, float *C )
// {
// // FILL-IN
//     for( int j = 0; j < strip; j++ ){
//         for( int k = 0; k < size; k++ ){
//             int unrollOps = strip/8;
//             int kStrip =  k*strip;
//             int jStrip = j*strip;
//             float AjkStrip =  A[j + kStrip];
//             int i8jStrip, i8kStrip;
//             for( int i = 0; i < unrollOps; i++ ) {
//                 i8jStrip = i*8 + jStrip;
//                 i8kStrip = i*8 + kStrip;
//                 __m128 single = _mm_set_ps(AjkStrip, AjkStrip, AjkStrip, AjkStrip);
 
//                 __m128 vecA0= _mm_load_ps(A+i8kStrip);
//                 vecA0=_mm_mul_ps(vecA0, single);
//                 //vecA0 *= AjkStrip;
//                 __m128 vecA4= _mm_load_ps(A+i8kStrip+4);
//                 vecA4=_mm_mul_ps(vecA4, single);
//                 //vecA4 *= AjkStrip;
//                 __m128 vecC0= _mm_load_ps(C+i8jStrip);
//                 __m128 vecC4= _mm_load_ps(C+i8jStrip+4);
 
//                 vecC0 = _mm_add_ps(vecC0, vecA0);
//                 vecC4 = _mm_add_ps(vecC4, vecA4);
 
//                 _mm_store_ps(C+i8jStrip, vecC0);
//                 _mm_store_ps(C+i8jStrip+4, vecC4);
 
               
//                 //     printf("Poopy: %f\n", vecC0[i]);
               
//                 //     printf("Poopy: %f\n", vecC0[i]);
               
 
               
//                 // C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
//                 // C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
//                 // C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
//                 // C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
//                 // C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
//                 // C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
//                 // C[i8jStrip++] += A[i8kStrip++] * AjkStrip;
//                 // C[i8jStrip] += A[i8kStrip] * AjkStrip;
//             }
//             for(int i = unrollOps*8; i < strip; i++){
//                 C[i + jStrip] += A[i + kStrip] * AjkStrip;
//             }
//         }
//     }
// }