/*
 ============================================================================
 Name        : GPUexplore.cu
 Author      : Anton Wijs
 Version     :
 Copyright   : Copyright Anton Wijs
 Description : CUDA GPUexplore: On the fly state space analysis
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// type of elements used
#define inttype uint32_t
// type of indices in hash table
#define indextype uint64_t

enum BucketEntryStatus { EMPTY, TAKEN, FOUND };
enum PropertyStatus { NONE, DEADLOCK, SAFETY, LIVENESS };

#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

// Nr of tiles processed in single kernel launch
//#define TILEITERS 10

static const int WARPSIZE = 32;
static const int HALFWARPSIZE = 16;
static const int INTSIZE = 32;
static const int BUFFERSIZE = 50;

// GPU constants
__constant__ inttype d_nrbuckets;
__constant__ inttype d_shared_q_size;
__constant__ inttype d_nr_procs;
__constant__ inttype d_max_buf_ints;
__constant__ inttype d_sv_nints;
__constant__ inttype d_bits_act;
__constant__ inttype d_nbits_offset;
__constant__ inttype d_kernel_iters;
__constant__ inttype d_nbits_syncbits_offset;
__constant__ PropertyStatus d_property;

// GPU shared memory array
extern __shared__ inttype shared[];

// thread ids
#define WARP_ID							(threadIdx.x / WARPSIZE)
#define GLOBAL_WARP_ID					(((blockDim.x / WARPSIZE)*blockIdx.x)+WARP_ID)
#define NR_WARPS						((blockDim.x / WARPSIZE)*gridDim.x)
#define LANE							(threadIdx.x % WARPSIZE)
#define HALFLANE						(threadIdx.x % HALFWARPSIZE)
//#define ENTRY_ID						(LANE % d_sv_nints)
#define ENTRY_ID						(HALFLANE % d_sv_nints)
#define GROUP_ID						(threadIdx.x % d_nr_procs)
#define GROUP_GID						(threadIdx.x / d_nr_procs)

//#define NREL_IN_BUCKET					((WARPSIZE / d_sv_nints))
#define NREL_IN_BUCKET					((HALFWARPSIZE / d_sv_nints)*2)
#define NREL_IN_BUCKET_HOST				((HALFWARPSIZE / sv_nints)*2)

// constant for cuckoo hashing (Alcantara et al)
static const inttype P = 334214459;
// Retry constant to determine number of retries for element insertion
#define RETRYFREQ 7
#define NR_HASH_FUNCTIONS 8
// Number of retries in local cache
#define CACHERETRYFREQ 20
// Maximum size of state vectors (in nr. of 32-bit integers)
#define MAX_SIZE 9
// Empty state vectors
static const inttype EMPTYVECT32 = 0x7FFFFFFF;
// Constant to indicate that no more work is required
# define EXPLORATION_DONE 0x7FFFFFFF
// offset in shared memory from which loaded data can be read
static const int SH_OFFSET = 5;
//static const int KERNEL_ITERS = 10;
//static const int NR_OF_BLOCKS = 3120;
//static const int BLOCK_SIZE = 512;
static const int KERNEL_ITERS = 1;
static const int NR_OF_BLOCKS = 1;
static const int BLOCK_SIZE = 32;
const size_t Mb = 1<<20;

// test macros
#define PRINTTHREADID()						{printf("Hello thread %d\n", (blockIdx.x*blockDim.x)+threadIdx.x);}
#define PRINTTHREAD(j, i)					{printf("%d: Seen by thread %d: %d\n", (j), (blockIdx.x*blockDim.x)+threadIdx.x, (i));}

// Offsets calculations for shared memory arrays
#define HASHCONSTANTSLEN				(2*NR_HASH_FUNCTIONS)
#define VECTORPOSLEN					(d_nr_procs+1)
#define LTSSTATESIZELEN					(d_nr_procs)
#define OPENTILELEN						(d_sv_nints*(blockDim.x/d_nr_procs))
#define TGTSTATELEN						(blockDim.x*d_sv_nints)
#define THREADBUFFERLEN					((blockDim.x/d_nr_procs)*(THREADBUFFERSHARED+(d_nr_procs*d_max_buf_ints)))

#define HASHCONSTANTSOFFSET 			(SH_OFFSET)
#define VECTORPOSOFFSET 				(HASHCONSTANTSOFFSET+HASHCONSTANTSLEN)
#define LTSSTATESIZEOFFSET 				(VECTORPOSOFFSET+VECTORPOSLEN)
#define OPENTILEOFFSET 					(LTSSTATESIZEOFFSET+LTSSTATESIZELEN)
#define TGTSTATEOFFSET		 			(OPENTILEOFFSET+OPENTILELEN)
#define THREADBUFFEROFFSET	 			(TGTSTATEOFFSET+TGTSTATELEN)
#define CACHEOFFSET 					(THREADBUFFEROFFSET+THREADBUFFERLEN)

// One int for sync action counter
// One int for POR counter
#define THREADBUFFERSHARED				2
// parameter is thread id
#define THREADBUFFERGROUPSTART(i)		(THREADBUFFEROFFSET+(((i) / d_nr_procs)*(THREADBUFFERSHARED+(d_nr_procs*d_max_buf_ints))))
// parameter is group id
#define THREADBUFFERGROUPPOS(i, j)		shared[THREADBUFFERGROUPSTART(threadIdx.x)+THREADBUFFERSHARED+((i)*d_max_buf_ints)+(j)]
#define THREADGROUPCOUNTER				shared[(THREADBUFFERGROUPSTART(threadIdx.x))]
#define THREADGROUPPOR					shared[(THREADBUFFERGROUPSTART(threadIdx.x)) + 1]
#define OPENTILESTATEPART(i)			shared[OPENTILEOFFSET+(d_sv_nints*(threadIdx.x / d_nr_procs))+(i)]

#define THREADINGROUP					(threadIdx.x < (blockDim.x/d_nr_procs)*d_nr_procs)

#define STATESIZE(i)					(shared[LTSSTATESIZEOFFSET+(i)])
#define VECTORSTATEPOS(i)				(shared[VECTORPOSOFFSET+(i)])
#define NR_OF_STATES_IN_TRANSENTRY(i)	((31 - d_bits_act) / shared[LTSSTATESIZEOFFSET+(i)])
// SM local progress flags
#define ITERATIONS						(shared[0])
#define CONTINUE						(shared[1])
#define OPENTILECOUNT					(shared[2])
#define WORKSCANRESULT					(shared[3])
#define SCAN							(shared[4])

// BIT MANIPULATION MACROS

#define SETBIT(i, x)							{(x) = ((1L<<(i)) | (x));}
#define GETBIT(i, x)							(((x) >> (i)) & 1L)
#define SETBITS(i, j, x)						{(x) = (x) | (((1L<<(j))-1)^((1L<<(i))-1));}
#define GETPROCTRANSACT(a, t)					{bitmask = 0; SETBITS(1, 1+d_bits_act, bitmask); (a) = ((t) & bitmask) >> 1;}
#define GETPROCTRANSSYNC(a, t)					{(a) = ((t) & 1);}
#define GETPROCTRANSSTATE(a, t, i, j)			{bitmask = 0; SETBITS(1+d_bits_act+((i)-1)*shared[LTSSTATESIZEOFFSET+(j)], \
								1+d_bits_act+(i)*shared[LTSSTATESIZEOFFSET+(j)],bitmask); \
								(a) = ((t) & bitmask) >> 1+d_bits_act+(((i)-1)*shared[LTSSTATESIZEOFFSET+(j)]);}
#define GETTRANSOFFSET(a, t, i)					{bitmask = 0; SETBITS((i)*d_nbits_offset, ((i)+1)*d_nbits_offset, bitmask); (a) = ((t) & bitmask) >> ((i)*d_nbits_offset);}
#define GETSYNCOFFSET(a, t, i)					{bitmask = 0; SETBITS((i)*d_nbits_syncbits_offset, ((i)+1)*d_nbits_syncbits_offset, bitmask); \
													(a) = ((t) & bitmask) >> ((i)*d_nbits_syncbits_offset);}
#define GETSTATEVECTORSTATE(a, t, i)			{bitmask = 0; 	if (shared[VECTORPOSOFFSET+(i)]/INTSIZE == (shared[VECTORPOSOFFSET+(i)+1]-1)/INTSIZE) { \
																	SETBITS((shared[VECTORPOSOFFSET+(i)] % INTSIZE), \
																			(((shared[VECTORPOSOFFSET+(i)+1]-1) % INTSIZE)+1), bitmask); \
																	(a) = ((t)[shared[VECTORPOSOFFSET+(i)]/INTSIZE] & bitmask) >> (shared[VECTORPOSOFFSET+(i)] % INTSIZE); \
																} \
																else { \
																	SETBITS(0,(shared[VECTORPOSOFFSET+(i)+1] % INTSIZE),bitmask); \
																	(a) = (t)[shared[VECTORPOSOFFSET+(i)]/INTSIZE] >> (shared[VECTORPOSOFFSET+(i)] % INTSIZE) \
																		 | \
																		((t)[shared[VECTORPOSOFFSET+(i)+1]/INTSIZE] & bitmask) << \
																		(INTSIZE - (shared[VECTORPOSOFFSET+(i)] % INTSIZE)); \
																} \
												}
#define SETPROCTRANSACT(t, x)					{bitmask = 0; SETBITS(1, d_bits_act+1,bitmask); (t) = ((t) & ~bitmask) | ((x) << 1);}
#define SETPROCTRANSSYNC(t, x)					{(t) = ((t) & ~1) | (x);}
#define SETPROCTRANSSTATE(t, i, x, j)			{bitmask = 0; SETBITS(1+d_bits_act+((i)-1)*shared[LTSSTATESIZEOFFSET+(j)],1+d_bits_act+(i)*shared[LTSSTATESIZEOFFSET+(j)],bitmask); \
													(t) = ((t) & ~bitmask) | ((x) << (1+d_bits_act+((i)-1)*shared[LTSSTATESIZEOFFSET+(j)]));}
#define SETSTATEVECTORSTATE(t, i, x)			{bitmask = 0; 	if (shared[VECTORPOSOFFSET+(i)]/INTSIZE == (shared[VECTORPOSOFFSET+(i)+1]-1)/INTSIZE) { \
																	SETBITS((shared[VECTORPOSOFFSET+(i)] % INTSIZE), \
																			(((shared[VECTORPOSOFFSET+(i)+1]-1) % INTSIZE)+1),bitmask); \
																	(t)[shared[VECTORPOSOFFSET+(i)]/INTSIZE] = ((t)[shared[VECTORPOSOFFSET+(i)]/INTSIZE] & ~bitmask) | \
																	((x) << (shared[VECTORPOSOFFSET+(i)] % INTSIZE)); \
																} \
																else { \
																	SETBITS(0,(shared[VECTORPOSOFFSET+(i)] % INTSIZE), bitmask); \
																	(t)[shared[VECTORPOSOFFSET+(i)]/INTSIZE] = ((t)[shared[VECTORPOSOFFSET+(i)]/INTSIZE] & bitmask) | \
																	((x) << (shared[VECTORPOSOFFSET+(i)] % INTSIZE)); \
																	bitmask = 0; \
																	SETBITS((shared[VECTORPOSOFFSET+(i)+1] % INTSIZE), INTSIZE, bitmask); \
																	(t)[shared[VECTORPOSOFFSET+(i)+1]/INTSIZE] = ((t)[shared[VECTORPOSOFFSET+(i)+1]/INTSIZE] & bitmask) | \
																		((x) >> (INTSIZE - (shared[VECTORPOSOFFSET+(i)] % INTSIZE))); \
																} \
												}
// NEEDS FIX: USE BIT 32 OF FIRST INTEGER TO INDICATE STATE OR NOT (1 or 0), IN CASE MULTIPLE INTEGERS ARE USED FOR STATE VECTOR!!!
//#define ISSTATE(t)								((t)[(d_sv_nints-1)] != EMPTYVECT32)
#define ISSTATE(t)								((t)[0] != EMPTYVECT32)
#define SETNEWSTATE(t)							{	(t)[(d_sv_nints-1)] = (t)[(d_sv_nints-1)] | 0x80000000;}
#define SETOLDSTATE(t)							{	(t)[(d_sv_nints-1)] = (t)[(d_sv_nints-1)] & 0x7FFFFFFF;}
#define ISNEWSTATE(t)							((t)[(d_sv_nints-1)] >> 31)
#define ISNEWSTATE_HOST(t)						((t)[(sv_nints-1)] >> 31)
#define ISNEWINT(t)								((t) >> 31)
#define OLDINT(t)								((t) & 0x7FFFFFFF)
#define NEWINT(t)								((t) | 0x80000000)
#define STRIPSTATE(t)							{(t)[(d_sv_nints-1)] = (t)[(d_sv_nints-1)] & 0x7FFFFFFF;}
#define STRIPPEDSTATE(t, i)						((i == d_sv_nints-1) ? ((t)[i] & 0x7FFFFFFF) : (t)[i])
#define STRIPPEDENTRY(t, i)						((i == d_sv_nints-1) ? ((t) & 0x7FFFFFFF) : (t))
#define STRIPPEDENTRY_HOST(t, i)				((i == sv_nints-1) ? ((t) & 0x7FFFFFFF) : (t))
#define NEWSTATEPART(t, i)						(((i) == d_sv_nints-1) ? ((t)[d_sv_nints-1] | 0x80000000) : (t)[(i)])
#define COMPAREENTRIES(t1, t2)					(((t1) & 0x7FFFFFFF) == ((t2) & 0x7FFFFFFF))
#define OWNSSYNCRULE(a, t, i)					{if (GETBIT((i),(t))) { \
													bitmask = 0; SETBITS(0,(i),bitmask); if ((t & bitmask) > 0) {(a) = 0;} else {(a) = 1;}} \
													else {(a) = 0;}}
#define GETSYNCRULE(a, t, i)					{bitmask = 0; SETBITS((i)*d_nr_procs,((i)+1)*d_nr_procs,bitmask); (a) = ((t) & bitmask) >> ((i)*d_nr_procs);}
#define SYNCRULEISAPPLICABLE(a, t, ac)			{(a) = 1; for (bk = 0; bk < d_nr_procs; bk++) { \
													if (GETBIT(bk,(t))) { \
														bj = THREADBUFFERGROUPPOS((inttype) bk,0); \
														if (bj == 0) { \
															(a) = 0; \
														} \
														else { \
															GETPROCTRANSACT(k, bj); \
															if (k != (ac)) { \
																(a) = 0; \
															} \
														}\
													} \
												} \
												}

// HASH TABLE MACROS

// Return 0 if not found, 1 if found, 2 if cache is full
__device__ inttype STOREINCACHE(inttype* t, inttype* d_q, inttype* address) {
	inttype bi, bj, bk, bl, bitmask;
	indextype hashtmp;
	STRIPSTATE(t);
	hashtmp = 0;
	for (bi = 0; bi < d_sv_nints; bi++) {
		hashtmp += t[bi];
	}
	bitmask = d_sv_nints*((inttype) (hashtmp % ((d_shared_q_size - CACHEOFFSET) / d_sv_nints)));
	SETNEWSTATE(t);
	bl = 0;
	while (bl < CACHERETRYFREQ) {
		bi = atomicCAS((inttype *) &shared[CACHEOFFSET+bitmask+(d_sv_nints-1)], EMPTYVECT32, t[d_sv_nints-1]);
		if (bi == EMPTYVECT32) {
			for (bj = 0; bj < d_sv_nints-1; bj++) {
				shared[CACHEOFFSET+bitmask+bj] = t[bj];
			}
			*address = bitmask;
			return 0;
		}
		if (COMPAREENTRIES(bi, t[d_sv_nints-1])) {
			if (d_sv_nints == 1) {
				*address = bitmask;
				return 1;
			}
			else {
				for (bj = 0; bj < d_sv_nints-1; bj++) {
					if (shared[CACHEOFFSET+bitmask+bj] != (t)[bj]) {
						break;
					}
				}
				if (bj == d_sv_nints-1) {
					*address = bitmask;
					return 1;
				}
			}
		}
		if (!ISNEWINT(bi)) {
			bj = atomicCAS((inttype *) &shared[CACHEOFFSET+bitmask+(d_sv_nints-1)], bi, t[d_sv_nints-1]);
			if (bi == bj) {
				for (bk = 0; bk < d_sv_nints-1; bk++) {
					shared[CACHEOFFSET+bitmask+bk] = t[bk];
				}
				*address = bitmask;
				return 0;
			}
		}
		bl++;
		bitmask += d_sv_nints;
		if ((bitmask+(d_sv_nints-1)) >= (d_shared_q_size - CACHEOFFSET)) {
			bitmask = 0;
		}
	}
	return 2;
}

// hash functions use bj variable
#define FIRSTHASH(a, t)							{	hashtmp = 0; \
													for (bj = 0; bj < d_sv_nints; bj++) { \
														hashtmp += (indextype) (d_h[0]*(STRIPPEDSTATE(t, bj))+d_h[1]); \
													} \
													(a) = WARPSIZE*((inttype) ((hashtmp % P) % d_nrbuckets)); \
												}
#define FIRSTHASHHOST(a)						{	indextype hashtmp = 0; \
													hashtmp = (indextype) h[1]; \
													(a) = WARPSIZE*((inttype) ((hashtmp % P) % q_size/WARPSIZE)); \
												}
#define HASHALL(a, i, t)						{	hashtmp = 0; \
													for (bj = 0; bj < d_sv_nints; bj++) { \
														hashtmp += (indextype) (shared[HASHCONSTANTSOFFSET+(2*(i))]*(STRIPPEDSTATE(t, bj))+shared[HASHCONSTANTSOFFSET+(2*(i))+1]); \
													} \
													(a) = WARPSIZE*((inttype) ((hashtmp % P) % d_nrbuckets)); \
												}
#define HASHFUNCTION(a, i, t)					((HASHALL((a), (i), (t))))

#define COMPAREVECTORS(a, t1, t2)				{	(a) = 1; \
													for (bk = 0; bk < d_sv_nints-1; bk++) { \
														if ((t1)[bk] != (t2)[bk]) { \
															(a) = 0; break; \
														} \
													} \
													if ((a)) { \
														if (STRIPPEDSTATE((t1),bk) != STRIPPEDSTATE((t2),bk)) { \
															(a) = 0; \
														} \
													} \
												}

// check if bucket element associated with lane is a valid position to store data
#define LANEPOINTSTOVALIDBUCKETPOS						(HALFLANE < ((HALFWARPSIZE / d_sv_nints)*d_sv_nints))
//#define LANEPOINTSTOVALIDBUCKETPOS						true

__device__ inttype LANE_POINTS_TO_EL(inttype i)	{
	if (i < HALFWARPSIZE / d_sv_nints) {
		return (LANE >= i*d_sv_nints && LANE < (i+1)*d_sv_nints);
	}
	else {
		return (LANE >= HALFWARPSIZE+(i-(HALFWARPSIZE / d_sv_nints))*d_sv_nints && LANE < HALFWARPSIZE+(i-(HALFWARPSIZE / d_sv_nints)+1)*d_sv_nints);
	}
}

//__device__ inttype LANE_POINTS_TO_EL(inttype i)	{
//	return (LANE >= i*d_sv_nints && LANE < (i+1)*d_sv_nints);
//}

// start position of element i in bucket
#define STARTPOS_OF_EL_IN_BUCKET(i)			((i < (HALFWARPSIZE / d_sv_nints)) ? (i*d_sv_nints) : (HALFWARPSIZE + (i-(HALFWARPSIZE/d_sv_nints))*d_sv_nints))
//#define STARTPOS_OF_EL_IN_BUCKET(i)			(i*d_sv_nints)
#define STARTPOS_OF_EL_IN_BUCKET_HOST(i)	((i < (HALFWARPSIZE / sv_nints)) ? (i*sv_nints) : (HALFWARPSIZE + (i-(HALFWARPSIZE/sv_nints))*sv_nints))
//#define STARTPOS_OF_EL_IN_BUCKET_HOST(i)	(i*sv_nints)

// find or put element, single thread version.
__device__ inttype FINDORPUT_SINGLE(inttype* t, inttype* d_q, volatile inttype* d_newstate_flags) {
	inttype bi, bj, bk, bl;
	indextype hashtmp;
	for (bi = 0; bi < NR_HASH_FUNCTIONS; bi++) {
		HASHFUNCTION(hashtmp, bi, t);
		for (bj = 0; bj < NREL_IN_BUCKET; bj++) {
			bl = d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)+(d_sv_nints-1)];
			if (bl == EMPTYVECT32) {
				bl = atomicCAS(&d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)+(d_sv_nints-1)], EMPTYVECT32, t[d_sv_nints-1]);
				if (bl == EMPTYVECT32) {
					// Write was successful
					if (d_sv_nints > 1) {
						for (bk = 0; bk < d_sv_nints-1; bk++) {
							d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)+bk] = t[bk];
						}
					}
					threadfence();
					// There is work available for some block
					d_newstate_flags[(hashtmp / blockDim.x) % gridDim.x] = 1;
				}
			}
			if (bl != EMPTYVECT32) {
				COMPAREVECTORS(bk, &d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)], t); \
				if (bk == 1) {
					// Found state in global memory
					return 1;
				}
			}
			else {
				SETOLDSTATE(t);
				return 1;
			}
		}
	}
	return 0;
}

// find or put element, warp version. t is element stored in block cache
__device__ inttype FINDORPUT_WARP(inttype* t, inttype* d_q, volatile inttype* d_newstate_flags)	{
	inttype bi, bj, bk, bl, bitmask;
	indextype hashtmp;
	BucketEntryStatus threadstatus;
	// prepare bitmask once to reason about results of threads in the same (state vector) group
	bitmask = 0;
	if (LANEPOINTSTOVALIDBUCKETPOS) {
		SETBITS(LANE-ENTRY_ID, LANE-ENTRY_ID+d_sv_nints, bitmask);
	}
	for (bi = 0; bi < NR_HASH_FUNCTIONS; bi++) {
		HASHFUNCTION(hashtmp, bi, t);
		bl = d_q[hashtmp+LANE];
		bk = __ballot(STRIPPEDENTRY(bl, ENTRY_ID) == STRIPPEDSTATE(t, ENTRY_ID));
		// threadstatus is used to determine whether full state vector has been found
		threadstatus = EMPTY;
		if (LANEPOINTSTOVALIDBUCKETPOS) {
			if ((bk & bitmask) == bitmask) {
				threadstatus = FOUND;
			}
		}
		if (__ballot(threadstatus == FOUND) != 0) {
			// state vector has been found in bucket. mark local copy as old.
			if (LANE == 0) {
				SETOLDSTATE(t);
			}
			return 1;
		}
		// try to find empty position to insert new state vector
		threadstatus = (bl == EMPTYVECT32 && LANEPOINTSTOVALIDBUCKETPOS) ? EMPTY : TAKEN;
		// let bk hold the smallest index of an available empty position
		bk = __ffs(__ballot(threadstatus == EMPTY));
		while (bk != 0) {
			// write the state vector
			bk--;
			if (LANE >= bk && LANE < bk+d_sv_nints) {
				bl = atomicCAS(&(d_q[hashtmp+LANE]), EMPTYVECT32, t[ENTRY_ID]);
				if (bl == EMPTYVECT32) {
					// success
					if (ENTRY_ID == d_sv_nints-1) {
						SETOLDSTATE(t);
					}
					// try to claim the state vector for future work
					bl = OPENTILELEN;
					if (ENTRY_ID == d_sv_nints-1) {
						// try to increment the OPENTILECOUNT counter
						bl = atomicAdd((inttype *) &OPENTILECOUNT, d_sv_nints);
						if (bl < OPENTILELEN) {
							d_q[hashtmp+LANE] = t[d_sv_nints-1];
						} else {
							// There is work available for some block
							__threadfence();
							d_newstate_flags[(hashtmp / blockDim.x) % gridDim.x] = 1;
						}
					}
					// all active threads read the OPENTILECOUNT value of the first thread, and possibly store their part of the vector in the shared memory
					bl = __shfl(bl, LANE-ENTRY_ID+d_sv_nints-1);
					if (bl < OPENTILELEN) {
						// write part of vector to shared memory
						shared[OPENTILEOFFSET+bl+ENTRY_ID] = NEWSTATEPART(t, ENTRY_ID);
					}
					// write was successful. propagate this to the whole warp by setting threadstatus to FOUND
					threadstatus = FOUND;
				}
				else {
					// write was not successful. check if the state vector now in place equals the one we are trying to insert
					bk = __ballot(STRIPPEDENTRY(bl, ENTRY_ID) == STRIPPEDSTATE(t, ENTRY_ID));
					if ((bk & bitmask) == bitmask) {
						// state vector has been found in bucket. mark local copy as old.
						if (LANE == bk) {
							SETOLDSTATE(t);
						}
						// propagate this result to the whole warp
						threadstatus = FOUND;
					}
					else {
						// state vector is different, and position in bucket is taken
						threadstatus = TAKEN;
					}
				}
			}
			// check if the state vector was either encountered or inserted
			if (__ballot(threadstatus == FOUND) != 0) {
				return 1;
			}
			// recompute bk
			bk = __ffs(__ballot(threadstatus == EMPTY));
		}
	}
	return 0;
}

// find element, warp version. t is element stored in block cache
__device__ inttype FIND_WARP(inttype* t, inttype* d_q)	{
	inttype bi, bj, bk, bl, bitmask;
	indextype hashtmp;
	BucketEntryStatus threadstatus;
	// prepare bitmask once to reason about results of threads in the same (state vector) group
	bitmask = 0;
	if (LANEPOINTSTOVALIDBUCKETPOS) {
		SETBITS(LANE-ENTRY_ID, LANE-ENTRY_ID+d_sv_nints, bitmask);
	}
	for (bi = 0; bi < NR_HASH_FUNCTIONS; bi++) {
		HASHFUNCTION(hashtmp, bi, t);
		bl = d_q[hashtmp+LANE];
		bk = __ballot(STRIPPEDENTRY(bl, ENTRY_ID) == STRIPPEDSTATE(t, ENTRY_ID));
		// threadstatus is used to determine whether full state vector has been found
		threadstatus = EMPTY;
		if (LANEPOINTSTOVALIDBUCKETPOS) {
			if ((bk & bitmask) == bitmask) {
				threadstatus = FOUND;
			}
		}
		if (__ballot(threadstatus == FOUND) != 0) {
			// state vector has been found in bucket. mark local copy as old.
			if (threadstatus == FOUND & ISNEWINT(bl) == 0 & ENTRY_ID == d_sv_nints - 1) {
				SETOLDSTATE(t);
			}
			return __ballot(threadstatus == FOUND & ISNEWINT(bl) == 0 & ENTRY_ID == d_sv_nints - 1);
		}
		// try to find empty position
		threadstatus = (bl == EMPTYVECT32 && LANEPOINTSTOVALIDBUCKETPOS) ? EMPTY : TAKEN;
		// let bk hold the smallest index of an available empty position
		bk = __ffs(__ballot(threadstatus == EMPTY));
		if(bk < 32) {
			// There is an empty slot in this bucket and the state vector was not found
			// State will also not be found after rehashing, so we return 0
			return 0;
		}
	}
	return 0;
}

__device__ inttype FINDORPUT_WARP_ORIG(inttype* t, inttype* d_q, inttype bi, inttype bj, inttype bk, inttype bl, inttype bitmask, indextype hashtmp) {
	for (bi = 0; bi < NR_HASH_FUNCTIONS; bi++) {
		HASHFUNCTION(hashtmp, bi, t);
		bl = d_q[hashtmp+LANE];
		if (ENTRY_ID == (d_sv_nints-1)) {
			if (bl != EMPTYVECT32) {
				COMPAREVECTORS(bl, &d_q[hashtmp+LANE-(d_sv_nints-1)], (t));
				if (bl) {
					SETOLDSTATE((t));
				}
			}
		}
		if (ISNEWSTATE(t)) {
			for (bj = 0; bj < NREL_IN_BUCKET; bj++) {
				if (d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)+(d_sv_nints-1)] == EMPTYVECT32) {
					if (LANE == 0) {
						bl = atomicCAS(&d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)+(d_sv_nints-1)], EMPTYVECT32, t[d_sv_nints-1]);
						if (bl == EMPTYVECT32) {
							SETOLDSTATE(t);
							shared[THREADBUFFEROFFSET+WARP_ID] = OPENTILELEN;
							if (ITERATIONS < d_kernel_iters-1) {
								bk = atomicAdd((inttype *) &OPENTILECOUNT, d_sv_nints);
								if (bk < OPENTILELEN) {
									shared[THREADBUFFEROFFSET+WARP_ID] = bk;
									d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)+(d_sv_nints-1)] = t[d_sv_nints-1];
								}
							}
						}
					}
					if (!ISNEWSTATE(t)) {
						if (LANE < d_sv_nints - 1) {
							d_q[hashtmp+STARTPOS_OF_EL_IN_BUCKET(bj)+LANE] = t[LANE];
						}
						bk = shared[THREADBUFFEROFFSET+WARP_ID];
						if (bk != OPENTILELEN) {
							if (LANE < d_sv_nints) {
								shared[OPENTILEOFFSET+bk+LANE] = NEWSTATEPART(t, LANE);
							}
							if (LANE == 0) {
								shared[THREADBUFFEROFFSET+WARP_ID] = 0;
							}
						}
					}
				}
				if (!ISNEWSTATE((t))) {
					return 1;
				}
			}
		}
		if (!ISNEWSTATE((t))) {
			return 1;
		}
	}
	return 0;
}

// macro to print state vector
#define PRINTVECTOR(s) 							{	printf ("("); \
													for (bk = 0; bk < d_nr_procs; bk++) { \
														GETSTATEVECTORSTATE(bj, (s), bk) \
														printf ("%d", bj); \
														if (bk < (d_nr_procs-1)) { \
															printf (","); \
														} \
													} \
													printf (")\n"); \
												}


//#define INCRSTATEVECTOR(t)						(sv_nints == 1 ? t[0]++ : (t[0] == EMPTYVECTOR ? t[1]++ : t[0]++))
//#define DECRSTATEVECTOR(t)						(sv_nints == 1 ? t[0]-- : (t[0] == 0 ? (t[1]--; t[0] = EMPTYVECTOR) : t[0]--))

int vmem = 0;

// GPU textures
texture<inttype, 1, cudaReadModeElementType> tex_proc_offsets_start;
texture<inttype, 1, cudaReadModeElementType> tex_proc_offsets;
texture<inttype, 1, cudaReadModeElementType> tex_proc_trans_start;
texture<inttype, 1, cudaReadModeElementType> tex_proc_trans;
texture<inttype, 1, cudaReadModeElementType> tex_syncbits_offsets;
texture<inttype, 1, cudaReadModeElementType> tex_syncbits;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

//wrapper around cudaMalloc to count allocated memory and check for error while allocating
int cudaMallocCount ( void ** ptr,int size) {
	cudaError_t err = cudaSuccess;
	vmem += size;
	err = cudaMalloc(ptr,size);
	if (err) {
		printf("Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__);
		exit(1);
	}
	fprintf (stdout, "allocated %d\n", size);
	return size;
}

//test function to print a given state vector
void print_statevector(FILE* stream, inttype *state, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints) {
	inttype i, s, bitmask;

	for (i = 0; i < nr_procs; i++) {
		bitmask = 0;
		if (firstbit_statevector[i]/INTSIZE == firstbit_statevector[i+1]/INTSIZE) {
			SETBITS(firstbit_statevector[i] % INTSIZE,firstbit_statevector[i+1] % INTSIZE, bitmask);
			s = (state[firstbit_statevector[i]/INTSIZE] & bitmask) >> (firstbit_statevector[i] % INTSIZE);
		}
		else {
			SETBITS(0, firstbit_statevector[i+1] % INTSIZE, bitmask);
			s = (state[firstbit_statevector[i]/INTSIZE] >> (firstbit_statevector[i] % INTSIZE)
					| (state[firstbit_statevector[i+1]/INTSIZE] & bitmask) << (INTSIZE - (firstbit_statevector[i] % INTSIZE))); \
		}
		fprintf (stream, "%d", s);
		if (i < (nr_procs-1)) {
			fprintf (stream, ",");
		}
	}
	fprintf (stream, " ");
	for (i = 0; i < sv_nints; i++) {
		fprintf (stream, "%d ", STRIPPEDENTRY_HOST(state[i], i));
	}
	fprintf (stream, "\n");
}

//test function to print the contents of the device queue
void print_queue(inttype *d_q, inttype q_size, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints) {
	inttype *q_test = (inttype*) malloc(sizeof(inttype)*q_size);
	cudaMemcpy(q_test, d_q, q_size*sizeof(inttype), cudaMemcpyDeviceToHost);
	inttype nw;
	int count = 0;
	int newcount = 0;
	for (inttype i = 0; i < (q_size/WARPSIZE); i++) {
		for (inttype j = 0; j < NREL_IN_BUCKET_HOST; j++) {
			if (q_test[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)+(sv_nints-1)] != EMPTYVECT32) {
				count++;
				nw = ISNEWSTATE_HOST(&q_test[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)]);
				if (nw) {
					newcount++;
					fprintf (stdout, "new: ");
				}
				print_statevector(stdout, &(q_test[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)]), firstbit_statevector, nr_procs, sv_nints);
			}
		}
	}
	fprintf (stdout, "nr. of states in hash table: %d (%d unexplored states)\n", count, newcount);
}

//test function to print the contents of the device queue
void print_local_queue(FILE* stream, inttype *q, inttype q_size, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints) {
	int count = 0, newcount = 0;
	inttype nw;
	for (inttype i = 0; i < (q_size/WARPSIZE); i++) {
		for (inttype j = 0; j < NREL_IN_BUCKET_HOST; j++) {
			if (q[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)+(sv_nints-1)] != EMPTYVECT32) {
				count++;

//				if (j == 0) {
//					fprintf (stdout, "-----------\n");
//				}
				nw = ISNEWSTATE_HOST(&q[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)]);
				if (nw) {
					newcount++;
					fprintf (stream, "new: ");
					//print_statevector(&(q[(i*WARPSIZE)+(j*sv_nints)]), firstbit_statevector, nr_procs);
				}
				print_statevector(stream, &(q[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)]), firstbit_statevector, nr_procs, sv_nints);
			}
		}
	}
	fprintf (stream, "nr. of states in hash table: %d (%d unexplored states)\n", count, newcount);
}

//test function to count the contents of the device queue
void count_queue(inttype *d_q, inttype q_size, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints) {
	inttype *q_test = (inttype*) malloc(sizeof(inttype)*q_size);
	cudaMemcpy(q_test, d_q, q_size*sizeof(inttype), cudaMemcpyDeviceToHost);

	int count = 0;
	for (inttype i = 0; i < (q_size/WARPSIZE); i++) {
		for (inttype j = 0; j < NREL_IN_BUCKET_HOST; j++) {
			if (q_test[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)+(sv_nints-1)] != EMPTYVECT32) {
				count++;
			}
		}
	}
	fprintf (stdout, "nr. of states in hash table: %d\n", count);
}

//test function to count the contents of the host queue
void count_local_queue(inttype *q, inttype q_size, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints) {
	int count = 0, newcount = 0;
	inttype nw;
	inttype nrbuckets = q_size / WARPSIZE;
	inttype nrels = NREL_IN_BUCKET_HOST;
	for (inttype i = 0; i < nrbuckets; i++) {
		for (inttype j = 0; j < nrels; j++) {
			inttype elpos = STARTPOS_OF_EL_IN_BUCKET_HOST(j);
			inttype abselpos = (i*WARPSIZE)+elpos+sv_nints-1;
			inttype q_abselpos = q[abselpos];
			if (q_abselpos != EMPTYVECT32) {
				count++;
				nw = ISNEWSTATE_HOST(&q[(i*WARPSIZE)+elpos]);
				if (nw) {
					newcount++;
				}
			}
		}
	}
	fprintf (stdout, "nr. of states in hash table: %d (%d unexplored states)\n", count, newcount);
}

/**
 * CUDA kernel function to initialise the queue
 */
__global__ void init_queue(inttype *d_q, inttype n_elem) {
    inttype nthreads = blockDim.x*gridDim.x;
    inttype i = (blockIdx.x *blockDim.x) + threadIdx.x;

    for(; i < n_elem; i += nthreads) {
    	d_q[i] = (inttype) EMPTYVECT32;
    }
}

/**
 * CUDA kernel to store initial state in hash table
 */
__global__ void store_initial(inttype *d_q, inttype *d_h, inttype *d_newstate_flags, inttype blockdim, inttype griddim) {
	inttype bj;
	indextype hashtmp;
	inttype state[MAX_SIZE];

	for (bj = 0; bj < d_sv_nints; bj++) {
		state[bj] = 0;
	}
	SETNEWSTATE(state);
	FIRSTHASH(hashtmp, state);
	for (bj = 0; bj < d_sv_nints; bj++) {
		d_q[hashtmp+bj] = state[bj];
	}
	d_newstate_flags[(hashtmp / blockdim) % griddim] = 1;
}

/**
 * CUDA kernel function for BFS iteration state gathering
 * Order of data in the shared queue:
 * (0. index of process LTS states sizes)
 * (1. index of sync rules offsets)
 * (2. index of sync rules)
 * (1. index of open queue tile)
 * 0. the 'iterations' flag to count the number of iterations so far (nr of tiles processed by SM)
 * 1. the 'continue' flag for thread work
 * (4. index of threads buffer)
 * (5. index of hash table)
 * 2. constants for d_q hash functions (2 per function, in total 8 by default)
 * 3. state vector offsets (nr_procs+1 elements)
 * 4. sizes of states in process LTS states (nr_procs elements)
 * (9. sync rules + offsets (nr_syncbits_offsets + nr_syncbits elements))
 * 5. tile of open queue to be processed by block (sv_nints*(blockDim.x / nr_procs) elements)
 * 6. buffer for threads ((blockDim.x*max_buf_ints)+(blockDim.x/nr_procs) elements)
 * 7. hash table
 */
__global__ void
__launch_bounds__(512, 2)
gather(inttype *d_q, inttype *d_h, inttype *d_bits_state,
						inttype *d_firstbit_statevector, inttype *d_proc_offsets_start,
						inttype *d_proc_offsets, inttype *d_proc_trans, inttype *d_syncbits_offsets,
						inttype *d_syncbits, inttype *d_contBFS, inttype *d_property_violation,
						volatile inttype *d_newstate_flags, inttype scan) {
	//inttype global_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	//inttype group_nr = threadIdx.x / nr_procs;
	inttype i, k, l, index, offset1, offset2, tmp, cont, act, sync_offset1, sync_offset2;
	int32_t local_action_counter;
	inttype* src_state = &shared[OPENTILEOFFSET+(threadIdx.x/d_nr_procs)*d_sv_nints];
	inttype* tgt_state = &shared[TGTSTATEOFFSET+threadIdx.x*d_sv_nints];
	inttype bitmask, bi, bj, bk;
	int pos;
	// TODO: remove this
	inttype TMPVAR;
	// is at least one outgoing transition enabled for a given state (needed to detect deadlocks)
	inttype outtrans_enabled;

	for (i = threadIdx.x; i < d_shared_q_size; i += blockDim.x) {
		shared[i] = 0;
	}
	// Locally store the state sizes and syncbits
	i = threadIdx.x;
	if (i == 0) {
		ITERATIONS = 0;
		OPENTILECOUNT = 0;
		WORKSCANRESULT = 0;
		SCAN = 0;
	}
	if ((blockIdx.x*blockDim.x)+threadIdx.x == 0) {
		(*d_contBFS) = 0;
	}
	for (i = threadIdx.x; i < HASHCONSTANTSLEN; i += blockDim.x) {
		shared[i+HASHCONSTANTSOFFSET] = d_h[i];
	}
	for (i = threadIdx.x; i < VECTORPOSLEN; i += blockDim.x) {
		shared[i+VECTORPOSOFFSET] = d_firstbit_statevector[i];
	}
	for (i = threadIdx.x; i < LTSSTATESIZELEN; i += blockDim.x) {
		shared[i+LTSSTATESIZEOFFSET] = d_bits_state[i];
	}
	// Reset the open queue tile
	if (threadIdx.x < d_sv_nints*(blockDim.x / d_nr_procs)) {
		shared[OPENTILEOFFSET+threadIdx.x] = EMPTYVECT32;
	}
	// Clean the cache
	i = threadIdx.x;
	while (i < (d_shared_q_size - CACHEOFFSET)) {
		shared[CACHEOFFSET + i] = EMPTYVECT32;
		i += blockDim.x;
	}
	__syncthreads();
	if(scan) {
		//Copy the work tile from global mem
		if (threadIdx.x < OPENTILELEN) {
			shared[OPENTILEOFFSET+threadIdx.x] = d_q[d_nrbuckets*WARPSIZE + (OPENTILELEN+1) * blockIdx.x + threadIdx.x];
		}
		if(threadIdx.x == 0) {
			OPENTILECOUNT = d_q[d_nrbuckets*WARPSIZE + (OPENTILELEN+1) * blockIdx.x + OPENTILELEN];
		}
	}
	__syncthreads();
	inttype last_search_location = 0;
	while (ITERATIONS < d_kernel_iters) {
		if (threadIdx.x == 0 && OPENTILECOUNT < OPENTILELEN && d_newstate_flags[blockIdx.x]) {
			d_newstate_flags[blockIdx.x] = 2;
			SCAN = 1;
		}
		__syncthreads();
		// Scan the open set for work; we use the OPENTILECOUNT flag at this stage to count retrieved elements
		if (SCAN) {
			// This block should be able to find a new state
			int found_new_state = 0;
			for (i = GLOBAL_WARP_ID; i < d_nrbuckets && OPENTILECOUNT < OPENTILELEN; i += NR_WARPS) {
				int loc = i + last_search_location;
				if(loc >= d_nrbuckets) {
					last_search_location = -i + GLOBAL_WARP_ID;
					loc = i + last_search_location;
				}
				tmp = d_q[loc*WARPSIZE+LANE];
				l = EMPTYVECT32;
				if (ENTRY_ID == (d_sv_nints-1)) {
					if (ISNEWINT(tmp)) {
						found_new_state = 1;
						// try to increment the OPENTILECOUNT counter, if successful, store the state
						l = atomicAdd((uint32_t *) &OPENTILECOUNT, d_sv_nints);
						if (l < OPENTILELEN) {
							d_q[loc*WARPSIZE+LANE] = OLDINT(tmp);
						}
					}
				}
				// all threads read the OPENTILECOUNT value of the 'tail' thread, and possibly store their part of the vector in the shared memory
				if (LANEPOINTSTOVALIDBUCKETPOS) {
					l = __shfl(l, LANE-ENTRY_ID+d_sv_nints-1);
					if (l < OPENTILELEN) {
						// write part of vector to shared memory
						shared[OPENTILEOFFSET+l+ENTRY_ID] = tmp;
					}
				}
			}
			if(i < d_nrbuckets) {
				last_search_location = i - GLOBAL_WARP_ID;
			} else {
				last_search_location = 0;
			}
			if(found_new_state || i < d_nrbuckets) {
				WORKSCANRESULT = 1;
			}
		}
		__syncthreads();
		// if work has been retrieved, indicate this
		if (threadIdx.x == 0) {
			if (OPENTILECOUNT > 0) {
				(*d_contBFS) = 1;
			}
			if(SCAN && WORKSCANRESULT == 0 && d_newstate_flags[blockIdx.x] == 2) {
				// No new states were found by this block, save this information to prevent
				// unnecessary scanning later on
				d_newstate_flags[blockIdx.x] = 0;
			} else {
				WORKSCANRESULT = 0;
			}
			scan = 0;
		}
		// is the thread part of an 'active' group?
		offset1 = 0;
		offset2 = 0;
		if (threadIdx.x == 0) {
			OPENTILECOUNT = 0;
		}
		__syncthreads();
		if (THREADINGROUP) {
			// Is there work?
			if (ISSTATE(src_state)) {
				// Gather the required transition information for all states in the tile
				i = tex1Dfetch(tex_proc_offsets_start, GROUP_ID);
				// Determine process state
				GETSTATEVECTORSTATE(cont, src_state, GROUP_ID);
				// TODO: remove
				TMPVAR = cont;
				// Offset position
				index = cont/(INTSIZE/d_nbits_offset);
				pos = cont - (index*(INTSIZE/d_nbits_offset));
				tmp = tex1Dfetch(tex_proc_offsets, i+index);
				GETTRANSOFFSET(offset1, tmp, pos);
				if (pos == (INTSIZE/d_nbits_offset)-1) {
					tmp = tex1Dfetch(tex_proc_offsets, i+index+1);
					GETTRANSOFFSET(offset2, tmp, 0);
				}
				else {
					GETTRANSOFFSET(offset2, tmp, pos+1);
				}
			}
			if (GROUP_ID == 0) {
				// for later, when constructing successors for this state, set action counter to maximum
				THREADGROUPCOUNTER = (1 << d_bits_act);
				THREADGROUPPOR = 0;
			}
		}
		// iterate over the outgoing transitions of state 'cont'
		// variable cont is reused to indicate whether the buffer content of this thread still needs processing
		cont = 0;
		if (threadIdx.x == 0) {
			CONTINUE = 1;
		}
		__syncthreads();
		// while there is work to be done
		//int loopcounter = 0;
		outtrans_enabled = 0;
		local_action_counter = 0;
		while (CONTINUE == 1) {
			if (offset1 < offset2 || cont) {
				if (!cont) {
					// reset act
					act = (1 << (d_bits_act));
					// reset buffer of this thread
					for (l = 0; l < d_max_buf_ints; l++) {
						THREADBUFFERGROUPPOS(GROUP_ID, l) = 0;
					}
					// if not sync, store in hash table
					while (offset1 < offset2) {
						tmp = tex1Dfetch(tex_proc_trans, offset1);
						GETPROCTRANSSYNC(bitmask, tmp);
						if (bitmask == 0) {
							// no deadlock
							outtrans_enabled = 1;
							// construct state
							for (l = 0; l < d_sv_nints; l++) {
								tgt_state[l] = src_state[l];
							}
							for (l = 1; l <= NR_OF_STATES_IN_TRANSENTRY(GROUP_ID); l++) {
								GETPROCTRANSSTATE(pos, tmp, l, GROUP_ID);
								if (pos > 0) {
									SETSTATEVECTORSTATE(tgt_state, GROUP_ID, pos-1);
									// check for violation of safety property, if required
									if (d_property == SAFETY) {
										if (GROUP_ID == d_nr_procs-1) {
											// pos contains state id + 1
											// error state is state 1
											if (pos == 2) {
												// error state found
												(*d_property_violation) = 1;
											}
										}
									}
									// store tgt_state in cache; if i == d_shared_q_size, state was found, duplicate detected
									// if i == d_shared_q_size+1, cache is full, immediately store in global hash table
									k = STOREINCACHE(tgt_state, d_q, &bi);
									if (k == 2) {
										// cache time-out; store directly in global hash table
										if (FINDORPUT_SINGLE(tgt_state, d_q, d_newstate_flags) == 0) {
											// ERROR! hash table too full. Set CONTINUE to 2
											CONTINUE = 2;
										}
									} else if(local_action_counter != -1){
										// Only keep unique pointers in the buffer
										for(bk = bj = 0; bj < local_action_counter && bk == 0; bj++) {
											if((THREADBUFFERGROUPPOS(GROUP_ID, bj) & 0x7FFFFFFF) == bi) {
												bk = 1;
											}
										}
										if(bk == 0) {
											// Set the most-significant bit if we are not the owner
											THREADBUFFERGROUPPOS(GROUP_ID,local_action_counter) = bi | (k << 31);
											local_action_counter++;
										}
									}
								}
								else {
									break;
								}
							}
							offset1++;
						}
						else {
							break;
						}
					}

					// i is the current relative position in the buffer for this thread
					i = 0;
					if (offset1 < offset2) {
						GETPROCTRANSACT(act, tmp);
						// store transition entry
						THREADBUFFERGROUPPOS(GROUP_ID,i) = tmp;
						cont = 1;
						i++;
						offset1++;
						while (offset1 < offset2) {
							tmp = tex1Dfetch(tex_proc_trans, offset1);
							GETPROCTRANSACT(bitmask, tmp);
							if (act == bitmask) {
								THREADBUFFERGROUPPOS(GROUP_ID,i) = tmp;
								i++;
								offset1++;
							}
							else {
								break;
							}
						}
					} else {
						THREADGROUPPOR = 0x80000000 | d_nr_procs;
					}
				}
			}
			// All states following from local transitions
			// are now checked in global memory to see if
			// they are new.
			k = (d_shared_q_size-CACHEOFFSET)/d_sv_nints;
			int por_possible = 0;
			int32_t index = -1;
			int32_t c = 0;
			while(1) {
				if(__all(c >= local_action_counter)) {
					break;
				}
				if(c < local_action_counter) {
					index = THREADBUFFERGROUPPOS(GROUP_ID,c);
				}
				int active_lane = __ffs(__ballot((index & 0x80000000) == 0)) - 1;

				if(active_lane == -1) {
					c++;
					continue;
				}
				int cache_index = __shfl(index, active_lane);
				if(FIND_WARP((inttype*) &shared[CACHEOFFSET+cache_index], d_q) == 0) {
					por_possible |= active_lane == LANE;
				}
				if(active_lane == LANE || ((index & 0x80000000) != 0)) {
					c++;
					index = -1;
				}
			}
			__syncthreads();
			if(por_possible) {
				// At least one state following from a local
				// transition is new, report that POR
				// can be applied.
				atomicMin((unsigned int*)&THREADGROUPPOR, 0x80000000 | GROUP_ID);
			}
			__syncthreads();
			// Apply partial-order reduction by only retaining
			// new states from one thread as new.
			// Reset some variables to prevent further successor
			// generation.
			int do_por = 0;
			if(THREADINGROUP) {
				do_por = THREADGROUPPOR < (0x80000000 | d_nr_procs) && THREADGROUPPOR != 0;
			}
			__syncthreads();
			if(do_por) {
				// Cycle proviso is satisfied
				for(int32_t c = 0; c < local_action_counter; c++) {
					if((THREADGROUPPOR & 0x7FFFFFFF) != GROUP_ID) {
						SETOLDSTATE( &shared[CACHEOFFSET+ (THREADBUFFERGROUPPOS(GROUP_ID,c) & 0x7FFFFFFF)] );
					}
					THREADBUFFERGROUPPOS(GROUP_ID,c) = 0;
				}
				cont = 0;
				offset1 = offset2;
				if (THREADINGROUP && GROUP_ID == 0) {
					THREADGROUPCOUNTER = EXPLORATION_DONE;
					THREADGROUPPOR = 0;
				}
			}
			__syncthreads();
			if(do_por == 0){
				for(int32_t c = 0; c < local_action_counter; c++) {
					SETNEWSTATE( &shared[CACHEOFFSET+ (THREADBUFFERGROUPPOS(GROUP_ID,c) & 0x7FFFFFFF)] );
					THREADBUFFERGROUPPOS(GROUP_ID,c) = 0;
				}
			}
			local_action_counter = -1;
			// group leaders now need to set the counter to the next minimal action value.
			// To avoid bank conflicts afterwards when threads would need to read GETTHREADGROUPCOUNTER,
			// the leader disables the SYNC bit of transition entries for those threads which need
			// to perform work next. In this way, threads can determine locally that they should proceed
			// without reading the counter.
			__syncthreads();
			if (THREADINGROUP) {
				if (GROUP_ID == 0) {
					if (THREADGROUPCOUNTER != EXPLORATION_DONE) {
						pos = (1 << d_bits_act);
						for (i = 0; i < d_nr_procs; i++) {
							l = THREADBUFFERGROUPPOS(i,0);
							if (l != 0) {
								GETPROCTRANSACT(bitmask, l);
								if (THREADGROUPCOUNTER == (1 << d_bits_act)) {
									if (bitmask < pos) {
										pos = bitmask;
									}
								}
								else {
									if (bitmask < pos && bitmask > THREADGROUPCOUNTER) {
										pos = bitmask;
									}
								}
							}
						}
						// if pos = THREADGROUPCOUNTER, no more work for this state is required, now or in future iterations
						if (pos == THREADGROUPCOUNTER) {
							THREADGROUPCOUNTER = EXPLORATION_DONE;
						}
						// store the value
						if (pos < (1 << d_bits_act)) {
							THREADGROUPCOUNTER = pos;
						}
						// notify threads to work
						for (i = 0; i < d_nr_procs; i++) {
							l = THREADBUFFERGROUPPOS(i,0);
							if (l != 0) {
								GETPROCTRANSACT(bitmask, l);
								if (bitmask == THREADGROUPCOUNTER) {
									// notify
									SETPROCTRANSSYNC(THREADBUFFERGROUPPOS(i,0),0);
								}
							}
						}
					}
				}
			}
			__syncthreads();
			// only active threads should do something
			if (cont) {
				// Now, we have obtained the info needed to combine process transitions
				// if the sync bit has been disabled, come into action, creating successors
				GETPROCTRANSSYNC(index, THREADBUFFERGROUPPOS(GROUP_ID,0));
				if (index == 0) {
					// syncbits Offset position
					i = act/(INTSIZE/d_nbits_syncbits_offset);
					pos = act - (i*(INTSIZE/d_nbits_syncbits_offset));
					l = tex1Dfetch(tex_syncbits_offsets, i);
					GETSYNCOFFSET(sync_offset1, l, pos);
					if (pos == (INTSIZE/d_nbits_syncbits_offset)-1) {
						l = tex1Dfetch(tex_syncbits_offsets, i+1);
						GETSYNCOFFSET(sync_offset2, l, 0);
					}
					else {
						GETSYNCOFFSET(sync_offset2, l, pos+1);
					}
					// iterate through the relevant syncbit filters
					tmp = 1;
					for (;sync_offset1 < sync_offset2 && tmp; sync_offset1++) {
						index = tex1Dfetch(tex_syncbits, sync_offset1);
						for (i = 0; i < (INTSIZE/d_nr_procs); i++) {
							GETSYNCRULE(tmp, index, i);
							if (tmp != 0) {
								OWNSSYNCRULE(bitmask, tmp, GROUP_ID);
							}
							else {
								bitmask = 0;
							}
							if (bitmask) {
								// start combining entries in the buffer to create target states
								// if sync rule applicable, construct the first successor
								// copy src_state into tgt_state
								SYNCRULEISAPPLICABLE(l, tmp, act);
								if (l) {
									// source state is not a deadlock
									outtrans_enabled = 1;
									for (pos = 0; pos < d_sv_nints; pos++) {
										tgt_state[pos] = src_state[pos];
									}
									// construct first successor
									for (pos = 0; pos < d_nr_procs; pos++) {
										if (GETBIT(pos, tmp)) {
											// get first state
											GETPROCTRANSSTATE(k, THREADBUFFERGROUPPOS(pos,0), 1, pos);
											SETSTATEVECTORSTATE(tgt_state, pos, k-1);
										}
									}
									SETNEWSTATE(tgt_state);
									// while we keep getting new states, store them
									while (ISNEWSTATE(tgt_state)) {
										// check for violation of safety property, if required
										if (d_property == SAFETY) {
											GETSTATEVECTORSTATE(pos, tgt_state, d_nr_procs-1);
											if (pos == 1) {
												// error state found
												(*d_property_violation) = 1;
											}
										}

										// store tgt_state in cache; if i == d_shared_q_size, state was found, duplicate detected
										// if i == d_shared_q_size+1, cache is full, immediately store in global hash table
										TMPVAR = STOREINCACHE(tgt_state, d_q, &bitmask);
										if (TMPVAR == 2) {
											// cache time-out; store directly in global hash table
											if (FINDORPUT_SINGLE(tgt_state, d_q, d_newstate_flags) == 0) {
												// ERROR! hash table too full. Set CONTINUE to 2
												CONTINUE = 2;
											}
										}
										// get next successor
										for (pos = d_nr_procs-1; pos > (int) GROUP_ID-1; pos--) {
											if (GETBIT(pos,tmp)) {
												GETSTATEVECTORSTATE(cont, tgt_state, pos);
												int st = 0;
												for (k = 0; k < d_max_buf_ints; k++) {
													for (l = 1; l <= NR_OF_STATES_IN_TRANSENTRY(pos); l++) {
														GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,k), l, pos);
														if (cont == (st-1)) {
															break;
														}
													}
													if (cont == (st-1)) {
														break;
													}
												}
												// Assumption: element has been found (otherwise, 'last' was not a valid successor)
												// Try to get the next element
												if (l == NR_OF_STATES_IN_TRANSENTRY(pos)) {
													if (k >= d_max_buf_ints-1) {
														st = 0;
													}
													else {
														k++;
														l = 1;
													}
												}
												else {
													l++;
												}
												// Retrieve next element, insert it in 'tgt_state' if it is not 0, and return result, otherwise continue
												if (st != 0) {
													GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,k), l, pos);
													if (st > 0) {
														SETSTATEVECTORSTATE(tgt_state, pos, st-1);
														SETNEWSTATE(tgt_state);
														break;
													}
												}
												// else, set this process state to first one, and continue to next process
												GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,0), 1, pos);
												SETSTATEVECTORSTATE(tgt_state, pos, st-1);
											}
										}
										// did we find a successor? if not, set tgt_state to old
										if (pos == (int) GROUP_ID-1) {
											SETOLDSTATE(tgt_state);
										}
									}
								}
							}
						}
					}
					cont = 0;
				}
			}
			// finished an iteration of adding states.
			// Is there still work? (is another iteration required?)
			if (threadIdx.x == 0) {
				if (CONTINUE != 2) {
					CONTINUE = 0;
				}
			}
			__syncthreads();
			if (THREADINGROUP) {
				if ((offset1 < offset2) || cont) {
					if (CONTINUE != 2) {
						CONTINUE = 1;
					}
				}
			}
			// FOR TEST PURPOSES!
//			if (threadIdx.x == 0) {
//				CONTINUE++;
//			}
			__syncthreads();
		} // END WHILE CONTINUE == 1
		// have we encountered a deadlock state?
		// we use the shared memory to communicate this to the group leaders
		if (d_property == DEADLOCK) {
			if (THREADINGROUP) {
				if (ISSTATE(src_state)) {
					THREADBUFFERGROUPPOS(GROUP_ID, 0) = outtrans_enabled;
					// group leader collects results
					l = 0;
					if (GROUP_ID == 0) {
						for (i = 0; i < d_nr_procs; i++) {
							l += THREADBUFFERGROUPPOS(i, 0);
						}
						if (l == 0) {
							// deadlock state found
							(*d_property_violation) = 1;
						}
					}
				}
			}
		}
		// Reset the open queue tile
		if (threadIdx.x < OPENTILELEN) {
			shared[OPENTILEOFFSET+threadIdx.x] = EMPTYVECT32;
		}
		if (threadIdx.x == 0) {
			OPENTILECOUNT = 0;
		}
		__syncthreads();
		// start scanning the local cache and write results to the global hash table
		k = (d_shared_q_size-CACHEOFFSET)/d_sv_nints;
		int c;
		for (i = WARP_ID; i * WARPSIZE < k; i += (blockDim.x / WARPSIZE)) {
			int have_new_state = i * WARPSIZE + LANE < k && ISNEWSTATE(&shared[CACHEOFFSET+(i*WARPSIZE+LANE)*d_sv_nints]);
			while (c = __ballot(have_new_state)) {
				int active_lane = __ffs(c) - 1;
				if(FINDORPUT_WARP((inttype*) &shared[CACHEOFFSET + (i*WARPSIZE+active_lane)*d_sv_nints], d_q, d_newstate_flags) == 0) {
					CONTINUE = 2;
				}
				if (LANE == active_lane) {
					have_new_state = 0;
				}
			}
		}
		__syncthreads();
		// Ready to start next iteration, if error has not occurred
		if (threadIdx.x == 0) {
			if (CONTINUE == 2) {
				(*d_contBFS) = 2;
				ITERATIONS = d_kernel_iters;
			}
			else {
				ITERATIONS++;
			}
			CONTINUE = 0;
		}
		__syncthreads();
	}

	//Copy the work tile to global mem
	if (threadIdx.x < OPENTILELEN) {
		d_q[d_nrbuckets*WARPSIZE + (OPENTILELEN+1) * blockIdx.x + threadIdx.x] = shared[OPENTILEOFFSET+threadIdx.x];
	}
	if(threadIdx.x == 0) {
		d_q[d_nrbuckets*WARPSIZE + (OPENTILELEN+1) * blockIdx.x + OPENTILELEN] = OPENTILECOUNT;
	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char** argv) {
	FILE *fp;
	inttype nr_procs, bits_act, bits_statevector, sv_nints, nr_trans, proc_nrstates, nbits_offset, max_buf_ints, nr_syncbits_offsets, nr_syncbits, nbits_syncbits_offset;
	inttype *bits_state, *firstbit_statevector, *proc_offsets, *proc_trans, *proc_offsets_start, *syncbits_offsets, *syncbits;
	inttype contBFS;
	char stmp[BUFFERSIZE], fn[50];
	// to store constants for closed set hash functions
	int h[NR_HASH_FUNCTIONS*2];
	// size of global hash table
	size_t q_size = 0;
	PropertyStatus check_property = NONE;
	// nr of iterations in single kernel run
	int kernel_iters = KERNEL_ITERS;
	int nblocks = NR_OF_BLOCKS;
	int nthreadsperblock = BLOCK_SIZE;
	// level of verbosity (1=print level progress)
	int verbosity = 0;
	// clock to measure time
	clock_t start, stop;
	double runtime = 0.0;

	// Start timer
	assert((start = clock())!=-1);

	cudaDeviceProp prop;
	int nDevices;

	// GPU side versions of the input
	inttype *d_bits_state, *d_firstbit_statevector, *d_proc_offsets_start, *d_proc_offsets, *d_proc_trans, *d_syncbits_offsets, *d_syncbits, *d_h;
	// flag to keep track of progress and whether hash table errors occurred (value==2)
	inttype *d_contBFS;
	// flags to track which blocks have new states
	inttype *d_newstate_flags;
	// flag to keep track of property verification outcome
	inttype *d_property_violation;

	// GPU datastructures for calculation
	inttype *d_q;

	if (argc == 1) {
		fprintf(stderr, "ERROR: No input network given!\n");
		exit(1);
	}

	strcpy(fn, argv[1]);
	strcat(fn, ".gpf");

	int i = 2;
	while (i < argc) {
		printf ("%s\n", argv[i]);
		if (!strcmp(argv[i],"-k")) {
			// if nr. of iterations per kernel run is given, store it
			kernel_iters = atoi(argv[i+1]);
			i += 2;
		}
		else if (!strcmp(argv[i],"-b")) {
			// store nr of blocks to be used
			nblocks = atoi(argv[i+1]);
			i += 2;
		}
		else if (!strcmp(argv[i],"-t")) {
			// store nr of threads per block to be used
			nthreadsperblock = atoi(argv[i+1]);
			i += 2;
		}
		else if (!strcmp(argv[i],"-q")) {
			// store hash table size
			q_size = atoi(argv[i+1]);
			i += 2;
		}
		else if (!strcmp(argv[i],"-v")) {
			// store verbosity level
			verbosity = atoi(argv[i+1]);
			if (verbosity > 3) {
				verbosity = 3;
			}
			i += 2;
		}
		else if (!strcmp(argv[i],"-d")) {
			// check for deadlocks
			check_property = DEADLOCK;
			i += 1;
		}
		else if (!strcmp(argv[i],"-p")) {
			// check a property
			check_property = SAFETY;
			i += 1;
		}
	}

	fp = fopen(fn, "r");
	if (fp) {
		// Read the input
		fgets(stmp, BUFFERSIZE, fp);
		if (check_property == SAFETY) {
			i = atoi(stmp);
			fprintf(stdout, "Property to check is ");
			if (i == 0) {
				fprintf(stdout, "not ");
			}
			fprintf(stdout, "a liveness property\n");
			if (i == 1) {
				check_property = LIVENESS;
			}
		}
		fgets(stmp, BUFFERSIZE, fp);
		nr_procs = atoi(stmp);
		fprintf(stdout, "nr of procs: %d\n", nr_procs);
		fgets(stmp, BUFFERSIZE, fp);
		bits_act = atoi(stmp);
		fprintf(stdout, "nr of bits for transition label: %d\n", bits_act);
		fgets(stmp, BUFFERSIZE, fp);
		proc_nrstates = atoi(stmp);
		fprintf(stdout, "min. nr. of proc. states that fit in 32-bit integer: %d\n", proc_nrstates);
		fgets(stmp, BUFFERSIZE, fp);
		bits_statevector = atoi(stmp);
		fprintf(stdout, "number of bits needed for a state vector: %d\n", bits_statevector);
		firstbit_statevector = (inttype*) malloc(sizeof(inttype)*(nr_procs+1));
		for (int i = 0; i <= nr_procs; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			firstbit_statevector[i] = atoi(stmp);
			fprintf(stdout, "statevector offset %d: %d\n", i, firstbit_statevector[i]);
		}
		// determine the number of integers needed for a state vector
		sv_nints = (bits_statevector+31) / INTSIZE;
		bits_state = (inttype*) malloc(sizeof(inttype)*nr_procs);
		for (int i = 0; i < nr_procs; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			bits_state[i] = atoi(stmp);
			fprintf(stdout, "bits for states of process LTS %d: %d\n", i, bits_state[i]);
		}
		fgets(stmp, BUFFERSIZE, fp);
		nbits_offset = atoi(stmp);
		fprintf(stdout, "size of offset in process LTSs: %d\n", nbits_offset);
		fgets(stmp, BUFFERSIZE, fp);
		max_buf_ints = atoi(stmp);
		fprintf(stdout, "maximum label-bounded branching factor: %d\n", max_buf_ints);
		proc_offsets_start = (inttype*) malloc(sizeof(inttype)*(nr_procs+1));
		for (int i = 0; i <= nr_procs; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			proc_offsets_start[i] = atoi(stmp);
		}
		proc_offsets = (inttype*) malloc(sizeof(inttype)*proc_offsets_start[nr_procs]);
		for (int i = 0; i < proc_offsets_start[nr_procs]; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			proc_offsets[i] = atoi(stmp);
		}
		fgets(stmp, BUFFERSIZE, fp);
		nr_trans = atoi(stmp);
		fprintf(stdout, "total number of transition entries in network: %d\n", nr_trans);
		proc_trans = (inttype*) malloc(sizeof(inttype)*nr_trans);
		for (int i = 0; i < nr_trans; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			proc_trans[i] = atoi(stmp);
		}

		fgets(stmp, BUFFERSIZE, fp);
		nbits_syncbits_offset = atoi(stmp);
		//fprintf(stdout, "size of offset in sync rules: %d\n", nbits_syncbits_offset);
		fgets(stmp, BUFFERSIZE, fp);
		nr_syncbits_offsets = atoi(stmp);
		syncbits_offsets = (inttype*) malloc(sizeof(inttype)*nr_syncbits_offsets);
		for (int i = 0; i < nr_syncbits_offsets; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			syncbits_offsets[i] = atoi(stmp);
			//fprintf(stdout, "syncbits offset %d: %d\n", i, syncbits_offsets[i]);
		}
		fgets(stmp, BUFFERSIZE, fp);
		nr_syncbits = atoi(stmp);
		syncbits = (inttype*) malloc(sizeof(inttype)*nr_syncbits);
		for (int i = 0; i < nr_syncbits; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			syncbits[i] = atoi(stmp);
			//fprintf(stdout, "syncbits %d: %d\n", i, syncbits[i]);
		}
	}
	else {
		fprintf(stderr, "ERROR: input network does not exist!\n");
		exit(1);
	}

	// Randomly define the closed set hash functions
//	srand(time(NULL));
//	for (int i = 0; i < NR_HASH_FUNCTIONS*2; i++) {
//		h[i] = rand();
//	}
	// TODO: make random again
	h[0] = 483319424;
	h[1] = 118985421;
	h[2] = 1287157904;
	h[3] = 1162380012;
	h[4] = 1231274815;
	h[5] = 1344969351;
	h[6] = 527997957;
	h[7] = 735456672;
	h[8] = 1774251664;
	h[9] = 23102285;
	h[10] = 2089529600;
	h[11] = 2083003102;
	h[12] = 908039861;
	h[13] = 1913855526;
	h[14] = 1515282600;
	h[15] = 1691511413;

	// continue flags
	contBFS = 1;

	// Query the device properties and determine data structure sizes
	cudaGetDeviceCount(&nDevices);
	if (nDevices == 0) {
		fprintf (stderr, "ERROR: No CUDA compatible GPU detected!\n");
		exit(1);
	}
	cudaGetDeviceProperties(&prop, 0);
	fprintf (stdout, "global mem: %lu\n", (uint64_t) prop.totalGlobalMem);
	fprintf (stdout, "shared mem per block: %d\n", (int) prop.sharedMemPerBlock);
	fprintf (stdout, "max. threads per block: %d\n", (int) prop.maxThreadsPerBlock);
	fprintf (stdout, "max. grid size: %d\n", (int) prop.maxGridSize[0]);
	fprintf (stdout, "nr. of multiprocessors: %d\n", (int) prop.multiProcessorCount);

	// determine actual nr of blocks
	nblocks = MAX(1,MIN(prop.maxGridSize[0],nblocks));

	// Allocate memory on GPU
	cudaMallocCount((void **) &d_contBFS, sizeof(inttype));
	cudaMallocCount((void **) &d_property_violation, sizeof(inttype));
	cudaMallocCount((void **) &d_h, NR_HASH_FUNCTIONS*2*sizeof(inttype));
	cudaMallocCount((void **) &d_bits_state, nr_procs*sizeof(inttype));
	cudaMallocCount((void **) &d_firstbit_statevector, (nr_procs+1)*sizeof(inttype));
	cudaMallocCount((void **) &d_proc_offsets_start, (nr_procs+1)*sizeof(inttype));
	cudaMallocCount((void **) &d_proc_offsets, proc_offsets_start[nr_procs]*sizeof(inttype));
	cudaMallocCount((void **) &d_proc_trans, nr_trans*sizeof(inttype));
	cudaMallocCount((void **) &d_syncbits_offsets, nr_syncbits_offsets*sizeof(inttype));
	cudaMallocCount((void **) &d_syncbits, nr_syncbits*sizeof(inttype));
	cudaMallocCount((void **) &d_newstate_flags, nblocks*sizeof(inttype));

	// Copy data to GPU
	CUDA_CHECK_RETURN(cudaMemcpy(d_contBFS, &contBFS, sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_h, h, NR_HASH_FUNCTIONS*2*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_bits_state, bits_state, nr_procs*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_firstbit_statevector, firstbit_statevector, (nr_procs+1)*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_proc_offsets_start, proc_offsets_start, (nr_procs+1)*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_proc_offsets, proc_offsets, proc_offsets_start[nr_procs]*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_proc_trans, proc_trans, nr_trans*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_syncbits_offsets, syncbits_offsets, nr_syncbits_offsets*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(d_syncbits, syncbits, nr_syncbits*sizeof(inttype), cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemset(d_newstate_flags, 0, nblocks*sizeof(inttype)));

	// Bind data to textures
	cudaBindTexture(NULL, tex_proc_offsets_start, d_proc_offsets_start, (nr_procs+1)*sizeof(inttype));
	cudaBindTexture(NULL, tex_proc_offsets, d_proc_offsets, proc_offsets_start[nr_procs]*sizeof(inttype));
	cudaBindTexture(NULL, tex_proc_trans, d_proc_trans, nr_trans*sizeof(inttype));
	cudaBindTexture(NULL, tex_syncbits_offsets, d_syncbits_offsets, nr_syncbits_offsets*sizeof(inttype));
	cudaBindTexture(NULL, tex_syncbits, d_syncbits, nr_syncbits*sizeof(inttype));

	size_t available, total;
	cudaMemGetInfo(&available, &total);
	if (q_size == 0) {
		q_size = total / sizeof(inttype);
	}
	size_t el_per_Mb = Mb / sizeof(inttype);


	while(cudaMalloc((void**)&d_q,  q_size * sizeof(inttype)) == cudaErrorMemoryAllocation)	{
		q_size -= el_per_Mb;
		if( q_size  < el_per_Mb) {
			// signal no free memory
			break;
		}
	}

	fprintf (stdout, "global mem queue size: %lu, number of entries: %lu\n", q_size*sizeof(inttype), (indextype) q_size);

	inttype shared_q_size = (int) prop.sharedMemPerBlock / sizeof(inttype);
	fprintf (stdout, "shared mem queue size: %lu, number of entries: %u\n", shared_q_size*sizeof(inttype), shared_q_size);
	fprintf (stdout, "nr. of blocks: %d, block size: %d, nr of kernel iterations: %d\n", nblocks, nthreadsperblock, kernel_iters);

	// copy symbols
	inttype tablesize = q_size - nblocks * (sv_nints*(nthreadsperblock/nr_procs)+1);
	inttype nrbuckets = tablesize / WARPSIZE;
	cudaMemcpyToSymbol(d_nrbuckets, &nrbuckets, sizeof(inttype));
	cudaMemcpyToSymbol(d_shared_q_size, &shared_q_size, sizeof(inttype));
	cudaMemcpyToSymbol(d_nr_procs, &nr_procs, sizeof(inttype));
	cudaMemcpyToSymbol(d_max_buf_ints, &max_buf_ints, sizeof(inttype));
	cudaMemcpyToSymbol(d_sv_nints, &sv_nints, sizeof(inttype));
	cudaMemcpyToSymbol(d_bits_act, &bits_act, sizeof(inttype));
	cudaMemcpyToSymbol(d_nbits_offset, &nbits_offset, sizeof(inttype));
	cudaMemcpyToSymbol(d_nbits_syncbits_offset, &nbits_syncbits_offset, sizeof(inttype));
	cudaMemcpyToSymbol(d_kernel_iters, &kernel_iters, sizeof(inttype));
	cudaMemcpyToSymbol(d_property, &check_property, sizeof(inttype));

	// init the queue
	init_queue<<<nblocks, nthreadsperblock>>>(d_q, q_size);
	store_initial<<<1,1>>>(d_q, d_h, d_newstate_flags,nthreadsperblock,nblocks);
	for (int i = 0; i < 2*NR_HASH_FUNCTIONS; i++) {
		fprintf (stdout, "hash constant %d: %d\n", i, h[i]);
	}
	FIRSTHASHHOST(i);
	fprintf (stdout, "hash of initial state: %d\n", i);

	inttype zero = 0;
	inttype *q_test = (inttype*) malloc(sizeof(inttype)*tablesize);
	int j = 0;
	inttype scan = 0;
	CUDA_CHECK_RETURN(cudaMemcpy(d_property_violation, &zero, sizeof(inttype), cudaMemcpyHostToDevice))
	inttype property_violation = 0;
	while (contBFS == 1) {
		CUDA_CHECK_RETURN(cudaMemcpy(d_contBFS, &zero, sizeof(inttype), cudaMemcpyHostToDevice))
		gather<<<nblocks, nthreadsperblock, shared_q_size*sizeof(inttype)>>>(d_q, d_h, d_bits_state, d_firstbit_statevector, d_proc_offsets_start,
																		d_proc_offsets, d_proc_trans, d_syncbits_offsets, d_syncbits, d_contBFS, d_property_violation, d_newstate_flags, scan);
		// copy progress result
		//CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpy(&contBFS, d_contBFS, sizeof(inttype), cudaMemcpyDeviceToHost))
		if (check_property > 0) {
			CUDA_CHECK_RETURN(cudaMemcpy(&property_violation, d_property_violation, sizeof(inttype), cudaMemcpyDeviceToHost))
			if (property_violation == 1) {
				contBFS = 0;
			}
		}
		if (verbosity > 0) {
			if (verbosity == 1) {
				printf ("%d\n", j++);
			}
			else if (verbosity == 2) {
				cudaMemcpy(q_test, d_q, tablesize*sizeof(inttype), cudaMemcpyDeviceToHost);
				count_local_queue(q_test, tablesize, firstbit_statevector, nr_procs, sv_nints);
			}
			else if (verbosity == 3) {
				cudaMemcpy(q_test, d_q, tablesize*sizeof(inttype), cudaMemcpyDeviceToHost);
				print_local_queue(stdout, q_test, tablesize, firstbit_statevector, nr_procs, sv_nints);
			}
		}
		scan = 1;
	}
	// determine runtime
	stop = clock();
	runtime = (double) (stop-start)/CLOCKS_PER_SEC;
	fprintf (stdout, "Run time: %f\n", runtime);

	if (property_violation == 1) {
		switch (check_property) {
			case DEADLOCK:
				printf ("deadlock detected!\n");
				break;
			case SAFETY:
				printf ("safety property violation detected!\n");
				break;
			case LIVENESS:
				printf ("liveness property violation detected!\n");
				break;
		}
	}
	// report error if required
	if (contBFS == 2) {
		fprintf (stderr, "ERROR: problem with hash table\n");
	}
	count_queue(d_q, tablesize, firstbit_statevector, nr_procs, sv_nints);

	// Debugging functionality: print states to file
//	FILE* fout;
//	fout = fopen("/tmp/gpuexplore.debug", "w");
//	cudaMemcpy(q_test, d_q, tablesize*sizeof(inttype), cudaMemcpyDeviceToHost);
//	print_local_queue(fout, q_test, tablesize, firstbit_statevector, nr_procs, sv_nints);
//	fclose(fout);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
	//CUDA_CHECK_RETURN(cudaGetLastError());

	return 0;
}
