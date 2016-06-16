/*
 ============================================================================
 Name        : GPUexplore.cu
 Author      : Anton Wijs and Thomas Neele
 Version     :
 Copyright   : Copyright Anton Wijs and Thomas Neele
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
__constant__ inttype d_apply_por;
__constant__ inttype d_check_cycle_proviso;

// GPU shared memory array
extern __shared__ volatile inttype shared[];

// thread ids
#define WARP_ID							(threadIdx.x / WARPSIZE)
#define GLOBAL_WARP_ID					(((blockDim.x / WARPSIZE)*blockIdx.x)+WARP_ID)
#define NR_WARPS						((blockDim.x / WARPSIZE)*gridDim.x)
#define LANE							(threadIdx.x % WARPSIZE)
#define HALFLANE						(threadIdx.x % HALFWARPSIZE)
//#define ENTRY_ID						(LANE % d_sv_nints)
#define ENTRY_ID						(HALFLANE % d_sv_nints)
#define GROUP_ID						(LANE % d_nr_procs)
#define GROUP_GID						(WARP_ID * GROUPS_PER_WARP + LANE / d_nr_procs)
#define NR_GROUPS						((blockDim.x / WARPSIZE) * GROUPS_PER_WARP)
#define GROUPS_PER_WARP                 (WARPSIZE / d_nr_procs)
// Group id to lane and lane to group id macros
#define GTL(i)							(LANE - GROUP_ID + (i))
#define LTG(i)							((i) - (LANE - GROUP_ID))

//#define NREL_IN_BUCKET					((WARPSIZE / d_sv_nints))
#define NREL_IN_BUCKET					((HALFWARPSIZE / d_sv_nints)*2)
#define NREL_IN_BUCKET_HOST				((HALFWARPSIZE / sv_nints)*2)

// constant for cuckoo hashing (Alcantara et al)
static const inttype P = 979946131;
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
#define OPENTILELEN						(d_sv_nints*NR_GROUPS)
#define LASTSEARCHLEN					(blockDim.x/WARPSIZE)
#define TGTSTATELEN						(blockDim.x*d_sv_nints)
#define THREADBUFFERLEN					(NR_GROUPS*(THREADBUFFERSHARED+(d_nr_procs*d_max_buf_ints)))

#define HASHCONSTANTSOFFSET 			(SH_OFFSET)
#define VECTORPOSOFFSET 				(HASHCONSTANTSOFFSET+HASHCONSTANTSLEN)
#define LTSSTATESIZEOFFSET 				(VECTORPOSOFFSET+VECTORPOSLEN)
#define OPENTILEOFFSET 					(LTSSTATESIZEOFFSET+LTSSTATESIZELEN)
#define LASTSEARCHOFFSET				(OPENTILEOFFSET+OPENTILELEN)
#define TGTSTATEOFFSET		 			(LASTSEARCHOFFSET+LASTSEARCHLEN)
#define THREADBUFFEROFFSET	 			(TGTSTATEOFFSET+TGTSTATELEN)
#define CACHEOFFSET 					(THREADBUFFEROFFSET+THREADBUFFERLEN)

// One int for sync action counter
// One int for POR counter
#define THREADBUFFERSHARED				2
// parameter is thread id
#define THREADBUFFERGROUPSTART(i)		(THREADBUFFEROFFSET+ (((i) / WARPSIZE)*GROUPS_PER_WARP+(((i) % WARPSIZE) / d_nr_procs)) * (THREADBUFFERSHARED+(d_nr_procs*d_max_buf_ints)))
// parameter is group id
#define THREADBUFFERGROUPPOS(i, j)		shared[tbgs+THREADBUFFERSHARED+((i)*d_max_buf_ints)+(j)]
#define THREADGROUPCOUNTER				shared[tbgs]
#define THREADGROUPPOR					shared[tbgs + 1]

#define THREADINGROUP					(LANE < (GROUPS_PER_WARP)*d_nr_procs)

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

#define SETBIT(i, x)							{(x) = ((1<<(i)) | (x));}
#define GETBIT(i, x)							(((x) >> (i)) & 1)
#define SETBITS(i, j, x)						{(x) = (x) | (((1<<(j))-1)^((1<<(i))-1));}
#define GETBITS(x, y, start, len)				{(x) = ((y) >> (start)) & ((1 << (len)) - 1);}
#define GETPROCTRANSACT(a, t)					GETBITS(a, t, 1, d_bits_act)
#define GETPROCTRANSSYNC(a, t)					{(a) = ((t) & 1);}
#define GETPROCTRANSSTATE(a, t, i, j)			GETBITS(a, t, 1+d_bits_act+(i)*STATESIZE(j), STATESIZE(j))
#define GETTRANSOFFSET(a, t, i)					GETBITS(a, t, (i)*d_nbits_offset, d_nbits_offset)
#define GETSYNCOFFSET(a, t, i)					GETBITS(a, t, (i)*d_nbits_syncbits_offset, d_nbits_syncbits_offset)
//GETBITS(a, (t)[shared[VECTORPOSOFFSET+(i)]/INTSIZE], \
//		shared[VECTORPOSOFFSET+(i)] % INTSIZE, shared[LTSSTATESIZEOFFSET+(i)]);
#define GETSTATEVECTORSTATE(a, t, i)			{bitmask = 0; 	if (VECTORSTATEPOS(i)/INTSIZE == (VECTORSTATEPOS((i)+1)-1)/INTSIZE) { \
																	SETBITS((VECTORSTATEPOS(i) % INTSIZE), \
																			(((VECTORSTATEPOS((i)+1)-1) % INTSIZE)+1), bitmask); \
																	(a) = ((t)[VECTORSTATEPOS(i)/INTSIZE] & bitmask) >> (VECTORSTATEPOS(i) % INTSIZE); \
																} \
																else { \
																	SETBITS(0,(VECTORSTATEPOS((i)+1) % INTSIZE),bitmask); \
																	(a) = (t)[VECTORSTATEPOS(i)/INTSIZE] >> (VECTORSTATEPOS(i) % INTSIZE) \
																		 | \
																		((t)[VECTORSTATEPOS((i)+1)/INTSIZE] & bitmask) << \
																		(INTSIZE - (VECTORSTATEPOS(i) % INTSIZE)); \
																} \
												}
#define SETSTATEVECTORSTATE(t, i, x)			{bitmask = 0; 	if (VECTORSTATEPOS(i)/INTSIZE == (VECTORSTATEPOS((i)+1)-1)/INTSIZE) { \
																	SETBITS((VECTORSTATEPOS(i) % INTSIZE), \
																			(((VECTORSTATEPOS((i)+1)-1) % INTSIZE)+1),bitmask); \
																	(t)[VECTORSTATEPOS(i)/INTSIZE] = ((t)[VECTORSTATEPOS(i)/INTSIZE] & ~bitmask) | \
																	((x) << (VECTORSTATEPOS(i) % INTSIZE)); \
																} \
																else { \
																	SETBITS(0,(VECTORSTATEPOS(i) % INTSIZE), bitmask); \
																	(t)[VECTORSTATEPOS(i)/INTSIZE] = ((t)[VECTORSTATEPOS(i)/INTSIZE] & bitmask) | \
																	((x) << (VECTORSTATEPOS(i) % INTSIZE)); \
																	bitmask = -1 << (VECTORSTATEPOS((i)+1) % INTSIZE); \
																	(t)[VECTORSTATEPOS((i)+1)/INTSIZE] = ((t)[VECTORSTATEPOS((i)+1)/INTSIZE] & bitmask) | \
																		((x) >> (INTSIZE - (VECTORSTATEPOS(i) % INTSIZE))); \
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

#define SETPORSTATE(t)							{	(t)[(d_sv_nints-1)] = (t)[(d_sv_nints-1)] | 0x40000000;}
#define SETOTHERSTATE(t)						{	(t)[(d_sv_nints-1)] = (t)[(d_sv_nints-1)] & 0xBFFFFFFF;}
#define ISPORSTATE(t)							(ISPORINT((t)[(d_sv_nints-1)))
#define ISPORSTATE_HOST(t)						(ISPORINT((t)[(sv_nints-1)))
#define ISPORINT(t)								(((t) & 0x40000000) >> 30)
#define OTHERINT(t)								((t) & 0xBFFFFFFF)
#define PORINT(t)								((t) | 0x40000000)

#define STATE_FLAGS_MASK                        (d_apply_por ? 0x3FFFFFFF : 0x7FFFFFFF)
#define STRIPSTATE(t)							{(t)[(d_sv_nints-1)] = (t)[(d_sv_nints-1)] & STATE_FLAGS_MASK;}
#define STRIPPEDSTATE(t, i)						((i == d_sv_nints-1) ? ((t)[i] & STATE_FLAGS_MASK) : (t)[i])
#define STRIPPEDENTRY(t, i)						((i == d_sv_nints-1) ? ((t) & STATE_FLAGS_MASK) : (t))
#define STRIPPEDENTRY_HOST(t, i)				((i == sv_nints-1) ? ((t) & (apply_por ? 0x3FFFFFFF : 0x7FFFFFFF)) : (t))
#define NEWSTATEPART(t, i)						(((i) == d_sv_nints-1) ? ((t)[d_sv_nints-1] | 0x80000000) : (t)[(i)])
#define COMPAREENTRIES(t1, t2)					(((t1) & STATE_FLAGS_MASK) == ((t2) & STATE_FLAGS_MASK))
#define GETSYNCRULE(a, t, i)					GETBITS(a, t, (i)*d_nr_procs, d_nr_procs)

// HASH TABLE MACROS

// Return 0 if not found, bit 2 is flag for new state, bit 3 is a flag for POR state, 8 if cache is full
__device__ inttype STOREINCACHE(volatile inttype* t, inttype* cache, inttype* address) {
	inttype bi, bj, bk, bl, bitmask;
	indextype hashtmp;
	STRIPSTATE(t);
	hashtmp = 0;
	for (bi = 0; bi < d_sv_nints; bi++) {
		hashtmp += t[bi];
		hashtmp <<= 5;
	}
	bitmask = d_sv_nints*((inttype) (hashtmp % ((d_shared_q_size - CACHEOFFSET) / d_sv_nints)));
	SETNEWSTATE(t);
	bl = 0;
	while (bl < CACHERETRYFREQ) {
		bi = atomicCAS((inttype *) &cache[bitmask+(d_sv_nints-1)], EMPTYVECT32, t[d_sv_nints-1]);
		if (bi == EMPTYVECT32) {
			for (bj = 0; bj < d_sv_nints-1; bj++) {
				cache[bitmask+bj] = t[bj];
			}
			*address = bitmask;
			return 0;
		}
		if (COMPAREENTRIES(bi, t[d_sv_nints-1])) {
			if (d_sv_nints == 1) {
				*address = bitmask;
				return 1 + (ISNEWINT(bi) << 1) + (ISPORINT(bi) << 2);
			}
			else {
				for (bj = 0; bj < d_sv_nints-1; bj++) {
					if (cache[bitmask+bj] != (t)[bj]) {
						break;
					}
				}
				if (bj == d_sv_nints-1) {
					*address = bitmask;
					return 1 + (ISNEWINT(bi) << 1) + (ISPORINT(bi) << 2);
				}
			}
		}
		if (!ISNEWINT(bi)) {
			bj = atomicCAS((inttype *) &cache[bitmask+(d_sv_nints-1)], bi, t[d_sv_nints-1]);
			if (bi == bj) {
				for (bk = 0; bk < d_sv_nints-1; bk++) {
					cache[bitmask+bk] = t[bk];
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
	return 8;
}

// Mark the state in the cache according to markNew
// This function is used while applying POR to decide whether the cycle proviso
// is satisfied.
__device__ void MARKINCACHE(volatile inttype* t, inttype* cache, int markNew) {
	inttype bi, bj, bl, bitmask;
	indextype hashtmp;
	STRIPSTATE(t);
	hashtmp = 0;
	for (bi = 0; bi < d_sv_nints; bi++) {
		hashtmp += t[bi];
		hashtmp <<= 5;
	}
	bitmask = d_sv_nints*((inttype) (hashtmp % ((d_shared_q_size - CACHEOFFSET) / d_sv_nints)));
	SETNEWSTATE(t);
	bl = 0;
	while (bl < CACHERETRYFREQ) {
		bi = cache[bitmask+(d_sv_nints-1)];
		if (COMPAREENTRIES(bi, t[d_sv_nints-1])) {
			for (bj = 0; bj < d_sv_nints-1; bj++) {
				if (cache[bitmask+bj] != (t)[bj]) {
					break;
				}
			}
			if (bj == d_sv_nints-1) {
				if(markNew) {
					cache[bitmask+(d_sv_nints-1)] = NEWINT(OTHERINT(cache[bitmask+(d_sv_nints-1)] & STATE_FLAGS_MASK));
				} else if(ISPORINT(bi) && ISNEWINT(bi)){
					atomicCAS((inttype*) &cache[bitmask+(d_sv_nints-1)], bi, OLDINT(bi));
				}
				return;
			}
		}
		bl++;
		bitmask += d_sv_nints;
		if ((bitmask+(d_sv_nints-1)) >= (d_shared_q_size - CACHEOFFSET)) {
			bitmask = 0;
		}
	}
}

// hash functions use bj variable
#define FIRSTHASH(a, t)							{	hashtmp = 0; \
													for (bj = 0; bj < d_sv_nints; bj++) { \
														hashtmp += STRIPPEDSTATE(t,bj); \
														hashtmp <<= 5; \
													} \
													hashtmp = (indextype) (d_h[0]*hashtmp+d_h[1]); \
													(a) = WARPSIZE*((inttype) ((hashtmp % P) % d_nrbuckets)); \
												}
#define FIRSTHASHHOST(a)						{	indextype hashtmp = 0; \
													hashtmp = (indextype) h[1]; \
													(a) = WARPSIZE*((inttype) ((hashtmp % P) % q_size/WARPSIZE)); \
												}
#define HASHALL(a, i, t)						{	hashtmp = 0; \
													for (bj = 0; bj < d_sv_nints; bj++) { \
														hashtmp += STRIPPEDSTATE(t,bj); \
														hashtmp <<= 5; \
													} \
													hashtmp = (indextype) (shared[HASHCONSTANTSOFFSET+(2*(i))]*(hashtmp)+shared[HASHCONSTANTSOFFSET+(2*(i))+1]); \
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

__device__ inttype LANE_POINTS_TO_EL(inttype i)	{
	if (i < HALFWARPSIZE / d_sv_nints) {
		return (LANE >= i*d_sv_nints && LANE < (i+1)*d_sv_nints);
	}
	else {
		return (LANE >= HALFWARPSIZE+(i-(HALFWARPSIZE / d_sv_nints))*d_sv_nints && LANE < HALFWARPSIZE+(i-(HALFWARPSIZE / d_sv_nints)+1)*d_sv_nints);
	}
}

// start position of element i in bucket
#define STARTPOS_OF_EL_IN_BUCKET(i)			((i < (HALFWARPSIZE / d_sv_nints)) ? (i*d_sv_nints) : (HALFWARPSIZE + (i-(HALFWARPSIZE/d_sv_nints))*d_sv_nints))
#define STARTPOS_OF_EL_IN_BUCKET_HOST(i)	((i < (HALFWARPSIZE / sv_nints)) ? (i*sv_nints) : (HALFWARPSIZE + (i-(HALFWARPSIZE/sv_nints))*sv_nints))


// find or put element, warp version. t is element stored in block cache
__device__ inttype FINDORPUT_WARP(inttype* t, inttype* d_q, volatile inttype* d_newstate_flags, inttype claim_work)	{
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
						if (claim_work && (bl = atomicAdd((inttype *) &OPENTILECOUNT, d_sv_nints)) < OPENTILELEN) {
							d_q[hashtmp+LANE] = t[d_sv_nints-1];
						} else {
							// There is work available for some block
							__threadfence();
							d_newstate_flags[(hashtmp / blockDim.x) % gridDim.x] = 1;
						}
					}
					// all active threads read the OPENTILECOUNT value of the last thread, and possibly store their part of the vector in the shared memory
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
// return 0 if not found or found and new, 1 if found and old
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
			SETPORSTATE(t);
			return __ballot(threadstatus == FOUND & ISNEWINT(bl) == 0 & ENTRY_ID == d_sv_nints - 1);
		}
		// try to find empty position
		threadstatus = (bl == EMPTYVECT32 && LANEPOINTSTOVALIDBUCKETPOS) ? EMPTY : TAKEN;
		if(__any(threadstatus == EMPTY)) {
			// There is an empty slot in this bucket and the state vector was not found
			// State will also not be found after rehashing, so we return 0
			SETPORSTATE(t);
			return 0;
		}
	}
	SETPORSTATE(t);
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
void print_statevector(FILE* stream, inttype *state, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints, inttype apply_por) {
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
void print_queue(inttype *d_q, inttype q_size, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints, inttype apply_por) {
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
				print_statevector(stdout, &(q_test[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)]), firstbit_statevector, nr_procs, sv_nints, apply_por);
			}
		}
	}
	fprintf (stdout, "nr. of states in hash table: %d (%d unexplored states)\n", count, newcount);
}

//test function to print the contents of the device queue
void print_local_queue(FILE* stream, inttype *q, inttype q_size, inttype *firstbit_statevector, inttype nr_procs, inttype sv_nints, inttype apply_por) {
	int count = 0, newcount = 0;
	inttype nw;
	for (inttype i = 0; i < (q_size/WARPSIZE); i++) {
		for (inttype j = 0; j < NREL_IN_BUCKET_HOST; j++) {
			if (q[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)+(sv_nints-1)] != EMPTYVECT32) {
				count++;

				nw = ISNEWSTATE_HOST(&q[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)]);
				if (nw) {
					newcount++;
					fprintf (stream, "new: ");
				}
				print_statevector(stream, &(q[(i*WARPSIZE)+STARTPOS_OF_EL_IN_BUCKET_HOST(j)]), firstbit_statevector, nr_procs, sv_nints, apply_por);
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
 * Kernel that counts the amount of states in global memory
 */
__global__ void count_states(inttype *d_q, inttype *result) {
	if(threadIdx.x == 0) {
		shared[0] = 0;
	}
	__syncthreads();
	int localResult = 0;
	for(int i = GLOBAL_WARP_ID; i < d_nrbuckets; i += NR_WARPS) {
		int tmp = d_q[i*WARPSIZE+LANE];
		if (ENTRY_ID == (d_sv_nints-1) && tmp != EMPTYVECT32) {
			localResult++;
		}
	}
	atomicAdd((unsigned int*)shared, localResult);
	__syncthreads();
	if(threadIdx.x == 0) {
		atomicAdd(result, shared[0]);
	}
}

// When the cache overflows, use the whole warp to store states to global memory
__device__ void store_cache_overflow_warp(inttype *d_q, volatile inttype *d_newstate_flags, int has_overflow) {
	while(int c = __ballot(has_overflow)) {
		int active_lane = __ffs(c) - 1;
		int bj = FINDORPUT_WARP((inttype*) &shared[TGTSTATEOFFSET + (threadIdx.x-LANE+active_lane)*d_sv_nints], d_q, d_newstate_flags, 0);
		if(LANE == active_lane) {
			has_overflow = 0;
			if(bj == 0) {
				CONTINUE = 2;
			}
		}
	}
}

// Copy all states from the cache to global memory
__device__ void copy_cache_to_global(inttype *d_q, inttype* cache, volatile inttype *d_newstate_flags) {
	int k = (d_shared_q_size-CACHEOFFSET)/d_sv_nints;
	for (int i = WARP_ID; i * WARPSIZE < k; i += (blockDim.x / WARPSIZE)) {
		int have_new_state = i * WARPSIZE + LANE < k && ISNEWSTATE(&cache[(i*WARPSIZE+LANE)*d_sv_nints]);
		while (int c = __ballot(have_new_state)) {
			int active_lane = __ffs(c) - 1;
			if(FINDORPUT_WARP((inttype*) &cache[(i*WARPSIZE+active_lane)*d_sv_nints], d_q, d_newstate_flags, 1) == 0) {
				CONTINUE = 2;
			}
			if (LANE == active_lane) {
				have_new_state = 0;
			}
		}
	}
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
						volatile inttype *d_newstate_flags, inttype *d_worktiles, inttype scan) {
	inttype i, k, l, index, offset1, offset2, tmp, cont, act, sync_offset1, sync_offset2;
	volatile inttype* src_state = &shared[OPENTILEOFFSET+d_sv_nints*GROUP_GID];
	volatile inttype* tgt_state = &shared[TGTSTATEOFFSET+threadIdx.x*d_sv_nints];
	inttype* cache = (inttype*) &shared[CACHEOFFSET];
	inttype bitmask, bi;
	int pos;
	int tbgs = THREADBUFFERGROUPSTART(threadIdx.x);
	// TODO
	// is at least one outgoing transition enabled for a given state (needed to detect deadlocks)
	inttype outtrans_enabled;

	// Locally store the state sizes and syncbits
	if (threadIdx.x < SH_OFFSET) {
		shared[threadIdx.x] = 0;
	}
	for (i = threadIdx.x; i < HASHCONSTANTSLEN; i += blockDim.x) {
		shared[i+HASHCONSTANTSOFFSET] = d_h[i];
	}
	for (i = threadIdx.x; i < VECTORPOSLEN; i += blockDim.x) {
		VECTORSTATEPOS(i) = d_firstbit_statevector[i];
	}
	for (i = threadIdx.x; i < LTSSTATESIZELEN; i += blockDim.x) {
		STATESIZE(i) = d_bits_state[i];
	}
	// Clean the cache
	for (i = threadIdx.x; i < (d_shared_q_size - (cache-shared)); i += blockDim.x) {
		cache[i] = EMPTYVECT32;
	}
	if(scan) {
		// Copy the work tile from global mem
		if (threadIdx.x < OPENTILELEN + LASTSEARCHLEN) {
			shared[OPENTILEOFFSET+threadIdx.x] = d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + threadIdx.x];
		}
		if(threadIdx.x == 0) {
			OPENTILECOUNT = d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + OPENTILELEN + LASTSEARCHLEN];
		}
	} else if (threadIdx.x < OPENTILELEN+LASTSEARCHLEN) {
		// On first run: initialize the work tile to empty
		shared[OPENTILEOFFSET+threadIdx.x] = threadIdx.x < OPENTILELEN ? EMPTYVECT32 : 0;
	}
	__syncthreads();
	while (ITERATIONS < d_kernel_iters) {
		if (threadIdx.x == 0 && OPENTILECOUNT < OPENTILELEN && d_newstate_flags[blockIdx.x]) {
			// Indicate that we are scanning
			d_newstate_flags[blockIdx.x] = 2;
			SCAN = 1;
		}
		__syncthreads();
		// Scan the open set for work; we use the OPENTILECOUNT flag at this stage to count retrieved elements
		if (SCAN) {
			inttype last_search_location = shared[LASTSEARCHOFFSET + WARP_ID];
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
			if(LANE == 0) {
				shared[LASTSEARCHOFFSET + WARP_ID] = last_search_location;
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
				// Scanning has completed and no new states were found by this block,
				// save this information to prevent unnecessary scanning later on
				d_newstate_flags[blockIdx.x] = 0;
			} else {
				WORKSCANRESULT = 0;
			}
			scan = 0;
		}
		// is the thread part of an 'active' group?
		offset1 = 0;
		offset2 = 0;
		// Reset the whole thread buffer (shared + private)
		int start = THREADBUFFEROFFSET;
		int end = THREADBUFFEROFFSET + THREADBUFFERLEN;
		for(i = start + threadIdx.x; i < end; i+=blockDim.x) {
			shared[i] = 0;
		}
		if (THREADINGROUP) {
			// Is there work?
			if (ISSTATE(src_state)) {
				// Gather the required transition information for all states in the tile
				i = tex1Dfetch(tex_proc_offsets_start, GROUP_ID);
				// Determine process state
				GETSTATEVECTORSTATE(cont, src_state, GROUP_ID);
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
		}
		// iterate over the outgoing transitions of state 'cont'
		// variable cont is reused to indicate whether the buffer content of this thread still needs processing
		cont = 0;
		// while there is work to be done
		outtrans_enabled = 0;
		// if not sync, store in hash table
		// loop over all transentries
		while (1) {
			i = 1;
			if(offset1 < offset2) {
				tmp = tex1Dfetch(tex_proc_trans, offset1);
				GETPROCTRANSSYNC(i, tmp);
			}
			if (__any(i == 0)) {
				if(i == 0) {
					// no deadlock
					outtrans_enabled = 1;
					// construct state
					for (l = 0; l < d_sv_nints; l++) {
						tgt_state[l] = src_state[l];
					}
					offset1++;
				}
				// loop over this transentry
				for (l = 0; __any(i == 0 && l < NR_OF_STATES_IN_TRANSENTRY(GROUP_ID)); l++) {
					if(i == 0) {
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
							// store tgt_state in cache
							// if k == 8, cache is full, immediately store in global hash table
							k = STOREINCACHE(tgt_state, cache, &bi);
						} else {
							i = 1;
						}
					}
					store_cache_overflow_warp(d_q, d_newstate_flags, i == 0 && k == 8);
				}
			} else {
				break;
			}
		}
		act = 1 << d_bits_act;
		while (CONTINUE != 2 && __any(offset1 < offset2 || cont)) {

			// i is the current relative position in the buffer for this thread
			i = 0;
			if (offset1 < offset2 && !cont) {
				tmp = tex1Dfetch(tex_proc_trans, offset1);
				GETPROCTRANSACT(act, tmp);
				// store transition entry
				THREADBUFFERGROUPPOS(GROUP_ID,i) = tmp;
				cont = 1;
				i++;
				offset1++;
				while (i < d_max_buf_ints && offset1 < offset2) {
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
				while(i < d_max_buf_ints) {
					THREADBUFFERGROUPPOS(GROUP_ID,i) = 0;
					i++;
				}
			}
			int sync_act = act;
			if ((__ballot(cont) >> (LANE - GROUP_ID)) & ((1 << d_nr_procs) - 1)) {
				for(i = 1; i < d_nr_procs; i<<=1) {
					sync_act = min(__shfl(sync_act, GTL((GROUP_ID + i) % d_nr_procs)), sync_act);
				}
			}
			// Now, we have obtained the info needed to combine process transitions
			sync_offset1 = sync_offset2 = 0;
			int proc_enabled = (__ballot(act == sync_act) >> (LANE - GROUP_ID)) & ((1 << d_nr_procs) - 1);
			if(THREADINGROUP && sync_act < (1 << d_bits_act) && (__popc(proc_enabled) >= 2)) {
				// syncbits Offset position
				i = sync_act/(INTSIZE/d_nbits_syncbits_offset);
				pos = sync_act - (i*(INTSIZE/d_nbits_syncbits_offset));
				l = tex1Dfetch(tex_syncbits_offsets, i);
				GETSYNCOFFSET(sync_offset1, l, pos);
				if (pos == (INTSIZE/d_nbits_syncbits_offset)-1) {
					l = tex1Dfetch(tex_syncbits_offsets, i+1);
					GETSYNCOFFSET(sync_offset2, l, 0);
				}
				else {
					GETSYNCOFFSET(sync_offset2, l, pos+1);
				}
			}
			// iterate through the relevant syncbit filters
			tmp = 1;
			for (int j = GROUP_ID;__any(sync_offset1 + j / (INTSIZE/d_nr_procs) < sync_offset2 && tmp);) {

				tmp = 0;
				while(THREADINGROUP && !(tmp != 0 && (tmp & proc_enabled) == tmp) && sync_offset1 + j / (INTSIZE/d_nr_procs) < sync_offset2) {
					index = tex1Dfetch(tex_syncbits, sync_offset1 + j / (INTSIZE/d_nr_procs));
					GETSYNCRULE(tmp, index, j % (INTSIZE/d_nr_procs));
					j += d_nr_procs - __popc((__ballot(tmp != 0 && (tmp & proc_enabled) == tmp) >> (LANE - GROUP_ID)) & ((1 << GROUP_ID) - 1));
				}
				if(THREADINGROUP && j >= d_nr_procs - 1 && THREADGROUPCOUNTER < j) {
					atomicMax((inttype*) &THREADGROUPCOUNTER, j);
				}

				int work_remaining = 0;
				int has_second_succ = 0;
				if (tmp != 0 && (tmp & proc_enabled) == tmp) {
					// source state is not a deadlock
					outtrans_enabled = 1;
					// start combining entries in the buffer to create target states
					// if sync rule applicable, construct the first successor
					// copy src_state into tgt_state
					for (pos = 0; pos < d_sv_nints; pos++) {
						tgt_state[pos] = src_state[pos];
					}
					// construct first successor
					for (int rule = tmp; rule;) {
						pos = __ffs(rule) - 1;
						// get first state
						GETPROCTRANSSTATE(k, THREADBUFFERGROUPPOS(pos,0), 0, pos);
						SETSTATEVECTORSTATE(tgt_state, pos, k-1);
						GETPROCTRANSSTATE(k, THREADBUFFERGROUPPOS(pos,0), 1, pos);
						has_second_succ |= k;
						if(d_max_buf_ints > 1 && !k) {
							GETPROCTRANSSTATE(k, THREADBUFFERGROUPPOS(pos,1), 0, pos);
							has_second_succ |= k;
						}
						rule &= ~(1 << pos);
					}
					work_remaining = 1 + has_second_succ;
				}
				// while we keep getting new states, store them
				while (__any(work_remaining)) {
					l = 0;
					if(work_remaining) {
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
						l = STOREINCACHE(tgt_state, cache, &bitmask);
						if(work_remaining == 1) {
							// There will be no second successor
							work_remaining = 0;
						}
					}
					store_cache_overflow_warp(d_q, d_newstate_flags, l == 8);
					if(work_remaining) {
						// get next successor
						int rule;
						for (rule = tmp; rule;) {
							pos = __ffs(rule) - 1;
							int curr_st;
							GETSTATEVECTORSTATE(curr_st, tgt_state, pos);
							int st = 0;
							int num_states_in_trans = NR_OF_STATES_IN_TRANSENTRY(pos);
							// We search for the position of the current state in the buffer
							// We don't have to compare the last position: if curr_st has not been found yet,
							// then it has to be in the last position
							for (k = 0; k < d_max_buf_ints * num_states_in_trans - 1; k++) {
								GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,k / num_states_in_trans), k % num_states_in_trans, pos);
								if (curr_st == (st-1) || st == 0) {
									break;
								}
							}
							// Try to get the next element
							k++;
							if (k < d_max_buf_ints * num_states_in_trans && st != 0) {
								// Retrieve next element, insert it in 'tgt_state' if it is not 0, and return result, otherwise continue
								GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,k / num_states_in_trans), k % num_states_in_trans, pos);
								if (st > 0) {
									SETSTATEVECTORSTATE(tgt_state, pos, st-1);
									break;
								}
							}
							// else, set this process state to first one, and continue to next process
							if (d_max_buf_ints * num_states_in_trans > 1) {
								GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,0), 0, pos);
								SETSTATEVECTORSTATE(tgt_state, pos, st-1);
							}
							rule &= ~(1 << pos);
						}
						// did we find a successor? if not, all successors have been generated
						if (rule == 0) {
							work_remaining = 0;
						}
					}
				}

				j = THREADGROUPCOUNTER + GROUP_ID + 1;
			}

			// only active threads should reset 'cont'
			if (cont && sync_act == act) {
				cont = 0;
				act = 1 << d_bits_act;
				THREADGROUPCOUNTER = 0;
			}
		}

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
		int performed_work = OPENTILECOUNT != 0;
		__syncthreads();
		// Reset the work tile count
		if (threadIdx.x == 0) {
			OPENTILECOUNT = 0;
		}
		__syncthreads();
		// start scanning the local cache and write results to the global hash table
		if(performed_work) {
			copy_cache_to_global(d_q, cache, d_newstate_flags);
		}
		__syncthreads();
		// Write empty state vector to part of the work tile that is not used
		if (threadIdx.x < OPENTILELEN - OPENTILECOUNT) {
			shared[OPENTILEOFFSET+OPENTILECOUNT+threadIdx.x] = EMPTYVECT32;
		}
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
	if (threadIdx.x < OPENTILELEN+LASTSEARCHLEN) {
		d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + threadIdx.x] = shared[OPENTILEOFFSET+threadIdx.x];
	}
	if(threadIdx.x == 0) {
		d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + OPENTILELEN+LASTSEARCHLEN] = OPENTILECOUNT;
	}
}

__global__ void
__launch_bounds__(512, 2)
gather_por(inttype *d_q, inttype *d_h, inttype *d_bits_state,
						inttype *d_firstbit_statevector, inttype *d_proc_offsets_start,
						inttype *d_proc_offsets, inttype *d_proc_trans, inttype *d_syncbits_offsets,
						inttype *d_syncbits, inttype *d_contBFS, inttype *d_property_violation,
						volatile inttype *d_newstate_flags, inttype *d_worktiles, inttype scan) {
	inttype i, k, l, index, offset1, offset2, tmp, cont, act, sync_offset1, sync_offset2;
	volatile inttype* src_state = &shared[OPENTILEOFFSET+d_sv_nints*GROUP_GID];
	volatile inttype* tgt_state = &shared[TGTSTATEOFFSET+threadIdx.x*d_sv_nints];
	inttype* cache = (inttype*) &shared[CACHEOFFSET];
	inttype bitmask, bi, bj;
	int pos;
	int tbgs = THREADBUFFERGROUPSTART(threadIdx.x);
	// TODO: remove this
	inttype TMPVAR;
	// is at least one outgoing transition enabled for a given state (needed to detect deadlocks)
	inttype outtrans_enabled;

	// Locally store the state sizes and syncbits
	if (threadIdx.x < SH_OFFSET) {
		shared[threadIdx.x] = 0;
	}
	for (i = threadIdx.x; i < HASHCONSTANTSLEN; i += blockDim.x) {
		shared[i+HASHCONSTANTSOFFSET] = d_h[i];
	}
	for (i = threadIdx.x; i < VECTORPOSLEN; i += blockDim.x) {
		VECTORSTATEPOS(i) = d_firstbit_statevector[i];
	}
	for (i = threadIdx.x; i < LTSSTATESIZELEN; i += blockDim.x) {
		STATESIZE(i) = d_bits_state[i];
	}
	// Clean the cache
	for (i = threadIdx.x; i < (d_shared_q_size - CACHEOFFSET); i += blockDim.x) {
		cache[i] = EMPTYVECT32;
	}
	if(scan) {
		// Copy the work tile from global mem
		if (threadIdx.x < OPENTILELEN + LASTSEARCHLEN) {
			shared[OPENTILEOFFSET+threadIdx.x] = d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + threadIdx.x];
		}
		if(threadIdx.x == 0) {
			OPENTILECOUNT = d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + OPENTILELEN + LASTSEARCHLEN];
		}
	} else if (threadIdx.x < OPENTILELEN+LASTSEARCHLEN) {
		// On first run: initialize the work tile to empty
		shared[OPENTILEOFFSET+threadIdx.x] = threadIdx.x < OPENTILELEN ? EMPTYVECT32 : 0;
	}
	__syncthreads();
	while (ITERATIONS < d_kernel_iters) {
		if (threadIdx.x == 0 && OPENTILECOUNT < OPENTILELEN && d_newstate_flags[blockIdx.x]) {
			// Indicate that we are scanning
			d_newstate_flags[blockIdx.x] = 2;
			SCAN = 1;
		}
		__syncthreads();
		// Scan the open set for work; we use the OPENTILECOUNT flag at this stage to count retrieved elements
		if (SCAN) {
			inttype last_search_location = shared[LASTSEARCHOFFSET + WARP_ID];
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
			if(LANE == 0) {
				shared[LASTSEARCHOFFSET + WARP_ID] = last_search_location;
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
				// Scanning has completed and no new states were found by this block,
				// save this information to prevent unnecessary scanning later on
				d_newstate_flags[blockIdx.x] = 0;
			} else {
				WORKSCANRESULT = 0;
			}
			scan = 0;
		}
		// is the thread part of an 'active' group?
		offset1 = 0;
		offset2 = 0;
		// Reset the whole thread buffer (shared + private)
		int start = THREADBUFFEROFFSET;
		int end = THREADBUFFEROFFSET + THREADBUFFERLEN;
		for(i = start + threadIdx.x; i < end; i+=blockDim.x) {
			shared[i] = 0;
		}
		if (THREADINGROUP) {
			act = 1 << d_bits_act;
			// Is there work?
			if (ISSTATE(src_state)) {
				// Gather the required transition information for all states in the tile
				i = tex1Dfetch(tex_proc_offsets_start, GROUP_ID);
				// Determine process state
				GETSTATEVECTORSTATE(cont, src_state, GROUP_ID);
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
				THREADGROUPPOR = 0;
			}
		}
		// iterate over the outgoing transitions of state 'cont'
		// variable cont is reused to indicate whether the buffer content of this thread still needs processing
		cont = 0;
		// while there is work to be done
		outtrans_enabled = 0;
		char generate = 1;
		char proviso_satisfied = 0;
		int cluster_trans = 1 << GROUP_ID;
		int orig_offset1 = offset1;
		while(generate > -1) {
			while (CONTINUE != 2 && __any(offset1 < offset2 || cont)) {
				if (offset1 < offset2 && !cont) {
					// reset act
					act = (1 << (d_bits_act));
					// reset buffer of this thread
					for (l = 0; l < d_max_buf_ints; l++) {
						THREADBUFFERGROUPPOS(GROUP_ID, l) = 0;
					}
				}
				// if not sync, store in hash table
				// loop over all transentries
				while (1) {
					i = 1;
					if(offset1 < offset2  && !cont) {
						tmp = tex1Dfetch(tex_proc_trans, offset1);
						GETPROCTRANSSYNC(i, tmp);
					}
					if (__any(i == 0)) {
						if(i == 0) {
							// no deadlock
							outtrans_enabled = 1;
							// construct state
							for (l = 0; l < d_sv_nints; l++) {
								tgt_state[l] = src_state[l];
							}
							offset1++;
						}
						// loop over this transentry
						for (l = 0; __any(i == 0 && l < NR_OF_STATES_IN_TRANSENTRY(GROUP_ID)); l++) {
							if(i == 0) {
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

									if (!d_check_cycle_proviso) {
										// Set proviso to 1 to indicate at least one state has been found
										proviso_satisfied = 1;
									}
									// store tgt_state in cache
									// if k == 8, cache is full, immediately store in global hash table
									if(generate == 1) {
										k = STOREINCACHE(tgt_state, cache, &bi);
										if(k >> 2) {
											proviso_satisfied |= (k >> 1) & 1;
										} else if (!d_check_cycle_proviso) {
											SETPORSTATE(&cache[bi]);
										}
									} else {
										MARKINCACHE(tgt_state, cache, (THREADGROUPPOR >> GROUP_ID) & 1);
									}
								} else {
									i = 1;
								}
							}
							store_cache_overflow_warp(d_q, d_newstate_flags, i == 0 && k == 8);
							int c;
							// Check cycle proviso with the whole warp
							while(generate && d_check_cycle_proviso && (c = __ballot(i == 0 && (k >> 2 == 0)))) {
								int active_lane = __ffs(c) - 1;
								int cache_index = __shfl(bi, active_lane);
								bj = FIND_WARP((inttype*) &cache[cache_index], d_q);
								if(LANE == active_lane) {
									i = 1;
									if(bj == 0) {
										proviso_satisfied = 1;
									}
								}
							}
						}
					} else {
						break;
					}
				}

				// i is the current relative position in the buffer for this thread
				i = 0;
				if (offset1 < offset2 && !cont) {
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
				}
				int sync_act = cont ? act : (1 << d_bits_act);
				for(i = 1; i < d_nr_procs; i<<=1) {
					sync_act = min(__shfl(sync_act, GTL((GROUP_ID + i) % d_nr_procs)), sync_act);
				}
				// Now, we have obtained the info needed to combine process transitions
				sync_offset1 = sync_offset2 = 0;
				int proc_enabled = (__ballot(act == sync_act) >> (LANE - GROUP_ID)) & ((1 << d_nr_procs) - 1);
				if(THREADINGROUP && sync_act < (1 << d_bits_act)) {
					// syncbits Offset position
					i = sync_act/(INTSIZE/d_nbits_syncbits_offset);
					pos = sync_act - (i*(INTSIZE/d_nbits_syncbits_offset));
					l = tex1Dfetch(tex_syncbits_offsets, i);
					GETSYNCOFFSET(sync_offset1, l, pos);
					if (pos == (INTSIZE/d_nbits_syncbits_offset)-1) {
						l = tex1Dfetch(tex_syncbits_offsets, i+1);
						GETSYNCOFFSET(sync_offset2, l, 0);
					}
					else {
						GETSYNCOFFSET(sync_offset2, l, pos+1);
					}
				}
				// iterate through the relevant syncbit filters
				tmp = 1;
				for (int j = GROUP_ID;__any(sync_offset1 + j / (INTSIZE/d_nr_procs) < sync_offset2 && tmp); j+=d_nr_procs) {
					index = 0;
					if(THREADINGROUP && sync_act < (1 << d_bits_act) && sync_offset1 + j / (INTSIZE/d_nr_procs) < sync_offset2 && tmp) {
						index = tex1Dfetch(tex_syncbits, sync_offset1 + j / (INTSIZE/d_nr_procs));
					}
					SETOLDSTATE(tgt_state);
					int has_second_succ = 0;
					GETSYNCRULE(tmp, index, j % (INTSIZE/d_nr_procs));
					if (tmp != 0 && (tmp & proc_enabled) == tmp) {
						// source state is not a deadlock
						outtrans_enabled = 1;
						// start combining entries in the buffer to create target states
						// if sync rule applicable, construct the first successor
						// copy src_state into tgt_state
						for (pos = 0; pos < d_sv_nints; pos++) {
							tgt_state[pos] = src_state[pos];
						}
						// construct first successor
						for (int rule = tmp; rule;) {
							pos = __ffs(rule) - 1;
							// get first state
							GETPROCTRANSSTATE(k, THREADBUFFERGROUPPOS(pos,0), 0, pos);
							SETSTATEVECTORSTATE(tgt_state, pos, k-1);
							GETPROCTRANSSTATE(k, THREADBUFFERGROUPPOS(pos,0), 1, pos);
							has_second_succ |= k;
							if(d_max_buf_ints > 1 && !k) {
								GETPROCTRANSSTATE(k, THREADBUFFERGROUPPOS(pos,1), 0, pos);
								has_second_succ |= k;
							}
							rule &= ~(1 << pos);
						}
						SETNEWSTATE(tgt_state);
					}
					int rule_proviso = 0;
					// while we keep getting new states, store them
					while (__any(ISNEWSTATE(tgt_state))) {
						l = k = TMPVAR = bitmask = 0;
						if(ISNEWSTATE(tgt_state)) {
							// check for violation of safety property, if required
							if (d_property == SAFETY) {
								GETSTATEVECTORSTATE(pos, tgt_state, d_nr_procs-1);
								if (pos == 1) {
									// error state found
									(*d_property_violation) = 1;
								}
							}

							if (!d_check_cycle_proviso) {
								// Set rule_proviso to 1 to indicate at least one state has been found
								rule_proviso = 1;
							}
							// store tgt_state in cache; if i == d_shared_q_size, state was found, duplicate detected
							// if i == d_shared_q_size+1, cache is full, immediately store in global hash table
							if(generate == 1) {
								TMPVAR = STOREINCACHE(tgt_state, cache, &bitmask);
								if(TMPVAR >> 2) {
									rule_proviso |= (TMPVAR >> 1) & 1;
								} else if (!d_check_cycle_proviso) {
									SETPORSTATE(&cache[bitmask]);
								}
							} else {
								MARKINCACHE(tgt_state, cache, (THREADGROUPPOR & tmp) == tmp);
							}
							l = 1;
							k = has_second_succ;
							if(!has_second_succ) {
								SETOLDSTATE(tgt_state);
							}
						}
						store_cache_overflow_warp(d_q, d_newstate_flags, l && TMPVAR == 8);
						int c;
						// Check cycle proviso with the whole warp
						while(generate && d_check_cycle_proviso && (c = __ballot(l && (TMPVAR >> 2 == 0)))) {
							int active_lane = __ffs(c) - 1;
							int cache_index = __shfl(bitmask, active_lane);
							bj = FIND_WARP((inttype*) &cache[cache_index], d_q);
							if(LANE == active_lane) {
								l = 0;
								if(bj == 0) {
									rule_proviso = 1;
								}
							}
						}
						if(k) {
							// get next successor
							int rule;
							for (rule = tmp; rule;) {
								pos = __ffs(rule) - 1;
								int curr_st;
								GETSTATEVECTORSTATE(curr_st, tgt_state, pos);
								int st = 0;
								for (k = 0; k < d_max_buf_ints; k++) {
									for (l = 0; l < NR_OF_STATES_IN_TRANSENTRY(pos); l++) {
										GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,k), l, pos);
										if (curr_st == (st-1)) {
											break;
										}
									}
									if (curr_st == (st-1)) {
										break;
									}
								}
								// Assumption: element has been found (otherwise, 'last' was not a valid successor)
								// Try to get the next element
								if (l == NR_OF_STATES_IN_TRANSENTRY(pos) - 1) {
									if (k >= d_max_buf_ints-1) {
										st = 0;
									}
									else {
										k++;
										l = 0;
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
								GETPROCTRANSSTATE(st, THREADBUFFERGROUPPOS(pos,0), 0, pos);
								SETSTATEVECTORSTATE(tgt_state, pos, st-1);
								rule &= ~(1 << pos);
							}
							// did we find a successor? if not, set tgt_state to old
							if (rule == 0) {
								SETOLDSTATE(tgt_state);
							}
						}
					}
					for (l = 0; l < d_nr_procs; l++) {
						// Exchange the sync rules so every thread can update its cluster_trans
						int sync_rule = __shfl(tmp, GTL((GROUP_ID + l) % d_nr_procs));
						int proviso = __shfl(rule_proviso, GTL((GROUP_ID + l) % d_nr_procs));
						if(GETBIT(GROUP_ID, sync_rule) && sync_act == act) {
							cluster_trans |= sync_rule;
							proviso_satisfied |= proviso;
						}
					}
				}

				// only active threads should reset 'cont'
				if (cont && sync_act == act) {
					cont = 0;
				}
			} // END WHILE CONTINUE == 1

			if(generate == 1 && THREADINGROUP) {
				// Choose a cluster for reduction
				if(!proviso_satisfied) {
					cluster_trans = cluster_trans & ~(1 << GROUP_ID);
				}
				THREADBUFFERGROUPPOS(GROUP_ID,0) = cluster_trans;
				__syncthreads();
				proviso_satisfied = 0;
				int to_check = cluster_trans;
				while (to_check) {
					i = __ffs(to_check) - 1;
					to_check &= ~(1 << i);
					int cluster = THREADBUFFERGROUPPOS(i, 0);
					proviso_satisfied |= GETBIT(i, cluster);
					to_check |= cluster & ~cluster_trans & ~(1 << i);
					cluster_trans |= cluster;
				}
				__syncthreads();
				if(!proviso_satisfied) {
					THREADBUFFERGROUPPOS(GROUP_ID,0) = 0;
				} else {
					THREADBUFFERGROUPPOS(GROUP_ID,0) = cluster_trans;
				}
				__syncthreads();
				if(GROUP_ID == 0) {
					int min = d_nr_procs;
					int cluster = 0xFFFFFFFF >> (INTSIZE - d_nr_procs);
					for(i = 0; i < d_nr_procs; i++) {
						if(THREADBUFFERGROUPPOS(i,0) > 0 && __popc(THREADBUFFERGROUPPOS(i,0)) < min) {
							min = __popc(THREADBUFFERGROUPPOS(i,0));
							cluster = THREADBUFFERGROUPPOS(i,0);
						}
					}
					THREADGROUPPOR = cluster;
					if(cluster < (0xFFFFFFFF >> (INTSIZE - d_nr_procs))) {
//						printf("Selected cluster %d for POR\n",cluster);
					}
				}
				__syncthreads();
			}
			offset1 = orig_offset1;
			generate--;
		} // END while(generate > -1)

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
		int performed_work = OPENTILECOUNT != 0;
		__syncthreads();
		// Reset the open queue tile
		if (threadIdx.x < OPENTILELEN) {
			shared[OPENTILEOFFSET+threadIdx.x] = EMPTYVECT32;
		}
		if (threadIdx.x == 0) {
			OPENTILECOUNT = 0;
		}
		__syncthreads();
		// start scanning the local cache and write results to the global hash table
		if(performed_work) {
			copy_cache_to_global(d_q, cache, d_newstate_flags);
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
	if (threadIdx.x < OPENTILELEN+LASTSEARCHLEN) {
		d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + threadIdx.x] = shared[OPENTILEOFFSET+threadIdx.x];
	}
	if(threadIdx.x == 0) {
		d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * blockIdx.x + OPENTILELEN+LASTSEARCHLEN] = OPENTILECOUNT;
	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char** argv) {
	FILE *fp;
	inttype nr_procs, bits_act, bits_statevector, sv_nints, nr_trans, proc_nrstates, nbits_offset, max_buf_ints, nr_syncbits_offsets, nr_syncbits, nbits_syncbits_offset;
	inttype *bits_state, *firstbit_statevector, *proc_offsets, *proc_trans, *proc_offsets_start, *syncbits_offsets, *syncbits;
	inttype contBFS, counted_states;
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
	// POR options
	int apply_por = 0;
	int use_cycle_proviso = 0;
	// level of verbosity (1=print level progress)
	int verbosity = 0;
	char* dump_file = NULL;
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
	// Integer to store the amount of states counted in the hash table
	inttype *d_counted_states;
	// Space to temporarily store work tiles
	inttype *d_worktiles;

	// GPU datastructures for calculation
	inttype *d_q;

	const char* help_text =
		"Usage: GPUexplore <model> [OPTIONS]\n"
		"Run state-space exploration on model (do not include the file extension).\n"
		"options:\n"
		"  -d                 Check for deadlocks\n"
		"  -p                 Check a safety property (should be embedded in the model)\n"
		"  --por              Apply partial-order reduction\n"
		"  --cycle-proviso    Apply the cycle proviso during partial-order reduction\n"
		"  -k NUM             Run NUM iterations per kernel launch (default 1)\n"
		"  -b NUM             Run the kernel on NUM blocks (default 1)\n"
		"  -t NUM             Use NUM threads per block (default 32)\n"
		"  -q NUM             Allocate NUM integers for the hash table\n"
		"  --dump FILE        Dump the state space to FILE after completing the exploration\n"
		"  -v NUM             Change the verbosity:\n"
		"                        0 - minimal output\n"
		"                        1 - print sequence number of each kernel launch\n"
		"                        2 - print number of states in the hash table after each kernel launch\n"
		"                        3 - print state vectors after each kernel launch\n"
		"  -h, --help         Show this help message\n";

	if (argc == 1) {
		fprintf(stderr, "ERROR: No input network given!\n");
		fprintf(stdout, help_text);
		exit(1);
	} else if(!strcmp(argv[1],"--help") || !strcmp(argv[1],"-h") || !strcmp(argv[1],"-?")) {
		fprintf(stdout, help_text);
		exit(0);
	}

	strcpy(fn, argv[1]);
	strcat(fn, ".gpf");

	int i = 2;
	while (i < argc) {
		if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h") || !strcmp(argv[i],"-?")) {
			fprintf(stdout, help_text);
			exit(0);
		}
		else if (!strcmp(argv[i],"-k")) {
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
			use_cycle_proviso = 0;
			i += 1;
		}
		else if (!strcmp(argv[i],"-p")) {
			// check a property
			check_property = SAFETY;
			use_cycle_proviso = 1;
			i += 1;
		}
		else if (!strcmp(argv[i],"--por")) {
			// apply partial-order reduction
			apply_por = 1;
			i += 1;
		}
		else if (!strcmp(argv[i],"--cycle-proviso")) {
			// use cycle proviso
			if (check_property == NONE) {
				use_cycle_proviso = 1;
			}
			i += 1;
		}
		else if (!strcmp(argv[i],"--dump")) {
			dump_file = argv[i+1];
			i += 2;
		} else {
			fprintf(stderr, "ERROR: unrecognized option %s\n", argv[i]);
			fprintf(stdout, help_text);
			exit(1);
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
		bits_statevector = atoi(stmp) + apply_por;
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
		fgets(stmp, BUFFERSIZE, fp);
		nr_syncbits_offsets = atoi(stmp);
		syncbits_offsets = (inttype*) malloc(sizeof(inttype)*nr_syncbits_offsets);
		for (int i = 0; i < nr_syncbits_offsets; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			syncbits_offsets[i] = atoi(stmp);
		}
		fgets(stmp, BUFFERSIZE, fp);
		nr_syncbits = atoi(stmp);
		syncbits = (inttype*) malloc(sizeof(inttype)*nr_syncbits);
		for (int i = 0; i < nr_syncbits; i++) {
			fgets(stmp, BUFFERSIZE, fp);
			syncbits[i] = atoi(stmp);
		}
	}
	else {
		fprintf(stderr, "ERROR: input network does not exist!\n");
		exit(1);
	}

	// Randomly define the closed set hash functions
	srand(time(NULL));
	for (int i = 0; i < NR_HASH_FUNCTIONS*2; i++) {
		h[i] = rand();
	}

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
	cudaMallocCount((void **) &d_counted_states, sizeof(inttype));
	cudaMallocCount((void **) &d_h, NR_HASH_FUNCTIONS*2*sizeof(inttype));
	cudaMallocCount((void **) &d_bits_state, nr_procs*sizeof(inttype));
	cudaMallocCount((void **) &d_firstbit_statevector, (nr_procs+1)*sizeof(inttype));
	cudaMallocCount((void **) &d_proc_offsets_start, (nr_procs+1)*sizeof(inttype));
	cudaMallocCount((void **) &d_proc_offsets, proc_offsets_start[nr_procs]*sizeof(inttype));
	cudaMallocCount((void **) &d_proc_trans, nr_trans*sizeof(inttype));
	cudaMallocCount((void **) &d_syncbits_offsets, nr_syncbits_offsets*sizeof(inttype));
	cudaMallocCount((void **) &d_syncbits, nr_syncbits*sizeof(inttype));
	cudaMallocCount((void **) &d_newstate_flags, nblocks*sizeof(inttype));
	cudaMallocCount((void **) &d_worktiles, nblocks * (sv_nints*(nthreadsperblock/nr_procs)+nthreadsperblock/WARPSIZE+1)*sizeof(inttype));


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
	CUDA_CHECK_RETURN(cudaMemset(d_worktiles, 0, nblocks * (sv_nints*(nthreadsperblock/nr_procs)+nthreadsperblock/WARPSIZE+1)*sizeof(inttype)));
	CUDA_CHECK_RETURN(cudaMemset(d_counted_states, 0, sizeof(inttype)));

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
	inttype tablesize = q_size;
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
	cudaMemcpyToSymbol(d_apply_por, &apply_por, sizeof(inttype));
	cudaMemcpyToSymbol(d_check_cycle_proviso, &use_cycle_proviso, sizeof(inttype));

	// init the hash table
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

	clock_t exploration_start;
	assert((exploration_start = clock())!=-1);

	while (contBFS == 1) {
		CUDA_CHECK_RETURN(cudaMemcpy(d_contBFS, &zero, sizeof(inttype), cudaMemcpyHostToDevice))
		if(apply_por) {
			gather_por<<<nblocks, nthreadsperblock, shared_q_size*sizeof(inttype)>>>(d_q, d_h, d_bits_state, d_firstbit_statevector, d_proc_offsets_start,
																		d_proc_offsets, d_proc_trans, d_syncbits_offsets, d_syncbits, d_contBFS, d_property_violation, d_newstate_flags, d_worktiles, scan);
		} else {
			gather<<<nblocks, nthreadsperblock, shared_q_size*sizeof(inttype)>>>(d_q, d_h, d_bits_state, d_firstbit_statevector, d_proc_offsets_start,
																		d_proc_offsets, d_proc_trans, d_syncbits_offsets, d_syncbits, d_contBFS, d_property_violation, d_newstate_flags, d_worktiles, scan);
		}
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
				print_local_queue(stdout, q_test, tablesize, firstbit_statevector, nr_procs, sv_nints, apply_por);
			}
		}
		scan = 1;
	}
	// determine runtime
	stop = clock();
	runtime = (double) (stop-start)/CLOCKS_PER_SEC;
	fprintf (stdout, "Run time: %f\n", runtime);
	runtime = (double) (stop-exploration_start)/CLOCKS_PER_SEC;
	fprintf(stdout, "Exploration time %f\n", runtime);

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

	CUDA_CHECK_RETURN(cudaMemset(d_counted_states, 0, sizeof(inttype)));
	count_states<<<((int) prop.multiProcessorCount)*8, 512, 1>>>(d_q, d_counted_states);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(&counted_states, d_counted_states, sizeof(inttype), cudaMemcpyDeviceToHost));
	fprintf (stdout, "nr. of states in hash table: %d\n", counted_states);

	// Debugging functionality: print states to file
	if(dump_file) {
		FILE* fout;
		if((fout = fopen(dump_file, "w")) != NULL) {
			fprintf(stdout, "Dumping state space to file...\n");
			cudaMemcpy(q_test, d_q, tablesize*sizeof(inttype), cudaMemcpyDeviceToHost);
			print_local_queue(fout, q_test, tablesize, firstbit_statevector, nr_procs, sv_nints, apply_por);
			fclose(fout);
		} else {
			fprintf(stderr, "Could not open file to dump the state space\n");
		}
	}

	return 0;
}
