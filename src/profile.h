#include "rt/rt_api.h"

// Profiling
#define ACTIVE			//# of total and active cycles
//#define EXTACC		//# of loads and stores in EXT memory (L2)
//#define INTACC		//# of loads and stores in INT memory (L1)
//#define STALL			//# number of core stalls
//#define INSTRUCTION	//# number of instructions
//#define TCDM			//# of conflicts in TCDM (L1 memory) between cores 

void profile_start(rt_perf_t *perf);
void profile_stop(rt_perf_t *perf);
