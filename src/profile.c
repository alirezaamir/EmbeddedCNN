#include "profile.h"	
#include "rt/rt_api.h"

void profile_start(rt_perf_t *perf){
 
		int id = rt_core_id();
		if(id==0){
			printf("\nstart profile FC\n");
		}
	 	rt_perf_init(&perf[id]); 
		 
#ifdef ACTIVE
	  	rt_perf_conf(&perf[id],(1<<RT_PERF_CYCLES) | (1<<RT_PERF_ACTIVE_CYCLES));
#endif
#ifdef STALL
		rt_perf_conf(&perf[id],(1<<RT_PERF_LD_STALL) | (1<<RT_PERF_IMISS));
#endif
#ifdef EXTACC
		rt_perf_conf(&perf[id],(1<<RT_PERF_LD_EXT_CYC) | (1<<RT_PERF_ST_EXT_CYC));
#endif
#ifdef INTACC
		rt_perf_conf(&perf[id],(1<<RT_PERF_LD) | (1<<RT_PERF_ST));
#endif
#ifdef INSTRUCTION
		rt_perf_conf(&perf[id],(1<<RT_PERF_INSTR));
#endif
#ifdef TCDM
		rt_perf_conf(&perf[id],(1<<RT_PERF_TCDM_CONT));
#endif
		rt_perf_reset(&perf[id]);
		rt_perf_start(&perf[id]);
 	
}

void profile_stop(rt_perf_t *perf){
 
	    int id = rt_core_id();
		
		rt_perf_stop(&perf[id]);
		rt_perf_save(&perf[id]);

#ifdef ACTIVE
		printf("[%d] cycles = %d\n", id, rt_perf_read (RT_PERF_CYCLES));
 		printf("[%d] active cycles = %d\n\n", id, rt_perf_read (RT_PERF_ACTIVE_CYCLES));
#endif
#ifdef STALL
 		printf("[%d] LD stall = %d\n", id, rt_perf_read (RT_PERF_LD_STALL));
 		printf("[%d] IMISS = %d\n", id, rt_perf_read (RT_PERF_IMISS));
#endif
#ifdef EXTACC
 		printf("[%d] RT_PERF_LD_EXT_CYC = %d\n", id, rt_perf_read (RT_PERF_LD_EXT_CYC));
 		printf("[%d] RT_PERF_ST_EXT_CYC = %d\n", id, rt_perf_read (RT_PERF_ST_EXT_CYC));
#endif
#ifdef INTACC
 		printf("[%d] RT_PERF_LD_CYC = %d\n", id, rt_perf_read (RT_PERF_LD));
 		printf("[%d] RT_PERF_ST_CYC = %d\n", id, rt_perf_read (RT_PERF_ST));
#endif
#ifdef INSTRUCTION
 		printf("[%d] instr = %d\n", id, rt_perf_read (RT_PERF_INSTR));	
#endif
#ifdef TCDM
 		printf("[%d] RT_PERF_TCDM_CONT = %d\n", id, rt_perf_read (RT_PERF_TCDM_CONT));	
#endif

}
 