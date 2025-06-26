#include <sys/time.h>

extern struct timeval start;
extern struct timeval end_;
extern unsigned long diff;

extern float rx_gate_time_ms;

extern float csqs_time_ms;

extern float transmission_time_ms;

#define MEASURET_START \
    do {gettimeofday(&start,NULL);} while (0)
    
#define MEASURET_END \
    do { \
        gettimeofday(&end_,NULL); \
        diff += 1000000 * (end_.tv_sec-start.tv_sec) + \
        end_.tv_usec-start.tv_usec; \
    } while (0)

