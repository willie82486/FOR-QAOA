#include <sys/time.h>

struct timeval start;
struct timeval end_;
unsigned long diff;

#define MEASURET_START \
    do {gettimeofday(&start,NULL);} while (0)
    
#define MEASURET_END \
    do { \
        gettimeofday(&end_,NULL); \
        diff = 1000000 * (end_.tv_sec-start.tv_sec) + \
        end_.tv_usec-start.tv_usec; \
    } while (0)
