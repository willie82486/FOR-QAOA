#ifndef _UTILS_H_
#define _UTILS_H_

#include <sys/time.h>
#include <vector>
struct timeval start;
struct timeval end;
unsigned long diff;

#define MEASURET_START \
    do {gettimeofday(&start,NULL);} while (0)
    
#define MEASURET_END \
    do { \
        gettimeofday(&end,NULL); \
        diff = 1000000 * (end.tv_sec-start.tv_sec) + \
        end.tv_usec-start.tv_usec; \
    } while (0)


void printGateBlock(std::vector<std::vector<int>>& GBs) {
    for (size_t i = 0; i < GBs.size(); i++) {
        printf("GateBlock %ld: ", i);
        for (size_t j = 0; j < GBs[i].size(); j++) {
            printf("%d ", GBs[i][j]);
        }
        printf("\n");
    }
}

void printQubitMap(std::vector<int>& qubitMap) {
    for (size_t i = 0; i < qubitMap.size(); i++) {
        printf("%d ", qubitMap[i]);
    }
    printf("\n");
}

#endif