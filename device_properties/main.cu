#include <cstdio>
#include <ostream>
#include <sstream>
#include <iostream>


int main(){
    cudaDeviceProp prop{};
    //query device
    cudaGetDeviceProperties(&prop,0);
    //store device parameters as variables
    std::string name = prop.name;
    double clockRate = prop.clockRate;
    double globalMem = prop.totalGlobalMem;
    double memFreq = prop.memoryClockRate;
    double maxThreads = prop.maxThreadsPerBlock;
    double maxBlocks = prop.maxBlocksPerMultiProcessor;
    //print results
    std::cout
        << "Name: "<< name << std::endl
        << "Clock Rate: "<< clockRate << std::endl
        << "Memory Available: "<< globalMem << std::endl
        << "Memory Frequency: "<< memFreq << std::endl
        << "Max Threads: "<< maxThreads << std::endl
        << "Max Blocks: "<< maxBlocks << std::endl;
    printf("done");
}