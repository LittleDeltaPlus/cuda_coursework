#include <iostream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <fstream>

__global__ void gaussian_block (double delta, double a, double len, const double *x, double *y){
    unsigned int idx = blockIdx.x;
    if (idx < len){
        //almost verbatim the summation formula
        y[idx] = delta*std::exp(-((a + delta*(x[idx]-0.5))*(a + delta*(x[idx]-0.5))));
    }
}


int main() {
    //remove existing benchmark file
    if(remove("gpuOut.txt") !=0){
        perror("error deleting file");
    }
    //Open Refrence Benchmark file
    std::ofstream outputFile ("gpuOut.txt");
    std::ifstream infile ("../cpuOut_omp.txt");

    //Initialise Numerical Integration Variables
    const double b = 10000000,
            a = -b;
    const int  nUpLim = (int)b*10, nLoLim =(int)(b*0.1);// (int)b;


    for(int n = nLoLim; n < nUpLim; n+=nLoLim){
        //load the corresponding benchmark
        std::string line;
        getline(infile, line);
        std::istringstream iss(line);
        long double delta_f, sum_lf, duration_lf, sum_of, duration_of, speedup_of;
        if(!(iss >> delta_f >> sum_lf >> duration_lf >> sum_of >> duration_of >> speedup_of)){std::cout << "oops"<<std::endl;}
        //ensure device is ready
        cudaDeviceSynchronize();
        //initialise Device vairables
        double *d_x, *d_y;
        cudaMalloc(&d_x, n*sizeof(double));
        cudaMalloc(&d_y, n*sizeof(double));
        //initialise Host variables
        auto *x = new double[n], *y = new double[n];
        
        //start timer
        long double t1 = omp_get_wtime();
        //calculate delta
        double delta = (b-a)/n;
        
        //populate host variables
        for (int i=0; i<=n ; i++){
            x[i]=i;
            y[i]=0;
        }
        //copy host variables to the device
        cudaMemcpy(d_x, x, n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, n*sizeof(double), cudaMemcpyHostToDevice);
        //evaluate the gaussian function
        gaussian_block<<<n, 1>>>(delta, a, n, d_x, d_y);
        //copy the result back
        cudaMemcpy(y, d_y, n*sizeof(double), cudaMemcpyDeviceToHost);
        //Sum the output
        double block_sum=0;
        for (int i=0; i<=n ; i++){
            block_sum+=y[i];
        }
        //end the timer
        long double t2 = omp_get_wtime();
        long double block_duration = ( t2 - t1 );
        //write the benchmark for this delta
        outputFile << std::setprecision(18) << delta_f << " " << sum_lf << " " << duration_lf << " " <<sum_of << " " << duration_of << " " << speedup_of
                   << " --> " << block_sum << " <--" << block_duration << " " << duration_lf/block_duration<< " " << duration_of/block_duration << std::endl;
        //indicate this pass is done
        std::cout << n << std::endl;

    }
    //close the benchmark output and finish
    outputFile.close();
    return 0;
}
