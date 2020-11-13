#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <omp.h>
#include <sstream>



using namespace std;


//Thrust functor to evaultate summation term for a value of i 
template <typename T> struct gaussian_functor {
    const T d, a;
    explicit gaussian_functor(T _d, T _a) : d(_d), a(_a) {}
     __device__
        T operator()(const T& x) const {
        int gid = blockDim.x * gridDim.x;
        //evaluate summation term, using delta (d) and a (a) 
        return d*exp(-((a + d*(x-0.5))*(a + d*(x-0.5))));
    }
};



int main() {
    //remove existing benchmark
    if(remove("gpuOut.txt") !=0){
        perror("error deleting file");
    }
    //set output file
    ofstream outputFile ("gpuOut.txt");
    //load refrence benchmark
    std::ifstream infile ("../cpuOut_omp.txt");
    //initialise integration parameters
    const double b = 10000000,
    a = -b;
    const int  nUpLim = (int)b*10, nLoLim =(int)(b*0.1);

    for(int n = nLoLim; n < nUpLim; n+=nLoLim){
        //Load corresponding benchmark
        string line;
        getline(infile, line);
        istringstream iss(line);
        long double delta_f, sum_lf, duration_lf, sum_of, duration_of, speedup_of;
        if(!(iss >> delta_f >> sum_lf >> duration_lf >> sum_of >> duration_of >> speedup_of)){cout << "oops"<<endl;}
        //ensure device is ready
        cudaDeviceSynchronize();
        //start timer
        long double t1 = omp_get_wtime();
        //calculate delta for this value of n
        double delta = (b-a)/n;
        //initalise Thrust functor
        gaussian_functor<double> Gaussian(delta, a);
        //create device vecotors
        thrust::device_vector<double> d_x(n);
        thrust::device_vector<double> d_y(n);
        //populate vector x with values of i
        thrust::sequence(d_x.begin(), d_x.end());
        //evaluate each summation term
        thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), Gaussian);
        //sum all evalutations
        double sum = thrust::reduce(d_y.begin(), d_y.end());
        //clear device memory
        d_x.clear();
        d_y.clear();
        //stop timer
        long double t2 = omp_get_wtime();
        long double duration = ( t2 - t1 );
        //write benchmark
        outputFile << setprecision(18) << delta_f << " " << sum_lf << " " << duration_lf << " " <<sum_of << " " << duration_of << " " << speedup_of
        << " " << sum << " " << duration << " " << duration_lf/duration<< " " << duration_of/duration <<endl;
        //indicate pass finished
        cout << n << endl;

    }
    //close output file, exit
    outputFile.close();
    return 0;
}
