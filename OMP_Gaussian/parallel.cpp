#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <cmath>

using namespace std;

//evaluate gaussian function at x
long double Evaluate(long double x){
    return expl(-x*x);
}


int main(){
    //set number of threads according to host specification
    omp_set_num_threads(4);
    //initialise integration parameters
    const long double b = 10000000, a = -b;
    const int  nUpLim = (int)b*10, nLoLim =(int)(b*0.1);
    //get rid of any existing benchmarks
    if(remove("cpuOut_omp.txt") !=0){
        perror("error deleting file");
    }
    //set output benchmark
    ofstream outputFile ("cpuOut_omp.txt");
    //set input benchmark
    ifstream infile ("cpuOut_noPar.txt");
    for(int n=nLoLim; n < nUpLim; n+=nLoLim){
        //get corresponding benchmark
        string line;
        getline(infile, line);
        istringstream iss(line);
        long double delta_f, sum_f, duration_f;
        if(!(iss >> delta_f >> sum_f >> duration_f)){throw;}
        //start timer
        long double t1 = omp_get_wtime();
        //calculate delta
        const long double delta = (b-a)/n;
        long double sum = 0;
        //start parallel summations
#pragma omp parallel for shared(n) default(none) reduction(+:sum)
        for (int i=1; i < n; i++){
            sum = sum + ( Evaluate(a +(i-0.5)*delta))*delta;
        }
        //end timer
        auto t2 = omp_get_wtime();
        long double duration = ( t2 - t1 );
        //write benchmarl
        outputFile << setprecision(18) << delta_f << " " << sum_f << " " << duration_f << sum << " " << duration <<  duration_f/duration <<endl;
        //inicate pass finished
        cout << n << endl;
    }
    outputFile.close();

    return 0;
}
