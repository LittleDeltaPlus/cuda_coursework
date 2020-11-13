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
    //initialise Integration parameters
    const long double b = 10000000, a = -b;
    const int  nUpLim = (int)b*10, nLoLim =(int)(b*0.1);
    //get rid of any existing benchmarks
    if(remove("cpuOut_noPar.txt") !=0){
        perror("error deleting file");
    }
    //set output file
    ofstream outputFile ("cpuOut_noPar.txt");
    for(int n=nLoLim; n < nUpLim; n+=nLoLim){
        //start timer
        long double t1 = omp_get_wtime();
        //calculate delta
        const long double delta = (b-a)/n;
        //evaluate integration
        long double sum = 0;
        for (int i=1; i < n; i++){
            sum = sum + ( Evaluate(a +(i-0.5)*delta))*delta;
        }
        //stop timer
        auto t2 = omp_get_wtime();
        long double duration = ( t2 - t1 );
        //write benchmark
        outputFile << setprecision(18) << delta << " " << sum << " " << duration << endl;
        //indicate pass finished
        cout << n << endl;
    }
    //close ouptut, exit
    outputFile.close();
    return 0;
}
