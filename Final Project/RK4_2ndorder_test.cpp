#include <iostream>
#include <vector>
#include <chrono>
#include "single_definitions.h"
#include <math.h>

//#define d(x,y) (y*y-x*x)/(y*y+x*x);
#define PI 3.14159

using namespace std;

int main(){
    // Initialize Solution Variables
    float phi, tf, t0, h, yn, k1[2], k2[2], k3[2], k4[2], k[2], q[2],qdot[2], min_norm;
    int n;
    segment vals;
    tpulseinfo segt;

    q[0] = -5*PI/180; // Initial value of q1 = phi = 45 degrees
    q[1] = 0*PI/180; // Initial value of q2 = phidot = 0 degrees/s;
    //torque = 40; // torque value at pivot joint [Newtons]
    t0 = 0; //initial value of deriving variable (time)
    tf = 0.5; // final value of deriving variable (time)
    //n = 50; // Number of steps
    //h = (tf-t0)/n; //step size
    h = .01;
    n = tf/h;

    float* norms = new float[n];
    float* qn = new float[2*n];
    segt.duration = 0.06275; //Length of torque pulse [s]
    segt.amp = 40.7; // maximum amplitude of torque pulse sin wave [N*m]
    

    vals.l = 0.867; // anthro table length of ankle to hip
    vals.lc = 0.589; // anthro table lenth of ankle to CM of legs
    vals.m = 26.30; // anthro table mass of lower leg segments
    vals.I = 1.4; // anthro table moment of intertia of leg segments
    vals.Icm = vals.I+vals.m*vals.lc*vals.lc;

    // Initialize Variables for timing
    std::chrono::duration<double, std::milli> ms;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

start = std::chrono::high_resolution_clock::now();
    // Calculate toruqes
    float torque_i[n];
    for (int i = 0; i<n; i++){
        torque_i[i] = tpulsecalc(segt,t0+(h*i));         //move this outside of for loop, parallelize separately for efficiency and load to shared mem?
    }

    // Calculate norm value for initial positions
    //min_norm = sqrt(q[0]*q[0] + q[1]*q[1]);
    min_norm = 10;
    for (int i = 0; i<n; i++){

        
        fsingle(t0,q,qdot,torque_i[i], vals,0,0);
        k1[0] = h * qdot[0];
        k1[1] = h * qdot[1];
        
        fsingle(t0+h/2,q,qdot,torque_i[i],vals,k1[0]/2,k1[1]/2);
        k2[0] = h * qdot[0];
        k2[1] = h * qdot[1];

        fsingle(t0+h/2,q,qdot,torque_i[i],vals,k2[0]/2,k2[1]/2);
        k3[0] = h * qdot[0];
        k3[1] = h * qdot[1];

        fsingle(t0+h,q,qdot,torque_i[i],vals,k3[0],k3[1]);
        k4[0] = h * qdot[0];
        k4[1] = h * qdot[1];
        //std::cout<<k4[0]<<" "<<k4[1]<<"\n";
        k[0] = (k1[0]+2*k2[0]+2*k3[0]+k4[0])/6;
        k[1] = (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6;
        //k3 = h * fn((x0+h/2),(y0+k2/2));
        //k4 = h * fn((x0+h),(y0+k3)); 
        //k = (k1+2*k2+2*k3+k4)/6;
        qn[i*2] = q[0]+k[0];
        qn[i*2 + 1] = q[1]+k[1];
        norms[i] = sqrt(qn[i*2]*qn[i*2]+qn[i*2+1]*qn[i*2+1])*180/PI;
        
        std::cout << qn[i*2]*180/PI <<"  "<< qn[i*2 + 1]*180/PI << "  " << torque_i[i] << "  " << t0 << "  " << norms[i] << endl;
        
        t0 = t0+h;
        q[0] = qn[i*2];
        q[1] = qn[i*2+1];
    }
    
    for (int i = 0; i < n-1; i++){
        if (norms[i]<min_norm) min_norm = norms[i];
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout<<"\nValue of q[1] at t = "<< t0 << " is "<<q[1]*180/PI << "\n";
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start);
    std::cout << "time for full calculation: "<< ms.count() <<"\n";
    std::cout << "min norm: "<< min_norm <<"\n";
    
    delete[] norms;
    delete[] qn;
    return 0;
}



        

   
        