#include <iostream>
#include <vector>
#include <chrono>
#include "definitions.h"
#include <math.h>

//#define d(x,y) (y*y-x*x)/(y*y+x*x);
#define PI 3.14159

std::vector<float> f(float t, float *q,segment vals,float torque, float k, float l){
    std::vector<float> fvec(2,0);
    fvec[0] = q[1]+k;
    fvec[1] = torque/vals.I+vals.m*9.81*vals.lc/vals.I*sin(q[0]+l);
    return fvec;
} 

float tcalc(tpulse torque, float t){
    float val;
    if (t>=torque.duration) return(0);
    else
    val = torque.amp * sin(PI*t/torque.duration);
    return val;
    }
using namespace std;

int main(){
    // Initialize Solution Variables
    float phi, tf, t0, h, yn, k1[2], k2[2], k3[2], k4[2], k[2], q[2], qn[2], oldnorm, newnorm;
    int n;
    segment vals;
    tpulse segt;
    vector <float> fun1(2,0), fun2(2,0), fun3(2,0), fun4(2,0);

    oldnorm = 20;
    q[0] = -5*PI/180; // Initial value of q1 = phi = 45 degrees
    q[1] = 0; // Initial value of q2 = phidot = 0 degrees/s;
    //torque = 40; // torque value at pivot joint [Newtons]
    t0 = 0; //initial value of deriving variable (time)
    tf = 0.4; // final value of deriving variable (time)
    n = 100; // Number of steps
    h = (tf-t0)/n; //step size
    
    segt.duration = 0.05; //Length of torque pulse [s]
    segt.amp = 50; // maximum amplitude of torque pulse sin wave [N*m]
    

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
        torque_i[i] = tcalc(segt,t0+(h*i));
    }


    for (int i = 0; i<n; i++){
        //move this outside of for loop, parallelize separately for efficiency and load to shared mem?
        
        fun1 = f(t0,q,vals,torque_i[i], 0, 0);
        k1[0] = h * fun1[0];
        k1[1] = h * fun1[1];
        
        fun2 = f(t0+h/2,q,vals,torque_i[i],k1[0]/2,k1[1]/2);
        k2[0] = h * fun2[0];
        k2[1] = h * fun2[1];

        fun3 = f(t0+h/2,q,vals,torque_i[i],k2[0]/2,k2[1]/2);
        k3[0] = h * fun3[0];
        k3[1] = h *fun3[1];

        fun4 = f(t0+h,q,vals,torque_i[i],k3[0],k3[1]);
        k4[0] = h * fun4[0];
        k4[1] = h * fun4[1];

        k[0] = (k1[0]+2*k2[0]+2*k3[0]+k4[0])/6;
        k[1] = (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6;
        //k3 = h * fn((x0+h/2),(y0+k2/2));
        //k4 = h * fn((x0+h),(y0+k3)); 
        //k = (k1+2*k2+2*k3+k4)/6;
        qn[0] = q[0]+k[0];
        qn[1] = q[1]+k[1];
        std::cout << qn[0]*180/PI <<"\t"<< qn[1]*180/PI << "\t" << torque_i[i] << "\t" << t0 <<endl;
        newnorm = sqrt(qn[0]*qn[0]+qn[1]*qn[1]);
        //if (newnorm >= oldnorm) break;
        t0 = t0+h;
        q[0] = qn[0];
        q[1] = qn[1];
        oldnorm = newnorm;
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout<<"\nValue of q[1] at t = "<< t0 << " is "<<q[1]*180/PI << "\n";
    ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start);
    std::cout << "time for full calculation: "<< ms.count() <<"\n";
    return 0;
}



        

   
        