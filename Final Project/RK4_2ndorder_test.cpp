#include <iostream>
#include <vector>
#include "definitions.h"

//#define d(x,y) (y*y-x*x)/(y*y+x*x);
#define PI 3.1415

std::vector<float> f(float t, float *q,segment vals,float torque, float k, float l){
    std::vector<float> fvec;
    fvec[0] = q[1]+k;
    fvec[1] = torque/vals.I+vals.m*9.81*vals.lc/vals.I*sin(q[0]+l);
    return fvec;
} 

using namespace std;

int main(){
    float phi, tf, t0, h, yn, k1[2], k2[2], k3[2], k4[2], k[2], torque, q[2], qn[2];
    int n;
    segment vals;
    vector <float> fun1, fun2, fun3, fun4;
    //cout<<"Enter Initial Conditions"<<endl;
    //cout<<"x0 = ";
    //cin>> x0;
    //cout<<"y0 = ";
    //cin>> y0;
    //cout<< "Enter calculation point xn = ";
    //cin>>xn;
    //cout<<"Enter number of steps: ";
    //cin>> n;

    q[0] = 45*PI/180; // Initial value of q1 = phi = 45 degrees
    q[1] = 0; // Initial value of q2 = phidot = 0 degrees/s;
    torque = -2; // torque value at pivot joint
    t0 = 0; //initial value of deriving variable (time)
    tf = 2; // final value of deriving variable (time)
    n = 100; // Number of steps

    h = (tf-t0)/n; //step size

    //cout<<"\nx0\ty0\tyn\n";
    //cout<<"-------------------\n";
    for (int i = 0; i<n; i++){
        fun1 = f(t0,q,vals,torque, 0, 0);
        k1[0] = h * fun1[0];
        k1[1] = h * fun1[1];
        
        fun2 = f(t0+h/2,q,vals,torque,k1[0]/2,k1[1]/2);
        k2[0] = h * fun2[0];
        k2[1] = h * fun2[1];

        fun3 = f(t0+h/2,q,vals,torque,k2[0]/2,k2[1]/2);
        k3[0] = h * fun3[0];
        k3[1] = h *fun3[1];

        fun4 = f(t0+h,q,vals,torque,k3[0],k3[1]);
        k4[0] = h * fun4[0];
        k4[1] = h * fun4[1];

        k[0] = (k1[0]+2*k2[0]+2*k3[0]+k4[0])/6;
        k[1] = (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6;
        //k3 = h * fn((x0+h/2),(y0+k2/2));
        //k4 = h * fn((x0+h),(y0+k3));
        //k = (k1+2*k2+2*k3+k4)/6;
        qn[0] = q[0]+k[0];
        qn[1] = q[1]+k[1];
        //cout<< q <<"\t"<<y0<<"\t"<<yn<<endl;
        t0 = t0+h;
        q[0] = qn[0];
        q[1] = qn[1];
    }
    cout<<"\nValue of q at t = "<< tf << " is "<<q[1];

    return 0;
}