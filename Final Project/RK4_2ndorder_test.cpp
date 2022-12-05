#include <iostream>
#include <vector>
#include "definitions.h"

//#define d(x,y) (y*y-x*x)/(y*y+x*x);
#define PI 3.1415

vector<float> f(float t, float *q,segment vals,float torque){
    vector<float> fvec;
    fvec[0] = q[0];
    fvec[1] = torque/vals.I+vals.m*9.81*vals.lc/vals.I*sin(q[1]);
    return fvec;
} 

using namespace std;

int main(){
    float phi, tf, t0, h, yn, k1[2], k2[2], k3[2], k4[2], k[2], torque;
    int n;
    float q[2];
    segment vals;
    vector <float> fun0, fun1, fun2, fun3;
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
    torque = -2; // torque value at pivot joint, assume constant for now
    t0 = 0; //initial value of deriving variable (time)
    tf = 2; // final value of deriving variable (time)
    n = 100;

    h = (tf-t0)/n; //step size

    cout<<"\nx0\ty0\tyn\n";
    cout<<"-------------------\n";
    for (int i = 0; i<n; i++){
        fun1 = f(t0,q,vals,torque);
        fun2 = f(t0+h/2,q+k1/2,vals,torque);
        
        
        k1[0] = h * fun0[0];
        k1[1] = h * fun0[1];
        
        k2[0] = h * f((x0+h/2),(y0+k1/2));
        k3 = h * fn((x0+h/2),(y0+k2/2));
        k4 = h * fn((x0+h),(y0+k3));
        k = (k1+2*k2+2*k3+k4)/6;
        yn = y0+k;
        cout<< x0 <<"\t"<<y0<<"\t"<<yn<<endl;
        x0 = x0+h;
        y0 = yn;
    }
    cout<<"\nValue of y at x = "<< xn<< " is "<<yn;

    return 0;
}