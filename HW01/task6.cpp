#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{

    int N = atoi(argv[1]);

    for (int i = 0; i <= N; i++)
    {
        printf("%d ", i);
    }
    printf("\n");
    for (int i = N; i >= 0; i--)
    {
        std::cout << i << " ";
    }
    cout << "\n";
}