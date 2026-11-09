
#include <iostream>
#include <thread>
using namespace std;

void task1()
{
    cout << "Task 1 is executing\n";
}

void task2()
{
    cout << "Task 2 is executing\n";
}

int main()
{
    thread t1(task1);
    thread t2(task2);

    t1.join();
    t2.join();

    cout << "Main function finished\n";

    return 0;
}
