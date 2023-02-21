#include <iostream>
#include "picograd.h"

// temp
#include <string>
#include <map>
#include <memory>

using std::cout, std::cin, std::endl;
using namespace ajs;

//template<typename T>
//class Foo {
//public:
//    T val;
//    std::tuple<Foo*, Foo*> children_{nullptr, nullptr};  // Error: Field has incomplete type 'Foo<int>'
//};

//int main() {
//    Foo<int> f;
//}


void sanity_check() {
    auto x = Value(-4.0f);
    auto x2 = Value(2.0f);
    auto x3 = Value(2.0f);
    auto z_x1 = x3 * x;
    auto z1 = z_x1 + x2;
    auto z2 = z1 + x;
    auto z_relu = z2.relu();
    auto z_x2 = z2 * x;
    auto q = z_relu + z_x2;
    auto h = (z2 * z2).relu();
    auto q_x = q * x;
    auto y = h + q + q_x;
    y.backward();

//    cout << "x: " << x << endl;
//    cout << "x2: " << x2 << endl;
//    cout << "x3: " << x3 << endl;
//    cout << "z_x1: " << z_x1 << endl;
//    cout << "z1: " << z1 << endl;
//    cout << "z2: " << z2 << endl;
//    cout << "z_relu: " << z_relu << endl;
//    cout << "z_x2: " << z_x2 << endl;
//    cout << "q: " << q << endl;
//    cout << "h: " << h << endl;
//    cout << "q_x: " << q_x << endl;
//    cout << "y: " << y << "\n" << endl;

//    auto a = Value(0.8813742f);
//    auto two1 = Value(2.0f);
//    auto two2 = Value(2.0f);
//    auto minusone = Value(-1.0f);
//    auto one2 = Value(1.0f);
//    auto exp1 = (two1 * a).exp();
//    auto nom = exp1 - minusone;
//    auto denom = ((two2 * a).exp() + one2);
//    auto b = nom / denom; //a.tanh();
//    cout << "Starting backward pass" << endl;
//    b.backward();
//    cout << a << endl;  // grad must be 0.5

//    auto x = Value(4.0f);
//    auto y = Value(2.0f);
//    auto relu = x.relu() + y;
//    relu.backward();
//    cout << x << endl;
//    cout << relu << endl;


//    cout << "x.get_grad()=" << x.get_grad() << endl;
//    cout << "x2.get_grad()=" << x2.get_grad() << endl;
//    cout << "x3.get_grad()=" << x3.get_grad() << endl;
//    cout << "z1.get_grad()=" << z1.get_grad() << endl;
//    cout << "z2.get_grad()=" << z2.get_grad() << endl;
//    cout << "z_relu.get_grad()=" << z_relu.get_grad() << endl;
//    cout << "z_x2.get_grad()=" << z_x2.get_grad() << endl;
//    cout << "q.get_grad()=" << q.get_grad() << endl;
//    cout << "h.get_grad()=" << h.get_grad() << endl;
//    cout << "y.get_grad()=" << y.get_grad() << "\n" << endl;

    cout << "x should be -4.0 and it is " << x << endl;
    cout << "y should be -20.0 and it is " << y << endl;
    cout << "x.get_grad() should be 46 and it is " << x.get_grad() << endl;

    cout << "end of sanity_check" << endl;
}



struct bla {
    int x_, y_;
    bla(int x, int y): x_{x}, y_{y} {};
    ~bla() {
        cout << "bla destructor!" << endl;
    }

};


std::map<size_t, size_t> sizes;
bool stop_recursion = false;

void* operator new(size_t sz) {
    void* ptr = std::malloc(sz);
    if (!stop_recursion && sz >= 100) {
        stop_recursion = true;
        sizes[(size_t)ptr] = sz;
        stop_recursion = false;
        cout << "-- new " << sz << " bytes @ " << ptr << " --" << endl;
    }
    return ptr;
}
void operator delete(void* ptr) noexcept {
    long size = sizes.contains((size_t)ptr) ? sizes[(size_t)ptr] : -1;
    if (size >= 100)
        cout << "-- delete " << size << " bytes @ " << ptr << " --" << endl;
    std::free(ptr);
    ptr = nullptr;
}
void* operator new[](size_t sz) {
    void* ptr = std::malloc(sz);
    if (!stop_recursion) {
        stop_recursion = true;
        sizes[(size_t)ptr] = sz;
        stop_recursion = false;
    }
    cout << "-- new[] " << sz << " bytes @ " << ptr << " --" << endl;
    return ptr;
}
void operator delete[](void* ptr) noexcept {
    cout << "-- delete[] " << ptr << " --" << endl;
    std::free(ptr);
}


int main()
{

    /*
    auto a = Value(2);
    auto b = Value(3);
    int exp = 2;
    auto be = b.pow(exp);
    Value<int> c = a + be;

    cout << a << " + " << b << "^" << exp << " (" << be << ")" << " = " << c << endl;

    cout << "before c.backward()" << endl;
    c.backward();
    cout << "after c.backward()" << endl;
    c.backward();
    cout << "after second c.backward()" << endl;


    cout << a << " + " << b << "^" << exp << " (" << be << ")" << " = " << c << endl;
*/
    sanity_check();

    {
    Value<double> c{};
    {
    auto a = Value(2.0);
    auto b = Value(3.0);
    c = a - b;

    cout << a << " + " << b << "^" << 2 << " = " << c << endl;
    cout << "before backward" << endl;
    c.backward();
    cout << "after backward" << endl;
    cout << a << " + " << b << "^" << 2 << " = " << c << endl;
    }
    cout << "after inner braces" << endl;
    }
    cout << "after outer braces" << endl;

    /*
    void* ptr2;
    {
    cout << 1 << endl;
    auto hey = std::make_shared<bla>(3,4);
    cout << 2 << endl;
    auto ho = std::make_shared<bla>(4,5);
    cout << 3 << endl;
    cout << 4 << endl;

    ptr2 = (void*)hey.get();
    }
    auto hey = std::make_shared<bla>(10,100);
    auto obj2 = (bla*)ptr2;
    cout << obj2->x_ << ", " << obj2->y_ << endl;
*/

    cout << "end of main" << endl;
    return 0;
}
