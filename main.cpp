#include <iostream>
#include "picograd.h"

// for some tests and debugging
#include <map>
#include <memory>

using std::cout, std::cin, std::endl;
using namespace ajs;

void sanity_check() {
    cout << "beginning of sanity_check" << endl;

    // without rvalues/temporaries:
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
    assert(std::abs(x.get_data() - (-4.0)) < 0.00001);
    cout << "y should be -20.0 and it is " << y << endl;
    assert(std::abs(y.get_data() - (-20.0)) < 0.00001);
    cout << "x.get_grad() should be 46 and it is " << x.get_grad() << endl;
    assert(std::abs(x.get_grad() - 46.0) < 0.00001);

    cout << "end of sanity_check" << endl;
}

void sanity_check_rvalues() {
    cout << "beginning of sanity_check_rvalues" << endl;

    // with rvalues:
    auto x = Value(-4.0);
    auto z = Value(2.0) * x + 2.0 + x;
    auto q = z.relu() + z * x;
    auto h = (z * z).relu();
    auto y = h + q + q * x;
    y.backward();

    cout << "x should be -4.0 and it is " << x << endl;
    assert(std::abs(x.get_data() - (-4.0)) < 0.00001);
    cout << "y should be -20.0 and it is " << y << endl;
    assert(std::abs(y.get_data() - (-20.0)) < 0.00001);
    cout << "x.get_grad() should be 46 and it is " << x.get_grad() << endl;
    assert(std::abs(x.get_grad() - 46.0) < 0.00001);

    cout << "end of sanity_check_rvalues" << endl;
}


void test_more_ops() {
    cout << "beginning of test_more_ops" << endl;

    auto a = Value(-4.0);
    auto b = Value(2.0);
    auto c = a + b;
    auto d = a * b + b.pow(3);
    c += c + 1;
    c += Value(1.0) + c + (-a);
    d += d * 2 + (b + a).relu();
    d += Value(3.0) * d + (b - a).relu();
    auto e = c - d;
    auto f = e.pow(2);
    auto g = f / 2.0;
    g += Value(10.0) / f;
    g.backward();

    // pytorch's g.data: 24.70408163265306
    // pytorch's a.grad: 138.83381924198252
    // pytorch's b.grad: 645.5772594752186

    cout << "g should be 24.704082 and it is " << g << endl;
    assert(std::abs(g.get_data() - 24.704082) < 0.00001);
    cout << "a.get_grad() should be 138.833819 and it is " << a.get_grad() << endl;
    assert(std::abs(a.get_grad() - 138.833819) < 0.00001);
    cout << "b.get_grad() should be 645.577259 and it is " << b.get_grad() << endl;
    assert(std::abs(b.get_grad() - 645.577259) < 0.00001);

    cout << "end of test_more_ops" << endl;
}


// Just for tracking object creating and destruction. Remove this, as well as the new/delete overloads
// (the crash at the end of main is because of delete trying to access it)
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
    sanity_check();
    sanity_check_rvalues();
    test_more_ops();

    cout << "end of main" << endl;
    return 0;
}
