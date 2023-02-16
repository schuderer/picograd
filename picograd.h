#pragma once

#include <functional>
#include <string>
#include <cmath>
#include <vector>
#include <set>

namespace ajs {

template<typename T>
class Value
{
public:
    Value();
    Value(const T& data);
    Value(const T& data, const std::string& op, const std::tuple<Value*, Value*> children);
    ~Value();

    Value operator+(Value& other);
    Value operator*(Value& other);
    Value operator-();
    Value operator-(Value& other);
    Value operator/(Value& other);
    Value pow(int exponent);  // todo: support int, float (not Value though)
    Value exp();
    Value tanh();
    Value relu();

    void backward();
    void graph();

    T get_data() const;
    T get_grad() const;
    void set_data(T data);
    void set_grad(T grad);

protected:
    T data_;
    T grad_{0};
    std::string op_{""};
    std::tuple<Value*, Value*> children_{nullptr, nullptr};
    void topological_order(Value* node, std::vector<Value*>& topo, std::set<Value*>& visited);
    std::function<void()> backward_{nullptr};
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Value<T>& v);

} // namespace ajs

// Needed because this is a template library
#include "picograd.cpp"
