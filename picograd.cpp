#include "picograd.h"
#include <iostream>
#include <vector>
#include <set>
#include <cmath>

#ifndef __OPTIMIZE__
#define LOG(x) std::cout << x << std::endl
#define LOGVAR(x) std::cout << #x"=" << x << std::endl
#else
#define LOG(x)
#define LOGVAR(x)
#endif

namespace ajs {

template<typename T>
Value<T>::Value(): Value{0} {
    LOG("Default constructor: " << *this);
}

template<typename T>
Value<T>::Value(const T& data): data_{data} {
    LOG("Number-only constructor: " << *this);
}

template<typename T>
Value<T>::Value(const T& data, const std::string& op, const std::tuple<Value<T>*, Value<T>*> children)
    :data_{data}, op_{op}, children_{children} {
    LOG("Full constructor: " << *this);
}


template<typename T>
Value<T>::Value(const Value& other) {
    this->data_ = other.data_;
    LOG("Copy constructor: " << other << " -> " << *this);
}

template<typename T>
Value<T>::Value(Value&& other) {
    LOG("Move constructor: " << other << " -> " << *this);
}

template<typename T>
Value<T> Value<T>::operator=(const Value& other) {
    LOG("Assignment constructor: " << other << " -> " << *this);
}

template<typename T>
Value<T> Value<T>::operator=(Value&& other) {
    LOG("Move assignment constructor: " << other << " -> " << *this);
}


template<typename T>
Value<T>::~Value() {
    LOG("Destroying " << *this);
    if (this->data_ == -1000) {  // just to temporarily make sure it's not called twice
        exit(1);
    }
    this->data_ = -1000;
    this->grad_ = -1000;
    this->backward_ = nullptr;
    this->children_ = {nullptr, nullptr};
}



template<typename T>
Value<T> Value<T>::operator+(Value<T>& other) {
    LOG("at " << *this << " + " << other);
    Value out{data_ + other.data_, "+", {this, &other}};

    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += 1 * out.grad_ + 0 * out.grad_;  // for clarity
        b->grad_ += 0 * out.grad_ + 1 * out.grad_;
        LOG(out.op_ << " backward result: " << *a << ", " << *b);
    };
    return out;
}

template<typename T>
Value<T> Value<T>::pow(float exponent) {
    LOG("at " << *this << ".pow(" << exponent << ")");
    Value out{std::pow(data_, exponent), "pow", {this, nullptr}};

    out.backward_ = [&out, exponent](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * exponent*std::pow(a->data_, exponent-1);
        LOG(out.op_ << " backward result: " << *a);
    };
    return out;
}

template<typename T>
Value<T> Value<T>::operator*(Value<T>& other) {
    LOG("at " << *this << " * " << other);
    Value out{data_ * other.data_, "*", {this, &other}};

    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * b->data_;
        b->grad_ += out.grad_ * a->data_;
        LOG(out.op_ << " backward result: " << *a << ", " << (b ? *b : 0));
    };
    return out;
}

template<typename T>
Value<T> Value<T>::operator-() {
    LOG("at -" << *this);
    Value temp{-1}; // todo: will this be destroyed before backward pass?
    return *this * temp;
}

template<typename T>
Value<T> Value<T>::operator-(Value<T>& other) {
    LOG("at " << *this << " - " << other);
    Value temp{-other}; // todo: will this be destroyed before backward pass?
    return *this + temp;
}

template<typename T>
Value<T> Value<T>::operator/(Value<T>& other) {
    LOG("at " << *this << " / " << other);
    Value temp{other.pow(-1)}; // todo: will this be destroyed before backward pass?
    return *this * temp;
}

template<typename T>
Value<T> Value<T>::exp() {
    LOG("at " << *this << ".exp()");
    Value out{std::exp(data_), "exp", {this, nullptr}};

    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * out.data_;
        LOG(out.op_ << " backward result: " << *a << ", " << (b ? *b : 0));
    };
    return out;
}

template<typename T>
Value<T> Value<T>::tanh() {
    LOG("at " << *this << ".tanh()");
    T exp_val = std::exp(2 * this->data_);
    Value out{(exp_val - 1) / (exp_val + 1), "tanh", {this, nullptr}};

    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * (1 - std::pow(out.data_, 2));
        LOG(out.op_ << " backward result: " << *a << ", " << (b ? *b : 0));
    };
    return out;
}

template<typename T>
Value<T> Value<T>::relu() {
    LOG("at " << *this << ".relu()");
    Value out{data_ > 0 ? data_ : 0, "relu", {this, nullptr}};

    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * (out.data_ > 0 ? 1 : 0);
        LOG(out.op_ << " backward result: " << *a << ", " << (b ? *b : 0));
    };
    return out;
}


template<typename T>
void Value<T>::topological_order(Value<T>* node, std::vector<Value<T>*>& topo, std::set<Value<T>*>& visited) {
    if (!visited.contains(node)) {
        visited.insert(node);
        auto [a, b] = node->children_;
        if (a != nullptr) {  // todo: use std::optional?
            topological_order(a, topo, visited);
        }
        if (b != nullptr) {  // todo: use std::optional?
            topological_order(b, topo, visited);
        }
        topo.push_back(node);  // AFTER pushing the dependent nodes
    }
}

template<typename T>
void Value<T>::backward() {
    std::vector<Value<T>*> topo{};
    std::set<Value<T>*> visited{};
    topological_order(this, topo, visited);
    //    std::cout << *this << " vs " << *topo.front() << std::endl;
    this->grad_ = 1;
    for (auto val_iter = topo.rbegin(); val_iter != topo.rend(); val_iter++) {
        if ((*val_iter)->backward_ != nullptr) {
            LOG("Calling " << **val_iter << ".backward():");
            (*val_iter)->backward_();
        }
    }
}

template<typename T>
T Value<T>::get_data() const {
    return data_;
}
template<typename T>
T Value<T>::get_grad() const {
    return grad_;
}
template<typename T>
void Value<T>::set_data(T data) {
    data_ = data;
}
template<typename T>
void Value<T>::set_grad(T grad) {
    grad_ = grad;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Value<T>& v) {
    return (os << "Value(" << v.get_data() << ", grad=" << v.get_grad() << ")@" << &v);
}

} // namespace ajs
