#include "picograd.h"
#include <iostream>
#include <vector>
#include <set>
#include <cmath>

namespace ajs {

template<typename T>
Value<T>::Value(): Value{0} {
    std::cout << "Default constructor: " << *this << std::endl;
}

template<typename T>
Value<T>::Value(const T& data): data_{data} {
    std::cout << "Number-only constructor: " << *this << std::endl;
}

template<typename T>
Value<T>::Value(const T& data, const std::string& op, const std::tuple<Value<T>*, Value<T>*> children)
    :data_{data}, op_{op}, children_{children} {
    std::cout << "Full constructor: " << *this << std::endl;
}

template<typename T>
Value<T>::~Value() {
    std::cout << "Destroying " << *this << std::endl;
}



template<typename T>
Value<T> Value<T>::operator+(Value<T>& other) {
    Value out = Value(data_ + other.data_, "+", {this, &other});
    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        auto a_obj = *a;
        auto b_obj = *b;
        a->grad_ += 1 * out.grad_ + 0 * out.grad_;  // for clarity
        b->grad_ += 0 * out.grad_ + 1 * out.grad_;
//        std::cout << out.op_ << ": " << *a << ", " << (b ? *b : 0) << std::endl;
    };
    return out;
}

template<typename T>
Value<T> Value<T>::pow(int exponent) {
    Value out = Value(std::pow(data_, exponent), "^", {this, nullptr});
    out.backward_ = [&out, exponent](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * exponent*std::pow(a->data_, exponent-1);
//        std::cout << out.op_ << ": " << *a << ", " << (b ? *b : 0) << std::endl;
    };
    return out;
}

template<typename T>
Value<T> Value<T>::operator*(Value<T>& other) {
    Value out = Value(data_ * other.data_, "*", {this, &other});
    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * b->data_;
        b->grad_ += out.grad_ * a->data_;
//        std::cout << out.op_ << ": " << *a << ", " << (b ? *b : 0) << std::endl;
    };
    return out;
}

template<typename T>
Value<T> Value<T>::operator-() {
    auto temp = Value<T>(-1); // todo: will this be destroyed before backward pass?
    return *this * temp;
}

template<typename T>
Value<T> Value<T>::operator-(Value<T>& other) {
    auto temp = -other; // todo: will this be destroyed before backward pass?
    return *this + temp;
}

template<typename T>
Value<T> Value<T>::operator/(Value<T>& other) {
    auto temp = other.pow(-1); // todo: will this be destroyed before backward pass?
    return *this * temp;
}

template<typename T>
Value<T> Value<T>::exp() {
    Value out = Value(std::exp(data_), "exp", {this, nullptr});
    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * out.data_;
//        std::cout << out.op_ << ": " << *a << ", " << (b ? *b : 0) << std::endl;
    };
    return out;
}

template<typename T>
Value<T> Value<T>::tanh() {
    T exp_val = std::exp(2 * this->data_);
    T out_val = (exp_val - 1) / (exp_val + 1);
    Value out = Value(out_val, "tanh", {this, nullptr});
    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * (1 - std::pow(out.data_, 2));
//        std::cout << out.op_ << ": " << *a << ", " << (b ? *b : 0) << std::endl;
    };
    return out;
}

template<typename T>
Value<T> Value<T>::relu() {
    T out_val = data_ > 0 ? data_ : 0;
    Value out = Value(out_val, "relu", {this, nullptr});
    out.backward_ = [&out](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * (out.data_ > 0 ? 1 : 0);
//        std::cout << out.op_ << ": " << *a << ", " << (b ? *b : 0) << std::endl;
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
//            std::cout << **val_iter << std::endl;
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
