#include "value.h"
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <memory>

namespace ajs {

template<typename T>
Value<T>::Value(): Value{0} {
    LOG("Default constructor deferring to number-only constructor: " << *this);
}

template<typename T>
Value<T>::Value(const T data): node_{std::make_shared<Node>(data)} {
    LOG("Number-only constructor: " << *this);
}

template<typename T>
Value<T>::Value(std::shared_ptr<Node> node): node_{node} {
    LOG("Constructor with node: " << *this);
}

// TODO clean up constructors and destructor (I *think* the auto-created default ones should be fine in our class)
template<typename T>
Value<T>::Value(const Value& other) {
    LOG("Copy constructor: " << other << " -> " << *this);
    node_ = other.node_;
}

template<typename T>
Value<T>::Value(Value&& other) {
    LOG("Move constructor: " << other << " -> " << *this);
    node_ = std::move(other.node_);
}

template<typename T>
Value<T>& Value<T>::operator=(const Value& other) {
    LOG("Copy assignment constructor: " << other << " -> " << *this);
    node_ = other.node_;
    return *this;

}

template<typename T>
Value<T>& Value<T>::operator=(Value&& other) {
    LOG("Move assignment constructor: " << other << " -> " << *this);
    node_ = std::move(other.node_);
    return *this;
}

template<typename T>
Value<T>::~Value() {
    LOG("Destroying " << *this);
}



template<typename T>
Value<T> Value<T>::operator+(const Value<T>& other) const {
    LOG("forward pass for " << *this << " + " << other);
    auto out_node = std::make_shared<Node>(get_data() + other.get_data(), Op::add, node_, other.node_);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        void operator()() {
            auto a = out_node->child1;
            auto b = out_node->child2;
            auto grad = out_node->grad;
            a->grad += 1 * grad + 0 * grad;  // for clarity
            b->grad += 0 * grad + 1 * grad;
            LOG(out_node->op_str() << " backward result: " << a->str() << ", " << b->str());
        };
    };

    // TODO: maybe change to struct-based functors for easier debugging
    out_node->backward = lambda{out_node};
    return Value(out_node);
}


/*  // attempt to use std::variant to avoid code duplication. giving up for now :(
template<typename T>
struct PowVisitor {
    T& data_;
    int to_add_;
    PowVisitor(const T& data): data_{data}, to_add_{0} {};
    PowVisitor(const T& data, const int& to_add): data_{data}, to_add_{to_add} {};

    T operator()(int exponent) {
        return std::pow(data_, exponent + to_add_);
    }
    T operator()(float exponent) {
        return std::pow(data_, exponent + to_add_);
    }
};

// another try. sadly, visitors with varying return types are not allowed.
struct Visitor {
    int operator()(int x) {
        return x;
    }
    float operator()(float x) {
        return x;
    }
};

template<typename T>
Value<T> Value<T>::pow(std::variant<int, float> exponent) {
    using ExpType = std::decay_t<decltype(exponent)>;
    ExpType exp_val;  // Trying to use decltype to get type and use exp_val as out-param did not work either :( :(
    std::visit([&](auto x){ exp_val = x; }, exponent);
    LOG("at " << *this << ".pow(" << exp_val << ")");
    T out_val = std::pow(data_, exp_val);
    Value out{out_val, "pow", {this, nullptr}};

    out.backward_ = [&out, exp_val](){
        auto [a, b] = out.children_;
        a->grad_ += out.grad_ * exp_val*std::pow(a->data_, exp_val-1);
        LOG(out.op_ << " backward result: " << *a);
    };
    return out;
}
*/

template<typename T>
Value<T> Value<T>::pow(int exponent) const {  // TODO get rid of code duplication. How to keep it working with std::pow? std::variant and std::visit?
    LOG("forward pass for " << *this << ".pow(" << exponent << ")");
    auto out_node = std::make_shared<Node>(std::pow(get_data(), exponent), Op::pow, node_, nullptr);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        int exponent;
        void operator()() {
            auto a = out_node->child1;
            auto grad = out_node->grad;
            a->grad += grad * exponent*std::pow(a->data, exponent-1);
            LOG(out_node->op_str() << " backward result: " << a->str());
        };
    };

    out_node->backward = lambda{out_node, exponent};
    return Value(out_node);
}

template<typename T>
Value<T> Value<T>::pow(float exponent) const {  // TODO get rid of code duplication. How to keep it working with std::pow? std::variant and std::visit?
    LOG("forward pass for " << *this << ".pow(" << exponent << ")");
    auto out_node = std::make_shared<Node>(std::pow(get_data(), exponent), Op::pow, node_, nullptr);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        float exponent;
        void operator()() {
            auto a = out_node->child1;
            auto grad = out_node->grad;
            a->grad += grad * exponent*std::pow(a->data, exponent-1);
            LOG(out_node->op_str() << " backward result: " << a->str());
        };
    };

    out_node->backward = lambda{out_node, exponent};
    return Value(out_node);
}

template<typename T>
Value<T> Value<T>::operator*(const Value<T>& other) const {
    LOG("forward pass for " << *this << " * " << other);
    auto out_node = std::make_shared<Node>(get_data() * other.get_data(), Op::mult, node_, other.node_);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        void operator()() {
            auto a = out_node->child1;
            auto b = out_node->child2;
            auto grad = out_node->grad;
            a->grad += grad * b->data;
            b->grad += grad * a->data;
            LOG(out_node->op_str() << " backward result: " << a->str() << ", " << b->str());
        };
    };

    out_node->backward = lambda{out_node};
    return Value(out_node);
}

template<typename T>
Value<T> Value<T>::operator-() const {
    LOG("forward pass for -" << *this);
    return *this * -1;
}

template<typename T>
Value<T> Value<T>::operator-(const Value<T>& other) const {
    LOG("forward pass for " << *this << " - " << other);
    return *this + (-other);
}

template<typename T>
Value<T> Value<T>::operator/(const Value<T>& other) const {
    LOG("forward pass for " << *this << " / " << other);
    return *this * other.pow(-1);
}

template<typename T>
Value<T> Value<T>::exp() const {
    LOG("forward pass for " << *this << ".exp()");
    auto out_node = std::make_shared<Node>(std::exp(get_data()), Op::exp, node_, nullptr);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        void operator()() {
            auto a = out_node->child1;
            auto grad = out_node->grad;
            a->grad += grad * out_node->data;
            LOG(out_node->op_str() << " backward result: " << a->str());
        };
    };

    out_node->backward = lambda{out_node};
    return Value(out_node);
}

template<typename T>
Value<T> Value<T>::log() const {
    LOG("forward pass for " << *this << ".log()");
    auto out_node = std::make_shared<Node>(std::log(get_data()), Op::log, node_, nullptr);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        void operator()() {
            auto a = out_node->child1;
            auto grad = out_node->grad;
            a->grad += grad * (1 / a->data);
            LOG(out_node->op_str() << " backward result: " << a->str());
        };
    };

    out_node->backward = lambda{out_node};
    return Value(out_node);
}

template<typename T>
Value<T> Value<T>::tanh() const {
    LOG("forward pass for " << *this << ".tanh()");
    T exp_val = std::exp(get_data() * 2);
    auto out_node = std::make_shared<Node>((exp_val - 1) / (exp_val + 1), Op::tanh, node_, nullptr);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        void operator()() {
            auto a = out_node->child1;
            auto grad = out_node->grad;
            a->grad += grad * (1 - std::pow(out_node->data, 2));
            LOG(out_node->op_str() << " backward result: " << a->str());
        };
    };

    out_node->backward = lambda{out_node};
    return Value(out_node);
}

template<typename T>
Value<T> Value<T>::relu() const {
    LOG("forward pass for " << *this << ".relu()");
    auto out_node = std::make_shared<Node>(get_data() > 0 ? get_data() : 0, Op::relu, node_, nullptr);

    struct lambda {  // native lambda function work just as well but the debugger refuses to jump into them (for performance, it does not matter): https://stackoverflow.com/questions/50346822/does-lambda-object-construction-cost-a-lot
        std::shared_ptr<Node> out_node;
        void operator()() {
            auto a = out_node->child1;
            auto grad = out_node->grad;
            a->grad += grad * (out_node->data > 0 ? 1 : 0);
            LOG(out_node->op_str() << " backward result: " << a->str());
        };
    };

    out_node->backward = lambda{out_node};
    return Value(out_node);
}

template<typename T>
Value<T> Value<T>::operator+=(const Value& other) {
    *this = *this + other;
    return *this;
}

template<typename T>
Value<T> Value<T>::operator*=(const Value& other) {
    *this = *this * other;
    return *this;
}

template<typename T>
Value<T> Value<T>::operator-=(const Value& other) {
    *this = *this - other;
    return *this;
}

template<typename T>
Value<T> Value<T>::operator/=(const Value& other) {
    *this = *this / other;
    return *this;
}


template<typename T>
void Value<T>::topological_order(
        const std::shared_ptr<Node>& node,  // we only ever want to look at them, but don't own them (and they won't disappear mid-processing)
        std::vector<std::shared_ptr<Node>>& inout_topo,
        std::unordered_set<std::shared_ptr<Node>>& inout_visited) {
    if (!inout_visited.contains(node)) {
        inout_visited.insert(node);
        if (node->child1 != nullptr) {  // todo: use std::optional?
            topological_order(node->child1, inout_topo, inout_visited);
        }
        if (node->child2 != nullptr) {  // todo: use std::optional?
            topological_order(node->child2, inout_topo, inout_visited);
        }
        inout_topo.push_back(node);  // AFTER pushing the dependent nodes
        // TODO: push_back shares ownership by copying -- necessary? Alternatives?
    }
}

template<typename T>
void Value<T>::backward() {
    std::vector<std::shared_ptr<Node>> topo{};
    std::unordered_set<std::shared_ptr<Node>> visited{};
    topological_order(node_, topo, visited);
//        std::cout << *this << " vs " << *topo.front() << std::endl;
    set_grad(1);
    for (auto iter_node = topo.rbegin(); iter_node != topo.rend(); ++iter_node) {
        LOG("Processing " << (*iter_node)->str());
        auto back_func = (*iter_node)->backward;
        if (back_func != nullptr) {
            LOG("Calling " << (**iter_node).str() << ".backward():");
            back_func();
        }
    }
}


template<typename T>
T Value<T>::get_data() const {
    return node_->data;
}
template<typename T>
T Value<T>::get_grad() const {
    return node_->grad;
}
template<typename T>
auto Value<T>::get_node() const {
    return node_;
}
template<typename T>
void Value<T>::set_data(T data) {
    node_->data = data;
}
template<typename T>
void Value<T>::set_grad(T grad) {
    node_->grad = grad;
}

template<typename T>
Value<T>::operator double() const {
    return static_cast<double>(get_data());
}
template<typename T>
Value<T>::operator int() const {  // explicit
    return static_cast<int>(get_data());
}
template<typename T>
Value<T>::operator float() const {  // explicit
    return static_cast<float>(get_data());
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Value<T>& v) {
    auto node = v.get_node();
    if (node != nullptr) {
        return (os << "Value(" << v.get_data() << ",grad=" << v.get_grad() << ",node=" << v.get_node() << ")@" << &v);
    }
    else {
        return (os << "Value(?,grad=?,node=nullptr)@" << &v);
    }
}

} // namespace ajs
