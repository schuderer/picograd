#pragma once

#include <functional>
#include <string>
#include <cmath>
#include <vector>
#include <set>
#include <memory>

namespace ajs {

template<typename T>
class Value
{
    enum class Op {
        none, add, sub, mult, div, neg, pow, exp, tanh, relu
    };

    struct Node {
        T m_data;
        T m_grad{0};
        Op m_op{Op::none};
        std::shared_ptr<Node> m_child1{nullptr};
        std::shared_ptr<Node> m_child2{nullptr};
        std::function<void()> m_backward{nullptr};
    };

public:
    Value();
    Value(const T& data);
    Value(const T& data, const std::string& op, const std::tuple<Value*, Value*> children);
    Value(const Value& other);  // copy constructor
    Value(Value&& other);  // move constructor
    Value operator=(const Value& other);  // copy assignment constructor
    Value operator=(Value&& other);  // move assignment constructor
    ~Value();

    Value operator+(Value& other);
    Value operator*(Value& other);
    Value operator-();
    Value operator-(Value& other);
    Value operator/(Value& other);
    Value pow(int exponent);
    Value pow(float exponent);
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
    std::shared_ptr<Node> m_node;
    T data_;
    T grad_{0};
    std::string op_{""};
    std::tuple<Value*, Value*> children_{nullptr, nullptr};
    std::function<void()> backward_{nullptr};
    void topological_order(Value* node, std::vector<Value*>& topo, std::set<Value*>& visited);
    void topological_order_nodes(Node* node, std::vector<Node*>& topo, std::set<Node*>& visited);
    T wrapped_std_pow(int exp);
    T wrapped_std_pow(float exp);
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Value<T>& v);

} // namespace ajs

// Needed because this is a template library
#include "picograd.cpp"
