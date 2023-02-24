#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <memory>

namespace ajs {

template<typename T>
class Value
{
    enum class Op {
        none, add, sub, mult, div, neg, pow, exp, tanh, relu
    };

    struct Node {
        Node(T d) : data{d} {
            std::cout << "node value constructor with data=" << d << " at " << this << std::endl;
        }
        Node(T d, Op o, std::shared_ptr<Node> ch1, std::shared_ptr<Node> ch2) : data{d}, op{o}, child1{ch1}, child2{ch2} {
            std::cout << "full node constructor " << this << std::endl;
        }
        ~Node() {
            std::cout << "destroying node " << this << std::endl;
        }
        std::string op_str() {
            switch (op) {
            case Op::none:
                return "none";
            case Op::add:
                return "+";
            case Op::sub:
                return "-";
            case Op::mult:
                return "*";
            case Op::div:
                return "/";
            case Op::neg:
                return "neg";
            case Op::pow:
                return "pow";
            case Op::exp:
                return "exp";
            case Op::tanh:
                return "tanh";
            case Op::relu:
                return "relu";
            default:
                return "unknown";
            }
        }
        std::string str() {
            return std::string("Node(") + std::to_string(data) + ",grad=" + std::to_string(grad) + ",op=" + op_str() + ")";// + std::to_string(this);
        }

        T data;
        T grad{0};
        Op op{Op::none};
        std::shared_ptr<Node> child1{nullptr};
        std::shared_ptr<Node> child2{nullptr};
        std::function<void()> backward{nullptr};
    };

public:
    Value();
    Value(const T data);
    Value(std::shared_ptr<Node> node);
    Value(const Value& other);  // copy constructor
    Value(Value&& other);  // move constructor
    Value& operator=(const Value& other);  // copy assignment constructor
    Value& operator=(Value&& other);  // move assignment constructor
    ~Value();

    // Arithmetic operator overloads
    // Need const reference here to deal with rvalues/temporaries (e.g. c = a + Value(3);). Without const, only accepts lvalues.
    // This also enables our number constructor to act as implicit converting constructor (c = a + 3) -- which is not exactly good style, though;
    Value operator+(const Value& other) const;
    Value operator*(const Value& other) const;
    Value operator-() const;
    Value operator-(const Value& other) const;
    Value operator/(const Value& other) const;
    Value pow(int exponent) const;
    Value pow(float exponent) const;
    Value exp() const;
    Value tanh() const;
    Value relu() const;

    Value operator+=(const Value& other);
    Value operator*=(const Value& other);
    Value operator-=(const Value& other);
    Value operator/=(const Value& other);

    void backward();
    void graph();

    T get_data() const;
    T get_grad() const;
    auto get_node() const;
    void set_data(T data);
    void set_grad(T grad);

    // User-defined conversion functions (so you can go int(Value(3)), float(Value(3)) etc.)
    // Implicit conversion functions seem to break gradient descent: Values are implicitly reduced to doubles and then copy-constructed, losing the gradient information
    explicit operator double() const;  // originally allowed implicit here, but this led to creation of new objects from old objects, losing object state
    explicit operator int() const;
    explicit operator float() const;

protected:
    std::shared_ptr<Node> node_;
    void topological_order(const std::shared_ptr<Node>& node, std::vector<std::shared_ptr<Node>>& topo, std::unordered_set<std::shared_ptr<Node>>& visited);
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Value<T>& v);

} // namespace ajs

// Needed because this is a template library
#include "picograd.cpp"
