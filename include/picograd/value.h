#pragma once

#include <functional>
#include <string>
#include <vector>           // std:vector
#include <algorithm>        // std:find
#include <unordered_set>
#include <cmath>            // std::log, std::pow, std::exp
#include <memory>           // smart pointers
#include <sstream>          // std::ostringstream

#if defined(VERBOSE_picograd) && !defined(NDEBUG)
#include <iostream>
#define FILENAME strrchr("/" __FILE__, '/') + 1
#define LOG(x) std::cout << FILENAME << " (" << __LINE__ << "): " << x << std::endl
#define LOGVAR(x) std::cout << FILENAME << " (" << __LINE__ << "): " << #x"=" << x << std::endl
#else
#define LOG(x)
#define LOGVAR(x)
#endif

namespace ajs {

template<typename T>
class Value
{
    enum class Op {
        none, add, sub, mult, div, neg, pow, exp, log, tanh, relu, sigmoid
    };

    struct Node {
        Node(T d) : data{d} {
            LOG("node value constructor with data=" << d << " at " << this);
        }
        Node(T d, Op o, std::shared_ptr<Node> ch1, std::shared_ptr<Node> ch2) : data{d}, op{o}, child1{ch1}, child2{ch2} {
            LOG("full node constructor " << this);
        }
        ~Node() {
            LOG("destroying node " << this);
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
            case Op::sigmoid:
                return "sigmoid";
            default:
                return "unknown";
            }
        }
        std::string str() {
            std::ostringstream stream;
            stream << "Node(" << data << ", grad=" << grad << ", op=" << op_str() << ")";
            return stream.str();
//            return std::string("Node(") + std::to_string(data) + ",grad=" + std::to_string(grad) + ",op=" + op_str() + ")";// + std::to_string(this);
        }

        void print_graph(uint8_t level=0, std::vector<int> running_levels={}) {
            std::ostringstream indent_os;
            for (uint8_t i=1; i<level; ++i) {
                if (std::find(running_levels.begin(), running_levels.end(), i-1) != running_levels.end()) {
                    indent_os << " |  ";
                }
                else {
                    indent_os << "    ";
                }
            }
            if (level > 0) indent_os << " |__";
            auto indent_str = indent_os.str();
            std::cout << indent_str << str() << "\n";

            running_levels.push_back(level);
            if (child1) child1->print_graph(level + 1, running_levels);
            running_levels.pop_back();
            if (child2) child2->print_graph(level + 1, running_levels);

            if (level == 0) std::cout << std::endl;
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
    Value(const Value& other);  // Copy constructor (new Value points to same inner Node as old value)
    Value(Value&& other);  // Move constructor
    Value& operator=(const Value& other);  // Copy assignment constructor (new Value points to same inner Node as old value)
    Value& operator=(Value&& other);  // Move assignment constructor
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
    Value log() const;
    Value tanh() const;
    Value relu() const;
    Value sigmoid() const;

    Value operator+=(const Value& other);
    Value operator*=(const Value& other);
    Value operator-=(const Value& other);
    Value operator/=(const Value& other);

    bool operator==(const Value& other) const;
    bool operator!=(const Value& other) const;
    bool operator<(const Value& other) const;

    void backward();
    void print_graph();

//    Value get_view() const;  // redundant with how copy constructor works --Get a new Value object that points to the same node and structure as this one

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
    std::shared_ptr<Node> node_{nullptr};
    void topological_order(const std::shared_ptr<Node>& node, std::vector<std::shared_ptr<Node>>& topo, std::unordered_set<std::shared_ptr<Node>>& visited);
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Value<T>& v);

} // namespace ajs

// Needed because this is a template library
#include "value.hpp"
