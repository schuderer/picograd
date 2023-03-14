#include "gtest/gtest.h"
#include <iostream>

#include "picograd/value.h"

using namespace ajs;

TEST(Value, SanityCheckLValue) {
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

    EXPECT_FLOAT_EQ(x.get_data(), -4.0f);
    EXPECT_FLOAT_EQ(y.get_data(), -20.0f);
    EXPECT_FLOAT_EQ(x.get_grad(), 46.0f);
}

TEST(Value, SanityCheckRValue) {
    // with rvalues:
    auto x = Value(-4.0);
    auto z = Value(2.0) * x + 2.0 + x;
    auto q = z.relu() + z * x;
    auto h = (z * z).relu();
    auto y = h + q + q * x;
    y.backward();

    EXPECT_DOUBLE_EQ(x.get_data(), -4.0);
    EXPECT_DOUBLE_EQ(y.get_data(), -20.0);
    EXPECT_DOUBLE_EQ(x.get_grad(), 46.0);
}

TEST(Value, TestMoreOps) {
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

    EXPECT_DOUBLE_EQ(g.get_data(), 24.70408163265306);
    EXPECT_DOUBLE_EQ(a.get_grad(), 138.83381924198252);
    EXPECT_DOUBLE_EQ(b.get_grad(), 645.5772594752186);
}

TEST(Value, Log) {
    auto a = Value(7.0);
    auto b = a.log();
    LOGVAR(a);
    LOGVAR(b);
    b.backward();
    LOGVAR(a);
    LOGVAR(b);
    EXPECT_DOUBLE_EQ(a.get_grad(), 1.0/7.0);
}

TEST(Value, TestPrinting) {
    auto a = (-Value(7.0) - 10) + Value(3.0) * -Value(-2.0).pow(2);
    a.print_graph();  // sorry, you'll have to check visually :)
}
