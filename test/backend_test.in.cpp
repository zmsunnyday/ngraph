/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static const vector<element::Type> s_known_element_types = {element::from<float>(),
                                                            element::from<double>(),
                                                            element::from<int8_t>(),
                                                            element::from<int16_t>(),
                                                            element::from<int32_t>(),
                                                            element::from<int64_t>(),
                                                            element::from<uint8_t>(),
                                                            element::from<uint16_t>(),
                                                            element::from<uint32_t>(),
                                                            element::from<uint64_t>()};

TEST(${BACKEND_NAME}, function_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, op::ParameterVector{A, B}, "funky func name");

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

TEST(${BACKEND_NAME}, node_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    C->set_name("a node name");
    auto f = make_shared<Function>(C, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

TEST(${BACKEND_NAME}, component_cleanup)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    shared_ptr<runtime::Backend> backend;
    shared_ptr<runtime::ExternalFunction> external;
    shared_ptr<runtime::CallFrame> cf;
    {
        Shape shape{2, 2};
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(A + B, op::ParameterVector{A, B});

        auto manager = runtime::Manager::get("${BACKEND_NAME}");
        external = manager->compile(f);
        backend = manager->allocate_backend();
        cf = backend->make_call_frame(external);
    }
    EXPECT_EQ(cf.use_count(), 1);
    cf = nullptr;
    EXPECT_EQ(backend.use_count(), 1);
    backend = nullptr;
    EXPECT_EQ(external.use_count(), 1);
}

TEST(${BACKEND_NAME}, aliased_output)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    auto D = A * B;
    auto E = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto f = make_shared<Function>(NodeVector{C, C, D, D, C, E, E}, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> out1 = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> out2 = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> out3 = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> out4 = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> out5 = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> out6 = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> out7 = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, vector<float>{0, 1, 2, 3});
    copy_data(b, vector<float>{1, 2, 3, 4});
    vector<float> expectedC{1, 3, 5, 7};
    vector<float> expectedD{0, 2, 6, 12};
    vector<float> expectedE{1, 2, 3, 4};

    cf->call({out1, out2, out3, out4, out5, out6, out7}, {a, b});
    EXPECT_EQ(expectedC, read_vector<float>(out1));
    EXPECT_EQ(expectedC, read_vector<float>(out2));
    EXPECT_EQ(expectedD, read_vector<float>(out3));
    EXPECT_EQ(expectedD, read_vector<float>(out4));
    EXPECT_EQ(expectedC, read_vector<float>(out5));
    EXPECT_EQ(expectedE, read_vector<float>(out6));
    EXPECT_EQ(expectedE, read_vector<float>(out7));
}

TEST(${BACKEND_NAME}, parameter_as_output)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");

    Shape shape{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->make_primary_tensor_view(element::f32, shape);

    vector<float> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> zero(shape_size(shape), 0);
    copy_data(a, expected);

    cf->call({result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(${BACKEND_NAME}, ab)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

TEST(${BACKEND_NAME}, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    shared_ptr<runtime::ExternalFunction> external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    cf->call({result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    cf->call({result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

TEST(${BACKEND_NAME}, abc_int64)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto C = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::i64, shape);
    copy_data(b, vector<int64_t>{5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::i64, shape);
    copy_data(c, vector<int64_t>{9, 10, 11, 12});
    auto result = backend->make_primary_tensor_view(element::i64, shape);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    cf->call({result}, {b, a, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    cf->call({result}, {a, c, b});
    EXPECT_EQ((vector<int64_t>{50, 72, 98, 128}), read_vector<int64_t>(result));
}

// Multiple retrive values
TEST(${BACKEND_NAME}, multiple_result)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto A_add_B = make_shared<op::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::Multiply>(A_add_B, C);

    auto f =
        make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(c, vector<float>{9, 10, 11, 12});

    auto r0 = backend->make_primary_tensor_view(element::f32, shape);
    auto r1 = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({r0, r1}, {a, b, c});

    EXPECT_EQ((vector<float>{6, 8, 10, 12}), read_vector<float>(r0));
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(r1));
}

TEST(${BACKEND_NAME}, abs)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 0, 4.75f}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, ceiling)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Ceiling>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{-2.0f, -2.0f, 1.0f, 5.0f}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, concat_matrix_colwise)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 8};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 1),
                                   op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, concat_matrix_rowwise)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, concat_matrix_int64)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::i64, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::i64, shape_b);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::i64, shape_c);
    copy_data(c, vector<int64_t>{2, 3, 5, 7, 11, 13});
    auto result = backend->make_primary_tensor_view(element::i64, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<int64_t>(result));
}

TEST(${BACKEND_NAME}, concat_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{6};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{12};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::f32, shape_c);
    copy_data(c, vector<float>{18, 19});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}), read_vector<float>(result));
}

// from numpy import *
// a=linspace(1,2*3*4*3*2,2*3*4*3*2)
// b=linspace(1000+1,1000+2*3*3*3*2,2*3*3*3*2)
// c=linspace(2000+1,2000+2*3*2*3*2,2*3*2*3*2)
// a.shape=(2,3,4,3,2)
// b.shape=(2,3,3,3,2)
// c.shape=(2,3,2,3,2)
// z=concatenate((a,b,c),axis=2)
// z.shape=(2*3*(4+3+2)*3*2)
// set_printoptions(suppress=True)
// print(z)
//
// [    1.     2.     3.     4.     5.     6.     7.     8.     9.    10.
//     11.    12.    13.    14.    15.    16.    17.    18.    19.    20.
//     21.    22.    23.    24.  1001.  1002.  1003.  1004.  1005.  1006.
//   1007.  1008.  1009.  1010.  1011.  1012.  1013.  1014.  1015.  1016.
//   1017.  1018.  2001.  2002.  2003.  2004.  2005.  2006.  2007.  2008.
//   2009.  2010.  2011.  2012.    25.    26.    27.    28.    29.    30.
//     31.    32.    33.    34.    35.    36.    37.    38.    39.    40.
//     41.    42.    43.    44.    45.    46.    47.    48.  1019.  1020.
//   1021.  1022.  1023.  1024.  1025.  1026.  1027.  1028.  1029.  1030.
//   1031.  1032.  1033.  1034.  1035.  1036.  2013.  2014.  2015.  2016.
//   2017.  2018.  2019.  2020.  2021.  2022.  2023.  2024.    49.    50.
//     51.    52.    53.    54.    55.    56.    57.    58.    59.    60.
//     61.    62.    63.    64.    65.    66.    67.    68.    69.    70.
//     71.    72.  1037.  1038.  1039.  1040.  1041.  1042.  1043.  1044.
//   1045.  1046.  1047.  1048.  1049.  1050.  1051.  1052.  1053.  1054.
//   2025.  2026.  2027.  2028.  2029.  2030.  2031.  2032.  2033.  2034.
//   2035.  2036.    73.    74.    75.    76.    77.    78.    79.    80.
//     81.    82.    83.    84.    85.    86.    87.    88.    89.    90.
//     91.    92.    93.    94.    95.    96.  1055.  1056.  1057.  1058.
//   1059.  1060.  1061.  1062.  1063.  1064.  1065.  1066.  1067.  1068.
//   1069.  1070.  1071.  1072.  2037.  2038.  2039.  2040.  2041.  2042.
//   2043.  2044.  2045.  2046.  2047.  2048.    97.    98.    99.   100.
//    101.   102.   103.   104.   105.   106.   107.   108.   109.   110.
//    111.   112.   113.   114.   115.   116.   117.   118.   119.   120.
//   1073.  1074.  1075.  1076.  1077.  1078.  1079.  1080.  1081.  1082.
//   1083.  1084.  1085.  1086.  1087.  1088.  1089.  1090.  2049.  2050.
//   2051.  2052.  2053.  2054.  2055.  2056.  2057.  2058.  2059.  2060.
//    121.   122.   123.   124.   125.   126.   127.   128.   129.   130.
//    131.   132.   133.   134.   135.   136.   137.   138.   139.   140.
//    141.   142.   143.   144.  1091.  1092.  1093.  1094.  1095.  1096.
//   1097.  1098.  1099.  1100.  1101.  1102.  1103.  1104.  1105.  1106.
//   1107.  1108.  2061.  2062.  2063.  2064.  2065.  2066.  2067.  2068.
//   2069.  2070.  2071.  2072.]
TEST(${BACKEND_NAME}, concat_5d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    vector<float> a_data(2 * 3 * 4 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 4 * 3 * 2; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 3 * 2; i++)
    {
        b_data[i] = 1000 + float(i + 1);
    }

    vector<float> c_data(2 * 3 * 2 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 2 * 3 * 2; i++)
    {
        c_data[i] = 2000 + float(i + 1);
    }

    Shape shape_a{2, 3, 4, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3, 2, 3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 3, 9, 3, 2};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 2);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->make_primary_tensor_view(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ(
        (vector<float>{
            1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,   12.,
            13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,
            1001., 1002., 1003., 1004., 1005., 1006., 1007., 1008., 1009., 1010., 1011., 1012.,
            1013., 1014., 1015., 1016., 1017., 1018., 2001., 2002., 2003., 2004., 2005., 2006.,
            2007., 2008., 2009., 2010., 2011., 2012., 25.,   26.,   27.,   28.,   29.,   30.,
            31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,   41.,   42.,
            43.,   44.,   45.,   46.,   47.,   48.,   1019., 1020., 1021., 1022., 1023., 1024.,
            1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036.,
            2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024.,
            49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,
            61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
            1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044., 1045., 1046., 1047., 1048.,
            1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028., 2029., 2030.,
            2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,   76.,   77.,   78.,
            79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,
            91.,   92.,   93.,   94.,   95.,   96.,   1055., 1056., 1057., 1058., 1059., 1060.,
            1061., 1062., 1063., 1064., 1065., 1066., 1067., 1068., 1069., 1070., 1071., 1072.,
            2037., 2038., 2039., 2040., 2041., 2042., 2043., 2044., 2045., 2046., 2047., 2048.,
            97.,   98.,   99.,   100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,
            109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,
            1073., 1074., 1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084.,
            1085., 1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054.,
            2055., 2056., 2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,  126.,
            127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,  136.,  137.,  138.,
            139.,  140.,  141.,  142.,  143.,  144.,  1091., 1092., 1093., 1094., 1095., 1096.,
            1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104., 1105., 1106., 1107., 1108.,
            2061., 2062., 2063., 2064., 2065., 2066., 2067., 2068., 2069., 2070., 2071., 2072.}),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, divide)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        auto external = manager->compile(f);
        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, divide_adjoint_stability)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        auto Y_out = f->get_output_op(0);
        auto Xs = f->get_parameters();
        auto C = std::make_shared<op::Parameter>(Y_out->get_element_type(), Y_out->get_shape());
        std::vector<std::shared_ptr<Node>> dYdXs(Xs.size());
        transform(Xs.begin(), Xs.end(), dYdXs.begin(), [C, Y_out](const std::shared_ptr<Node>& X) {
            return Y_out->backprop_node(X, C);
        });
        std::vector<std::shared_ptr<op::Parameter>> params(Xs);
        params.push_back(C);

        auto bf = std::make_shared<Function>(dYdXs, params);
        auto external = manager->compile(bf);

        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{0, 0, 1, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{2, 2, 2, 2});
    auto c = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(c, vector<float>{1, 1, 1, 1});

    auto resulta = backend->make_primary_tensor_view(element::f32, shape);
    auto resultb = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({resulta, resultb}, {a, b, c});
    EXPECT_EQ((vector<float>{0.5, 0.5, 0.5, 0.5}), read_vector<float>(resulta));
    EXPECT_EQ((vector<float>{-0.0, -0.0, -0.25, -0.25}), read_vector<float>(resultb));
}

TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        auto external = manager->compile(f);
        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, divide_by_zero_int32)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::i32, shape);
        auto B = make_shared<op::Parameter>(element::i32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        auto external = manager->compile(f);
        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape);
    copy_data(a, vector<int>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::i32, shape);
    copy_data(b, vector<int>{0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::i32, shape);

    EXPECT_ANY_THROW({ cf->call({result}, {a, b}); });
}

TEST(${BACKEND_NAME}, equal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, floor)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{-3.0f, -2.0f, 0.0f, 4.0f}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_0_0)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{0};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112});

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_matrix_2x0_0x2)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{2, 0};
    Shape shape_b{0, 2};
    Shape shape_r{2, 2};

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

        auto external = manager->compile(f);
        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112});

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{0, 0, 0, 0}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_matrix_0x2_2x0)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 0};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_matrix_3x2_2x0)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{3, 0};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_scalar_0x2)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_2x0_0)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{2, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112});

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{0, 0}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot1d)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{170}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot2d)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{19, 22, 43, 50}), read_vector<float>(result));
}

//
// Here is what numpy does:
//
// >>> a = linspace(1,2*2*2,2*2*2)
// >>> b = linspace(1,2*2*2,2*2*2)
//
// >>> a.shape=(2,2,2)
// >>> b.shape=(2,2,2)
//
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[[ 11.,  14.],
//          [ 17.,  20.]],
//
//         [[ 23.,  30.],
//          [ 37.,  44.]]],
//
//
//        [[[ 35.,  46.],
//          [ 57.,  68.]],
//
//         [[ 47.,  62.],
//          [ 77.,  92.]]]])
//
TEST(${BACKEND_NAME}, dot3d_3d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, 47, 62, 77, 92}),
              read_vector<float>(result));
}

//
// Here is what numpy does:
//
// >>> from numpy import *
// >>> a = linspace(0,4*2*3-1,4*2*3)
// >>> b = linspace(0,3*4-1,3*4)
//
// >>> a.shape=(4,2,3)
// >>> b.shape=(3,4)
//
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[  20.,   23.,   26.,   29.],
//         [  56.,   68.,   80.,   92.]],
//
//        [[  92.,  113.,  134.,  155.],
//         [ 128.,  158.,  188.,  218.]],
//
//        [[ 164.,  203.,  242.,  281.],
//         [ 200.,  248.,  296.,  344.]],
//
//        [[ 236.,  293.,  350.,  407.],
//         [ 272.,  338.,  404.,  470.]]])
//
TEST(${BACKEND_NAME}, dot3d_2d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 2, 4};
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{20,  23,  26,  29,  56,  68,  80,  92,  92,  113, 134,
                             155, 128, 158, 188, 218, 164, 203, 242, 281, 200, 248,
                             296, 344, 236, 293, 350, 407, 272, 338, 404, 470}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_scalar_tensor_arg0)
{
    Shape shape_a{};
    Shape shape_b{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape_b);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_scalar_tensor_arg1)
{
    Shape shape_a{2, 2, 2};
    Shape shape_b{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_a);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_scalar_scalar)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{8});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{48}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_matrix_vector)
{
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});
    Shape shape_r{4};

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{17, 18, 19, 20});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{190, 486, 782, 1078}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, dot_matrix_vector_int64)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 4};
    Shape shape_b{4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), op::ParameterVector{A, B});
    Shape shape_r{4};

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->make_primary_tensor_view(element::i64, shape_b);
    copy_data(b, vector<int64_t>{17, 18, 19, 20});
    auto result = backend->make_primary_tensor_view(element::i64, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<int64_t>{190, 486, 782, 1078}), read_vector<int64_t>(result));
}

TEST(${BACKEND_NAME}, greater)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Greater>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, greatereq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::GreaterEq>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 1, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, less)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Less>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, lesseq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 0, 1, 1, 0, 1}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, lesseq_bool)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::boolean, shape);
    copy_data(a, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});
    auto b = backend->make_primary_tensor_view(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 0, 0, 0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 0, 0, 0, 0, 0, 0}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, log)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Log>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(
        a, vector<float>{expf(1), expf(2), expf(3), expf(4), expf(5), expf(6), expf(7), expf(8)});
    vector<float> loga;
    for (auto elt : read_vector<float>(a))
    {
        loga.push_back(logf(elt));
    }
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_TRUE(test::all_close(loga, read_vector<float>(result)));
}

TEST(${BACKEND_NAME}, maximum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, minimum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, negative)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Negative>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f, 8.75f, -8.75f});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{-1, 2, 0, 4.75f, -8.75f, 8.75f}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, notequal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::NotEqual>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, select)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Select>(A, B, C), op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::boolean, shape);
    copy_data(a, vector<char>{0, 1, 1, 0, 0, 1, 0, 1});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(c, vector<float>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, subtract)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, tensor_constant)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(A, op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, tensor_constant_with_op)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {-1, 2, 3, -4, 5, -6, -7, 8});
    auto f = make_shared<Function>(make_shared<op::Abs>(A), op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, constant_broadcast)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    const string js =
        R"([{
       "name" : "Function_0",
       "ops" : [
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Parameter_4",
             "op" : "Parameter",
             "outputs" : ["Parameter_4"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Parameter_0",
             "op" : "Parameter",
             "outputs" : ["Parameter_0"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Constant_1",
             "op" : "Constant",
             "outputs" : ["Constant_1"],
             "shape" : [],
             "value" : ["0"]
           },
           {
             "axes" : [ 0, 1 ],
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : ["Constant_1"],
             "name" : "Broadcast_2",
             "op" : "Broadcast",
             "outputs" : ["Broadcast_2"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [ "Parameter_0", "Broadcast_2" ],
             "name" : "Maximum_3",
             "op" : "Maximum",
             "outputs" : ["Maximum_3"]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [ "Maximum_3", "Parameter_4" ],
             "name" : "Multiply_5",
             "op" : "Multiply",
             "outputs" : ["Multiply_5"]
           }
       ],
       "parameters" : [ "Parameter_0", "Parameter_4" ],
       "result" : ["Multiply_5"],
       "result_shape" : [ 3, 4 ],
       "result_type" :
           {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true}
    }])";
    stringstream ss(js);

    shared_ptr<Function> f = ngraph::deserialize(ss);

    // max(x,broadcast(Constant(0)))
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // If this compiles it works
}

TEST(${BACKEND_NAME}, function_call)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto g =
        make_shared<Function>(make_shared<op::FunctionCall>(f, NodeVector{X + Y, Y + Z, Z + X}) +
                                  make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z}),
                              op::ParameterVector{X, Y, Z});

    // Now call g on some test vectors.
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto x = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {x, y, z});
    EXPECT_EQ((vector<float>{254, 368, 502, 656}), read_vector<float>(result));

    cf->call({result}, {y, x, z});
    EXPECT_EQ((vector<float>{278, 400, 542, 704}), read_vector<float>(result));

    cf->call({result}, {x, z, y});
    EXPECT_EQ((vector<float>{194, 296, 418, 560}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_scalar_vector)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_scalar_matrix)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_scalar_tensor)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1, 2}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_trivial)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape, AxisSet{}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 6, 8, 16, 32, 64, 128});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_vector_colwise)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_vector_rowwise)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), read_vector<float>(result));
}

// Test hybrid mechanism after broadcast
TEST(${BACKEND_NAME}, broadcast_vector_rowwise_reversed)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");

    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto broadcast = make_shared<op::Broadcast>(A, shape_r, AxisSet{0});
    auto reverse = make_shared<op::Reverse>(broadcast, AxisSet{1});
    auto f = make_shared<Function>(reverse, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_vector_rowwise_int64)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::i64, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), read_vector<int64_t>(result));
}

TEST(${BACKEND_NAME}, broadcast_matrix_0)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_matrix_1)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 1, 2, 3, 4, 3, 4}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, broadcast_matrix_2)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{2}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 2, 2, 3, 3, 4, 4}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, convert_int32_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Convert>(A, element::f32), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, convert_int32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::boolean),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, convert_float32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::boolean),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

// Trivial case with no reduction axes.
TEST(${BACKEND_NAME}, reduce_trivial)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reduce_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::f32, Shape{});

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

TEST(${BACKEND_NAME}, reduce_matrix_columns)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};

    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->make_primary_tensor_view(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

TEST(${BACKEND_NAME}, reduce_matrix_rows)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->make_primary_tensor_view(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

TEST(${BACKEND_NAME}, reduce_matrix_rows_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, Shape{});
    copy_data(b, vector<float>{66});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{66, 66, 66}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{66}), read_vector<float>(b));
}

TEST(${BACKEND_NAME}, reduce_matrix_cols_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, Shape{});
    copy_data(b, vector<float>{77});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{77, 77}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{77}), read_vector<float>(b));
}

TEST(${BACKEND_NAME}, reduce_vector_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, Shape{});
    copy_data(b, vector<float>{88});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{88}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{88}), read_vector<float>(b));
}

TEST(${BACKEND_NAME}, reduce_matrix_to_scalar_zero_by_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::f32, Shape{});
    copy_data(b, vector<float>{99});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{99}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{99}), read_vector<float>(b));
}

TEST(${BACKEND_NAME}, reduce_3d_to_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}"); // Correct values but need to handle precisions

    // First, the reduction function (f(x:float32[],y:float32[]) = x*y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(make_shared<op::Multiply>(f_A, f_B), op::ParameterVector{f_A, f_B});

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(A, B, f, AxisSet{0, 1}),
                                   op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{1});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_t2v_012)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_t2s_012)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_t2s_120)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 2, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_s2t)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 1, 1, 1, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{42});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{42}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_v2m_col)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_v2m_row)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_v2t_middle)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 3, 1};
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_m2m_same)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_m2m_transpose)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reshape_m2m_dim_change_transpose)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 3};
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 3, 5, 2, 4, 6}), read_vector<float>(result));
}

//
// Numpy:
//
// >>> x = linspace(1,2*2*3*3*2*4,2*2*3*3*2*4)
// >>> x.shape=(2,2,3,3,2,4)
// >>> y = ascontiguousarray(transpose(x,(2,4,0,5,3,1)))
// >>> y.shape=2*2*3*3*2*4
// >>> y
// array([   1.,   73.,    9.,   81.,   17.,   89.,    2.,   74.,   10.,
//          82.,   18.,   90.,    3.,   75.,   11.,   83.,   19.,   91.,
//           4.,   76.,   12.,   84.,   20.,   92.,  145.,  217.,  153.,
//         225.,  161.,  233.,  146.,  218.,  154.,  226.,  162.,  234.,
//         147.,  219.,  155.,  227.,  163.,  235.,  148.,  220.,  156.,
//         228.,  164.,  236.,    5.,   77.,   13.,   85.,   21.,   93.,
//           6.,   78.,   14.,   86.,   22.,   94.,    7.,   79.,   15.,
//          87.,   23.,   95.,    8.,   80.,   16.,   88.,   24.,   96.,
//         149.,  221.,  157.,  229.,  165.,  237.,  150.,  222.,  158.,
//         230.,  166.,  238.,  151.,  223.,  159.,  231.,  167.,  239.,
//         152.,  224.,  160.,  232.,  168.,  240.,   25.,   97.,   33.,
//         105.,   41.,  113.,   26.,   98.,   34.,  106.,   42.,  114.,
//          27.,   99.,   35.,  107.,   43.,  115.,   28.,  100.,   36.,
//         108.,   44.,  116.,  169.,  241.,  177.,  249.,  185.,  257.,
//         170.,  242.,  178.,  250.,  186.,  258.,  171.,  243.,  179.,
//         251.,  187.,  259.,  172.,  244.,  180.,  252.,  188.,  260.,
//          29.,  101.,   37.,  109.,   45.,  117.,   30.,  102.,   38.,
//         110.,   46.,  118.,   31.,  103.,   39.,  111.,   47.,  119.,
//          32.,  104.,   40.,  112.,   48.,  120.,  173.,  245.,  181.,
//         253.,  189.,  261.,  174.,  246.,  182.,  254.,  190.,  262.,
//         175.,  247.,  183.,  255.,  191.,  263.,  176.,  248.,  184.,
//         256.,  192.,  264.,   49.,  121.,   57.,  129.,   65.,  137.,
//          50.,  122.,   58.,  130.,   66.,  138.,   51.,  123.,   59.,
//         131.,   67.,  139.,   52.,  124.,   60.,  132.,   68.,  140.,
//         193.,  265.,  201.,  273.,  209.,  281.,  194.,  266.,  202.,
//         274.,  210.,  282.,  195.,  267.,  203.,  275.,  211.,  283.,
//         196.,  268.,  204.,  276.,  212.,  284.,   53.,  125.,   61.,
//         133.,   69.,  141.,   54.,  126.,   62.,  134.,   70.,  142.,
//          55.,  127.,   63.,  135.,   71.,  143.,   56.,  128.,   64.,
//         136.,   72.,  144.,  197.,  269.,  205.,  277.,  213.,  285.,
//         198.,  270.,  206.,  278.,  214.,  286.,  199.,  271.,  207.,
//         279.,  215.,  287.,  200.,  272.,  208.,  280.,  216.,  288.])
//
TEST(${BACKEND_NAME}, reshape_6d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    vector<float> a_data(2 * 2 * 3 * 3 * 2 * 4);
    for (int i = 0; i < 2 * 2 * 3 * 3 * 2 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2, 2, 4, 3, 2};

    auto r = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, a_data);

    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ(
        (vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,   75.,
            11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217., 153., 225.,
            161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155., 227., 163., 235.,
            148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,   78.,
            14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,  8.,   80.,  16.,  88.,
            24.,  96.,  149., 221., 157., 229., 165., 237., 150., 222., 158., 230., 166., 238.,
            151., 223., 159., 231., 167., 239., 152., 224., 160., 232., 168., 240., 25.,  97.,
            33.,  105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,  35.,  107.,
            43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177., 249., 185., 257.,
            170., 242., 178., 250., 186., 258., 171., 243., 179., 251., 187., 259., 172., 244.,
            180., 252., 188., 260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  110.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  120.,
            173., 245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 247.,
            183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,  129.,
            65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,  139.,
            52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209., 281., 194., 266.,
            202., 274., 210., 282., 195., 267., 203., 275., 211., 283., 196., 268., 204., 276.,
            212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142.,
            55.,  127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144., 197., 269.,
            205., 277., 213., 285., 198., 270., 206., 278., 214., 286., 199., 271., 207., 279.,
            215., 287., 200., 272., 208., 280., 216., 288.}),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sin)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sin>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 6, -pi, pi};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinf(x); });

    cf->call({result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, cos)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 3, -pi, pi};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return cosf(x); });

    cf->call({result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, tan)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tan>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{pi / 4, 0.0f, -0.0f, 7 * pi / 4, 3 * pi / 4, 5 * pi / 4};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanf(x); });

    cf->call({result}, {a});
    EXPECT_TRUE(test::all_close(input, read_vector<float>(result)));
}

TEST(${BACKEND_NAME}, asin)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Asin>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return asinf(x); });

    cf->call({result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, acos)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Acos>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return acosf(x); });

    cf->call({result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, atan)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Atan>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return atanf(x); });

    cf->call({result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sinh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sinh>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinhf(x); });

    cf->call({result}, {a});
    EXPECT_EQ(input, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, cosh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return coshf(x); });

    cf->call({result}, {a});
    EXPECT_TRUE(test::all_close(input, read_vector<float>(result)));
}

TEST(${BACKEND_NAME}, tanh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tanh>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanhf(x); });

    cf->call({result}, {a});
    EXPECT_TRUE(test::all_close(input, read_vector<float>(result)));
}

TEST(${BACKEND_NAME}, exp)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Exp>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{-4, -3, -2, -1, 0, 1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ(
        (vector<float>{expf(-4), expf(-3), expf(-2), expf(-1), expf(0), expf(1), expf(2), expf(3)}),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, slice_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, slice_matrix)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, slice_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, slice_matrix_strided)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4}, Strides{2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{4, 7, 12, 15}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, slice_3d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, slice_3d_strided)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, slice_3d_strided_different_strides)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, scalar_constant_float32)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    auto r = op::Constant::create(element::f32, Shape{}, {4.75});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::f32, Shape{});

    cf->call({result}, {});
    EXPECT_EQ(vector<float>{4.75f}, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, scalar_constant_int64)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    auto r = op::Constant::create(element::i64, Shape{}, {2112});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::i64, Shape{});

    cf->call({result}, {});
    EXPECT_EQ(vector<int64_t>{2112}, read_vector<int64_t>(result));
}

TEST(${BACKEND_NAME}, tensor_constant_float32)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto r = op::Constant::create(element::f32, shape, {4.75, 4.7, -5.3, 0.0});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {});
    EXPECT_EQ((vector<float>{4.75f, 4.7f, -5.3f, 0.0f}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, tensor_constant_int64)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto r = op::Constant::create(element::i64, shape, {2112, 1848, 1776, 1964});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::i64, shape);

    cf->call({result}, {});
    EXPECT_EQ((vector<int64_t>{2112, 1848, 1776, 1964}), read_vector<int64_t>(result));
}

// Trivial case with no summed axes.
TEST(${BACKEND_NAME}, sum_trivial)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
TEST(${BACKEND_NAME}, sum_trivial_5d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sum_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, Shape{});

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, sum_matrix_columns)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, sum_matrix_rows)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, sum_matrix_rows_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, sum_matrix_cols_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, sum_vector_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, sum_3d_to_matrix_most_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19,
                             2 + 11 + 20,
                             3 + 12 + 21,
                             4 + 13 + 22,
                             5 + 14 + 23,
                             6 + 15 + 24,
                             7 + 16 + 25,
                             8 + 17 + 26,
                             9 + 18 + 27}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sum_3d_to_matrix_least_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1 + 2 + 3,
                             4 + 5 + 6,
                             7 + 8 + 9,
                             10 + 11 + 12,
                             13 + 14 + 15,
                             16 + 17 + 18,
                             19 + 20 + 21,
                             22 + 23 + 24,
                             25 + 26 + 27}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sum_3d_to_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                             2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                             3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sum_3d_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 +
                             8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sum_3d_eliminate_zero_dim)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0, 0, 0, 0}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sum_to_scalar_stable)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1e-6f, -1, 0, 1});
    auto result = backend->make_primary_tensor_view(element::f32, Shape{});

    cf->call({result}, {a});
    EXPECT_TRUE(test::all_close(read_vector<float>(result), vector<float>{1e-6f}, 5e-2f));
    // EXPECT_EQ(vector<float>{1e-6}, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, sum_3d_to_vector_stable)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 1,  1,  1,  1,  1,  1e-4f, 1e-5f, 1e-6f, 1,  1,  1,  1, 1,
                               1, -1, -1, -1, -1, -1, -1,    -1,    -1,    -1, -1, -1, -1});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_TRUE(
        test::all_close(read_vector<float>(result), vector<float>{1e-4f, 1e-5f, 1e-6f}, 5e-2f));
}

TEST(${BACKEND_NAME}, sign)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sign>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f, 4.8f, -0.0f});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, -1, 0, -1, 1, 0}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, power)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Power>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 5});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{2, 0, 6, 3});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a, b});
    EXPECT_TRUE(test::all_close(vector<float>{1, 1, 729, 125}, read_vector<float>(result)));
}

TEST(${BACKEND_NAME}, constant_equality_bool)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{4};
    // auto A = make_shared<op::Parameter>(element::boolean, shape);
    // auto B = make_shared<op::Parameter>(element::boolean, shape);
    // auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{A, B});

    auto A = op::Constant::create(element::boolean, shape, {true, false, true, false});
    auto B = op::Constant::create(element::boolean, shape, {true, true, true, true});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {});
    EXPECT_EQ((vector<char>{true, false, true, false}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, sqrt)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{16, 4, 81, 100, 10000, 0});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{4, 2, 9, 10, 100, 0}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, replace_slice_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{808});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{808}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, replace_slice_matrix)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 106, 107, 110, 111});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, replace_slice_matrix_step)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{64, 64};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 32};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{64, 64};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{0, 0}, Coordinate{32, 32});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(
        a,
        vector<float>{
            1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,
            15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,
            29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,
            43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,
            57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,
            71,   72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,
            85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,   98,
            99,   100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,  112,
            113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,  126,
            127,  128,  129,  130,  131,  132,  133,  134,  135,  136,  137,  138,  139,  140,
            141,  142,  143,  144,  145,  146,  147,  148,  149,  150,  151,  152,  153,  154,
            155,  156,  157,  158,  159,  160,  161,  162,  163,  164,  165,  166,  167,  168,
            169,  170,  171,  172,  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,
            183,  184,  185,  186,  187,  188,  189,  190,  191,  192,  193,  194,  195,  196,
            197,  198,  199,  200,  201,  202,  203,  204,  205,  206,  207,  208,  209,  210,
            211,  212,  213,  214,  215,  216,  217,  218,  219,  220,  221,  222,  223,  224,
            225,  226,  227,  228,  229,  230,  231,  232,  233,  234,  235,  236,  237,  238,
            239,  240,  241,  242,  243,  244,  245,  246,  247,  248,  249,  250,  251,  252,
            253,  254,  255,  256,  257,  258,  259,  260,  261,  262,  263,  264,  265,  266,
            267,  268,  269,  270,  271,  272,  273,  274,  275,  276,  277,  278,  279,  280,
            281,  282,  283,  284,  285,  286,  287,  288,  289,  290,  291,  292,  293,  294,
            295,  296,  297,  298,  299,  300,  301,  302,  303,  304,  305,  306,  307,  308,
            309,  310,  311,  312,  313,  314,  315,  316,  317,  318,  319,  320,  321,  322,
            323,  324,  325,  326,  327,  328,  329,  330,  331,  332,  333,  334,  335,  336,
            337,  338,  339,  340,  341,  342,  343,  344,  345,  346,  347,  348,  349,  350,
            351,  352,  353,  354,  355,  356,  357,  358,  359,  360,  361,  362,  363,  364,
            365,  366,  367,  368,  369,  370,  371,  372,  373,  374,  375,  376,  377,  378,
            379,  380,  381,  382,  383,  384,  385,  386,  387,  388,  389,  390,  391,  392,
            393,  394,  395,  396,  397,  398,  399,  400,  401,  402,  403,  404,  405,  406,
            407,  408,  409,  410,  411,  412,  413,  414,  415,  416,  417,  418,  419,  420,
            421,  422,  423,  424,  425,  426,  427,  428,  429,  430,  431,  432,  433,  434,
            435,  436,  437,  438,  439,  440,  441,  442,  443,  444,  445,  446,  447,  448,
            449,  450,  451,  452,  453,  454,  455,  456,  457,  458,  459,  460,  461,  462,
            463,  464,  465,  466,  467,  468,  469,  470,  471,  472,  473,  474,  475,  476,
            477,  478,  479,  480,  481,  482,  483,  484,  485,  486,  487,  488,  489,  490,
            491,  492,  493,  494,  495,  496,  497,  498,  499,  500,  501,  502,  503,  504,
            505,  506,  507,  508,  509,  510,  511,  512,  513,  514,  515,  516,  517,  518,
            519,  520,  521,  522,  523,  524,  525,  526,  527,  528,  529,  530,  531,  532,
            533,  534,  535,  536,  537,  538,  539,  540,  541,  542,  543,  544,  545,  546,
            547,  548,  549,  550,  551,  552,  553,  554,  555,  556,  557,  558,  559,  560,
            561,  562,  563,  564,  565,  566,  567,  568,  569,  570,  571,  572,  573,  574,
            575,  576,  577,  578,  579,  580,  581,  582,  583,  584,  585,  586,  587,  588,
            589,  590,  591,  592,  593,  594,  595,  596,  597,  598,  599,  600,  601,  602,
            603,  604,  605,  606,  607,  608,  609,  610,  611,  612,  613,  614,  615,  616,
            617,  618,  619,  620,  621,  622,  623,  624,  625,  626,  627,  628,  629,  630,
            631,  632,  633,  634,  635,  636,  637,  638,  639,  640,  641,  642,  643,  644,
            645,  646,  647,  648,  649,  650,  651,  652,  653,  654,  655,  656,  657,  658,
            659,  660,  661,  662,  663,  664,  665,  666,  667,  668,  669,  670,  671,  672,
            673,  674,  675,  676,  677,  678,  679,  680,  681,  682,  683,  684,  685,  686,
            687,  688,  689,  690,  691,  692,  693,  694,  695,  696,  697,  698,  699,  700,
            701,  702,  703,  704,  705,  706,  707,  708,  709,  710,  711,  712,  713,  714,
            715,  716,  717,  718,  719,  720,  721,  722,  723,  724,  725,  726,  727,  728,
            729,  730,  731,  732,  733,  734,  735,  736,  737,  738,  739,  740,  741,  742,
            743,  744,  745,  746,  747,  748,  749,  750,  751,  752,  753,  754,  755,  756,
            757,  758,  759,  760,  761,  762,  763,  764,  765,  766,  767,  768,  769,  770,
            771,  772,  773,  774,  775,  776,  777,  778,  779,  780,  781,  782,  783,  784,
            785,  786,  787,  788,  789,  790,  791,  792,  793,  794,  795,  796,  797,  798,
            799,  800,  801,  802,  803,  804,  805,  806,  807,  808,  809,  810,  811,  812,
            813,  814,  815,  816,  817,  818,  819,  820,  821,  822,  823,  824,  825,  826,
            827,  828,  829,  830,  831,  832,  833,  834,  835,  836,  837,  838,  839,  840,
            841,  842,  843,  844,  845,  846,  847,  848,  849,  850,  851,  852,  853,  854,
            855,  856,  857,  858,  859,  860,  861,  862,  863,  864,  865,  866,  867,  868,
            869,  870,  871,  872,  873,  874,  875,  876,  877,  878,  879,  880,  881,  882,
            883,  884,  885,  886,  887,  888,  889,  890,  891,  892,  893,  894,  895,  896,
            897,  898,  899,  900,  901,  902,  903,  904,  905,  906,  907,  908,  909,  910,
            911,  912,  913,  914,  915,  916,  917,  918,  919,  920,  921,  922,  923,  924,
            925,  926,  927,  928,  929,  930,  931,  932,  933,  934,  935,  936,  937,  938,
            939,  940,  941,  942,  943,  944,  945,  946,  947,  948,  949,  950,  951,  952,
            953,  954,  955,  956,  957,  958,  959,  960,  961,  962,  963,  964,  965,  966,
            967,  968,  969,  970,  971,  972,  973,  974,  975,  976,  977,  978,  979,  980,
            981,  982,  983,  984,  985,  986,  987,  988,  989,  990,  991,  992,  993,  994,
            995,  996,  997,  998,  999,  1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008,
            1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
            1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036,
            1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050,
            1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064,
            1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078,
            1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092,
            1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106,
            1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
            1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134,
            1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148,
            1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162,
            1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176,
            1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190,
            1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204,
            1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218,
            1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232,
            1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246,
            1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260,
            1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274,
            1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288,
            1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302,
            1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316,
            1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330,
            1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344,
            1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358,
            1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372,
            1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386,
            1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400,
            1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414,
            1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
            1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442,
            1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456,
            1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470,
            1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484,
            1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498,
            1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512,
            1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526,
            1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540,
            1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554,
            1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568,
            1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582,
            1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596,
            1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610,
            1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624,
            1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638,
            1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652,
            1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666,
            1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680,
            1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694,
            1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708,
            1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722,
            1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736,
            1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750,
            1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764,
            1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778,
            1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792,
            1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806,
            1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820,
            1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834,
            1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848,
            1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862,
            1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876,
            1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890,
            1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904,
            1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918,
            1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932,
            1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946,
            1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960,
            1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
            1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988,
            1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
            2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
            2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030,
            2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044,
            2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058,
            2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072,
            2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086,
            2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100,
            2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114,
            2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128,
            2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142,
            2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156,
            2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170,
            2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184,
            2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198,
            2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
            2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226,
            2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240,
            2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254,
            2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268,
            2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282,
            2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296,
            2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310,
            2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324,
            2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338,
            2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352,
            2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366,
            2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380,
            2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394,
            2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408,
            2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422,
            2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436,
            2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450,
            2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464,
            2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478,
            2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492,
            2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506,
            2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520,
            2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534,
            2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548,
            2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562,
            2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576,
            2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590,
            2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604,
            2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618,
            2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632,
            2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646,
            2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660,
            2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674,
            2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688,
            2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702,
            2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716,
            2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730,
            2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744,
            2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758,
            2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772,
            2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786,
            2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800,
            2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814,
            2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828,
            2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842,
            2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856,
            2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870,
            2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884,
            2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898,
            2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912,
            2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926,
            2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940,
            2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954,
            2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968,
            2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982,
            2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996,
            2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
            3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024,
            3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038,
            3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052,
            3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066,
            3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080,
            3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094,
            3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108,
            3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122,
            3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136,
            3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150,
            3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164,
            3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178,
            3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192,
            3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206,
            3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220,
            3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234,
            3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248,
            3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262,
            3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276,
            3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290,
            3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304,
            3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318,
            3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332,
            3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346,
            3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360,
            3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374,
            3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388,
            3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402,
            3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416,
            3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430,
            3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444,
            3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458,
            3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472,
            3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486,
            3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500,
            3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514,
            3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528,
            3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542,
            3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556,
            3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570,
            3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584,
            3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598,
            3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612,
            3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626,
            3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640,
            3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654,
            3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668,
            3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682,
            3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696,
            3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710,
            3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724,
            3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738,
            3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752,
            3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766,
            3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780,
            3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794,
            3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808,
            3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822,
            3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836,
            3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850,
            3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864,
            3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878,
            3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892,
            3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906,
            3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920,
            3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934,
            3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948,
            3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962,
            3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976,
            3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990,
            3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004,
            4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018,
            4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032,
            4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046,
            4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060,
            4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074,
            4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088,
            4089, 4090, 4091, 4092, 4093, 4094, 4095, 0});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);
    cf->call({result}, {a, b});
    EXPECT_EQ(
        (vector<float>{
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    33,   34,   35,   36,   37,   38,   39,   40,   41,   42,
            43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,
            57,   58,   59,   60,   61,   62,   63,   64,   0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    97,   98,
            99,   100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,  112,
            113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,  126,
            127,  128,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    161,  162,  163,  164,  165,  166,  167,  168,
            169,  170,  171,  172,  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,
            183,  184,  185,  186,  187,  188,  189,  190,  191,  192,  0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            225,  226,  227,  228,  229,  230,  231,  232,  233,  234,  235,  236,  237,  238,
            239,  240,  241,  242,  243,  244,  245,  246,  247,  248,  249,  250,  251,  252,
            253,  254,  255,  256,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    289,  290,  291,  292,  293,  294,
            295,  296,  297,  298,  299,  300,  301,  302,  303,  304,  305,  306,  307,  308,
            309,  310,  311,  312,  313,  314,  315,  316,  317,  318,  319,  320,  0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    353,  354,  355,  356,  357,  358,  359,  360,  361,  362,  363,  364,
            365,  366,  367,  368,  369,  370,  371,  372,  373,  374,  375,  376,  377,  378,
            379,  380,  381,  382,  383,  384,  0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    417,  418,  419,  420,
            421,  422,  423,  424,  425,  426,  427,  428,  429,  430,  431,  432,  433,  434,
            435,  436,  437,  438,  439,  440,  441,  442,  443,  444,  445,  446,  447,  448,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    481,  482,  483,  484,  485,  486,  487,  488,  489,  490,
            491,  492,  493,  494,  495,  496,  497,  498,  499,  500,  501,  502,  503,  504,
            505,  506,  507,  508,  509,  510,  511,  512,  0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    545,  546,
            547,  548,  549,  550,  551,  552,  553,  554,  555,  556,  557,  558,  559,  560,
            561,  562,  563,  564,  565,  566,  567,  568,  569,  570,  571,  572,  573,  574,
            575,  576,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    609,  610,  611,  612,  613,  614,  615,  616,
            617,  618,  619,  620,  621,  622,  623,  624,  625,  626,  627,  628,  629,  630,
            631,  632,  633,  634,  635,  636,  637,  638,  639,  640,  0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            673,  674,  675,  676,  677,  678,  679,  680,  681,  682,  683,  684,  685,  686,
            687,  688,  689,  690,  691,  692,  693,  694,  695,  696,  697,  698,  699,  700,
            701,  702,  703,  704,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    737,  738,  739,  740,  741,  742,
            743,  744,  745,  746,  747,  748,  749,  750,  751,  752,  753,  754,  755,  756,
            757,  758,  759,  760,  761,  762,  763,  764,  765,  766,  767,  768,  0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    801,  802,  803,  804,  805,  806,  807,  808,  809,  810,  811,  812,
            813,  814,  815,  816,  817,  818,  819,  820,  821,  822,  823,  824,  825,  826,
            827,  828,  829,  830,  831,  832,  0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    865,  866,  867,  868,
            869,  870,  871,  872,  873,  874,  875,  876,  877,  878,  879,  880,  881,  882,
            883,  884,  885,  886,  887,  888,  889,  890,  891,  892,  893,  894,  895,  896,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    929,  930,  931,  932,  933,  934,  935,  936,  937,  938,
            939,  940,  941,  942,  943,  944,  945,  946,  947,  948,  949,  950,  951,  952,
            953,  954,  955,  956,  957,  958,  959,  960,  0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    993,  994,
            995,  996,  997,  998,  999,  1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008,
            1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
            1023, 1024, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064,
            1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078,
            1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134,
            1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148,
            1149, 1150, 1151, 1152, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    1185, 1186, 1187, 1188, 1189, 1190,
            1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204,
            1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260,
            1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274,
            1275, 1276, 1277, 1278, 1279, 1280, 0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1313, 1314, 1315, 1316,
            1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330,
            1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386,
            1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400,
            1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1441, 1442,
            1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456,
            1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470,
            1471, 1472, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512,
            1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526,
            1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582,
            1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596,
            1597, 1598, 1599, 1600, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    1633, 1634, 1635, 1636, 1637, 1638,
            1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652,
            1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708,
            1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722,
            1723, 1724, 1725, 1726, 1727, 1728, 0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1761, 1762, 1763, 1764,
            1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778,
            1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834,
            1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848,
            1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1889, 1890,
            1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904,
            1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918,
            1919, 1920, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960,
            1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
            1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030,
            2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044,
            2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058,
            2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072,
            2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086,
            2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100,
            2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114,
            2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128,
            2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142,
            2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156,
            2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170,
            2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184,
            2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198,
            2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
            2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226,
            2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240,
            2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254,
            2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268,
            2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282,
            2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296,
            2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310,
            2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324,
            2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338,
            2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352,
            2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366,
            2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380,
            2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394,
            2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408,
            2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422,
            2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436,
            2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450,
            2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464,
            2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478,
            2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492,
            2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506,
            2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520,
            2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534,
            2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548,
            2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562,
            2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576,
            2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590,
            2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604,
            2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618,
            2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632,
            2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646,
            2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660,
            2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674,
            2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688,
            2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702,
            2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716,
            2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730,
            2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744,
            2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758,
            2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772,
            2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786,
            2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800,
            2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814,
            2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828,
            2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842,
            2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856,
            2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870,
            2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884,
            2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898,
            2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912,
            2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926,
            2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940,
            2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954,
            2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968,
            2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982,
            2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996,
            2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
            3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024,
            3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038,
            3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052,
            3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066,
            3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080,
            3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094,
            3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108,
            3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122,
            3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136,
            3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150,
            3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164,
            3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178,
            3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192,
            3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206,
            3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220,
            3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234,
            3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248,
            3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262,
            3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276,
            3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290,
            3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304,
            3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318,
            3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332,
            3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346,
            3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360,
            3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374,
            3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388,
            3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402,
            3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416,
            3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430,
            3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444,
            3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458,
            3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472,
            3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486,
            3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500,
            3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514,
            3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528,
            3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542,
            3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556,
            3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570,
            3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584,
            3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598,
            3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612,
            3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626,
            3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640,
            3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654,
            3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668,
            3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682,
            3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696,
            3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710,
            3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724,
            3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738,
            3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752,
            3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766,
            3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780,
            3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794,
            3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808,
            3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822,
            3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836,
            3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850,
            3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864,
            3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878,
            3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892,
            3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906,
            3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920,
            3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934,
            3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948,
            3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962,
            3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976,
            3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990,
            3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004,
            4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018,
            4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032,
            4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046,
            4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060,
            4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074,
            4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088,
            4089, 4090, 4091, 4092, 4093, 4094, 4095, 0}),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, replace_slice_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{12};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{16};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ(
        (vector<float>{0, 1, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 14, 15}),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, one_hot_scalar_2_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 0, 1}), read_vector<int32_t>(result));
}

TEST(${BACKEND_NAME}, one_hot_scalar_1_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 1, 0}), read_vector<int32_t>(result));
}

TEST(${BACKEND_NAME}, one_hot_scalar_0_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{0});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0}), read_vector<int32_t>(result));
}

TEST(${BACKEND_NAME}, one_hot_scalar_fp_nonint_in_3)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1.1f});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    try
    {
        cf->call({result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST(${BACKEND_NAME}, one_hot_scalar_oob_in_3)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{3000000});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    try
    {
        cf->call({result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST(${BACKEND_NAME}, one_hot_vector_0)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 8};
    auto r = make_shared<op::OneHot>(A, Shape{3, 8}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
        read_vector<int32_t>(result));
}

TEST(${BACKEND_NAME}, one_hot_vector_1)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<int32_t>(result));
}

TEST(${BACKEND_NAME}, one_hot_vector_1_barely_oob)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    try
    {
        cf->call({result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST(${BACKEND_NAME}, one_hot_vector_1_far_oob)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3000000, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    try
    {
        cf->call({result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST(${BACKEND_NAME}, one_hot_matrix_0)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 3, 3};
    auto r = make_shared<op::OneHot>(A, Shape{3, 3, 3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::i32, shape_a);
    copy_data(a,
              vector<int32_t>{
                  0, 1, 1, 2, 1, 0, 0, 2, 1,
              });
    auto result = backend->make_primary_tensor_view(element::i32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0, 0, 0, 1, 1, 0, 0,

                               0, 1, 1, 0, 1, 0, 0, 0, 1,

                               0, 0, 0, 1, 0, 0, 0, 1, 0}),
              read_vector<int32_t>(result));
}

TEST(${BACKEND_NAME}, one_hot_vector_1_fp)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ(
        (vector<float>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, one_hot_vector_1_fp_nonint)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1.01f, 0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    try
    {
        cf->call({result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

TEST(${BACKEND_NAME}, replace_slice_3d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{921, 922, 925, 926, 937, 938, 941, 942});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{0,  1,  2,  3,  4,  5,   6,   7,  8,  9,   10,  11, 12, 13, 14, 15,

                             16, 17, 18, 19, 20, 921, 922, 23, 24, 925, 926, 27, 28, 29, 30, 31,

                             32, 33, 34, 35, 36, 937, 938, 39, 40, 941, 942, 43, 44, 45, 46, 47,

                             48, 49, 50, 51, 52, 53,  54,  55, 56, 57,  58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, replace_slice_3d_strided)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{900, 902, 908, 910, 932, 934, 940, 942});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  902, 3,  4,  5,  6,  7,  908, 9,  910, 11, 12, 13, 14, 15,

                             16,  17, 18,  19, 20, 21, 22, 23, 24,  25, 26,  27, 28, 29, 30, 31,

                             932, 33, 934, 35, 36, 37, 38, 39, 940, 41, 942, 43, 44, 45, 46, 47,

                             48,  49, 50,  51, 52, 53, 54, 55, 56,  57, 58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, replace_slice_3d_strided_different_strides)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{900, 903, 908, 911, 932, 935, 940, 943});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  2,  903, 4,  5,  6,  7,  908, 9,  10, 911, 12, 13, 14, 15,

                             16,  17, 18, 19,  20, 21, 22, 23, 24,  25, 26, 27,  28, 29, 30, 31,

                             932, 33, 34, 935, 36, 37, 38, 39, 940, 41, 42, 943, 44, 45, 46, 47,

                             48,  49, 50, 51,  52, 53, 54, 55, 56,  57, 58, 59,  60, 61, 62, 63}),
              read_vector<float>(result));
}

//
// Numpy test:
//
// > from numpy import *
// > x = linspace(1,2*3*4,2*3*4)
// > y = linspace(1,3*4*5,3*4*5)
// > x.shape=(2,3,4)
// > y.shape=(3,4,5)
// > z = tensordot(x,y,([1,2],[0,1]))
// > z.shape = 2*5
// > z
// array([ 2938.,  3016.,  3094.,  3172.,  3250.,  7042.,  7264.,  7486.,
//         7708.,  7930.])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_3d_multi_axis)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    vector<float> a_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 5);
    for (int i = 0; i < 3 * 4 * 5; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 5};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 5};

    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{2938., 3016., 3094., 3172., 3250., 7042., 7264., 7486., 7708., 7930.}),
              read_vector<float>(result));
}

//
// Numpy test:
//
// > from numpy import *
// > x = array([6,61,2,3,5,21,75,23,23,0,23,2,35,67,1,2,9,16,2,3,6,1,8,0])
// > y = array([9,1,4,6,3,5,1,36,7,3,5,0,1,20,35,2,1,0,1,25,3,6,7,8])
// > x.shape=(2,4,3)
// > y.shape=(3,4,2)
// > z = tensordot(x,y,([2],[0]))
// > z.shape = 2*4*4*2
// > z
// array([ 483,  189,  331,   86,   85, 1262, 2155,  354,   83,   18,   58,
//         543,   77,  241,  325,  286,  859,  144,  438, 1025,  317,  973,
//        1041, 2930,  163,   69,  117,   50,   29,  472,  819,   62,  785,
//         236,  476,  235,  175, 1521, 2387, 1402,   97,   29,   69,  412,
//          63,  286,  429,  218,   45,   11,   29,  162,   27,  106,  149,
//         126,   65,   25,   44,    6,   11,  165,  281,   52])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_3d_one_axis_arbitrary)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    vector<float> a_data{6,  61, 2, 3, 5, 21, 75, 23, 23, 0, 23, 2,
                         35, 67, 1, 2, 9, 16, 2,  3,  6,  1, 8,  0};
    vector<float> b_data{9, 1,  4,  6, 3, 5, 1, 36, 7, 3, 5, 0,
                         1, 20, 35, 2, 1, 0, 1, 25, 3, 6, 7, 8};

    Shape shape_a{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 4, 4, 2};

    auto r = make_shared<op::Dot>(A, B);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{483,  189, 331, 86,  85,  1262, 2155, 354, 83,  18,   58,   543,  77,
                             241,  325, 286, 859, 144, 438,  1025, 317, 973, 1041, 2930, 163,  69,
                             117,  50,  29,  472, 819, 62,   785,  236, 476, 235,  175,  1521, 2387,
                             1402, 97,  29,  69,  412, 63,   286,  429, 218, 45,   11,   29,   162,
                             27,   106, 149, 126, 65,  25,   44,   6,   11,  165,  281,  52}),
              read_vector<float>(result));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,3*4*2*3*2,3*4*2*2*3)
// x.shape=(2,3,3,4)
// y.shape=(3,4,2,2,3)
// z = tensordot(x,y,([2,3],[0,1]))
// z.shape = 2*3*2*2*3
// z
//
// array([  6942.,   7020.,   7098.,   7176.,   7254.,   7332.,   7410.,
//          7488.,   7566.,   7644.,   7722.,   7800.,  16590.,  16812.,
//         17034.,  17256.,  17478.,  17700.,  17922.,  18144.,  18366.,
//         18588.,  18810.,  19032.,  26238.,  26604.,  26970.,  27336.,
//         27702.,  28068.,  28434.,  28800.,  29166.,  29532.,  29898.,
//         30264.,  35886.,  36396.,  36906.,  37416.,  37926.,  38436.,
//         38946.,  39456.,  39966.,  40476.,  40986.,  41496.,  45534.,
//         46188.,  46842.,  47496.,  48150.,  48804.,  49458.,  50112.,
//         50766.,  51420.,  52074.,  52728.,  55182.,  55980.,  56778.,
//         57576.,  58374.,  59172.,  59970.,  60768.,  61566.,  62364.,
//         63162.,  63960.])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 2 * 2 * 3);
    for (int i = 0; i < 3 * 4 * 2 * 2 * 3; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 4, 2, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 3, 2, 3, 2};

    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ(
        (vector<float>{6942.,  7020.,  7098.,  7176.,  7254.,  7332.,  7410.,  7488.,  7566.,
                       7644.,  7722.,  7800.,  16590., 16812., 17034., 17256., 17478., 17700.,
                       17922., 18144., 18366., 18588., 18810., 19032., 26238., 26604., 26970.,
                       27336., 27702., 28068., 28434., 28800., 29166., 29532., 29898., 30264.,
                       35886., 36396., 36906., 37416., 37926., 38436., 38946., 39456., 39966.,
                       40476., 40986., 41496., 45534., 46188., 46842., 47496., 48150., 48804.,
                       49458., 50112., 50766., 51420., 52074., 52728., 55182., 55980., 56778.,
                       57576., 58374., 59172., 59970., 60768., 61566., 62364., 63162., 63960.}),
        read_vector<float>(result));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,2*3*3*4*2,2*3*3*4*2)
// x.shape=(2,3,3,4)
// y.shape=(2,3,3,4,2)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// z
//
// array([ 251412.,  254040.])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis_more)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 4 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 4 * 2; i++)
    {
        b_data[i] = float(i + 1);
    }

    Shape shape_a{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 4, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2};

    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((vector<float>{251412., 254040.}), read_vector<float>(result));
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,20*30*30*40,20*30*30*40)
// y = linspace(1,20*30*30*40*20,20*30*30*40*20)
// x.shape=(20,30,30,40)
// y.shape=(20,30,30,40,20)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// set_printoptions(precision=20)
// z
//
// array([  2.48832025919525478400e+18,   2.48832051839533977600e+18,
//          2.48832077759658444800e+18,   2.48832103679413504000e+18,
//          2.48832129599669350400e+18,   2.48832155519793971200e+18,
//          2.48832181439802265600e+18,   2.48832207359808000000e+18,
//          2.48832233279813580800e+18,   2.48832259199822028800e+18,
//          2.48832285119946496000e+18,   2.48832311040043008000e+18,
//          2.48832336959957401600e+18,   2.48832362880081817600e+18,
//          2.48832388800090368000e+18,   2.48832414720096000000e+18,
//          2.48832440640101478400e+18,   2.48832466560109772800e+18,
//          2.48832492480234188800e+18,   2.48832518400031897600e+18])
//
// Disabled because this test is very slow.
//
TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis_big_fp64_VERY_SLOW)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    vector<double> a_data(20 * 30 * 30 * 40);
    for (int i = 0; i < 20 * 30 * 30 * 40; i++)
    {
        a_data[i] = double(i + 1);
    }

    vector<double> b_data(20 * 30 * 30 * 40 * 20);
    for (int i = 0; i < 20 * 30 * 30 * 40 * 20; i++)
    {
        b_data[i] = double(i + 1);
    }

    Shape shape_a{20, 30, 30, 40};
    auto A = make_shared<op::Parameter>(element::f64, shape_a);
    Shape shape_b{20, 30, 30, 40, 20};
    auto B = make_shared<op::Parameter>(element::f64, shape_b);
    Shape shape_r{20};

    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f64, shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::f64, shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::f64, shape_r);

    cf->call({result}, {a, b});
    EXPECT_TRUE(test::all_close(
        vector<double>{
            2.48832025919525478400e+18, 2.48832051839533977600e+18, 2.48832077759658444800e+18,
            2.48832103679413504000e+18, 2.48832129599669350400e+18, 2.48832155519793971200e+18,
            2.48832181439802265600e+18, 2.48832207359808000000e+18, 2.48832233279813580800e+18,
            2.48832259199822028800e+18, 2.48832285119946496000e+18, 2.48832311040043008000e+18,
            2.48832336959957401600e+18, 2.48832362880081817600e+18, 2.48832388800090368000e+18,
            2.48832414720096000000e+18, 2.48832440640101478400e+18, 2.48832466560109772800e+18,
            2.48832492480234188800e+18, 2.48832518400031897600e+18},
        read_vector<double>(result)));
}

TEST(${BACKEND_NAME}, max_pool_1d_1channel_1image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}}).get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}, {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_pool_1d_2channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 12};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                    {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 4, 3};
    auto f =
        make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_overpadded)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{1, 1, 5, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 0};
    Shape padding_above{1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 7, 5};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    auto min = std::numeric_limits<float>::lowest();
    EXPECT_EQ((test::NDArray<float, 4>({{{{min, min, min, min, min},
                                          {1, 2, 2, 2, 1},
                                          {3, 3, 2, 2, 1},
                                          {3, 3, 2, 1, 1},
                                          {2, 1, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 1, 0, 0}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_padded)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{1, 1, 5, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 0};
    Shape padding_above{1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 6, 5};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{1, 2, 2, 2, 1},
                                          {3, 3, 2, 2, 1},
                                          {3, 3, 2, 1, 1},
                                          {2, 1, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 1, 0, 0}}}})
                   .get_vector()),
              read_vector<float>(result));
}

// Test to make sure that negative elements and padding are handled properly. Added this because
// mkldnn calls its padding "zero padding" but apparently that is not technically true (negative
// values still "win" versus out-of-bounds values), which is good.
TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_padded_negative_values)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    auto shape_a = Shape{
        1,
        1,
        1,
        14}; // 1 image, 1 channel, 1 row, 14 columns (if it's 1D we don't get mkldnn as of this writing)
    Shape window_shape{1, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 1};
    Shape padding_above{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 1, 15};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>{{{{-1, -2, -3, -3, -2, -1, -3, -2, -2, -2, -2, -3, -4, -5}}}}
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 4>({{{{-1, -1, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2, -3, -4, -5}}}})
             .get_vector()),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_strided)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 1, 8, 8};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(A, window_shape, window_movement_strides), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}}}}).get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, not)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 2, 0});
    auto result = backend->make_primary_tensor_view(element::boolean, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<char>{0, 1, 0, 1}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, reverse_0d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_1d_nochange)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{0, 1, 2, 3, 4, 5, 6, 7}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_1d_0)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{7, 6, 5, 4, 3, 2, 1, 0}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_2d_nochange)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector()),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_2d_0)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}).get_vector()),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_2d_1)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}).get_vector()),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_2d_01)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}).get_vector()),
        read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_nochange)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                        {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_0)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}},
                                        {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_1)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}},
                                        {{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_2)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}},
                                        {{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_01)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}},
                                        {{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_02)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}},
                                        {{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_12)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1, 2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}},
                                        {{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reverse_3d_012)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1, 2}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}},
                                        {{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, numeric_float_nan)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::boolean, shape);
    cf->call({result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, numeric_double_nan)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::boolean, shape);
    cf->call({result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, numeric_float_inf)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::boolean, shape);
    cf->call({result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, numeric_double_inf)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::boolean, shape);
    cf->call({result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

TEST(${BACKEND_NAME}, abc_tbb)
{
    ONLY_ENABLE_TEST_FOR("CPU", "${BACKEND_NAME}");

    // Force TBB flow graph generation in the CPU backend
    // This has no effect on other backends
    bool use_tbb = (getenv("NGRAPH_CPU_USE_TBB") != nullptr);
    if (!use_tbb)
    {
        setenv("NGRAPH_CPU_USE_TBB", "1", 1);
    }

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    cf->call({result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    cf->call({result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());

    if (!use_tbb)
    {
        unsetenv("NGRAPH_CPU_USE_TBB");
    }
}

//
// The unit tests for ReduceWindow follow exactly what we test for MaxPool---but they use ReduceWindow to do it.
//
TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_1image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}}).get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}, {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_2channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                    {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_2channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 5, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 4, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_1channel_1image_strided)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 8, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 3, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}}}}).get_vector()),
              read_vector<float>(result));
}

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
TEST(${BACKEND_NAME}, select_and_scatter_with_overlap)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 5};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 2};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}, {1, 5, 7, 5, 6}, {0, 6, 2, 10, 2}})
                  .get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->make_primary_tensor_view(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 0}, {0, 0, 8, 0, 0}, {0, 0, 3, 0, 0}, {0, 0, 0, 1, 0}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
TEST(${BACKEND_NAME}, select_and_scatter_without_overlap)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 6};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}})
                  .get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->make_primary_tensor_view(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 3, 0, 0, 0}, {0, 0, 0, 0, 0, 1}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// Adapted from the XLA docs to provide an example in >2D: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
TEST(${BACKEND_NAME}, select_and_scatter_3d_without_overlap)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{2, 4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{1, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 4, 6};
    Shape window_shape{2, 2, 3};
    auto window_strides = Strides{2, 2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(
        a,
        test::NDArray<float, 3>(
            {{{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}},
             {{2, 5, 8, 3, 4, 2}, {1, 2, 8, 4, 5, 2}, {10, 2, 3, 4, 1, 0}, {4, 1, 2, 4, 5, 7}}})
            .get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 3>({{{2, 6}, {3, 1}}}).get_vector());
    auto c = backend->make_primary_tensor_view(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b, c});
    EXPECT_EQ(
        (test::NDArray<float, 3>(
             {{{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},
              {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {3, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}})
             .get_vector()),
        read_vector<float>(result));
}

template <typename OP>
void make_unary_empty_test(const string& backend_name)
{
    Shape shape{0};

    op::ParameterVector params;
    NodeVector result_list;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        shared_ptr<op::Parameter> p = make_shared<op::Parameter>(s_known_element_types[i], shape);
        params.push_back(p);
        result_list.push_back(make_shared<OP>(p));
    }

    auto f = make_shared<Function>(result_list, params);

    auto manager = runtime::Manager::get(backend_name);
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    vector<shared_ptr<runtime::TensorView>> inputs;
    vector<shared_ptr<runtime::TensorView>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->make_primary_tensor_view(s_known_element_types[i], shape));
        outputs.push_back(backend->make_primary_tensor_view(s_known_element_types[i], shape));
    }

    cf->call(outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
}

template <typename OP>
void make_binary_empty_test(const string& backend_name, bool is_comparison = false)
{
    Shape shape{0};
    op::ParameterVector A;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        A.push_back(make_shared<op::Parameter>(s_known_element_types[i], shape));
    }

    NodeVector result_list;
    for (shared_ptr<op::Parameter> p : A)
    {
        result_list.push_back(make_shared<OP>(p, p));
    }

    auto f = make_shared<Function>(result_list, A);

    auto manager = runtime::Manager::get(backend_name);
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    vector<shared_ptr<runtime::TensorView>> inputs;
    vector<shared_ptr<runtime::TensorView>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->make_primary_tensor_view(s_known_element_types[i], shape));
        if (is_comparison)
        {
            outputs.push_back(backend->make_primary_tensor_view(element::from<char>(), shape));
        }
        else
        {
            outputs.push_back(backend->make_primary_tensor_view(s_known_element_types[i], shape));
        }
    }

    cf->call(outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    if (is_comparison)
    {
        EXPECT_EQ(read_vector<char>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[9]).size(), 0);
    }
    else
    {
        EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
    }
}

TEST(${BACKEND_NAME}, zero_sized_abs)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Abs>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_ceiling)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Ceiling>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_exp)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Exp>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_floor)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Floor>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_log)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Log>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_negative)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Negative>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_not)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{0};
    auto A = make_shared<op::Parameter>(element::from<char>(), shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::from<char>(), shape);
    auto result = backend->make_primary_tensor_view(element::from<char>(), shape);

    cf->call({result}, {a});

    auto in_vec = read_vector<char>(a);
    auto out_vec = read_vector<char>(result);

    EXPECT_EQ(in_vec.size(), 0);
    EXPECT_EQ(out_vec.size(), 0);
}

TEST(${BACKEND_NAME}, zero_sized_sign)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Sign>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_sqrt)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Sqrt>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_sin)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Sin>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_sinh)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Sinh>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_cos)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Cos>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_cosh)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Cosh>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_tan)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Tan>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_tanh)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Tanh>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_asin)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Asin>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_acos)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Acos>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_atan)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_unary_empty_test<op::Atan>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_add)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Add>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_divide)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Divide>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_eq)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Equal>("${BACKEND_NAME}", true);
}

TEST(${BACKEND_NAME}, zero_sized_greater)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Greater>("${BACKEND_NAME}", true);
}

TEST(${BACKEND_NAME}, zero_sized_greatereq)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::GreaterEq>("${BACKEND_NAME}", true);
}

TEST(${BACKEND_NAME}, zero_sized_less)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Less>("${BACKEND_NAME}", true);
}

TEST(${BACKEND_NAME}, zero_sized_lesseq)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::LessEq>("${BACKEND_NAME}", true);
}

TEST(${BACKEND_NAME}, zero_sized_maximum)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Maximum>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_minimum)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Minimum>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_multiply)
{
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Multiply>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_not_equal)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::NotEqual>("${BACKEND_NAME}", true);
}

TEST(${BACKEND_NAME}, zero_sized_power)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Power>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, zero_sized_subtract)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");
    make_binary_empty_test<op::Subtract>("${BACKEND_NAME}");
}

TEST(${BACKEND_NAME}, convolution_outlining)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::Convolution>(A,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto conv2 = make_shared<op::Convolution>(conv1,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto f = make_shared<Function>(conv2, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    vector<float> expected_result{4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    cf->call({result}, {a, b});
    EXPECT_EQ(vector<float>{expected_result}, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, mkldnn_layouts)
{
    ONLY_ENABLE_TEST_FOR("CPU", "${BACKEND_NAME}");

    Shape shape_a{1, 16, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 32, 2, 2};
    auto conv1 = make_shared<op::Convolution>(A,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    Shape pool_shape{1, 1};
    auto pool1 = make_shared<op::AvgPool>(conv1, pool_shape);
    auto f = make_shared<Function>(pool1, op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    vector<float> input(64, 1.0f);
    copy_data(a, input);
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    vector<float> weights(512, 1.0f);
    copy_data(b, weights);
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    vector<float> expected_result(128, 16.0f);

    cf->call({result}, {a, b});
    EXPECT_EQ(vector<float>{expected_result}, read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_1d_1channel_1image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    float denom = 3.0;

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{1 / denom,
                                          3 / denom,
                                          3 / denom,
                                          3 / denom,
                                          4 / denom,
                                          5 / denom,
                                          5 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          0 / denom}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    float denom = 3.0;

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{1 / denom,
                                          3 / denom,
                                          3 / denom,
                                          3 / denom,
                                          4 / denom,
                                          5 / denom,
                                          5 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          0 / denom}},
                                        {{3 / denom,
                                          4 / denom,
                                          2 / denom,
                                          1 / denom,
                                          0 / denom,
                                          2 / denom,
                                          2 / denom,
                                          3 / denom,
                                          1 / denom,
                                          1 / denom,
                                          1 / denom,
                                          3 / denom}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_1d_2channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    float denom = 3.0;

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{1 / denom,
                                          3 / denom,
                                          3 / denom,
                                          3 / denom,
                                          4 / denom,
                                          5 / denom,
                                          5 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          0 / denom},
                                         {0 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          2 / denom,
                                          5 / denom,
                                          5 / denom,
                                          4 / denom,
                                          3 / denom,
                                          3 / denom,
                                          3 / denom,
                                          1 / denom}},

                                        {{3 / denom,
                                          4 / denom,
                                          2 / denom,
                                          1 / denom,
                                          0 / denom,
                                          2 / denom,
                                          2 / denom,
                                          3 / denom,
                                          1 / denom,
                                          1 / denom,
                                          1 / denom,
                                          3 / denom},
                                         {3 / denom,
                                          1 / denom,
                                          1 / denom,
                                          1 / denom,
                                          3 / denom,
                                          2 / denom,
                                          2 / denom,
                                          0 / denom,
                                          1 / denom,
                                          2 / denom,
                                          4 / denom,
                                          3 / denom}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 4, 3};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    float denom = 2 * 3;

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{6 / denom, 8 / denom, 5 / denom}, // img 0 chan 0
                                          {7 / denom, 5 / denom, 3 / denom},
                                          {5 / denom, 2 / denom, 5 / denom},
                                          {6 / denom, 5 / denom, 5 / denom}},

                                         {{5 / denom, 7 / denom, 6 / denom}, // img 0 chan 1
                                          {8 / denom, 6 / denom, 7 / denom},
                                          {7 / denom, 2 / denom, 3 / denom},
                                          {6 / denom, 1 / denom, 0 / denom}}},

                                        {{{5 / denom, 6 / denom, 5 / denom}, // img 1 chan 0
                                          {3 / denom, 5 / denom, 9 / denom},
                                          {3 / denom, 6 / denom, 9 / denom},
                                          {2 / denom, 3 / denom, 3 / denom}},

                                         {{5 / denom, 3 / denom, 1 / denom}, // img 1 chan 1
                                          {6 / denom, 5 / denom, 4 / denom},
                                          {7 / denom, 5 / denom, 6 / denom},
                                          {4 / denom, 2 / denom, 4 / denom}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_strided)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 1, 8, 8};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(A, window_shape, window_movement_strides), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    float denom = 2 * 3;

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{6 / denom, 5 / denom, 4 / denom},
                                          {6 / denom, 5 / denom, 8 / denom},
                                          {6 / denom, 2 / denom, 4 / denom}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_padded)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{1, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}}}).get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                          {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                          {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                          {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                          {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                          {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                          {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                         {{3.0f / 1, 8.0f / 2, 7.0f / 2, 2.0f / 1},
                                          {5.0f / 2, 10.0f / 4, 16.0f / 4, 11.0f / 2},
                                          {5.0f / 2, 11.0f / 4, 20.0f / 4, 14.0f / 2},
                                          {3.0f / 1, 9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_below)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2},
                                          {0.0f / 2, 4.0f / 4, 6.0f / 4},
                                          {2.0f / 2, 5.0f / 4, 5.0f / 4}},
                                         {{3.0f / 1, 8.0f / 2, 7.0f / 2},
                                          {5.0f / 2, 10.0f / 4, 16.0f / 4},
                                          {5.0f / 2, 11.0f / 4, 20.0f / 4}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_above)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{4.0f / 4, 6.0f / 4, 2.0f / 2},
                                          {5.0f / 4, 5.0f / 4, 2.0f / 2},
                                          {2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                         {{10.0f / 4, 16.0f / 4, 11.0f / 2},
                                          {11.0f / 4, 20.0f / 4, 14.0f / 2},
                                          {9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 5, 5};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 2, 0.0f / 1},
                                          {0.0f / 2, 4.0f / 4, 6.0f / 6, 6.0f / 4, 2.0f / 2},
                                          {2.0f / 3, 6.0f / 6, 8.0f / 9, 6.0f / 6, 2.0f / 3},
                                          {2.0f / 2, 5.0f / 4, 7.0f / 6, 5.0f / 4, 2.0f / 2},
                                          {2.0f / 1, 2.0f / 2, 2.0f / 3, 0.0f / 2, 0.0f / 1}},
                                         {{3.0f / 1, 8.0f / 2, 10.0f / 3, 7.0f / 2, 2.0f / 1},
                                          {5.0f / 2, 10.0f / 4, 21.0f / 6, 16.0f / 4, 11.0f / 2},
                                          {8.0f / 3, 19.0f / 6, 35.0f / 9, 27.0f / 6, 16.0f / 3},
                                          {5.0f / 2, 11.0f / 4, 25.0f / 6, 20.0f / 4, 14.0f / 2},
                                          {3.0f / 1, 9.0f / 2, 14.0f / 3, 11.0f / 2, 5.0f / 1}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 3, 0.0f / 1},
                                          {2.0f / 3, 8.0f / 9, 2.0f / 3},
                                          {2.0f / 1, 2.0f / 3, 0.0f / 1}},
                                         {{3.0f / 1, 10.0f / 3, 2.0f / 1},
                                          {8.0f / 3, 35.0f / 9, 16.0f / 3},
                                          {3.0f / 1, 14.0f / 3, 5.0f / 1}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided_uneven)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 3};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>(
                   {{{{0.0f / 1, 1.0f / 2}, {2.0f / 3, 6.0f / 6}, {2.0f / 1, 0.0f / 2}},
                     {{3.0f / 1, 7.0f / 2}, {8.0f / 3, 27.0f / 6}, {3.0f / 1, 11.0f / 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_interior_1d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{16};
    Shape padding_below{0};
    Shape padding_above{0};
    Shape padding_interior{2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>(
                   {1, 2112, 2112, 2, 2112, 2112, 3, 2112, 2112, 4, 2112, 2112, 5, 2112, 2112, 6})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_exterior_1d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{15};
    Shape padding_below{4};
    Shape padding_above{5};
    Shape padding_interior{0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>(
                   {2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_interior_exterior_1d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{25};
    Shape padding_below{4};
    Shape padding_above{5};
    Shape padding_interior{2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>({2112, 2112, 2112, 2112, 1,    2112, 2112, 2, 2112,
                                        2112, 3,    2112, 2112, 4,    2112, 2112, 5, 2112,
                                        2112, 6,    2112, 2112, 2112, 2112, 2112})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_interior_exterior_2d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{7, 6};
    Shape padding_below{1, 0};
    Shape padding_above{2, 1};
    Shape padding_interior{2, 1};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{9});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{9, 9, 9, 9, 9, 9},
                                        {1, 9, 2, 9, 3, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {4, 9, 5, 9, 6, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {9, 9, 9, 9, 9, 9}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_exterior_2d_0x0)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{2, 3};
    Shape padding_above{3, 2};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({{}}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_exterior_2d_0x3)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{2, 1};
    Shape padding_above{3, 1};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_exterior_2d_3x0)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{1, 3};
    Shape padding_above{1, 2};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 4, 4};
    Shape padding_below{0, 0, 1, 1};
    Shape padding_above{0, 0, 1, 1};
    Shape padding_interior{0, 0, 0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
    // clang-format on

    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{42});

    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    cf->call({result}, {a, b});
    // clang-format off
    EXPECT_EQ((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                },
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result));
    // clang-format on
}

// This is a regression test for one of TF's unit tests, which was failing.
// The problem was inappropriate handling of the shape computation for a
// zero-length axis with interior padding. Rather than subtract 1 from the
// source shape and multiply by the interior padding (which causes underflow),
// we should just count the pre-interior-padding length as zero.
TEST(${BACKEND_NAME}, pad_interior_exterior_4d_2x0x3x2)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{2, 0, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape padding_below{1, 0, 0, 0};
    Shape padding_above{0, 2, 0, 0};
    Shape padding_interior{2, 1, 0, 0};
    Shape shape_r{5, 2, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    //copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->make_primary_tensor_view(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->make_primary_tensor_view(element::f32, shape_r);

    vector<float> expected(5 * 2 * 3 * 2, 2112);

    cf->call({result}, {a, b});
    EXPECT_EQ(expected, read_vector<float>(result));
}

// Trivial case with no reduced axes.
TEST(${BACKEND_NAME}, product_trivial)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
TEST(${BACKEND_NAME}, product_trivial_5d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, product_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, Shape{});

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{24}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, product_matrix_columns)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{15, 48}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, product_matrix_rows)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{2, 12, 30}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, product_matrix_rows_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, product_matrix_cols_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, product_vector_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, product_matrix_to_scalar_zero_by_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, product_3d_to_matrix_most_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1 * 10 * 19,
                             2 * 11 * 20,
                             3 * 12 * 21,
                             4 * 13 * 22,
                             5 * 14 * 23,
                             6 * 15 * 24,
                             7 * 16 * 25,
                             8 * 17 * 26,
                             9 * 18 * 27}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, product_3d_to_matrix_least_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1 * 2 * 3,
                             4 * 5 * 6,
                             7 * 8 * 9,
                             10 * 11 * 12,
                             13 * 14 * 15,
                             16 * 17 * 18,
                             19 * 20 * 21,
                             22 * 23 * 24,
                             25 * 26 * 27}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, product_3d_to_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}"); // Correct values but OOB

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f =
        make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, product_3d_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}"); // Correct values but OOB

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1, 2}),
                                   op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 9.0f * 4.0f * 13.0f * 6.0f * 7.0f * 12.0f * 3.0f *
                             2.0f * 11.0f * 8.0f * 5.0f * 14.0f * 5.0f * 8.0f * 11.0f * 2.0f *
                             3.0f * 12.0f * 7.0f * 6.0f * 13.0f * 4.0f * 9.0f * 10.0f * 1.0f}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, product_3d_eliminate_zero_dim)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1}), read_vector<float>(result));
}

// Trivial case with no reduced axes.
TEST(${BACKEND_NAME}, max_trivial)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
TEST(${BACKEND_NAME}, max_trivial_5d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, Shape{});

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{4}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, max_matrix_columns)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{5, 6}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, max_matrix_rows)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{2, 4, 6}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, max_matrix_rows_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, max_matrix_cols_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, max_vector_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, max_matrix_to_scalar_zero_by_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_3d_to_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{25.0f, 26.0f, 27.0f}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_3d_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1, 2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{14.0f}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, max_3d_eliminate_zero_dim)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    float mi = -std::numeric_limits<float>::infinity();

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{mi, mi, mi, mi, mi, mi}), read_vector<float>(result));
}

// Trivial case with no reduced axes.
TEST(${BACKEND_NAME}, min_trivial)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
TEST(${BACKEND_NAME}, min_trivial_5d)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

TEST(${BACKEND_NAME}, min_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::f32, Shape{});

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, min_matrix_columns)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, min_matrix_rows)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 3, 5}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, min_matrix_rows_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, min_matrix_cols_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, min_vector_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, min_matrix_to_scalar_zero_by_zero)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

TEST(${BACKEND_NAME}, min_3d_to_matrix_most_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, min_3d_to_matrix_least_sig)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 4, 7, 10, 13, 16, 19, 22, 25}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, min_3d_to_vector)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, min_3d_to_scalar)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1, 2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, min_3d_eliminate_zero_dim)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    float inf = std::numeric_limits<float>::infinity();

    cf->call({result}, {a});
    EXPECT_EQ((vector<float>{inf, inf, inf, inf, inf, inf}), read_vector<float>(result));
}

TEST(${BACKEND_NAME}, relu_2Dfprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    cf->call({result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(${BACKEND_NAME}, relu_4Dfprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    cf->call({result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(${BACKEND_NAME}, fuse_max_with_constant_zero_input_as_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::Maximum>(A, B);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(max, op::ParameterVector{B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto b = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(b, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    cf->call({result}, {b});
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(${BACKEND_NAME}, relu_2Dbackprop)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, op::ParameterVector{A, delta_val});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto delta = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    vector<float> expected{1, 2, 0, 4, 0, 6, 7, 0, 9, 0};

    cf->call({result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(${BACKEND_NAME}, relu_4Dbackprop)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, op::ParameterVector{A, delta_val});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto delta = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    cf->call({result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(${BACKEND_NAME}, softmax_all)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{-3, -2, -1, 0, 1, 2});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);

    cf->call({result}, {a});
    vector<float> expected{
        expf(-3) / d, expf(-2) / d, expf(-1) / d, expf(0) / d, expf(1) / d, expf(2) / d};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));

    // empty AxisSet is the same as "full" AxisSet
    f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{}), op::ParameterVector{A});
    external = manager->compile(f);
    cf = backend->make_call_frame(external);

    cf->call({result}, {a});
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

TEST(${BACKEND_NAME}, softmax_axis)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    auto d0 = expf(-10) + expf(-20) + expf(-30);
    auto d1 = expf(-40) + expf(-50) + expf(-60);

    cf->call({result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d0,
                           expf(-30) / d0,
                           expf(-40) / d1,
                           expf(-50) / d1,
                           expf(-60) / d1};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

TEST(${BACKEND_NAME}, softmax_underflow)
{
    SKIP_TEST_FOR("GPU", "${BACKEND_NAME}");
    SKIP_TEST_FOR("NNP", "${BACKEND_NAME}");

    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto low = std::numeric_limits<float>::lowest();

    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{low, 1, 2, 3, 4, 5});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    auto d0 = expf(low) + expf(3);
    auto d1 = expf(1) + expf(4);
    auto d2 = expf(2) + expf(5);

    cf->call({result}, {a});
    vector<float> expected{
        expf(low) / d0, expf(1) / d1, expf(2) / d2, expf(3) / d0, expf(4) / d1, expf(5) / d2};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}
