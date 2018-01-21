// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"

#include "uncomment.hpp"

using namespace std;

TEST(resource_generator, uncomment)
{
    string sample = R"(1
2
3
//_4
//_5
6_//_7
8
9_/*_10_*/_11)";

    string result = uncomment(sample);
    string expected = R"(1
2
3


6_
8
9__11)";
    EXPECT_STREQ(expected.c_str(), result.c_str());
}
