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

#include <fstream>
#include <string>

#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/external_function.hpp"

using namespace ngraph;
using namespace std;

std::shared_ptr<ngraph::runtime::CallFrame> runtime::cpu::CPU_Backend::make_call_frame(
    const std::shared_ptr<ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

std::shared_ptr<ngraph::runtime::TensorView>
    runtime::cpu::CPU_Backend::make_primary_tensor_view(const ngraph::element::Type& element_type,
                                                        const Shape& shape)
{
    auto rc = make_shared<runtime::cpu::CPUTensorView>(element_type, shape);
    return dynamic_pointer_cast<runtime::TensorView>(rc);
}

// The following functions are here for use by the emitted code
// These functions are used nowhere else, only in the emitted code
// Their simple function signatures are designed to comppile quickly in the emitted code
template <typename T>
static void dump_tensor(const string& name, const T* data, size_t count)
{
    string result_file = file_util::path_join("dump_temporaries", name + ".txt");
    ofstream f(result_file);
    if (f)
    {
        f << data[0];
        for (size_t i = 1; i < count; i++)
        {
            f << ", " << data[i];
        }
    }
}

void dump_tensor_float(const char* name, const float* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_double(const char* name, const double* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_int8_t(const char* name, const int8_t* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_int16_t(const char* name, const int16_t* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_int32_t(const char* name, const int32_t* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_int64_t(const char* name, const int64_t* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_uint8_t(const char* name, const uint8_t* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_uint16_t(const char* name, const uint16_t* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_uint32_t(const char* name, const uint32_t* data, size_t count)
{
    dump_tensor(name, data, count);
}

void dump_tensor_uint64_t(const char* name, const uint64_t* data, size_t count)
{
    dump_tensor(name, data, count);
}
