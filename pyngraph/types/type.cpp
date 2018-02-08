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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <string>
#include "ngraph/common.hpp"             // ngraph::Shape
#include "ngraph/types/element_type.hpp" // ngraph::element::Type
#include "ngraph/types/type.hpp"         // ngraph::TensorViewType
#include "pyngraph/types/type.hpp"

namespace py = pybind11;

void regclass_pyngraph_TensorViewType(py::module m) {
    py::class_<ngraph::TensorViewType, std::shared_ptr<ngraph::TensorViewType>> tensorViewType(m, "TensorViewType");

    tensorViewType.def(py::init<const ngraph::element::Type&, const ngraph::Shape&>());
    tensorViewType.def("get_shape", &ngraph::TensorViewType::get_shape);
}
