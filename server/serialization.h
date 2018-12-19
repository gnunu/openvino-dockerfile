/*
// Copyright (c) 2018 Intel Corporation
//
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
// limitations under the License.
*/

#pragma once

#include <ostream>
#include <unordered_map>

#include <ie_icnn_network.hpp>
#include <ie_data.h>
#include <ie_iexecutable_network.hpp>
#include <ie_input_info.hpp>

// serialize

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const InferenceEngine::Data& data) {
    static const std::unordered_map<InferenceEngine::Layout, const char*, std::hash<uint8_t>> names = {
#define LAYOUT_NAME(s) {InferenceEngine::s, #s}
        LAYOUT_NAME(ANY),
        LAYOUT_NAME(NCHW),
        LAYOUT_NAME(NHWC),
        LAYOUT_NAME(OIHW),
        LAYOUT_NAME(C),
        LAYOUT_NAME(CHW),
        LAYOUT_NAME(HW),
        LAYOUT_NAME(NC),
        LAYOUT_NAME(CN),
        LAYOUT_NAME(BLOCKED),
#undef  LAYOUT_NAME
    };

    os << data.name << ": "
       << data.precision.name() << " "
       << names.at(data.layout);

    for (auto n: data.dims)
        os << " " << n;
    return os;
}

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const InferenceEngine::InputInfo& input_info) {
    auto info = const_cast<InferenceEngine::InputInfo&>(input_info);
    return os << *info.getInputData();
}

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const InferenceEngine::InputsDataMap& inputs_info) {
    for (auto& p: inputs_info)
        os << *p.second << "\n";
    return os;
}
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const InferenceEngine::ConstInputsDataMap& inputs_info) {
    for (auto& p: inputs_info)
        os << *p.second << "\n";
    return os;
}

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const InferenceEngine::OutputsDataMap& outputs_info) {
    for (auto& p: outputs_info)
        os << *p.second << "\n";
    return os;
}
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const InferenceEngine::ConstOutputsDataMap& outputs_info) {
    for (auto& p: outputs_info)
        os << *p.second << "\n";
    return os;
}

///////////////////////////////////////////////////////////////////////////

#include <istream>
#include <string>

// deserialize

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(
        std::basic_istream<CharT, Traits>& is,
        InferenceEngine::Data& data) {
    std::string name, p, l;
    if (!(is >> name >> p >> l))
        return is;

    // parse name
    if (name.back() == ':')  name.pop_back();

    // parse precision
    auto precision = InferenceEngine::Precision::FromStr(p);

    // parse layout
    static const std::unordered_map<std::string, InferenceEngine::Layout> names = {
#define LAYOUT_NAME(s) {#s, InferenceEngine::s}
        LAYOUT_NAME(ANY),
        LAYOUT_NAME(NCHW),
        LAYOUT_NAME(NHWC),
        LAYOUT_NAME(OIHW),
        LAYOUT_NAME(C),
        LAYOUT_NAME(CHW),
        LAYOUT_NAME(HW),
        LAYOUT_NAME(NC),
        LAYOUT_NAME(CN),
        LAYOUT_NAME(BLOCKED),
#undef  LAYOUT_NAME
    };
    auto it = names.find(l);
    if (it == names.end() || !precision) {
        is.setstate(std::ios_base::failbit);
        return is;
    }
    InferenceEngine::Layout layout = it->second;

    // parse dims
    InferenceEngine::SizeVector dims;
    for (int i = l.size(); i; --i) {
        size_t n;
        if (!(is >> n)) return is;
        dims.push_back(n);
    }

    data = InferenceEngine::Data(name, dims, precision, layout);
    return is;
}

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(
        std::basic_istream<CharT, Traits>& is,
        InferenceEngine::InputInfo& input_info) {
    const auto precision = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Data data("", precision);
    if (is >> data) {
        auto p = std::make_shared<InferenceEngine::Data>(std::move(data));
        input_info.setInputData(p);
    }
    return is;
}

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(
        std::basic_istream<CharT, Traits>& is,
        InferenceEngine::ConstInputsDataMap& inputs_info) {
    InferenceEngine::ConstInputsDataMap inputs;
    InferenceEngine::InputInfo info;
    while (is.good() && is >> info) {
        auto p = std::make_shared<InferenceEngine::InputInfo>(std::move(info));
        inputs.emplace(p->name(), p);

        if (!is.eof()) {
            if (is.get() != '\n')
                is.setstate(std::ios_base::failbit);
            else if (is.peek() == '\n')
                break;
        }
    }
    if (is)
        inputs_info.swap(inputs);
    return is;
}

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(
        std::basic_istream<CharT, Traits>& is,
        InferenceEngine::ConstOutputsDataMap& outputs_info) {
    InferenceEngine::ConstOutputsDataMap outputs;
    const auto precision = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Data data("", precision);
    while (is.good() && is >> data) {
        auto p = std::make_shared<InferenceEngine::Data>(std::move(data));
        outputs.emplace(p->getName(), p);

        if (!is.eof()) {
            if (is.get() != '\n')
                is.setstate(std::ios_base::failbit);
            else if (is.peek() == '\n')
                break;
        }
    }
    if (is)
        outputs_info.swap(outputs);
    return is;
}
