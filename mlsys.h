/*
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MLSYS_H_
#define MLSYS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/status/statusor.h"

////////////////////////////////////////////////////////////////////////////////
/////////  Basic definitions for problem & solution data structures.   /////////
/////////  Contest participants do not need to modify this code.       /////////
////////////////////////////////////////////////////////////////////////////////

namespace mlsys {

using BaseCost = int64_t;
using Depth = int64_t;
using FastMemoryCapacity = int64_t;
using Height = int64_t;
using Inputs = std::vector<size_t>;
using OpType = std::string;
using Outputs = std::vector<size_t>;
using SlowMemoryBandwidth = int64_t;
using SubgraphLatency = double;
using TotalLatency = double;
using TraversalOrder = std::vector<int64_t>;
using Width = int64_t;

struct Tensor {
  Width width;
  Height height;
  bool operator==(const Tensor& other) const = default;
};

struct Op {
  OpType op_type;
  Inputs inputs;
  Outputs outputs;
  BaseCost base_cost;
  bool operator==(const Op& other) const = default;
};

struct Granularity {
  Width width;
  Height height;
  Depth depth;
  bool operator==(const Granularity& other) const = default;
};

struct Problem {
  std::vector<Tensor> tensors;
  std::vector<Op> ops;
  FastMemoryCapacity fast_memory_capacity;
  SlowMemoryBandwidth slow_memory_bandwidth;
  Granularity native_granularity;
  bool operator==(const Problem& other) const = default;
};

absl::StatusOr<Problem> ReadProblem(const std::string& filename);

struct Subgraph {
  std::vector<size_t> ops;
  std::vector<size_t> tensors_to_retain;
  Granularity granularity;
  std::optional<TraversalOrder> traversal_order;
  SubgraphLatency subgraph_latency;
  bool operator==(const Subgraph& other) const = default;
};

struct Solution {
  std::vector<Subgraph> subgraphs;
  bool operator==(const Solution& other) const = default;
};

absl::StatusOr<Solution> ReadSolution(const std::string& filename);

absl::StatusOr<TotalLatency> Evaluate(const Problem& problem,
                                      const Solution& solution);

}  // namespace mlsys

#endif  // MLSYS_H_
