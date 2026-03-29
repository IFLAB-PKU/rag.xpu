// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/thread_pool.hpp"

#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
    int iterations = 32;
    if (argc > 1) {
        iterations = std::atoi(argv[1]);
    }

    std::vector<powerserve::ThreadConfig> configs(8);
    for (int iter = 0; iter < iterations; ++iter) {
        powerserve::ThreadPool pool(configs);
        pool.run([](size_t) {
            for (int i = 0; i < 64; ++i) {
                std::this_thread::yield();
            }
        });
    }

    std::cout << "thread_pool_shutdown_race: ok (" << iterations << " iterations)" << std::endl;
    return 0;
}
