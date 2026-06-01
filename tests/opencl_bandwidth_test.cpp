#include "backend/opencl/opencl_context.hpp"
#include "backend/opencl/opencl_memory.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <limits>
#include <memory>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using powerserve::opencl::OpenCLContext;
using powerserve::opencl::OpenCLMemoryPool;

namespace {

constexpr size_t kMiB = 1024ull * 1024ull;

struct Options {
    size_t bytes = 256ull * kMiB;
    int warmup = 5;
    int repeat = 30;
    int cpu_threads = 0;
};

struct Stats {
    double best_gbs = 0.0;
    double median_gbs = 0.0;
    double avg_gbs = 0.0;
};

volatile uint64_t g_cpu_read_sink = 0;

static void consume_cpu_value(uint64_t v) {
    g_cpu_read_sink = g_cpu_read_sink ^ v;
}

const char *kStreamKernelSource = R"CLC(
__kernel void bandwidth_stream_copy(__global const uint4 *src,
                                    __global uint4 *dst,
                                    const ulong n_vec) {
    const ulong i = (ulong)get_global_id(0);
    if (i < n_vec) {
        dst[i] = src[i];
    }
}

__kernel void bandwidth_checksum(__global const uint4 *src,
                                 __global uint *partial,
                                 const ulong n_vec) {
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const ulong global_size = (ulong)get_global_size(0);
    ulong i = (ulong)get_global_id(0);
    uint acc = 0;

    while (i < n_vec) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
        i += global_size;
    }

    __local uint scratch[256];
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] ^= scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[group] = scratch[0];
    }
}
)CLC";

static bool parse_size(const char *s, size_t *out) {
    if (!s || !out) {
        return false;
    }
    char *end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || (end && *end != '\0') || v == 0) {
        return false;
    }
    *out = static_cast<size_t>(v);
    return true;
}

static bool parse_int(const char *s, int *out) {
    if (!s || !out) {
        return false;
    }
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || (end && *end != '\0') || v < 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

static void print_usage(const char *argv0) {
    std::printf(
        "Usage: %s [--mib N] [--bytes N] [--warmup N] [--repeat N] [--cpu-threads N]\n"
        "  --mib N      Buffer size in MiB, default 256\n"
        "  --bytes N    Buffer size in bytes, overrides --mib\n"
        "  --warmup N   Warmup iterations, default 5\n"
        "  --repeat N   Measured iterations, default 30\n"
        "  --cpu-threads N  CPU worker threads for MT tests, default hardware_concurrency\n",
        argv0);
}

static bool parse_options(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if ((arg == "--mib" || arg == "--bytes" || arg == "--warmup" ||
             arg == "--repeat" || arg == "--cpu-threads") && i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
            return false;
        }
        if (arg == "--mib") {
            size_t mib = 0;
            if (!parse_size(argv[++i], &mib)) {
                std::fprintf(stderr, "Invalid --mib value\n");
                return false;
            }
            opt->bytes = mib * kMiB;
        } else if (arg == "--bytes") {
            if (!parse_size(argv[++i], &opt->bytes)) {
                std::fprintf(stderr, "Invalid --bytes value\n");
                return false;
            }
        } else if (arg == "--warmup") {
            if (!parse_int(argv[++i], &opt->warmup)) {
                std::fprintf(stderr, "Invalid --warmup value\n");
                return false;
            }
        } else if (arg == "--repeat") {
            if (!parse_int(argv[++i], &opt->repeat) || opt->repeat <= 0) {
                std::fprintf(stderr, "Invalid --repeat value\n");
                return false;
            }
        } else if (arg == "--cpu-threads") {
            if (!parse_int(argv[++i], &opt->cpu_threads) || opt->cpu_threads <= 0) {
                std::fprintf(stderr, "Invalid --cpu-threads value\n");
                return false;
            }
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (opt->cpu_threads == 0) {
        unsigned n = std::thread::hardware_concurrency();
        opt->cpu_threads = n > 0 ? static_cast<int>(n) : 1;
    }
    opt->bytes = (opt->bytes / 16) * 16;
    if (opt->bytes == 0) {
        std::fprintf(stderr, "Buffer size must be at least 16 bytes\n");
        return false;
    }
    return true;
}

static bool event_elapsed_ns(cl_event ev, double *out_ns) {
    cl_int err = clWaitForEvents(1, &ev);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clWaitForEvents failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        return false;
    }

    cl_ulong start = 0;
    cl_ulong end = 0;
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr,
                     "clGetEventProfilingInfo(START) failed: %s. Reconfigure with -DPOWERSERVE_OPENCL_PROFILING=ON.\n",
                     OpenCLContext::get_error_string(err).c_str());
        return false;
    }
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr,
                     "clGetEventProfilingInfo(END) failed: %s. Reconfigure with -DPOWERSERVE_OPENCL_PROFILING=ON.\n",
                     OpenCLContext::get_error_string(err).c_str());
        return false;
    }
    if (end <= start) {
        std::fprintf(stderr, "Invalid profiling interval: start=%llu end=%llu\n",
                     static_cast<unsigned long long>(start),
                     static_cast<unsigned long long>(end));
        return false;
    }
    *out_ns = static_cast<double>(end - start);
    return true;
}

static Stats make_stats(std::vector<double> ns_values, double traffic_bytes) {
    std::vector<double> gbs;
    gbs.reserve(ns_values.size());
    for (double ns : ns_values) {
        if (ns > 0.0) {
            gbs.push_back(traffic_bytes / ns);
        }
    }
    std::sort(gbs.begin(), gbs.end());
    Stats s;
    if (gbs.empty()) {
        return s;
    }
    double sum = 0.0;
    for (double v : gbs) {
        sum += v;
    }
    s.best_gbs = gbs.back();
    s.median_gbs = gbs[gbs.size() / 2];
    s.avg_gbs = sum / static_cast<double>(gbs.size());
    return s;
}

static void print_stats(const char *name, const Stats &s) {
    std::printf("%-18s best=%8.2f GB/s  median=%8.2f GB/s  avg=%8.2f GB/s\n",
                name, s.best_gbs, s.median_gbs, s.avg_gbs);
}

static size_t floor_power_of_two(size_t v) {
    size_t r = 1;
    while ((r << 1) <= v) {
        r <<= 1;
    }
    return r;
}

static double now_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count());
}

static uint64_t checksum_u64(const uint64_t *data, size_t count) {
    uint64_t acc = 0;
    for (size_t i = 0; i < count; ++i) {
        acc ^= data[i] + 0x9e3779b97f4a7c15ull + (acc << 6) + (acc >> 2);
    }
    return acc;
}

template <typename Fn>
static bool measure_wall_series(int warmup, int repeat, Fn &&fn, std::vector<double> *out_ns) {
    for (int i = 0; i < warmup; ++i) {
        if (fn() < 0.0) {
            return false;
        }
    }
    out_ns->clear();
    out_ns->reserve(static_cast<size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        const double elapsed = fn();
        if (elapsed < 0.0) {
            return false;
        }
        out_ns->push_back(elapsed);
    }
    return true;
}

template <typename Fn>
static void parallel_for_chunks(size_t count, int n_threads, Fn &&fn) {
    if (count == 0 || n_threads <= 1) {
        fn(0, count);
        return;
    }

    const size_t threads = std::min<size_t>(static_cast<size_t>(n_threads), count);
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (size_t t = 0; t < threads; ++t) {
        const size_t begin = (count * t) / threads;
        const size_t end = (count * (t + 1)) / threads;
        workers.emplace_back([begin, end, &fn]() {
            fn(begin, end);
        });
    }

    for (auto &worker : workers) {
        worker.join();
    }
}

static bool measure_event_series(int warmup,
                                 int repeat,
                                 const std::function<bool(cl_event *)> &enqueue,
                                 std::vector<double> *out_ns) {
    for (int i = 0; i < warmup; ++i) {
        cl_event ev = nullptr;
        if (!enqueue(&ev)) {
            return false;
        }
        if (ev) {
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }
    }

    out_ns->clear();
    out_ns->reserve(static_cast<size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        cl_event ev = nullptr;
        if (!enqueue(&ev) || !ev) {
            return false;
        }
        double ns = 0.0;
        bool ok = event_elapsed_ns(ev, &ns);
        clReleaseEvent(ev);
        if (!ok) {
            return false;
        }
        out_ns->push_back(ns);
    }
    return true;
}

static bool build_stream_kernel(const std::shared_ptr<OpenCLContext> &ctx,
                                cl_program *out_program,
                                cl_kernel *out_stream_kernel,
                                cl_kernel *out_checksum_kernel) {
    cl_int err = CL_SUCCESS;
    const char *src = kStreamKernelSource;
    const size_t len = std::strlen(kStreamKernelSource);
    cl_program program = clCreateProgramWithSource(ctx->get_context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clCreateProgramWithSource failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        return false;
    }

    cl_device_id device = ctx->get_device();
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, ctx->get_device(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        if (log_size > 0) {
            clGetProgramBuildInfo(program, ctx->get_device(), CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        }
        std::fprintf(stderr, "clBuildProgram failed: %s\n%s\n",
                     OpenCLContext::get_error_string(err).c_str(), log.data());
        clReleaseProgram(program);
        return false;
    }

    cl_kernel stream_kernel = clCreateKernel(program, "bandwidth_stream_copy", &err);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clCreateKernel(stream) failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        clReleaseProgram(program);
        return false;
    }

    cl_kernel checksum_kernel = clCreateKernel(program, "bandwidth_checksum", &err);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clCreateKernel(checksum) failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return false;
    }

    *out_program = program;
    *out_stream_kernel = stream_kernel;
    *out_checksum_kernel = checksum_kernel;
    return true;
}

} // namespace

int main(int argc, char **argv) {
#ifndef POWERSERVE_OPENCL_PROFILING
    std::fprintf(stderr,
                 "opencl_bandwidth_test requires -DPOWERSERVE_OPENCL_PROFILING=ON so OpenCL events contain timestamps.\n");
    return 1;
#endif

    Options opt;
    if (!parse_options(argc, argv, &opt)) {
        print_usage(argv[0]);
        return 1;
    }

    auto ctx = std::make_shared<OpenCLContext>();
    if (!ctx->initialize()) {
        return 1;
    }

    OpenCLMemoryPool pool(ctx);
    cl_mem src = pool.allocate(opt.bytes, CL_MEM_READ_WRITE);
    cl_mem dst = pool.allocate(opt.bytes, CL_MEM_READ_WRITE);
    if (!src || !dst) {
        std::fprintf(stderr, "Failed to allocate device buffers: %zu bytes each\n", opt.bytes);
        return 1;
    }

    std::vector<uint8_t> host(opt.bytes);
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    }

    std::vector<uint8_t> host_out(opt.bytes);
    std::vector<double> ns;

    std::printf("OpenCL bandwidth test\n");
    std::printf("  bytes=%zu (%.2f MiB), warmup=%d, repeat=%d\n",
                opt.bytes, static_cast<double>(opt.bytes) / static_cast<double>(kMiB),
                opt.warmup, opt.repeat);
    std::printf("  copy API traffic uses bytes; stream kernel traffic uses read+write = 2*bytes.\n");
    std::printf("  H2D consumed is wall-clock WriteBuffer + GPU checksum + small readback.\n");
    std::printf("  Map write consumed is wall-clock MapBuffer + CPU write + Unmap + GPU checksum.\n\n");

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  return pool.copy_host_to_device_async(src, host.data(), opt.bytes, 0, 0, nullptr, ev);
                              },
                              &ns)) {
        return 1;
    }
    print_stats("H2D event", make_stats(ns, static_cast<double>(opt.bytes)));

    cl_program program = nullptr;
    cl_kernel stream_kernel = nullptr;
    cl_kernel checksum_kernel = nullptr;
    if (!build_stream_kernel(ctx, &program, &stream_kernel, &checksum_kernel)) {
        return 1;
    }

    const cl_ulong n_vec = static_cast<cl_ulong>(opt.bytes / 16);
    size_t max_wg = 256;
    clGetDeviceInfo(ctx->get_device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
    const size_t local = floor_power_of_two(std::min<size_t>(256, std::max<size_t>(1, max_wg)));
    const size_t global = ((static_cast<size_t>(n_vec) + local - 1) / local) * local;
    const size_t checksum_groups = global / local;

    cl_mem partial = pool.allocate(checksum_groups * sizeof(cl_uint), CL_MEM_READ_WRITE);
    if (!partial) {
        std::fprintf(stderr, "Failed to allocate checksum partial buffer\n");
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(checksum_kernel, 0, sizeof(cl_mem), &src);
    err |= clSetKernelArg(checksum_kernel, 1, sizeof(cl_mem), &partial);
    err |= clSetKernelArg(checksum_kernel, 2, sizeof(cl_ulong), &n_vec);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clSetKernelArg(checksum) failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }

    std::vector<cl_uint> partial_host(checksum_groups);
    std::vector<double> h2d_consumed_ns;
    h2d_consumed_ns.reserve(static_cast<size_t>(opt.repeat));

    auto run_h2d_consumed_once = [&]() -> double {
        const double t0 = now_ns();
        cl_int e = clEnqueueWriteBuffer(ctx->get_queue(), src, CL_FALSE, 0, opt.bytes, host.data(), 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "clEnqueueWriteBuffer(H2D consumed) failed: %s\n",
                         OpenCLContext::get_error_string(e).c_str());
            return -1.0;
        }
        e = clEnqueueNDRangeKernel(ctx->get_queue(), checksum_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "clEnqueueNDRangeKernel(checksum) failed: %s\n",
                         OpenCLContext::get_error_string(e).c_str());
            return -1.0;
        }
        e = clEnqueueReadBuffer(ctx->get_queue(), partial, CL_TRUE, 0,
                                partial_host.size() * sizeof(cl_uint), partial_host.data(), 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "clEnqueueReadBuffer(checksum partial) failed: %s\n",
                         OpenCLContext::get_error_string(e).c_str());
            return -1.0;
        }
        const double t1 = now_ns();
        return t1 - t0;
    };

    for (int i = 0; i < opt.warmup; ++i) {
        if (run_h2d_consumed_once() < 0.0) {
            pool.free(partial);
            clReleaseKernel(checksum_kernel);
            clReleaseKernel(stream_kernel);
            clReleaseProgram(program);
            return 1;
        }
    }
    for (int i = 0; i < opt.repeat; ++i) {
        const double elapsed = run_h2d_consumed_once();
        if (elapsed < 0.0) {
            pool.free(partial);
            clReleaseKernel(checksum_kernel);
            clReleaseKernel(stream_kernel);
            clReleaseProgram(program);
            return 1;
        }
        h2d_consumed_ns.push_back(elapsed);
    }
    print_stats("H2D consumed", make_stats(h2d_consumed_ns, static_cast<double>(opt.bytes)));

#ifdef CL_MAP_WRITE_INVALIDATE_REGION
    const cl_map_flags map_write_flags = CL_MAP_WRITE_INVALIDATE_REGION;
#else
    const cl_map_flags map_write_flags = CL_MAP_WRITE;
#endif

    std::vector<double> map_write_ns;
    map_write_ns.reserve(static_cast<size_t>(opt.repeat));
    auto run_map_write_consumed_once = [&]() -> double {
        cl_int e = CL_SUCCESS;
        const double t0 = now_ns();
        void *mapped = clEnqueueMapBuffer(
            ctx->get_queue(), src, CL_TRUE, map_write_flags, 0, opt.bytes, 0, nullptr, nullptr, &e);
        if (e != CL_SUCCESS || !mapped) {
            std::fprintf(stderr, "clEnqueueMapBuffer(write) failed: %s\n",
                         OpenCLContext::get_error_string(e).c_str());
            return -1.0;
        }

        std::memcpy(mapped, host.data(), opt.bytes);

        e = clEnqueueUnmapMemObject(ctx->get_queue(), src, mapped, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "clEnqueueUnmapMemObject(write) failed: %s\n",
                         OpenCLContext::get_error_string(e).c_str());
            return -1.0;
        }
        e = clEnqueueNDRangeKernel(ctx->get_queue(), checksum_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "clEnqueueNDRangeKernel(map checksum) failed: %s\n",
                         OpenCLContext::get_error_string(e).c_str());
            return -1.0;
        }
        e = clEnqueueReadBuffer(ctx->get_queue(), partial, CL_TRUE, 0,
                                partial_host.size() * sizeof(cl_uint), partial_host.data(), 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "clEnqueueReadBuffer(map checksum partial) failed: %s\n",
                         OpenCLContext::get_error_string(e).c_str());
            return -1.0;
        }
        const double t1 = now_ns();
        return t1 - t0;
    };

    for (int i = 0; i < opt.warmup; ++i) {
        if (run_map_write_consumed_once() < 0.0) {
            pool.free(partial);
            clReleaseKernel(checksum_kernel);
            clReleaseKernel(stream_kernel);
            clReleaseProgram(program);
            return 1;
        }
    }
    for (int i = 0; i < opt.repeat; ++i) {
        const double elapsed = run_map_write_consumed_once();
        if (elapsed < 0.0) {
            pool.free(partial);
            clReleaseKernel(checksum_kernel);
            clReleaseKernel(stream_kernel);
            clReleaseProgram(program);
            return 1;
        }
        map_write_ns.push_back(elapsed);
    }
    print_stats("Map write consumed", make_stats(map_write_ns, static_cast<double>(opt.bytes)));

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  return pool.copy_device_to_host_async(host_out.data(), src, opt.bytes, 0, 0, nullptr, ev);
                              },
                              &ns)) {
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("D2H read", make_stats(ns, static_cast<double>(opt.bytes)));

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  return pool.copy_device_to_device_async(dst, src, opt.bytes, 0, 0, 0, nullptr, ev);
                              },
                              &ns)) {
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("D2D copy", make_stats(ns, static_cast<double>(opt.bytes)));

    err = CL_SUCCESS;
    err |= clSetKernelArg(stream_kernel, 0, sizeof(cl_mem), &src);
    err |= clSetKernelArg(stream_kernel, 1, sizeof(cl_mem), &dst);
    err |= clSetKernelArg(stream_kernel, 2, sizeof(cl_ulong), &n_vec);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clSetKernelArg(stream) failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  cl_int e = clEnqueueNDRangeKernel(
                                      ctx->get_queue(), stream_kernel, 1, nullptr, &global, &local, 0, nullptr, ev);
                                  if (e != CL_SUCCESS) {
                                      std::fprintf(stderr, "clEnqueueNDRangeKernel failed: %s\n",
                                                   OpenCLContext::get_error_string(e).c_str());
                                      return false;
                                  }
                                  return true;
                              },
                              &ns)) {
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("global stream", make_stats(ns, static_cast<double>(opt.bytes) * 2.0));

    std::printf("\nCPU memory test\n");
    std::printf("  cpu_threads=%d\n", opt.cpu_threads);

    std::vector<uint8_t> cpu_src(opt.bytes);
    std::vector<uint8_t> cpu_dst(opt.bytes);
    for (size_t i = 0; i < cpu_src.size(); ++i) {
        cpu_src[i] = static_cast<uint8_t>((i * 97u + 23u) & 0xffu);
    }

    std::vector<double> cpu_ns;
    if (!measure_wall_series(opt.warmup, opt.repeat,
                             [&]() -> double {
                                 const double t0 = now_ns();
                                 std::memcpy(cpu_dst.data(), cpu_src.data(), opt.bytes);
                                 const double t1 = now_ns();
                                 consume_cpu_value(static_cast<uint64_t>(cpu_dst[0]));
                                 consume_cpu_value(static_cast<uint64_t>(cpu_dst[opt.bytes - 1]));
                                 return t1 - t0;
                             },
                             &cpu_ns)) {
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("CPU memcpy", make_stats(cpu_ns, static_cast<double>(opt.bytes)));

    if (!measure_wall_series(opt.warmup, opt.repeat,
                             [&]() -> double {
                                 const double t0 = now_ns();
                                 parallel_for_chunks(opt.bytes, opt.cpu_threads, [&](size_t begin, size_t end) {
                                     std::memcpy(cpu_dst.data() + begin, cpu_src.data() + begin, end - begin);
                                 });
                                 const double t1 = now_ns();
                                 consume_cpu_value(static_cast<uint64_t>(cpu_dst[0]));
                                 consume_cpu_value(static_cast<uint64_t>(cpu_dst[opt.bytes - 1]));
                                 return t1 - t0;
                             },
                             &cpu_ns)) {
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("CPU memcpy MT", make_stats(cpu_ns, static_cast<double>(opt.bytes)));

    const size_t u64_count = opt.bytes / sizeof(uint64_t);
    auto *cpu_src64 = reinterpret_cast<const uint64_t *>(cpu_src.data());
    auto *cpu_dst64 = reinterpret_cast<uint64_t *>(cpu_dst.data());
    if (!measure_wall_series(opt.warmup, opt.repeat,
                             [&]() -> double {
                                 const double t0 = now_ns();
                                 for (size_t i = 0; i < u64_count; ++i) {
                                     cpu_dst64[i] = cpu_src64[i];
                                 }
                                 const double t1 = now_ns();
                                 consume_cpu_value(checksum_u64(cpu_dst64, std::min<size_t>(u64_count, 16)));
                                 return t1 - t0;
                             },
                             &cpu_ns)) {
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("CPU stream", make_stats(cpu_ns, static_cast<double>(opt.bytes) * 2.0));

    if (!measure_wall_series(opt.warmup, opt.repeat,
                             [&]() -> double {
                                 const double t0 = now_ns();
                                 parallel_for_chunks(u64_count, opt.cpu_threads, [&](size_t begin, size_t end) {
                                     for (size_t i = begin; i < end; ++i) {
                                         cpu_dst64[i] = cpu_src64[i];
                                     }
                                 });
                                 const double t1 = now_ns();
                                 consume_cpu_value(checksum_u64(cpu_dst64, std::min<size_t>(u64_count, 16)));
                                 return t1 - t0;
                             },
                             &cpu_ns)) {
        pool.free(partial);
        clReleaseKernel(checksum_kernel);
        clReleaseKernel(stream_kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("CPU stream MT", make_stats(cpu_ns, static_cast<double>(opt.bytes) * 2.0));

    pool.free(partial);
    clReleaseKernel(checksum_kernel);
    clReleaseKernel(stream_kernel);
    clReleaseProgram(program);
    pool.free(src);
    pool.free(dst);
    return 0;
}
