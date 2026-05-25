#include "backend/opencl/opencl_context.hpp"
#include "backend/opencl/opencl_memory.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <functional>
#include <string>
#include <vector>

using powerserve::opencl::OpenCLContext;
using powerserve::opencl::OpenCLMemoryPool;

namespace {

constexpr size_t kMiB = 1024ull * 1024ull;

struct Options {
    size_t bytes = 256ull * kMiB;
    int warmup = 5;
    int repeat = 30;
};

struct Stats {
    double best_gbs = 0.0;
    double median_gbs = 0.0;
    double avg_gbs = 0.0;
};

const char *kStreamKernelSource = R"CLC(
__kernel void bandwidth_stream_copy(__global const uint4 *src,
                                    __global uint4 *dst,
                                    const ulong n_vec) {
    const ulong i = (ulong)get_global_id(0);
    if (i < n_vec) {
        dst[i] = src[i];
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
        "Usage: %s [--mib N] [--bytes N] [--warmup N] [--repeat N]\n"
        "  --mib N      Buffer size in MiB, default 256\n"
        "  --bytes N    Buffer size in bytes, overrides --mib\n"
        "  --warmup N   Warmup iterations, default 5\n"
        "  --repeat N   Measured iterations, default 30\n",
        argv0);
}

static bool parse_options(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if ((arg == "--mib" || arg == "--bytes" || arg == "--warmup" || arg == "--repeat") && i + 1 >= argc) {
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
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
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
                                cl_kernel *out_kernel) {
    cl_int err = CL_SUCCESS;
    const char *src = kStreamKernelSource;
    const size_t len = std::strlen(kStreamKernelSource);
    cl_program program = clCreateProgramWithSource(ctx->get_context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clCreateProgramWithSource failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        return false;
    }

    err = clBuildProgram(program, 1, &ctx->get_device(), nullptr, nullptr, nullptr);
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

    cl_kernel kernel = clCreateKernel(program, "bandwidth_stream_copy", &err);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clCreateKernel failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        clReleaseProgram(program);
        return false;
    }

    *out_program = program;
    *out_kernel = kernel;
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
    std::printf("  copy API traffic uses bytes; stream kernel traffic uses read+write = 2*bytes.\n\n");

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  return pool.copy_host_to_device_async(src, host.data(), opt.bytes, 0, 0, nullptr, ev);
                              },
                              &ns)) {
        return 1;
    }
    print_stats("H2D write", make_stats(ns, static_cast<double>(opt.bytes)));

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  return pool.copy_device_to_host_async(host_out.data(), src, opt.bytes, 0, 0, nullptr, ev);
                              },
                              &ns)) {
        return 1;
    }
    print_stats("D2H read", make_stats(ns, static_cast<double>(opt.bytes)));

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  return pool.copy_device_to_device_async(dst, src, opt.bytes, 0, 0, 0, nullptr, ev);
                              },
                              &ns)) {
        return 1;
    }
    print_stats("D2D copy", make_stats(ns, static_cast<double>(opt.bytes)));

    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    if (!build_stream_kernel(ctx, &program, &kernel)) {
        return 1;
    }

    const cl_ulong n_vec = static_cast<cl_ulong>(opt.bytes / 16);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_vec);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "clSetKernelArg failed: %s\n", OpenCLContext::get_error_string(err).c_str());
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }

    size_t max_wg = 256;
    clGetDeviceInfo(ctx->get_device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
    const size_t local = std::min<size_t>(256, std::max<size_t>(1, max_wg));
    const size_t global = ((static_cast<size_t>(n_vec) + local - 1) / local) * local;

    if (!measure_event_series(opt.warmup, opt.repeat,
                              [&](cl_event *ev) {
                                  cl_int e = clEnqueueNDRangeKernel(
                                      ctx->get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, ev);
                                  if (e != CL_SUCCESS) {
                                      std::fprintf(stderr, "clEnqueueNDRangeKernel failed: %s\n",
                                                   OpenCLContext::get_error_string(e).c_str());
                                      return false;
                                  }
                                  return true;
                              },
                              &ns)) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return 1;
    }
    print_stats("global stream", make_stats(ns, static_cast<double>(opt.bytes) * 2.0));

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    pool.free(src);
    pool.free(dst);
    return 0;
}
