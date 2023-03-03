load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "parallel_sum",
    srcs = ["parallel_sum.cu"],
    hdrs = ["parallel_sum.h"],
)

cc_binary(
    name = "parallel_sum_main",
    srcs = ["parallel_sum_main.cc"],
    deps = [":parallel_sum"],
)
