load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "vector_addition",
    srcs = ["vector_addition.cu"],
    hdrs = ["vector_addition.h"],
)

cc_binary(
    name = "vector_addition_main",
    srcs = ["vector_addition_main.cc"],
    deps = [":vector_addition"],
)
