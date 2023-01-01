load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "vector_addition",
    srcs = ["vector_addition.cu"],
    hdrs = ["vector_addition.h"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [":vector_addition"],
)
