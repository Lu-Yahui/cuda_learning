load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "matrix_mul",
    srcs = ["matrix_mul.cu"],
    hdrs = ["matrix_mul.h"],
)

cc_binary(
    name = "matrix_mul_main",
    srcs = ["matrix_mul_main.cc"],
    deps = [":matrix_mul"],
)
