load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "conv",
    srcs = ["conv.cu"],
    hdrs = ["conv.h"],
)

cc_binary(
    name = "conv_main",
    srcs = ["conv_main.cc"],
    deps = [":conv"],
)
