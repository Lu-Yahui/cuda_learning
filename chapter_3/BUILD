load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "rgb_to_grayscale",
    srcs = ["rgb_to_grayscale.cu"],
    hdrs = ["rgb_to_grayscale.h"],
)

cc_binary(
    name = "rgb_to_grayscale_main",
    srcs = ["rgb_to_grayscale_main.cc"],
    deps = [
        ":rgb_to_grayscale",
        "//common:test_image",
    ],
)
