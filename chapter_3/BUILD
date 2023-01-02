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

cuda_library(
    name = "image_blur",
    srcs = ["image_blur.cu"],
    hdrs = ["image_blur.h"],
)

cc_binary(
    name = "image_blur_main",
    srcs = ["image_blur_main.cc"],
    deps = [
        ":image_blur",
        "//common:test_image",
    ],
)

cc_binary(
    name = "query_device_property_main",
    srcs = ["query_device_property_main.cc"],
    deps = [
        "@local_cuda//:cuda_runtime"
    ],
)
