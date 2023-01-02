package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_image_bin",
    srcs = [
        "cats.image.bin"
    ]
)

cc_library(
    name = "test_image",
    srcs = [],
    hdrs = ["test_image.h"],
    data = [
        ":test_image_bin"
    ]
)
