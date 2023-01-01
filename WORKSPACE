workspace(name = "cuda_learning")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cuda",
    sha256 = "536db6cc5adcc79ff6a65e6600c1a88f9f488ba9392d1053245775e0862be1b5",
    strip_prefix = "rules_cuda-0.1.0",
    urls = [
        "https://github.com/bazel-contrib/rules_cuda/archive/refs/tags/v0.1.0.zip"
    ],
)
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()
