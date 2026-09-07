# Changelog

## 0.1.0 (2026-09-07)


### ⚠ BREAKING CHANGES

* v2 rewrite with torch-free ONNX inference and real-time pipeline ([#5](https://github.com/fodorad/BlinkLinMulT/issues/5))
* torch is no longer a core dependency. Install the torch extra for the PyTorch path, or the onnx extra for the torch-free runtime. The 1.x model classes and the blinklinmult.models module layout have been replaced; see docs/migration.md.

### Features

* torch-free ONNX inference path, cutting the core install to numpy and pyyaml ([71ce770](https://github.com/fodorad/BlinkLinMulT/commit/71ce770986bf72247c7028090932f2b8af19a020))
* v2 rewrite with torch-free ONNX inference and real-time pipeline ([#5](https://github.com/fodorad/BlinkLinMulT/issues/5)) ([71ce770](https://github.com/fodorad/BlinkLinMulT/commit/71ce770986bf72247c7028090932f2b8af19a020))


### Bug Fixes

* made it easier to import ([3efa07f](https://github.com/fodorad/BlinkLinMulT/commit/3efa07ff030e0a708a2d5c82143f4689938080c2))
