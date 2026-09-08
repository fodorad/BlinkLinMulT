# Changelog

## [2.1.0](https://github.com/fodorad/BlinkLinMulT/compare/v2.0.2...v2.1.0) (2026-09-08)


### Features

* **demo:** add a fast pipeline route and a head-pose choice ([#21](https://github.com/fodorad/BlinkLinMulT/issues/21)) ([fd53bf2](https://github.com/fodorad/BlinkLinMulT/commit/fd53bf2f622908ace31fcea866a0da037084c62a))

## [2.0.2](https://github.com/fodorad/BlinkLinMulT/compare/v2.0.1...v2.0.2) (2026-09-08)


### Bug Fixes

* **deps:** drop exordium[video] so pip resolves a working timm ([#19](https://github.com/fodorad/BlinkLinMulT/issues/19)) ([b137652](https://github.com/fodorad/BlinkLinMulT/commit/b13765275571433bbcb2fd847080c19ceb04f317))

## [2.0.1](https://github.com/fodorad/BlinkLinMulT/compare/v2.0.0...v2.0.1) (2026-09-07)


### Bug Fixes

* **demo:** pin the HF Space to a released version and add a deploy target ([#17](https://github.com/fodorad/BlinkLinMulT/issues/17)) ([efae6b7](https://github.com/fodorad/BlinkLinMulT/commit/efae6b74390060a6382007127112834c0b68a099))

## [2.0.0](https://github.com/fodorad/BlinkLinMulT/compare/v1.0.4...v2.0.0) (2026-09-07)


### ⚠ BREAKING CHANGES

* v2 rewrite with torch-free ONNX inference and real-time pipeline ([#5](https://github.com/fodorad/BlinkLinMulT/issues/5))
* torch is no longer a core dependency. Install the torch extra for the PyTorch path, or the onnx extra for the torch-free runtime. The 1.x model classes and the blinklinmult.models module layout have been replaced; see docs/migration.md.

### Features

* torch-free ONNX inference path, cutting the core install to numpy and pyyaml ([71ce770](https://github.com/fodorad/BlinkLinMulT/commit/71ce770986bf72247c7028090932f2b8af19a020))
* v2 rewrite with torch-free ONNX inference and real-time pipeline ([#5](https://github.com/fodorad/BlinkLinMulT/issues/5)) ([71ce770](https://github.com/fodorad/BlinkLinMulT/commit/71ce770986bf72247c7028090932f2b8af19a020))
