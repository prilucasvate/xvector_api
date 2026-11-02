// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "XVector",
    platforms: [.iOS(.v15), .macOS(.v14)],
    products: [
        .library(name: "XVector", targets: ["XVector"]),
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.20.0"),
    ],
    targets: [
        .target(
            name: "XVector",
            dependencies: [.product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")]
        ),
        .testTarget(name: "XVectorTests", dependencies: ["XVector"]),
    ]
)
