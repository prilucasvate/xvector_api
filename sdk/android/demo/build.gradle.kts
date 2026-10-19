plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "tw.xvector.demo"
    compileSdk = 35
    defaultConfig {
        applicationId = "tw.xvector.demo"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { viewBinding = true }
    // ONNX 模型壓縮效果有限，不壓縮可直接 mmap 讀取
    androidResources { noCompress += listOf("onnx", "bin") }
    sourceSets["main"].assets.srcDirs(layout.buildDirectory.dir("generated/model-assets"))
}

// 將 mobile/export_onnx.py 產生的模型與 cohort 打包進 App
val copyModel by tasks.registering(Copy::class) {
    from("../../../mobile") { include("xvector_v14.onnx", "cohort.bin") }
    into(layout.buildDirectory.dir("generated/model-assets"))
    doFirst {
        require(file("../../../mobile/xvector_v14.onnx").exists()) {
            "找不到 mobile/xvector_v14.onnx，請先執行 python mobile/export_onnx.py"
        }
    }
}
tasks.named("preBuild") { dependsOn(copyModel) }

dependencies {
    implementation(project(":xvector"))
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
}
