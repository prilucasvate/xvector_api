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
    buildTypes {
        release {
            // R8 移除未使用的 AndroidX / Material 程式碼
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    // 壓縮 libonnxruntime.so (約 17 MB -> 7 MB)，安裝時才解壓
    packaging { jniLibs { useLegacyPackaging = true } }
    // ONNX Runtime 預設包含 4 種 CPU 架構的原生函式庫，依架構拆成獨立 APK：
    // arm64-v8a 給實體手機，x86_64 給模擬器
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }
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
