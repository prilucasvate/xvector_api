plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
}

android {
    namespace = "tw.xvector"
    compileSdk = 35
    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // 測試用模型與 golden 資料，由 mobile/ 與 tools/make_golden.py 產生
    sourceSets["androidTest"].assets.srcDirs("../../../mobile", "build/golden")
    publishing { singleVariant("release") }
}

dependencies {
    api("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "tw.xvector"
            artifactId = "xvector-android"
            version = "0.1.0"
            afterEvaluate { from(components["release"]) }
        }
    }
}
