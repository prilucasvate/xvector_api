# ONNX Runtime 透過 JNI 存取這些類別，不可被 R8 移除或改名
-keep class ai.onnxruntime.** { *; }
