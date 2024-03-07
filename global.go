package main

import (
	_ "embed"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed third_party/onnxruntime.dll
var OnnxruntimeDLL []byte

//go:embed third_party/onnxruntime_arm64.dylib
var OnnxruntimeARM64Dylib []byte

//go:embed third_party/onnxruntime_arm64.so
var OnnxruntimeARM64So []byte

//go:embed third_party/onnxruntime.so
var OnnxruntimeSo []byte

//go:embed yolov8m.onnx
var ModelData []byte
var (
	UseCoreML  = false
	UseCuda    = false
	Blank      []float32
	Yolo8Model ModelSession
)

type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}
