package main

import (
	ort "github.com/yalue/onnxruntime_go"
	"os"
	"path"
	"runtime"
)

func InitYolo8Session(input []float32) (ModelSession, error) {
	ort.SetSharedLibraryPath(getSharedLibPath())
	err := ort.InitializeEnvironment()
	if err != nil {
		return ModelSession{}, err
	}

	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return ModelSession{}, err
	}

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return ModelSession{}, err
	}

	options, e := ort.NewSessionOptions()
	if e != nil {
		return ModelSession{}, err
	}

	if UseCuda {
		opts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return ModelSession{}, err
		}
		err = opts.Update(map[string]string{"device_id": "1"})
		if err != nil {
			return ModelSession{}, err
		}
		err = options.AppendExecutionProviderCUDA(opts)
		if err != nil {
			options.Destroy()
			return ModelSession{}, err
		}
		defer options.Destroy()
	}
	if UseCoreML { // If CoreML is enabled, append the CoreML execution provider
		e = options.AppendExecutionProviderCoreML(0)
		if e != nil {
			options.Destroy()
			return ModelSession{}, err
		}
		defer options.Destroy()
	}

	session, err := ort.NewAdvancedSessionWithONNXData(ModelData,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, options)

	if err != nil {
		return ModelSession{}, err
	}

	modelSes := ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}

	return modelSes, err
}

func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			tmpPath := path.Join(os.TempDir(), "onnxruntime.dll")
			err := os.WriteFile(tmpPath, OnnxruntimeDLL, 0644)
			if err != nil {
				panic(err)
			}
			return tmpPath
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			tmpPath := path.Join(os.TempDir(), "onnxruntime_arm64.dylib")
			err := os.WriteFile(tmpPath, OnnxruntimeARM64Dylib, 0644)
			if err != nil {
				panic(err)
			}
			return tmpPath
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			tmpPath := path.Join(os.TempDir(), "onnxruntime_arm64.so")
			err := os.WriteFile(tmpPath, OnnxruntimeARM64So, 0644)
			if err != nil {
				panic(err)
			}

			return tmpPath
		}

		if UseCuda {
			tmpPath := path.Join(os.TempDir(), "libonnxruntime_providers_cuda.so")
			err := os.WriteFile(tmpPath, OnnxruntimeCUDASo, 0644)
			if err != nil {
				panic(err)
			}

			return tmpPath

		}
		tmpPath := path.Join(os.TempDir(), "onnxruntime.so")
		err := os.WriteFile(tmpPath, OnnxruntimeSo, 0644)
		if err != nil {
			panic(err)
		}

		return tmpPath
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}

// Array of YOLOv8 class labels
var yolo_classes = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

func runInference(modelSes ModelSession, input []float32) ([]float32, error) {
	inTensor := modelSes.Input.GetData()
	copy(inTensor, input)
	err := modelSes.Session.Run()
	if err != nil {
		return nil, err
	}
	return modelSes.Output.GetData(), nil
}
