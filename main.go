package main

import (
	"encoding/json"
	"fmt"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"os"
	"time"
)

// Main function that defines
// a web service endpoints a starts
// the web service
func main() {
	path := os.Args[1]
	if path == "" {
		log.Fatal("Path to image is not provided")
	}

	if os.Getenv("USE_CUDA") == "true" {
		UseCuda = true
	}
	n := time.Now()
	objs, err := detectPath(path)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(objs))
	fmt.Println("TOOK", time.Since(n))

}

// Handler of /detect POST endpoint
// Receives uploaded file with a name "image_file", passes it
// through YOLOv8 object detection network and returns and array
// of bounding boxes.
// Returns a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
func detectPath(filePath string) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("Error opening file: %w", err)
	}

	boxes, err := detect_objects_on_image(file)
	if err != nil {
		return nil, fmt.Errorf("Error processing image: %w", err)
	}
	buf, err := json.MarshalIndent(&boxes, "", " ")
	if err != nil {
		return nil, fmt.Errorf("Error marshalling data: %w", err)
	}
	return buf, nil
}

// Function receives an image,
// passes it through YOLOv8 neural network
// and returns an array of detected objects
// and their bounding boxes
// Returns Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
func detect_objects_on_image(buf io.Reader) ([][]interface{}, error) {
	input, img_width, img_height := prepare_input(buf)
	output, err := run_model(input)
	if err != nil {
		return nil, err
	}

	data := process_output(output, img_width, img_height)

	return data, nil
}

// Function used to pass provided input tensor to
// YOLOv8 neural network and return result
// Returns raw output of YOLOv8 network as a single dimension
// array
func run_model(input []float32) ([]float32, error) {

	var err error

	if Yolo8Model.Session == nil {
		Yolo8Model, err = InitYolo8Session(input)
		if err != nil {
			return nil, err
		}
	}

	return runInference(Yolo8Model, input)

}
