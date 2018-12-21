package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

func sigmoid(x float64) float64 {
	if x > -600 {
		return (1 / (1 + math.Exp(-x)))
	}
	return 0
}

func cutNegative(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func randomize() {
	fmt.Println("randomizing weights...")
	for ri := 0; ri <= len(layers)-2; ri++ {
		if ri != 0 {
			weights = append(weights, [][]float64{{}})
		}
		for n := 0; n < layers[ri+1]; n++ {
			if n != 0 {
				weights[ri] = append(weights[ri], []float64{})
			}
			for prv := 0; prv <= layers[ri]; prv++ {
				rand.Seed(time.Now().UTC().UnixNano() + int64(ri) + int64(n) + int64(prv))
				weights[ri][n] = append(weights[ri][n], -1+2*rand.Float64())
			}
		}
	}
}

func calcErrors(d int) {
	errors = [][]float64{{}}
	for l := 0; l < len(layers)-1; l++ {
		errors = append(errors, []float64{})
	}
	for lbw := len(errors) - 1; lbw >= 0; lbw-- {
		if lbw == len(errors)-1 {
			for n := 0; n < layers[lbw]; n++ {
				errors[lbw] = append(errors[lbw], dataset[d][1][n]-nodes[lbw][n])
			}
		} else if lbw != 0 {
			for h := 0; h <= layers[lbw]; h++ {

				var dotProduct float64 = 0
				for n := 0; n < layers[lbw+1]; n++ {
					dotProduct += errors[lbw+1][n] * weights[lbw][n][h]
				}

				errors[lbw] = append(errors[lbw], nodes[lbw][h]*(1-nodes[lbw][h])*dotProduct)
			}
		} else {
			for h := 0; h <= layers[lbw]; h++ {
				var dotProduct float64 = 0
				for n := 0; n < layers[lbw+1]; n++ {
					dotProduct += errors[lbw+1][n] * weights[lbw][n][h]
				}

				errors[lbw] = append(errors[lbw], nodes[lbw][h]*dotProduct)
			}
		}
	}

}

func forward(d int) {
	nodes = [][]float64{{}}
	nodes[0] = append(nodes[0], dataset[d][0]...)
	nodes[0] = append(nodes[0], 1)

	for ri := 0; ri < len(layers)-1; ri++ {
		nodes = append(nodes, []float64{})
		var sm float64 = 0
		for h := 0; h < layers[ri+1]; h++ {
			for n := 0; n < len(nodes[ri]); n++ {
				sm += weights[ri][h][n] * nodes[ri][n]
			}

			if ri != len(layers)-2 {
				nodes[ri+1] = append(nodes[ri+1], sigmoid(sm))
			} else {
				nodes[ri+1] = append(nodes[ri+1], cutNegative(sm))
			}
		}
		if ri != len(layers)-2 {
			nodes[ri+1] = append(nodes[ri+1], 1)
		}
	}

	calcErrors(d)

}

func backward(lr float64) {
	weightHistory = append(weightHistory, weights)
	newWeights := weights
	for ri := 0; ri < len(layers)-1; ri++ {
		for h := 0; h < layers[ri+1]; h++ {
			for n := 0; n < len(errors[ri]); n++ {
				newWeights[ri][h][n] += nodes[ri][n] * errors[ri+1][h] * lr
			}
		}
	}
}

func status(d int) {
	var sum float64 = 0

	for e := 0; e < len(nodes[len(nodes)-1]); e++ {
		for i := 0; i < len(nodes[len(nodes)-1]); i++ {
			sum += math.Abs(nodes[len(nodes)-1][e]-nodes[len(nodes)-1][i]) / float64(len(nodes[len(nodes)-1])) * 100
		}
	}

	var max float64 = 0
	var ind int = 0
	var avgNodeDifference float64 = 0
	for p := 0; p < len(nodes[len(nodes)-1]); p++ {
		if max < nodes[len(nodes)-1][p] {
			max = nodes[len(nodes)-1][p]
			ind = p
		}
	}
	var indc int = 0
	for p := 0; p < len(dataset[d][1]); p++ {
		if 1 == dataset[d][1][p] {
			indc = p
			break
		}
	}
	for p := 0; p < len(nodes[len(nodes)-1]); p++ {
		if p != ind {
			avgNodeDifference += nodes[len(nodes)-1][p]
		}
	}
	avgNodeDifference /= float64(len(nodes[len(nodes)-1]) - 1)

	if ind == indc {
		fmt.Print(d, " / ", len(dataset)-1, ": " /*dataset[d][0], "		-> ",*/, predictions[ind], "	correct!" /* nodes[len(nodes)-1], "	(", dataset[d][1], ") 	*/)

		endSuccess = append(endSuccess, 1)
		runSuccess = append(runSuccess, 1)
	} else {

		fmt.Print(d, " / ", len(dataset)-1, ": " /* dataset[d][0], "		-> ",*/, predictions[ind], "	wrong!	" /*	 nodes[len(nodes)-1], "	(", dataset[d][1], ") 	*/)
		endSuccess = append(endSuccess, 0)
		runSuccess = append(runSuccess, 0)
	}
	if sum > 100 {
		if len(runConfidence) < picWidth {
			runConfidence = append(runConfidence, 100)
		} else {
			for i := 1; i < len(runConfidence)-1; i++ {
				runConfidence[i] = runConfidence[i+1]
				runSuccess[i] = runSuccess[i+1]
			}
			runConfidence[len(runConfidence)-1] = 100
		}

		endConfidence = append(endConfidence, 1)

		runConfidenceAverage = append(runConfidenceAverage, average(runConfidence, int64(runConfidenceLength)))
		runSuccessAverage = append(runSuccessAverage, average(runSuccess, int64(runConfidenceLength)))

		fmt.Print("	", int(runConfidence[int(math.Min(float64(d), float64(picWidth-1)))]), "%	", int(runConfidenceAverage[int(math.Min(float64(d), float64(picWidth-1)))]))

	} else {
		if len(runConfidence) < picWidth {
			runConfidence = append(runConfidence, (nodes[len(nodes)-1][ind]-avgNodeDifference)*100)
		} else {
			for i := 1; i < len(runConfidence)-1; i++ {
				runConfidence[i] = runConfidence[i+1]
				runSuccess[i] = runSuccess[i+1]
			}
			runConfidence[len(runConfidence)-1] = (nodes[len(nodes)-1][ind] - avgNodeDifference) * 100
		}
		endConfidence = append(endConfidence, (nodes[len(nodes)-1][ind] - avgNodeDifference))

		runConfidenceAverage = append(runConfidenceAverage, average(runConfidence, int64(runConfidenceLength)))
		runSuccessAverage = append(runSuccessAverage, average(runSuccess, int64(runConfidenceLength)))

		fmt.Print("	", int(runConfidence[int(math.Min(float64(d), float64(picWidth-1)))]), "%	", int(runConfidenceAverage[int(math.Min(float64(d), float64(picWidth-1)))]))
	}

	fmt.Print("	", dataset[d][0], "\n")

}

func average(arr []float64, params ...int64) float64 {
	var maxBack int64
	if len(params) != 0 {
		maxBack = params[0]
	} else {
		maxBack = int64(len(arr))
	}
	var arrSum float64 = 0
	for p := 0; p < int(math.Min(float64(len(arr)), float64(maxBack))); p++ {
		arrSum += arr[len(arr)-1-p]
	}
	return arrSum / math.Min(float64(len(arr)), float64(maxBack))
}

func createData(n int) {
	fmt.Println("creating ramdom datasets... (", n, ")")
	dataset = [][][]float64{{{}}}

	for l := 0; l < n; l++ {

		rand.Seed(time.Now().UTC().UnixNano() + int64(l))
		var o = []float64{}

		var a float64 = float64(rand.Intn(16))
		var b float64 = float64(rand.Intn(16))
		var c float64 = 0

		if l%2 == 0 {
			c = a + b
			o = []float64{1, 0}
		} else {
			c = a * b
			o = []float64{0, 1}
		}
		//------------------------------------------------------------------------------------ logical outputs
		predictions[0] = "added     "
		predictions[1] = "multiplied"

		if l != 0 {
			dataset = append(dataset, [][]float64{{}})
		}
		dataset[l] = append(dataset[l], []float64{})

		dataset[l][0] = append(dataset[l][0], a)
		dataset[l][0] = append(dataset[l][0], b)
		dataset[l][0] = append(dataset[l][0], c)
		dataset[l][1] = append(dataset[l][1], o...)
	}

}

func learn(lr float64) {
	createData(datasets + amt) //------------------------------------------------------------------- datasets (learning)

	fmt.Println("learning...")

	f := 0
	for d := 0; d < len(dataset)-amt; d++ {
		forward(d)
		status(d)
		backward(lr)
		if average(runConfidence, int64(runConfidenceLength)) > runConfidenceThreshhold && d > len(dataset)/4 {
			fmt.Println("\nlearning session (", d, ") successful\n")
			break
		} else if d == len(dataset)-1 {
			fmt.Println("\nlearning session (", len(dataset), ") successful\n")
		} else if average(runConfidence, int64(runConfidenceLength)) < 20 && d > len(dataset)/4 {
			if f != 3 {
				randomize()
				runSuccess = []float64{}
				runSuccessAverage = []float64{}
				weightHistory = [][][][]float64{{{{}}}}
				endConfidence = []float64{}
				endSuccess = []float64{}
				runConfidenceAverage = []float64{}
				runConfidence = []float64{}

				d = -1
				f++
			} else {
				break
			}
		}
	}

}

func filename() string {
	name := "result_" + time.Now().Format("2-Jan-2006_15-04-05") + ".png"
	return name
}

var layers = []int{3, 5, 2}       //------------------------------------------------------ layout (!)
var weights = [][][]float64{{{}}} // weights[layer_left][node_in_layer_left][node_in_layer_right]
var nodes = [][]float64{{}}
var errors = [][]float64{{}}
var dataset = [][][]float64{{{}}}
var predictions = [2]string{} //------------------------------------------------------------ how many logical outputs

var datasets = 10000
var amt int = 100

var endConfidence = []float64{}
var endSuccess = []float64{}

var runConfidenceAverage = []float64{}
var runConfidence = []float64{}
var runConfidenceLength = 10000          // 97% will be based on this number of previous confidence results
var runConfidenceThreshhold float64 = 97 // learning done hitting 97%

var runSuccess = []float64{}
var runSuccessAverage = []float64{}

var weightHistory = [][][][]float64{{{{}}}}

var picWidth = datasets

func main() {

	randomize()

	learn(0.05)

	fmt.Println("making predictions...\n")
	time.Sleep(2 * time.Second)

	endConfidence = []float64{}
	endSuccess = []float64{}
	for p := datasets; p < datasets+amt; p++ {
		forward(p)
		status(p)
	}

	fmt.Println("\nOverall confidence: 	", average(endConfidence)*100, "%	 Success rate:	", average(endSuccess)*100, "%		Score:	", int(average(endConfidence)*(average(endSuccess)*10000)), "\n\n", "producing image...\n")

	height := 500
	width := int(math.Min(float64(picWidth), float64(len(runConfidence))))

	upLeft := image.Point{0, 0}
	lowRight := image.Point{width, height}

	img := image.NewRGBA(image.Rectangle{upLeft, lowRight})

	black := color.RGBA{10, 10, 10, 0xff}
	cyan := color.RGBA{100, 200, 200, 0xff}
	red := color.RGBA{255, 0, 0, 0xff}
	gray := color.RGBA{70, 70, 70, 0xff}
	darkg := color.RGBA{30, 30, 30, 0xff}

	occupied := false
	weightHistory[0] = weightHistory[1]

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			occupied = false
			switch {
			case x < len(runConfidence):
				if y == int(3*runConfidence[x]) {
					if runSuccess[x] == 1 {
						img.Set(x, height-y, cyan)
						occupied = true
					} else {
						img.Set(x, height-y, red)
						occupied = true
					}
				} else if y == int(3*runConfidenceAverage[x]*runSuccessAverage[x]) {
					img.Set(x, height-y, gray)
					occupied = true
				} else {
					for i := 0; i < len(weightHistory[x]); i++ {
						for j := 0; j < len(weightHistory[x][i]); j++ {
							for k := 0; k < len(weightHistory[x][i][j]); k++ {
								if y == int(250+math.Pow(weightHistory[x][i][j][k], 3)) && !occupied {
									img.Set(x, height-y, darkg)
									occupied = true
								}
							}
						}
					}
				}
				if !occupied {
					img.Set(x, height-y, black)
				}
			default:
				img.Set(x, height-y, black)
			}
		}
	}
	f, err := os.Create(filename())
	png.Encode(f, img)
	if err != nil {
		log.Fatal(err)
	}

	if err := png.Encode(f, img); err != nil {
		f.Close()
		log.Fatal(err)
	}

	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}
