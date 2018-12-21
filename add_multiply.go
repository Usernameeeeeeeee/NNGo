package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
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
		fmt.Print(d, " / ", len(dataset)-1, ": " /*dataset[d][0], "		-> ",*/, predictions[ind], "	(correct!)" /* nodes[len(nodes)-1], "	(", dataset[d][1], ") 	*/, "	Confidence: ")

		endSuccess = append(endSuccess, 1)
	} else {

		fmt.Print(d, " / ", len(dataset)-1, ": " /* dataset[d][0], "		-> ",*/, predictions[ind], "	(wrong!)" /*	 nodes[len(nodes)-1], "	(", dataset[d][1], ") 	*/, "	Confidence: ")
		endSuccess = append(endSuccess, 0)
	}
	if sum > 100 {
		fmt.Print(" ", 100, "%	", int(average(lastCouple)))
		if len(lastCouple) < lastCoupleLength {
			lastCouple = append(lastCouple, 100)
			lastCoupleRight = append(lastCoupleRight, ind == indc)
		} else {
			for i := 1; i < len(lastCouple)-1; i++ {
				lastCouple[i] = lastCouple[i+1]
				lastCoupleRight[i] = lastCoupleRight[i+1]
			}
			lastCouple[len(lastCouple)-1] = 100
			lastCoupleRight[len(lastCoupleRight)-1] = (ind == indc)
		}

		endConfidence = append(endConfidence, 1)
	} else {
		fmt.Print(" ", int((nodes[len(nodes)-1][ind]-avgNodeDifference)*100), "%	", int(average(lastCouple)))
		if len(lastCouple) < lastCoupleLength {
			lastCouple = append(lastCouple, (nodes[len(nodes)-1][ind]-avgNodeDifference)*100)
			lastCoupleRight = append(lastCoupleRight, ind == indc)
		} else {
			for i := 1; i < len(lastCouple)-1; i++ {
				lastCouple[i] = lastCouple[i+1]
				lastCoupleRight[i] = lastCoupleRight[i+1]
			}
			lastCouple[len(lastCouple)-1] = (nodes[len(nodes)-1][ind] - avgNodeDifference) * 100
			lastCoupleRight[len(lastCoupleRight)-1] = (ind == indc)
		}
		endConfidence = append(endConfidence, (nodes[len(nodes)-1][ind] - avgNodeDifference))
	}
	fmt.Print("	", dataset[d][0], "\n")

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
	newWeights := weights
	for ri := 0; ri < len(layers)-1; ri++ {
		for h := 0; h < layers[ri+1]; h++ {
			for n := 0; n < len(errors[ri]); n++ {
				newWeights[ri][h][n] += nodes[ri][n] * errors[ri+1][h] * lr
			}
		}
	}
}

func average(arr []float64) float64 {
	var arrSum float64 = 0
	for p := 0; p < len(arr); p++ {
		arrSum += arr[p]
	}
	return arrSum / float64(len(arr))
}

var layers = []int{3, 5, 2} //------------------------------------------------------ layout (!)
var weights = [][][]float64{{{}}}
var nodes = [][]float64{{}}
var errors = [][]float64{{}}
var dataset = [][][]float64{{{}}}
var predictions = [4]string{} //------------------------------------------------------------ how many logical outputs

var endConfidence = []float64{}
var endSuccess = []float64{}

var lastCouple = []float64{}

var lastCoupleLength = 50000
var lastCoupleRight = []bool{}

var lastCoupleThreshhold float64 = 97

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
	createData(100000) //------------------------------------------------------------------- datasets (learning)

	fmt.Println("learning...")

	f := 0
	for d := 0; d < len(dataset); d++ {
		forward(d)
		status(d)
		backward(lr)
		if average(lastCouple) > lastCoupleThreshhold && d > len(dataset)/4 {
			fmt.Println("\nlearning session (", d, ") successful\n")
			break
		} else if d == len(dataset)-1 {
			fmt.Println("\nlearning session (", len(dataset), ") successful\n")
		} else if average(lastCouple) < 20 && d > len(dataset)/4 {
			randomize()
			d = 0
			if f != 3 {
				f++
			} else {
				break
			}
		}
	}

}

func main() {

	randomize()

	learn(0.08)

	fmt.Println("making predictions...\n")
	time.Sleep(2 * time.Second)

	var amt int = 100

	createData(amt) //------------------------------------------------------------------------ dataset (show)
	endConfidence = []float64{}
	endSuccess = []float64{}
	for p := 0; p < amt; p++ {
		forward(p)
		status(p)
	}
	fmt.Println("\nOverall confidence: 	", average(endConfidence)*100, "%	 Success rate:	", average(endSuccess)*100, "%		Score:	", int(average(endConfidence)*(average(endSuccess)*100)), "\n", endConfidence)

	width := lastCoupleLength
	height := 500

	upLeft := image.Point{0, 0}
	lowRight := image.Point{width, height}

	img := image.NewRGBA(image.Rectangle{upLeft, lowRight})

	black := color.RGBA{10, 10, 10, 0xff}
	cyan := color.RGBA{100, 200, 200, 0xff}
	red := color.RGBA{255, 0, 0, 0xff}

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			switch {
			case x < len(lastCouple):
				if y == int(3*lastCouple[x]) {
					if lastCoupleRight[x] {
						img.Set(x, height-y, cyan)
					} else {
						img.Set(x, height-y, red)
					}
				} else {
					img.Set(x, height-y, black)
				}
			default:
				img.Set(x, y, black)
			}
		}
	}
	f, _ := os.Create("result_0_3.png")
	png.Encode(f, img)

}
