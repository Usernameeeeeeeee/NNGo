package main

import (
	"fmt"
	"math"
	"math/rand"
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

func CALCerrors(d int) {
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

func status(d int, show bool) {
	var sum float64 = 0

	for e := 0; e < len(nodes[len(nodes)-1]); e++ {
		for i := 0; i < len(nodes[len(nodes)-1]); i++ {
			sum += math.Abs(nodes[len(nodes)-1][e]-nodes[len(nodes)-1][i]) / float64(len(nodes[len(nodes)-1])) * 100
		}
	}

	var max float64 = 0
	var ind int = 0
	var wrong_sum float64 = 0
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
			wrong_sum += nodes[len(nodes)-1][p]
		}
	}
	wrong_sum /= float64(len(nodes[len(nodes)-1]) - 1)

	if show {
		fmt.Print(d, " / ", len(dataset)-1, ": ", dataset[d][0], " -> ", nodes[len(nodes)-1], " ", dataset[d][1], "\n")
	} else {
		if ind == indc {
			fmt.Print(d, " / ", len(dataset)-1, ": ", dataset[d][0], " -> ", predictions[ind], " (correct!) ", nodes[len(nodes)-1], " (", dataset[d][1], ") Confidence: ")

		} else {

			fmt.Print(d, " / ", len(dataset)-1, ": ", dataset[d][0], " -> ", predictions[ind], " (wrong!) ", nodes[len(nodes)-1], " (", dataset[d][1], ") Confidence: ")

		}
		if sum > 100 {
			fmt.Println(" ", 100, "%\n")
		} else {
			fmt.Println(" ", (nodes[len(nodes)-1][ind]-wrong_sum)*100, "%\n")
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

	CALCerrors(d)

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

var layers = []int{3, 8, 2} //------------------------------------------------------ layout (!)
var weights = [][][]float64{{{}}}
var nodes = [][]float64{{}}
var errors = [][]float64{{}}
var dataset = [][][]float64{{{}}}
var predictions = [4]string{} //------------------------------------------------------------ how many logical outputs

func strToArray(str string) []float64 {

	wordToArray := []float64{}

	for i := 0; i < len(str)-1; i++ {
		wordToArray = append(wordToArray, float64([]rune(str)[i]))
	}

	return wordToArray
}

func createData(n int) {
	fmt.Println("creating ramdom datasets...")
	dataset = [][][]float64{{{}}}

	for l := 0; l < n; l++ {

		rand.Seed(time.Now().UTC().UnixNano() + int64(l))
		var o = []float64{}

		var a float64 = float64(rand.Intn(20))
		var b float64 = float64(rand.Intn(20))
		var c float64 = 0

		if l%2 == 0 {
			c = a + b
			o = []float64{1, 0}
		} else {
			c = a * b
			o = []float64{0, 1}
		}
		//------------------------------------------------------------------------------------ logical outputs
		predictions[0] = "added"
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
	createData(5000) //------------------------------------------------------------------- datasets (learning)

	fmt.Println("learning...")

	for d := 0; d < len(dataset); d++ {
		forward(d)
		status(d, true)
		backward(lr)
	}

	fmt.Println("\nlearning session (", len(dataset), ") successful\n")

}

func main() {

	randomize()

	fmt.Println(weights)

	learn(0.05)

	fmt.Println("making predictions...\n")
	time.Sleep(2 * time.Second)

	var amt int = 20

	createData(amt) //------------------------------------------------------------------------ dataset (show)

	for p := 0; p < amt; p++ {
		forward(p)
		status(p, false)
	}
}
