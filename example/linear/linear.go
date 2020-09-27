package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"github.com/maxrafiandy/go-machine-learning/method"
)

const stdDev = 1e2

func testSet(l *method.LogisticRegression) {
	testCSV, err := os.Open("testset.csv")

	if err != nil {
		log.Fatal(err)
	}

	defer testCSV.Close()

	b := bufio.NewScanner(testCSV)

	var testData [][]float64
	for b.Scan() {
		var X [4]float64

		X[0] = 1
		_, err := fmt.Sscanf(b.Text(), "%f,%f,%f", &X[1], &X[2], &X[3])

		if err != nil {
			log.Fatal(err)
		}

		testData = append(testData, []float64{X[0], X[1] / stdDev, X[2] / stdDev, X[3] / stdDev})
	}

	for i, X := range testData {
		var stdDevX []float64
		for _, x := range X {
			stdDevX = append(stdDevX, x*stdDev)
		}
		fmt.Printf("Predict #%d on %v: %v\n", i+1, stdDevX[1:], l.Predict(X))
	}
}

func trainSet(l *method.LogisticRegression) {
	trainCSV, err := os.Open("trainset.csv")

	if err != nil {
		log.Fatal(err)
	}

	defer trainCSV.Close()

	b := bufio.NewScanner(trainCSV)

	for b.Scan() {
		var X [4]float64
		var y float64

		X[0] = 1
		_, err := fmt.Sscanf(b.Text(), "%f,%f,%f,%f", &X[1], &X[2], &X[3], &y)

		if err != nil {
			log.Fatal(err)
		}

		l.Features = append(l.Features, []float64{X[0], X[1] / stdDev, X[2] / stdDev, X[3] / stdDev})
		l.Output = append(l.Output, y)
	}
}

func main() {
	l := method.NewLogisticRegression()

	l.Theta = []float64{-0.1, -0.2, 0.1, 0.2}

	l.TrueDegree = 0.7

	trainSet(l)

	l.Minimize(method.LinearDefaultSetting())

	fmt.Printf("Optimal theta : %.3f\n", l.Theta)

	testSet(l)
}
