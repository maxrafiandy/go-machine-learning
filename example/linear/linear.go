package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"github.com/maxrafiandy/go-machine-learning/method"
)

func testSet(l *method.LogisticRegression) {
	test, _ := os.Open("testset.csv")
	defer test.Close()

	b := bufio.NewScanner(test)

	var testdata [][]float64
	for b.Scan() {
		var X [4]float64

		X[0] = 1
		_, err := fmt.Sscanf(b.Text(), "%f,%f,%f", &X[1], &X[2], &X[3])

		if err != nil {
			log.Fatal(err)
		}

		testdata = append(testdata, []float64{X[0], X[1], X[2], X[3]})
	}

	for _, X := range testdata {
		fmt.Printf("Predict on %v: %v\n", X[1:], l.Predict(X))
	}
}

func trainSet(l *method.LogisticRegression) {
	train, _ := os.Open("trainset.csv")
	defer train.Close()

	b := bufio.NewScanner(train)

	for b.Scan() {
		var X [4]float64
		var y float64

		X[0] = 1
		_, err := fmt.Sscanf(b.Text(), "%f,%f,%f,%f", &X[1], &X[2], &X[3], &y)

		if err != nil {
			log.Fatal(err)
		}

		l.Features = append(l.Features, []float64{X[0], X[1], X[2], X[3]})
		l.Output = append(l.Output, y)
	}
}

func main() {
	l := method.NewLogisticRegression()
	l.Theta = []float64{1, 1, 1, 1}
	l.TrueDegree = 0.7

	trainSet(l)

	l.Minimize(method.LinearDefaultSetting())
	fmt.Printf("Optimal theta : %v\n", l.Theta)

	testSet(l)
}
