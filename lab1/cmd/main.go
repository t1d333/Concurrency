package main

import (
	"flag"
	"math/rand"

	"github.com/t1d333/concurrency/lab1/internal/matrixops"
)

var (
	size        = 0
	chunksCount = 0
)

func init() {
	flag.IntVar(&size, "n", 0, "size of matrix")
	flag.IntVar(&chunksCount, "k", 0, "count of chunks")
}

func main() {
	flag.Parse()

	matrixLhs := make([][]int, size)
	matrixRhs := make([][]int, size)

	for i := 0; i < size; i++ {
		matrixLhs[i] = make([]int, size)
		matrixRhs[i] = make([]int, size)
		for j := 0; j < size; j++ {
			matrixLhs[i][j] = rand.Intn(size)
			matrixRhs[i][j] = rand.Intn(size)
		}
	}

	result := make([][]int, size)
	for i := 0; i < size; i++ {
		result[i] = make([]int, size)
	}

	matrixops.ConcurrentMul(matrixLhs, matrixRhs, size, chunksCount, result)
}
