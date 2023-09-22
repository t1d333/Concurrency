package matrixops

import "sync"

func ConcurrentMul(lhs, rhs [][]int, size int, chunksCount int, res [][]int) {
	wg := sync.WaitGroup{}

	chunkSize := size / chunksCount
	for i := chunkSize; i <= size; i += chunkSize {
		rightBoundI := i
		if size-i < chunkSize && i != size {
			rightBoundI = size
		}

		for j := chunkSize; j <= size; j += chunkSize {
			rightBoundJ := j
			if size-j < chunkSize && j != size {
				rightBoundJ = size
			}

			lhs := SplitByRows(lhs, i-chunkSize, rightBoundI)
			rhs := SplitByCols(rhs, j-chunkSize, rightBoundJ)
			tmp := SplitByCols(res[i-chunkSize:rightBoundI], j-chunkSize, rightBoundJ)

			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				MulMatrix(lhs, rhs, rightBoundI-i+chunkSize, size, size, rightBoundJ-j+chunkSize, tmp)
			}(i, j)

		}
	}

	wg.Wait()
}

func MulMatrix(lhs, rhs [][]int, lhsRows, lhsCols, rhsRows, rhsCols int, res [][]int) {
	for i := 0; i < lhsRows; i++ {
		for j := 0; j < rhsCols; j++ {
			for k := 0; k < lhsCols; k++ {
				res[i][j] += lhs[i][k] * rhs[k][j]
			}
		}
	}
}

func SplitByCols(matrix [][]int, i, j int) [][]int {
	res := make([][]int, len(matrix))
	for k := range matrix {
		res[k] = matrix[k][i:j]
	}
	return res
}

func SplitByRows(matrix [][]int, i, j int) [][]int {
	return matrix[i:j]
}
