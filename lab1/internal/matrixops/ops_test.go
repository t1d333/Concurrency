package matrixops_test

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/t1d333/concurrency/lab1/internal/matrixops"
)

func TestConncurentMul(t *testing.T) {
	size := 10
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

	connResult := make([][]int, size)
	defResult := make([][]int, size)

	for i := 0; i < size; i++ {
		connResult[i] = make([]int, size)
		defResult[i] = make([]int, size)
	}

	t.Run("2 chunks", func(t *testing.T) {
		matrixops.ConcurrentMul(matrixLhs, matrixRhs, size, 2, connResult)
		matrixops.MulMatrix(matrixLhs, matrixRhs, size, size, size, size, defResult)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				assert.Equal(t, defResult[i][j], connResult[i][j])
			}
		}
	})

	t.Run("3 chunks", func(t *testing.T) {
		matrixops.ConcurrentMul(matrixLhs, matrixRhs, size, 3, connResult)
		matrixops.MulMatrix(matrixLhs, matrixRhs, size, size, size, size, defResult)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				assert.Equal(t, defResult[i][j], connResult[i][j])
			}
		}
	})
}

func BenchmarkConcurrentMul(b *testing.B) {
	size := 1000
	lhs := make([][]int, size)
	rhs := make([][]int, size)
	res := make([][]int, size)
	for i := 0; i < size; i++ {
		res[i] = make([]int, size)
		lhs[i] = make([]int, size)
		rhs[i] = make([]int, size)
		for j := 0; j < size; j++ {
			lhs[i][j] = rand.Intn(size)
			rhs[i][j] = rand.Intn(size)
		}
	}

	b.Run("default mul", func(b *testing.B) {
		b.ResetTimer()
		matrixops.ConcurrentMul(lhs, rhs, size, 1, res)
	})

	b.Run("two chunks", func(b *testing.B) {
		b.ResetTimer()
		matrixops.ConcurrentMul(lhs, rhs, size, 2, res)
	})

	b.Run("four chunks", func(b *testing.B) {
		b.ResetTimer()
		matrixops.ConcurrentMul(lhs, rhs, size, 4, res)
	})

	b.Run("six chunks", func(b *testing.B) {
		b.ResetTimer()
		matrixops.ConcurrentMul(lhs, rhs, size, 6, res)
	})

	b.Run("eight chunks", func(b *testing.B) {
		b.ResetTimer()
		matrixops.ConcurrentMul(lhs, rhs, size, 8, res)
	})

	
	b.Run("ten chunks", func(b *testing.B) {
		b.ResetTimer()
		matrixops.ConcurrentMul(lhs, rhs, size, 10, res)
	})
}
