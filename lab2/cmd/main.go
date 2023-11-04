package main

import (
	"flag"
	"math/rand"

	"github.com/emer/empi/mpi"
	"github.com/t1d333/Concurrency/lab2/internal/solver"
	"gonum.org/v1/gonum/mat"
)

var (
	size = 30000
	eps  = 1e-4
)

func init() {
	flag.IntVar(&size, "n", size, "size of matrix")
}

func main() {
	defer mpi.Finalize()
	mpi.Init()
	comm, _ := mpi.NewComm(nil)

	flag.Parse()

	chunkSize := size / comm.Size()
	start := comm.Rank() * chunkSize
	end := (comm.Rank() + 1) * chunkSize

	buff := make([]float64, size)

	if comm.Rank() == 0 {
		buff = genUvec()
	}

	comm.BcastF64(0, buff)

	uvec := mat.NewVecDense(size, buff)
	coeffs := getCoeffs(start, end)

	freeCoeffs := getFreeCoeffs(uvec, coeffs)

	solver := solver.NewSolverWithVecSeparation(comm, coeffs, freeCoeffs, size, eps)

	s := solver.FindSolution()
	res := make([]float64, size)

	comm.AllGatherF64(res, s.RawVector().Data)
}

func getCoeffs(start, end int) *mat.Dense {
	rows := end - start
	res := mat.NewDense(rows, size, make([]float64, rows*size))

	for i := start; i < end; i++ {
		for j := 0; j < size; j++ {
			if i == j {
				res.Set(i-start, j, 2.0)
			} else {
				res.Set(i-start, j, 1.0)
			}
		}
	}

	return res
}

func genUvec() []float64 {
	res := make([]float64, size)

	for i := 0; i < size; i++ {
		res[i] = float64(rand.Intn(size))
	}

	return res
}

func getFreeCoeffs(u *mat.VecDense, coeffs *mat.Dense) *mat.VecDense {
	res := mat.NewVecDense(coeffs.RawMatrix().Rows, nil)

	res.MulVec(coeffs, u)

	return res
}
