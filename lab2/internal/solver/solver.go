package solver

import (
	"math"

	"github.com/emer/empi/mpi"
	"gonum.org/v1/gonum/mat"
)

type Solver struct {
	rank       int
	start      int
	end        int
	size       int
	tau        float64
	eps        float64
	comm       *mpi.Comm
	coeffs     *mat.Dense
	freeCoeffs *mat.VecDense
}

func NewSolver(comm *mpi.Comm, coeffs *mat.Dense, freeCoeffs *mat.VecDense, size int, eps float64) *Solver {
	chunkSize := size / comm.Size()
	rank := comm.Rank()
	rowStart := chunkSize * comm.Rank()
	rowEnd := chunkSize * (comm.Rank() + 1)

	return &Solver{
		rank:       rank,
		start:      rowStart,
		end:        rowEnd,
		size:       size,
		eps:        eps,
		comm:       comm,
		tau:        float64(1) / float64(size),
		coeffs:     coeffs,
		freeCoeffs: freeCoeffs,
	}
}

func (s *Solver) FindSolution() *mat.VecDense {
	res := mat.NewVecDense(s.size, nil)
	metric := math.MaxFloat64

	prev := metric

	buff := make([]float64, s.size)
	for metric > s.eps {
		s.comm.AllGatherF64(buff, s.iterateSolution(res).RawVector().Data)
		tmp := mat.NewVecDense(s.size, buff)

		res.SubVec(res, tmp)

		prev = metric
		metric = s.calcMetric(res)

		if prev < metric {
			s.tau *= -1
		}

	}

	return res
}

func (s *Solver) iterateSolution(chunk *mat.VecDense) *mat.VecDense {
	tmp := mat.NewVecDense(s.end-s.start, nil)

	tmp.MulVec(s.coeffs, chunk)
	tmp.SubVec(tmp, s.freeCoeffs.SliceVec(s.start, s.end))
	tmp.ScaleVec(s.tau, tmp)

	return tmp
}

func (s *Solver) calcMetric(xVec *mat.VecDense) float64 {
	tmp := mat.NewVecDense(s.end-s.start, nil)
	tmp.MulVec(s.coeffs, xVec)
	tmp.SubVec(tmp, s.freeCoeffs.SliceVec(s.start, s.end))

	numBuff := make([]float64, 0)
	num := float64(0)
	den := float64(0)

	for _, v := range tmp.RawVector().Data {
		num += v * v
	}

	for _, v := range s.freeCoeffs.RawVector().Data {
		den += v * v
	}

	s.comm.AllReduceF64(mpi.OpSum, numBuff, []float64{num})

	return math.Sqrt(numBuff[0]) / math.Sqrt(den)
}
