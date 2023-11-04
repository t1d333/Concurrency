package solver

import (
	"math"

	"github.com/emer/empi/mpi"
	"gonum.org/v1/gonum/mat"
)

type SolverWithVecSeparation struct {
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

func NewSolverWithVecSeparation(comm *mpi.Comm, coeffs *mat.Dense, freeCoeffs *mat.VecDense, size int, eps float64) *SolverWithVecSeparation {
	chunkSize := size / comm.Size()
	rank := comm.Rank()
	rowStart := chunkSize * comm.Rank()
	rowEnd := chunkSize * (comm.Rank() + 1)

	return &SolverWithVecSeparation{
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

func (s *SolverWithVecSeparation) FindSolution() *mat.VecDense {
	res := mat.NewVecDense(s.end-s.start, nil)
	metric := math.MaxFloat64

	prev := metric

	buff := make([]float64, s.size)
	for metric > s.eps {
		chunk := s.iterateSolution(res).RawVector().Data

		s.comm.AllGatherF64(buff, chunk)
		tmp := mat.NewVecDense(s.size, buff)

		res.SubVec(res, tmp.SliceVec(s.start, s.end))

		prev = metric
		metric = s.calcMetric(res)

		if prev < metric {
			s.tau *= -1
		}

	}

	return res
}

func (s *SolverWithVecSeparation) iterateSolution(chunk *mat.VecDense) *mat.VecDense {
	buff := make([]float64, s.size)
	s.comm.AllGatherF64(buff, chunk.RawVector().Data)

	xvec := mat.NewVecDense(s.size, buff)

	tmp := mat.NewVecDense(s.end-s.start, nil)

	tmp.MulVec(s.coeffs, xvec)
	tmp.SubVec(tmp, s.freeCoeffs)
	tmp.ScaleVec(s.tau, tmp)

	return tmp
}

func (s *SolverWithVecSeparation) calcMetric(chunk *mat.VecDense) float64 {
	buff := make([]float64, s.size)

	s.comm.AllGatherF64(buff, chunk.RawVector().Data)

	xvec := mat.NewVecDense(s.size, buff)

	tmp := mat.NewVecDense(s.end-s.start, nil)

	tmp.MulVec(s.coeffs, xvec)
	tmp.SubVec(tmp, s.freeCoeffs)

	frac := make([]float64, 2)
	num := float64(0)
	den := float64(0)

	for _, v := range tmp.RawVector().Data {
		num += v * v
	}

	for _, v := range s.freeCoeffs.RawVector().Data {
		den += v * v
	}

	s.comm.AllReduceF64(mpi.OpSum, frac, []float64{num, den})

	return math.Sqrt(frac[0]) / math.Sqrt(frac[1])
}
