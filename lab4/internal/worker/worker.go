package worker

import (
	"context"
	"math/rand"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

type WorkerState int

const (
	Work WorkerState = iota
	LockedFirstRes
	LockedSecondRes
	FreeResources
	Workless
)

func (s WorkerState) String() string {
	switch s {
	case Work:
		return "working"
	case LockedFirstRes:
		return "locked first resource"
	case LockedSecondRes:
		return "locked second resource"
	case FreeResources:
		return "freed resources"
	case Workless:
		return "not working"
	default:
		return "unknown state"
	}
}

type Worker struct {
	num       int
	ctx       context.Context
	firstRes  *sync.Mutex
	secondRes *sync.Mutex
	state     WorkerState
	logger    *zerolog.Logger
}

func NewWorker(workerNum int, logger *zerolog.Logger, ctx context.Context, firstRes, secondRes *sync.Mutex) Worker {
	return Worker{
		num:       workerNum,
		ctx:       ctx,
		firstRes:  firstRes,
		secondRes: secondRes,
		state:     Workless,
		logger:    logger,
	}
}

func (w *Worker) logState() {
	w.logger.Info().Int("worker_num", w.num).Str("worker_state", w.state.String()).Send()
	<-time.After(1 * time.Millisecond)
}

func (w *Worker) do(state WorkerState) {
	t := rand.Intn(10) + 1
	w.state = state
	ch := time.After(time.Duration(t) * time.Millisecond)

	for {
		select {
		case <-ch:
			return
		case <-w.ctx.Done():
			return
		default:
			w.logState()
		}
	}
}

func (w *Worker) freeResources() {
	w.secondRes.Unlock()
	w.firstRes.Unlock()
	w.state = FreeResources
	w.logState()
}

func (w *Worker) Run() {
	for {
		select {
		case <-w.ctx.Done():
			return
		default:
			for !w.firstRes.TryLock() {
				w.do(Workless)
			}

			w.state = LockedFirstRes
			w.logState()

			for !w.secondRes.TryLock() {
				w.do(Workless)
			}

			w.state = LockedSecondRes
			w.logState()

			w.do(Work)
			w.freeResources()

		}
	}
}
