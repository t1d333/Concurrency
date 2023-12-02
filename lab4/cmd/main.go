package main

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/t1d333/Concurrency/lab4/internal/worker"
)

func max(a, b int) int {
	if a > b {
		return a
	}

	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	wg := &sync.WaitGroup{}
	defer cancel()
	arr := make([]sync.Mutex, 5)
	workersCount := 5

	workers := make([]worker.Worker, workersCount)

	for i := 0; i < 5; i++ {
		i := i
		file, _ := os.Create(fmt.Sprintf("./logs/log_worker%d", i))

		logger := zerolog.New(file).With().Timestamp().Logger()

		wg.Add(1)
		firstRes := &arr[min(i, (i+1)%workersCount)]
		secondRes := &arr[max(i, (i+1)%workersCount)]
		workers[i] = worker.NewWorker(i, &logger, ctx, firstRes, secondRes)
		go func() {
			workers[i].Run()
			wg.Done()
		}()
	}

	wg.Wait()
}
