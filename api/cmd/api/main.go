package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
  // get a context depedning on the signal
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)

	// defaults cli flags.
	httpPort := flag.Int("port", 8899, "port to listen on")
	dataDir := flag.String("data", "./data", "directiory to store data")
	flag.Parse()

  // run the server
	status := run(ctx, cancel, *httpPort, *dataDir)

  // safety cancel
	cancel()
	os.Exit(status)
}

func run(ctx context.Context, cancel context.CancelFunc, httpPort int, dataDir string) int {
	s := newServer(httpPort, cancel)

  // a channel for errors from the goroutine
	errCh := make(chan error, 1)

  // start the server in a goroutine 
	go func() {
		errCh <- s.start()
	}()

  // block until a signal arrives
	<-ctx.Done()

  // fresh shutdown context
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := s.kill(shutdownCtx); err != nil {
		fmt.Fprintf(os.Stderr, "failed to shutdown server: %v\n", err)
		return 1
	}

  // catch server errirs without data race
	var serverErr error
	select {
	case serverErr = <-errCh:
	case <-time.After(5 * time.Second):
		fmt.Fprintf(os.Stderr, "server goroutine did not exit cleanly\n")
	}

	if serverErr != nil {
		fmt.Fprintf(os.Stderr, "server error: %v\n", serverErr)
		return 1
	}

	return 0
}
