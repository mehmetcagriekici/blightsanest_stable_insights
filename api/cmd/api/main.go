package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mehmetcagriekici/blightsanest_stable_insights/api/internal/config"
)

func main() {
	// get a context depedning on the signal
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		cancel()
		os.Exit(1)
	}

	// run the server
	status := run(ctx, cancel, cfg.Port)

	// safety cancel
	cancel()
	os.Exit(status)
}

func run(ctx context.Context, cancel context.CancelFunc, httpPort int) int {
	s := newServer(httpPort, cancel)

	// a channel for errors from the goroutine
	errCh := make(chan error, 1)

	// start the server in a goroutine
	go func() {
		errCh <- s.start()
	}()

	// block until either a signal arrives or the server exits on its own
	// (e.g. it failed to bind the port) - waiting on ctx.Done() alone would
	// hang forever in the latter case
	select {
	case <-ctx.Done():
		// fresh shutdown context
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()

		if err := s.kill(shutdownCtx); err != nil {
			fmt.Fprintf(os.Stderr, "failed to shutdown server: %v\n", err)
			return 1
		}

		// catch server errirs without data race
		select {
		case serverErr := <-errCh:
			if serverErr != nil {
				fmt.Fprintf(os.Stderr, "server error: %v\n", serverErr)
				return 1
			}
		case <-time.After(5 * time.Second):
			fmt.Fprintf(os.Stderr, "server goroutine did not exit cleanly\n")
		}

		return 0
	case serverErr := <-errCh:
		// the server stopped on its own before any shutdown signal, most
		// likely because it failed to start - there is nothing to shut down
		if serverErr != nil {
			fmt.Fprintf(os.Stderr, "server error: %v\n", serverErr)
			return 1
		}
		return 0
	}
}
