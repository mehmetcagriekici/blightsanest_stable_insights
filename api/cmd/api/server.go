package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
)

// server shape
type server struct {
	// wrap it here to add custom methods (start/kill) and dependencies (config, logger)
	httpServer *http.Server
	// for signalling with external context
	cancel context.CancelFunc
}

// create a new server
func newServer(port int, cancel context.CancelFunc) *server {
	// to mathc request to handlers
	mux := http.NewServeMux()

	srv := &http.Server{
		// host:port
		Addr: fmt.Sprintf(":%d", port),
		// request handler
		Handler: mux,
	}

	s := &server{
		httpServer: srv,
		cancel:     cancel,
	}

	// routes with mux

	return s
}

// start the server
func (s *server) start() error {
	// bind the tcp port and start accepting connections
	ln, err := net.Listen("tcp", s.httpServer.Addr)
	if err != nil {
		return err
	}

	// block handling request on listener intul server is stopped
	if err := s.httpServer.Serve(ln); !errors.Is(err, http.ErrServerClosed) {
		return err
	}

	return nil
}

// kill the server
func (s *server) kill(ctx context.Context) error {
	// Shutdown gracefully stops the server
	return s.httpServer.Shutdown(ctx)
}
