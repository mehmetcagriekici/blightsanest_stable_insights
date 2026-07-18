package main

import (
	"bufio"
	"log/slog"
	"os"
)

// initiate logger with a log file
func initLogger(filename string, customBufferSize int, env string) (*slog.Logger, *bufio.Writer, error) {
	// open the log file
	logFile, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return nil, nil, err
	}

	// create a new buffered writer with custom buffer customBufferSize
	w := bufio.NewWriterSize(logFile, customBufferSize)

	// debug logs are only useful in development; production stays at info
	// and above to avoid the noise/volume of debug-level logging
	level := slog.LevelInfo
	if env == "development" {
		level = slog.LevelDebug
	}

	// a single handler writing to w - two handlers with overlapping levels
	// (e.g. debug and info) would each write every info-and-above record,
	// duplicating log lines
	handler := slog.NewJSONHandler(w, &slog.HandlerOptions{
		Level: level,
	})

	logger := slog.New(handler)

	// static fields every log has
	hostname, _ := os.Hostname()
	logger = logger.With(
		slog.String("env", env),
		slog.String("hostname", hostname),
	)

	// return the logger and the writer
	return logger, w, nil
}
