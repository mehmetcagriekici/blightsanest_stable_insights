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
    return  nil, nil, err
  }

  // create a new buffered writer with custom buffer customBufferSize
  w := bufio.NewWriterSize(logFile, customBufferSize)

  // create log handller with default options for information level and debug level
  infoHandler := slog.NewJSONHandler(w, &slog.HandlerOptions{
    Level: slog.LevelInfo,
  })
  debugHandler := slog.NewJSONHandler(w, &slog.HandlerOptions{
    Level: slog.LevelDebug,
  })

  // main initLogger 
  logger := slog.New(slog.NewMultiHandler(
    debugHandler,
    infoHandler,
    ))

  // static fields every log has
  hostname, _ := os.Hostname()
  logger = logger.With(
    slog.String("env", env),
    slog.String("hostname", hostname),
    )

  // return the logger and the writer
  return logger, w, nil
}
