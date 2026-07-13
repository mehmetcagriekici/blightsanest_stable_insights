package config

import (
	"fmt"
	"os"
	"strconv"
)

const defaultPort      = 8080
const defaultCustomBufferSize = 8192
const defaultEnv = "development"

// Config holds runtime configuration for the API service, loaded once at
// startup from environment variables.
type Config struct {
	Port             int
  CustomBufferSize int
  Env              string
}

// Load builds a Config from environment variables, falling back to
// defaults for anything unset.
func Load() (*Config, error) {
	port := defaultPort
  customBufferSize := defaultCustomBufferSize
  env := defaultEnv

	if v := os.Getenv("PORT"); v != "" {
		p, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("invalid PORT %q: %w", v, err)
		}
		port = p
	}

  if v := os.Getenv("CUSTOM_BUFFER_SIZE"); v != "" {
		cbs, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("invalid CUSTOM BUFFER SIZE %q: %w", v, err)
		}
		customBufferSize = cbs
  }

  if v := os.Getenv("ENV"); v != "" {
    env = v
  }

  return &Config{Port: port, CustomBufferSize: customBufferSize, Env: env}, nil
}
