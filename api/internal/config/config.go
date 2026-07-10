package config

import (
	"fmt"
	"os"
	"strconv"
)

const defaultPort = 8080

// Config holds runtime configuration for the API service, loaded once at
// startup from environment variables.
type Config struct {
	Port int
}

// Load builds a Config from environment variables, falling back to
// defaults for anything unset.
func Load() (*Config, error) {
	port := defaultPort

	if v := os.Getenv("PORT"); v != "" {
		p, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("invalid PORT %q: %w", v, err)
		}
		port = p
	}

	return &Config{Port: port}, nil
}
