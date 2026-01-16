#!/usr/bin/env python3
# Thin wrapper that forwards to the real CLI entrypoint in the package.

from servo.cli import main

if __name__ == "__main__":
    main()
