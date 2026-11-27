#!/usr/bin/env bash
# Unfreeze the overnight jobs paused at 05:50 (SIGSTOP -> SIGCONT).
kill -CONT 65146 65709 68611 73029 && echo resumed
