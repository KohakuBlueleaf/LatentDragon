"""Taming GAN active with accumulation, which caches every micro-batch."""

from kohakuengine import use_config

use_config("bench_taming_active.py")

NAME = "bench_taming_acc"
GRAD_ACC = 4
