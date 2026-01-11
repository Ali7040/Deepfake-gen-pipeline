#!/usr/bin/env python3
"""
DeepTrace - Advanced Face Manipulation Framework
Optimized for performance and accuracy
"""

import os
import sys

# Performance optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from deeptrace import core

if __name__ == '__main__':
	try:
		core.cli()
	except KeyboardInterrupt:
		print("\nDeepTrace terminated by user")
		sys.exit(0)
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)
