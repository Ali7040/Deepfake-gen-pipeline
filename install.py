#!/usr/bin/env python3
"""
DeepTrace Installer
Handles dependency installation and environment setup
"""

import os
import sys

os.environ['SYSTEM_VERSION_COMPAT'] = '0'

from deeptrace import installer

if __name__ == '__main__':
	try:
		installer.cli()
	except KeyboardInterrupt:
		print("\nInstallation cancelled by user")
		sys.exit(0)
	except Exception as e:
		print(f"Installation error: {e}")
		sys.exit(1)
