#!/usr/bin/env python3

from QA4QUBO.tests.test import main
import sys

if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except:
        n = 8
    main(n)