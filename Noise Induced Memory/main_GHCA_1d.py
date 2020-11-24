import GHCA_1d 
import line_profiler

import line_profiler 
import atexit

import sys, trace

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

lattice = GHCA_1d.GHCA()
tracer = trace.Trace( trace=0, count=0, timing=True,  countcallers=True)
tracer.run('lattice.run()') # init your class and track the function calls
tracer.results().write_results(show_missing=False) # print result to the screen







