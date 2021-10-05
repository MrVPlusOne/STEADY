## reload SEDL module in this process
include("../src/SEDL.jl")
## setup and reload SEDL on all worker processes
using Distributed
n_workers = 8
addprocs(n_workers+1 - nprocs(), exeflags="--project")
@assert length(workers()) == n_workers

@everywhere workers() include("../src/SEDL.jl")
## remove worker processes when done
rmprocs(workers()...)