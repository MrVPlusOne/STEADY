using ProgressMeter: Progress, next!, @showprogress, progress_pmap
using Distributed

include("synthesis_utils.jl")

include("deterministic_synthesis.jl")
include("posterior_sampling.jl")
include("probabilistic_synthesis.jl")

include("sparse_regression.jl")
include("em_synthesis.jl")
include("sindy_regression.jl")
include("symbolic_regression.jl")
include("neural_regression.jl")

include("batched_em.jl")
