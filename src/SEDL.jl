"""
Simultaneous state estimation and dynamics learning.
"""
module SEDL

using AutoHashEquals
using StaticArrays
using UnPack
using Distributions
include("IOIndents.jl")
using .IOIndents
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

include("utils.jl")
include("numerical_integration.jl")
include("DSL.jl")
include("components.jl")
include("compiler.jl")
include("synthesis.jl")
include("Examples/Car1D.jl")

if false
    include("../scripts/car1d_example.jl")
    include("../scripts/test_synthesis.jl")
end

end