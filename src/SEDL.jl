"""
Simultaneous state estimation and dynamics learning.
"""
module SEDL

using AutoHashEquals
using StaticArrays
using UnPack
using ProgressLogging
using Distributions
include("IOIndents.jl")
using .IOIndents
using Statistics
using Measurements
using Optim
import ForwardDiff
using ThreadsX
using Transducers
using Bijectors
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
import Metatheory
using Metatheory: EGraph, EClassId, SaturationParams, AbstractRule 
using Metatheory: addexpr!, saturate!, @theory
using Metatheory.Library: commutative_monoid

Metatheory.@metatheory_init ()

include("utils.jl")
include("DSL.jl")
include("components.jl")
include("prunning.jl")
include("compiler.jl")
include("synthesis.jl")
include("Examples/Car1D_new.jl")
include("Examples/Car1D.jl")

if false
    include("../scripts/car1d_example.jl")
    include("../scripts/test_synthesis.jl")
end

end