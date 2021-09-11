"""
Simultaneous state estimation and dynamics learning.
"""
module SEDL

using Base: @kwdef
using AutoHashEquals
using MacroTools: @forward
using Formatting: format
using StaticArrays
using UnPack
using ProgressLogging
import Random
using Random: AbstractRNG
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
using SplitApplyCombine: group as groupby
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
import Metatheory
using Metatheory: EGraph, EClassId, SaturationParams, AbstractRule 
using Metatheory: addexpr!, saturate!, @theory
using Metatheory.Library: commutative_monoid

Metatheory.@metatheory_init()

include("utils.jl")
include("control_utils.jl")
include("DSL.jl")
include("components.jl")
include("enumeration.jl")
include("compiler.jl")
include("synthesis.jl")
include("Examples/example_utils.jl")
include("Examples/Car1D.jl")
include("Examples/Rocket2D.jl")
include("tests.jl")

if false
    include("../scripts/car1d_example.jl")
    include("../scripts/rocket_example.jl")
    include("../scripts/test_synthesis.jl")
end

end