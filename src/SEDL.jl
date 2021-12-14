"""
Simultaneous state estimation and dynamics learning.
"""
module SEDL

using Base: @kwdef
using BinaryTraits
using BinaryTraits.Prefix: Is, IsNot
using AutoHashEquals
using MacroTools: @forward
using Formatting: format
using StaticArrays
using DataStructures: OrderedDict
export OrderedDict
import Random
using Random: AbstractRNG
using Distributions
include("IOIndents.jl")
using .IOIndents
using Statistics
using Measurements
using Optim
import ForwardDiff
import ThreadPools
using Transducers
using Bijectors
using SplitApplyCombine: group as to_groups
using TimerOutputs
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
import Metatheory
using Metatheory: EGraph, EClassId, SaturationParams, AbstractRule 
using Metatheory: addexpr!, saturate!, @theory
using Metatheory.Library: commutative_monoid

Metatheory.@metatheory_init()


include("utils.jl")
include("distributions_utils.jl")
include("samplers/samplers.jl")
include("control_utils.jl")
include("DSL.jl")
include("components.jl")
include("enumeration/enumeration.jl")
include("compiler.jl")
include("synthesis/synthesis.jl")
include("Examples/Examples.jl")
include("tests.jl")

if false
    include("../scripts/hovercraft_example.jl")
    include("../scripts/car2d_example.jl")
    include("../scripts/iterate_example.jl")
    include("../scripts/test_synthesis.jl")
    # include("../scripts/rocket_example.jl")

    include("../scripts/archived/car1d_example.jl")
    include("../scripts/scratch.jl")
end

end # end module