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
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
import Metatheory
using Metatheory: EGraph, EClassId, SaturationParams, AbstractRule 
using Metatheory: addexpr!, saturate!, @theory
using Metatheory.Library: commutative_monoid

Metatheory.@metatheory_init()

# temporary fix for the julia 1.7-rc2 bug
function Base.collect(itr::Base.Generator)
    isz = Base.IteratorSize(itr.iter)
    et = Base.@default_eltype(itr)
    if isa(isz, Base.SizeUnknown)
        return Base.grow_to!(Vector{et}(), itr)
    else
        shp = Base._similar_shape(itr, isz)
        y = iterate(itr)
        if y === nothing
            return Base._array_for(et, isz, shp)
        end
        v1, st = y
        dest = Base._array_for(typeof(v1), isz, shp)
        Base.collect_to_with_first!(dest, v1, itr, st)
    end
end

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