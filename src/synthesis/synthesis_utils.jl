function structure_to_vec(v::Union{Tuple, NamedTuple})
    T = promote_numbers_type(v)
    vec = Vector{T}(undef, n_numbers(v))
    structure_to_vec!(vec, v)
    vec
end

function structure_to_vec!(arr, v::Union{AbstractVector, Tuple, NamedTuple})
    i = Ref(0)
    rec(r) = let
        if r isa Real
            arr[i[]+=1] = r
        elseif r isa Union{AbstractVector, Tuple, NamedTuple}
            foreach(rec, r)
        end
        nothing
    end
    rec(v)
    @assert i[] == length(arr)
    arr
end

function promote_numbers_type(v::Union{Tuple, NamedTuple})
    rec(x::Real) = typeof(x)
    rec(::AbstractVector{T}) where {T <: Real} = T
    rec(v::AbstractVector) = rec(v[1])
    rec(v::Union{Tuple, NamedTuple}) = Base.promote_eltype(rec.(values(v))...)
    Base.promote_eltype(rec.(values(v))...)
end

function structure_from_vec(template::NamedTuple{S}, vec)::NamedTuple{S} where S
    NamedTuple{keys(template)}(structure_from_vec(values(template), vec))
end

function structure_from_vec(template::Tuple, vec)::Tuple
    i::Ref{Int} = Ref(0)

    map(template) do x
        _read_structure(x, i, vec)
    end
end

_read_structure(x, i::Ref{Int}, vec) = let
    if x isa Real
        vec[i[]+=1]
    elseif x isa Union{AbstractVector, Tuple, NamedTuple}
        map(x) do x′
            _read_structure(x′, i, vec)
        end
    else
        error("don't know how to handle the template: $x")
    end
end

"""
Count how many numbers there are in the given NamedTuple.

```jldoctest
julia> n_numbers((0.0, @SVector[0.0, 0.0]))
3
```
"""
n_numbers(v::Union{Tuple, NamedTuple}) = sum(n_numbers, v)
n_numbers(::Real) = 1
n_numbers(v::AbstractVector{<:Real}) = length(v)
n_numbers(v::AbstractVector) = sum(n_numbers, v)

"""
Computes a vector of upper and lower bounds for a given distribution.
This can be useful for, e.g., box-constrained optimization.
"""
function _compute_bounds(prior_dist)
    lower, upper = Float64[], Float64[]
    rec(d) = let
        if d isa UnivariateDistribution
            (l, u) = Distributions.extrema(d)
            push!(lower, l)
            push!(upper, u)
        elseif d isa DistrIterator
            foreach(rec, d.distributions)
        elseif d isa SMvUniform
            foreach(rec, d.uniforms)
        else
            error("Don't know how to compute bounds for $d")
        end
        nothing
    end
    rec(prior_dist)
    lower, upper
end