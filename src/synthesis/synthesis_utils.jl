export prog_size_prior
function prog_size_prior(decay::Float64)
    (comps) -> log(decay) * sum(map(ast_size, comps); init=0) 
end

Base.size(v::Transducers.ProgressLoggingFoldable) = size(v.foldable)

function Distributions.logpdf(dist::NamedTuple{ks}, v::NamedTuple{ks})::Real where ks
    sum(logpdf.(values(dist), values(v)))
end

function structure_to_vec(v::Union{AbstractVector, Tuple, NamedTuple})
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
    @smart_assert i[] == length(arr)
    arr
end

promote_numbers_type(x::Real) = typeof(x)
promote_numbers_type(v::AbstractVector{T}) where T = 
    T <: Real ? T : promote_numbers_type(v[1])
promote_numbers_type(v::Union{Tuple, NamedTuple}) = 
    Base.promote_type(promote_numbers_type.(values(v))...)


function structure_from_vec(template::NamedTuple{S}, vec)::NamedTuple{S} where S
    NamedTuple{keys(template)}(structure_from_vec(values(template), vec))
end

function structure_from_vec(template, vec)
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
            foreach(rec, d.core)
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

"""
A very crude estimation by counting the number of unique particles.
"""
function n_effective_trajectoreis(trajectories::Matrix)
    T = length(trajectories[1])
    n_unique = 0
    for t in 1:T
        n_unique += length(Set(tr[t] for tr in trajectories))
    end
    n_unique / T
end

"""
- inputs shape: n_transition of NamedTuple
- outputs shape: (n_transition, n_outputs) of Float64
"""
function construct_inputs_outputs(
    trajectories::Matrix{<:Vector}, 
    obs_data_list::Vector,
    sketch::MotionModelSketch,
)
    @smart_assert length(obs_data_list) == size(trajectories, 2)
    (; inputs_transform, outputs_inv_transform) = sketch
    inputs = []
    outputs = []
    for j in 1:length(obs_data_list)
        (; times, controls) = obs_data_list[j]
        for i in 1:size(trajectories, 1)
            tr = trajectories[i, j]
            for t in 1:length(tr)-1
                push!(inputs, inputs_transform(tr[t], controls[t]))
                s = tr[t]
                s1 = tr[t+1]
                Δt = times[t+1] - times[t]
                o = outputs_inv_transform(s, s1, Δt)
                push!(outputs, transpose(collect(o)))
            end
        end
    end
    inputs = specific_elems(inputs)
    outputs = vcatreduce(outputs)
    (; inputs, outputs)
end

"""
Returns the average log probability of the outputs.
"""
function data_likelihood(
    dynamics::GaussianGenerator{names},
    inputs::AbsVec{<:NamedTuple},
    outputs::AbstractMatrix{Float64};
    output_type=Float64,
) where names
    @smart_assert length(names) == size(outputs, 2)
    @smart_assert length(inputs) == size(outputs, 1)

    ll::output_type = 0
    for t in 1:length(inputs)
        ll += logpdf(dynamics(inputs[t]), NamedTuple{names}(outputs[t, :]))
    end
    ll / length(inputs)
end

"""
Sample random inputs for the missing dynamics. This can be useful for [`IOPruner`](@ref).
"""
function sample_rand_inputs(sketch::DynamicsSketch, n::Integer)
    dist = DistrIterator(merge(
        inputs_distribution(sketch).core,
        params_distribution(sketch).core,
    ));
    [rand(dist) for _ in 1:n]
end