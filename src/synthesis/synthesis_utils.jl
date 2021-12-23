const TimeSeries{T} = Vector{T}
const ObservationData = NamedTuple{(
    :times, :obs_frames, :observations, :controls, :x0_dist
)}

export DynamicsSketch, params_distribution, compile_motion_model, to_p_motion_model


struct GaussianGenerator{names,D,F} <: Function
    μ_f::F
    σs::NamedTuple{names,NTuple{D,Float64}}
    meta::NamedTuple
end
@use_short_show GaussianGenerator

Base.length(gg::GaussianGenerator) = length(gg.σs)

function (gg::GaussianGenerator{names})(in) where {names}
    (; μ_f, σs) = gg
    μ = μ_f(in)::NamedTuple{names}
    map(values(μ), values(σs)) do m, s
        Normal(m, s)
    end |> NamedTuple{names} |> DistrIterator
end

function Base.print(io::IO, expr::GaussianGenerator)
    compact = get(io, :compact, false)
    !compact && print(io, "GaussianGenerator")
    print(io, "(σs = ", expr.σs)
    print(io, ", meta = ", expr.meta, ")")
end

Base.show(io::IO, comp::GaussianGenerator) = print(io, comp)
function Base.show(io::IO, ::MIME"text/plain", expr::GaussianGenerator)
    io = IOIndent(IOContext(io, :compact => true))
    println(io, "GaussianGenerator:", Indent())
    for (k, v) in pairs(expr.σs)
        println(io, k, ".σ: ", v)
    end
    println(io, "-------------------")

    for (k, v) in pairs(expr.meta)
        println(io, k, ": ", v)
    end
end

"""
The sketch of a motion model that enables supervised learning from input-output data. 
The output transformation therefore needs to be a bijection.

Use [mk_motion_model](@ref) to construct the motion model from the sketch and components.
"""
@kwdef(struct MotionModelSketch{F1,F2,F3}
    input_vars::Vector{Var}
    output_vars::Vector{Var}
    "genereate sketch inputs from state and control"
    inputs_transform::F1
    "transform (state, outputs, Δt) into next state"
    outputs_transform::F2
    "inversely compute sketch outputs from (state, next_state, Δt)"
    outputs_inv_transform::F3
end)
@use_short_show MotionModelSketch

struct GaussianMotionModel{Core<:GaussianGenerator,SK<:MotionModelSketch} <: Function
    sketch::SK
    core::Core
end

function (motion_model::GaussianMotionModel{<:GaussianGenerator{names}})(
    x::NamedTuple, u::NamedTuple, Δt::Real
) where {names}
    sketch, core = motion_model.sketch, motion_model.core
    inputs = sketch.inputs_transform(x, u)
    outputs_dist = core(inputs)
    GenericSamplable(
        rng -> begin
            local out = NamedTuple{names}(rand(rng, outputs_dist))
            sketch.outputs_transform(x, out, Δt)# |> assert_finite
        end,
        x1 -> begin
            local out = sketch.outputs_inv_transform(x, x1, Δt)# |> assert_finite
            logpdf(outputs_dist, values(out))
        end,
    )
end

"""
A regression algorith specifies how to synthesize the dynamics from a given set of 
state transition data.

Need to implement:
 - [`fit_candidate_dynamics`](@ref)
"""
abstract type AbstractRegerssionAlgorithm end

"""
Fit multiple dynamics and return the best one according to the validation set.
"""
function fit_best_dynamics(
    alg::AbstractRegerssionAlgorithm,
    sketch::MotionModelSketch,
    (inputs, outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    (valid_inputs, valid_outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    comps_σ_guess::Vector{Float64},
)::NamedTuple{(:dynamics, :model_info, :optimizer_info, :display_info)}
    error("Not implemented.")
end

"""
Split the provided trajectories into training and validation set. 
Fit multiple dynamics and return the best one according to the validation set.
"""
function fit_best_dynamics(
    alg::AbstractRegerssionAlgorithm,
    sketch::MotionModelSketch,
    trajectories::Matrix{<:Vector},
    obs_data_list::Vector{<:ObservationData},
    train_split::Int,
    comps_σ_guess::Vector{Float64};
    n_fit_trajs::Int,
    n_valid_trajs::Int=size(trajectories, 1),
)
    @smart_assert train_split < size(trajectories, 2)
    @smart_assert n_fit_trajs <= size(trajectories, 1)
    @smart_assert n_valid_trajs <= size(trajectories, 1)

    train_data = trajectories[1:n_fit_trajs, 1:train_split]
    valid_data = trajectories[1:n_valid_trajs, (train_split + 1):end]

    inputs, outputs = construct_inputs_outputs(
        train_data, obs_data_list[1:train_split], sketch
    )
    valid_inputs, valid_outputs = construct_inputs_outputs(
        valid_data, obs_data_list[(train_split + 1):end], sketch
    )

    fit_best_dynamics(
        alg, sketch, (inputs, outputs), (valid_inputs, valid_outputs), comps_σ_guess
    )
end

"""
A sketch for some missing dynamics. 

With the missing dynamics provided (by e.g., a synthesizer), this skech can then be 
compiled into a parameterized dynamics of the form
`params -> (state, control, Δt) -> distribution_of_state`.
"""
@kwdef struct DynamicsSketch{F,G}
    inputs::Vector{Var}
    outputs::Vector{Var}
    params::OrderedDict{Var,GDistr}
    "state_to_inputs(state::NamedTuple, control::NamedTuple) -> inputs"
    state_to_inputs::F
    "outputs_to_state_dist(outputs, (; state..., control..., inputs...), Δt) -> distribution_of_state"
    outputs_to_state_dist::G
end
@use_short_show DynamicsSketch

params_distribution(sketch::DynamicsSketch) = begin
    DistrIterator((; (v.name => dist for (v, dist) in sketch.params)...))
end

inputs_distribution(sketch::DynamicsSketch) = begin
    DistrIterator((; (v.name => dist_for_shape[v.type.shape] for v in sketch.inputs)...))
end

function check_params_logp(x0, x0_dist, params, params_dist)
    x0_logp = logpdf(x0_dist, x0)
    isfinite(x0_logp) || error("x0_logp = $x0_logp, some value may be out of its support.\n
        x0=$x0,\nx0_dist=$x0_dist")
    params_logp = logpdf(params_dist, params)
    isfinite(params_logp) ||
        error("params_logp = $params_logp, some value may be out of its support.")
end

function compile_motion_model(
    comps::NamedTuple, (; shape_env, comp_env, sketch, hide_type);
)
    funcs = map(comp -> compile(comp, shape_env, comp_env; hide_type), comps)
    sketch_core = _build_sketch_core(funcs)
    # hide_type && (sketch_core = WrappedFunc(sketch_core))
    WrappedFunc(to_p_motion_model(sketch_core, sketch))
end

function _build_sketch_core(funcs)
    x -> map(f -> f(x), funcs)
end

function to_p_motion_model(
    sketch_core::Function, (; state_to_inputs, outputs_to_state_dist)
)
    (params::NamedTuple) ->
        (x::NamedTuple, u::NamedTuple, Δt::Real) -> begin
            local inputs = state_to_inputs(x, u)
            local prog_inputs = merge(inputs, params)
            local outputs = sketch_core(prog_inputs)
            local others = merge(x, u, params, prog_inputs)
            outputs_to_state_dist(outputs, others, Δt)::GDistr
        end
end

export VariableData

struct VariableData
    state_vars::Vector{Var}
    action_vars::Vector{Var}
end


_check_variable_types(vdata::VariableData, shape_env::ShapeEnv) = begin
    check_type(var, value) = begin
        t = shape_env[var.type]
        @smart_assert value isa t "Prior distribution for \
            $var produced value $value, expected type: $t"
    end

    (; states, dynamics_params) = vdata
    foreach([states; dynamics_params]) do (v, d)
        check_type(v, rand(d))
    end
end

export SynthesisEnumerationResult, synthesis_enumeration

"""
Group the enumeration data needed for `map_synthesis`. Can be created by `bottom_up_enum`.
"""
@kwdef(struct SynthesisEnumerationResult{Combine}
    vdata::VariableData
    sketch::DynamicsSketch{Combine}
    shape_env::ShapeEnv
    comp_env::ComponentEnv
    enum_result::EnumerationResult
end)

Base.show(io::IO, mime::MIME"text/plain", r::SynthesisEnumerationResult) = begin
    (; comp_env, enum_result, vdata, sketch) = r
    (; state_vars, action_vars) = vdata
    param_vars = sketch.params |> keys |> collect
    hole_inputs, holes = sketch.inputs, sketch.outputs

    n_interest = prod(count_len(enum_result[v.type]) for v in holes)
    search_details = join((count_len(enum_result[v.type]) for v in holes), " * ")

    io = IOIndent(io)
    println(io, "===== Synthesis enumeration result =====")
    println(io, "search_space: $(pretty_number(n_interest)) = $search_details")
    println(io, "n_components: ", length(comp_env.signatures))
    println(io, "holes: $holes")
    println(io, "hole_inputs: $hole_inputs")
    println(io, "states: $state_vars")
    println(io, "actions: $action_vars")
    println(io, "params: $param_vars")

    println(io, "Candidate expressions:", Indent())
    for h in holes
        println(io, "---- For $h ----")
        show(io, DataFrame((expression=e,) for e in enum_result[h.type]))
        println(io)
    end
    print(io, Dedent())

    show(io, mime, enum_result)
end

"""
Enuemrate all programs with the correct return types up to some max AST size.
"""
function synthesis_enumeration(
    vdata::VariableData,
    sketch::DynamicsSketch,
    shape_env::ShapeEnv,
    comp_env::ComponentEnv,
    max_size;
    pruner=NoPruner(),
    type_pruning=true,
)
    param_vars = sketch.params |> keys |> collect
    dyn_vars = [sketch.inputs; param_vars]
    if type_pruning
        types_needed, _ = enumerate_types(
            comp_env,
            Set(v.type for v in dyn_vars),
            Set(v.type for v in sketch.outputs),
            max_size,
        )
    else
        types_needed = nothing
    end
    dyn_varset = Set(v for v in dyn_vars)
    enum_result = enumerate_terms(comp_env, dyn_varset, max_size; types_needed, pruner)
    SynthesisEnumerationResult(; vdata, sketch, shape_env, comp_env, enum_result)
end

export MapSynthesisResult
struct MapSynthesisResult{R}
    best_result::R
    stats::NamedTuple
    errored_programs::Vector
    sorted_results::Vector
end

get_top_results(r::MapSynthesisResult, top_k::Int) = begin
    rows = map(Iterators.take(r.sorted_results, top_k)) do (; score, comps, params)
        (; score, comps, params)
    end
    rows
end

export get_top_results, show_top_results
show_top_results(r::MapSynthesisResult, top_k::Int) = begin
    rs = get_top_results(r, top_k)
    println("Top $top_k solutions:")
    for (i, r) in enumerate(rs)
        println("---- solution $i ----")
        println(r)
    end
end

Base.show(io::IO, r::MapSynthesisResult) =
    print(io, "MapSynthesisResult(best=$(r.best_result))")

Base.show(io::IO, ::MIME"text/plain", r::MapSynthesisResult) = begin
    io = IOIndents.IOIndent(io)
    println(io, "==== MAP synthesis result ====")
    println(io, "Stats:", Indent())
    for (k, v) in pairs(r.stats)
        println(io, "$k: ", pretty_number(v))
    end
    print(io, Dedent())
    println(io, "Best estimation found:", Indent())
    let (; score, comps, params) = r.best_result
        println(io, "score: $score")
        println(io, "expressions: $(comps)")
        println(io, "params: $(params)")
    end
end


export prog_size_prior
function prog_size_prior(decay::Float64)
    (comps) -> log(decay) * sum(map(ast_size, comps); init=0)
end

Base.size(v::Transducers.ProgressLoggingFoldable) = size(v.foldable)

function Distributions.logpdf(dist::NamedTuple{ks}, v::NamedTuple{ks})::Real where {ks}
    sum(logpdf.(values(dist), values(v)))
end

function structure_to_vec(v::Union{AbstractVector,Tuple,NamedTuple})
    T = promote_numbers_type(v)
    vec = Vector{T}(undef, n_numbers(v))
    structure_to_vec!(vec, v)
    vec
end

function structure_to_vec!(arr, v::Union{AbstractVector,Tuple,NamedTuple})
    i = Ref(0)
    rec(r) =
        let
            if r isa Real
                arr[i[] += 1] = r
            elseif r isa Union{AbstractVector,Tuple,NamedTuple}
                foreach(rec, r)
            end
            nothing
        end
    rec(v)
    @smart_assert i[] == length(arr)
    arr
end

promote_numbers_type(x::Real) = typeof(x)
promote_numbers_type(v::AbstractVector{T}) where {T} =
    T <: Real ? T : promote_numbers_type(v[1])
promote_numbers_type(v::Union{Tuple,NamedTuple}) =
    Base.promote_type(promote_numbers_type.(values(v))...)


function structure_from_vec(template::NamedTuple{S}, vec)::NamedTuple{S} where {S}
    NamedTuple{keys(template)}(structure_from_vec(values(template), vec))
end

function structure_from_vec(template, vec)
    i::Ref{Int} = Ref(0)

    map(template) do x
        _read_structure(x, i, vec)
    end
end

_read_structure(x, i::Ref{Int}, vec) =
    let
        if x isa Real
            vec[i[] += 1]
        elseif x isa Union{AbstractVector,Tuple,NamedTuple}
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
n_numbers(v::Union{Tuple,NamedTuple}) = sum(n_numbers, v)
n_numbers(::Real) = 1
n_numbers(v::AbstractVector{<:Real}) = length(v)
n_numbers(v::AbstractVector) = sum(n_numbers, v)

"""
Computes a vector of upper and lower bounds for a given distribution.
This can be useful for, e.g., box-constrained optimization.
"""
function _compute_bounds(prior_dist)
    lower, upper = Float64[], Float64[]
    rec(d) =
        let
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
    trajectories::Matrix{<:Vector}, obs_data_list::Vector, sketch::MotionModelSketch
)
    @smart_assert length(obs_data_list) == size(trajectories, 2)
    (; inputs_transform, outputs_inv_transform) = sketch
    inputs = []
    outputs = []
    for j in 1:length(obs_data_list)
        (; times, controls) = obs_data_list[j]
        for i in 1:size(trajectories, 1)
            tr = trajectories[i, j]
            for t in 1:(length(tr) - 1)
                push!(inputs, inputs_transform(tr[t], controls[t]))
                s = tr[t]
                s1 = tr[t + 1]
                Δt = times[t + 1] - times[t]
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
) where {names}
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
    dist = DistrIterator(
        merge(inputs_distribution(sketch).core, params_distribution(sketch).core)
    )
    [rand(dist) for _ in 1:n]
end