using LinearAlgebra
using MLJLinearModels
using TimerOutputs

abstract type SparseOptimizer end

"""
Sequential thresholded optimizer.
"""
struct SeqThresholdOptimizer{Reg <: GLR} <: SparseOptimizer
    "Discarding threshold"
    ϵ::Float64
    regressor::Reg
end

Base.:*(opt::SeqThresholdOptimizer, s::Real) = 
    SeqThresholdOptimizer(opt.ϵ * s, opt.regressor * s)

Base.:*(reg::GLR, s::Real) = 
    GLR(reg.loss, reg.penalty / s, reg.fit_intercept, reg.penalize_intercept)

"""
return `(; coeffs, intercept, active_ids, iterations)`.
"""
function regression(
    opt::SeqThresholdOptimizer, 
    target::AbsVec{Float64},
    basis::AbsMat{Float64}, 
    active_ids::Vector{Int64} = collect(1:size(basis, 2)),
    iterations::Int=1,
)
    X = @views basis[:, active_ids]
    θ = MLJLinearModels.fit(opt.regressor, X, target)
    (coeffs, intercept) = if opt.regressor.fit_intercept
        (θ[1:end-1], θ[end])
    else
        (θ, 0.0)
    end
    
    is_active = abs.(coeffs) .> opt.ϵ
    if all(is_active)
        return (; coeffs, intercept, active_ids, iterations)
    end
    active_ids = active_ids[is_active]
    return regression(opt, target, basis, active_ids, iterations+1)
end

struct LinearExpression{N}
    shift::Float64
    coeffs::SVector{N, Float64}
    basis::SVector{N, <:CompiledFunc}
    type::PType
end

num_terms(::LinearExpression{N}) where N = N

function LinearExpression(
    shift::Float64, coeffs::AbstractVector, basis::AbstractVector, type::PType
)
    @smart_assert (n=length(coeffs)) == length(basis)
    LinearExpression(
        shift, SVector{n, Float64}(coeffs), SVector{n, CompiledFunc}(basis), type)
end

function Base.print(io::IO, expr::LinearExpression)
    (; shift, coeffs, basis, type) = expr
    compact = get(io, :compact, false)
    !compact && print(io, "LinearExpression(")
    print(io, "`")
    shift != 0.0 && print(io, shift)
    for i in 1:length(coeffs)
        op = coeffs[i] >= 0 ? " + " : " - "
        print(io, op, abs(coeffs[i]), " ", basis[i].ast)
    end
    print(io, "`")
    !compact && print(io, ")::", type)
end

Base.show(io::IO, expr::LinearExpression) = print(io, expr)

function compile(lexpr::LinearExpression, shape_env=ℝenv())
    annot = shape_env.type_annots[lexpr.type.shape]
    rtype = shape_env.return_type[lexpr.type.shape]

    (; shift, coeffs, basis) = lexpr

    terms = Any[]
    if isempty(coeffs) || shift != 0.0
        push!(terms, :(shift::$annot))
    end
    foreach(enumerate(basis)) do (i, e)
        push!(terms, :(coeffs[$i] * $(e.julia)))
    end
    body_ex = Expr(:call, :+, terms...)
    f_ex = :((args, shift, coeffs) -> $body_ex)
    rgf = compile_julia_expr(f_ex)

    param_f = args -> rgf(args, lexpr.shift, lexpr.coeffs)

    CompiledFunc{rtype, typeof(param_f)}(param_f, lexpr, body_ex)
end

struct GaussianComponent{F<:Function, R<:Real} <: Function
    μ_f::F
    σ::R
end

(gp::GaussianComponent)(in) = begin
    (; μ_f, σ) = gp
    Normal(μ_f(in), σ)
end

function Base.print(io::IO, expr::GaussianComponent)
    compact = get(io, :compact, false)
    (; μ_f, σ) = expr
    !compact && print(io, "GaussianComponent") 
    print(io, "(σ = ", σ)
    print(io, ", μ = ", μ_f, ")")
end

function Base.show(io::IO, ::Type{<:GaussianComponent})
    print(io, "GaussianComponent{...}")
end

Base.show(io::IO, comp::GaussianComponent) = print(io, comp)

"""
The output transformation needs to be a bijection.

Use [sindy_motion_model](@ref) to construct the motion model from the sketch and components.
"""
@kwdef(
struct SindySketch{F1, F2, F3}
    input_vars::Vector{Var}
    output_vars::Vector{Var}
    "genereate sketch inputs from state and control"
    inputs_transform::F1
    "transform (state, outputs, Δt) into next state"
    outputs_transform::F2
    "inversely compute sketch outputs from (state, next_state, Δt)"
    outputs_inv_transform::F3
end)

function Base.show(io::IO, ::Type{<:SindySketch})
    print(io, "SindySketch{...}")
end

function sindy_motion_model(sketch::SindySketch, comps::NamedTuple{names}) where names
    (x::NamedTuple, u::NamedTuple, Δt::Real) -> begin
        local inputs = sketch.inputs_transform(x, u)
        local outputs_dist = DistrIterator(map(f -> f(inputs), values(comps)))
        GenericSamplable(
            rng -> begin
                local out = NamedTuple{names}(rand(rng, outputs_dist))
                sketch.outputs_transform(x, out, Δt)# |> assert_finite
            end, 
            x1 -> begin 
                local out = sketch.outputs_inv_transform(x, x1, Δt)# |> assert_finite
                logpdf(outputs_dist, values(out))
            end
        )
    end
end

struct SindySynthesis <: SynthesisAlgorithm
    comp_env::ComponentEnv
    basis::Vector{<:CompiledFunc}
    sketch::SindySketch
    "A list of optimizers with different levels of regularization strength."
    optimizer_list::Vector{<:SparseOptimizer}
    optimizer_param_list::Vector{<:NamedTuple}
end

function compile_component(comp::GaussianComponent)
    if comp.μ_f isa LinearExpression
        GaussianComponent(compile(comp.μ_f), comp.σ)
    else
        comp
    end
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
Synthesize the dynamics using an Expectation Maximization-style loop and SINDy.  
"""
function em_sindy(
    train_split::Int, # the number of training trajectories
    obs_data_list::Vector{<:ObservationData},
    sindy_synthesis::SindySynthesis,
    comps_guess::OrderedDict{Symbol, <:GaussianComponent};
    obs_model::Function,
    sampler::PosteriorSampler,
    n_fit_trajs::Int, # the number of trajectories to use for fitting the dynamics
    iteration_callback = (data::NamedTuple{(:iter, :trajectories, :dyn_est)}) -> nothing,
    logger=Base.current_logger(),
    max_iters::Int=100,
)
    (; comp_env, basis, sketch, optimizer_list) = sindy_synthesis
    timer = TimerOutput()
    
    @smart_assert train_split < length(obs_data_list)
    n_valid = length(obs_data_list) - train_split

    sampler_states = [new_state(sampler) for _ in obs_data_list]

    function sample_data(motion_model)
        local systems = map(obs_data_list) do obs_data
            MarkovSystem(obs_data.x0_dist, motion_model, obs_model)
        end
        
        sample_posterior_parallel(sampler, systems, obs_data_list, sampler_states)
    end
    
    dyn_history = OrderedDict{Symbol, <:GaussianComponent}[]
    logp_history = Vector{Float64}[]
    dyn_est = comps_guess

    try @progress "fit_dynamics_iterative" for iter in 1:max_iters+1
        comps_compiled = map(compile_component, NamedTuple(dyn_est))
        motion_model = sindy_motion_model(sketch, comps_compiled)
        (; trajectories, log_obs) = @timeit timer "sample_data" sample_data(motion_model)
        trajectories::Matrix{<:Vector}

        push!(dyn_history, dyn_est)
        push!(logp_history, log_obs)

        @timeit timer "iteration_callback" begin
            iteration_callback((;iter, trajectories, dyn_est))
        end

        (iter == max_iters+1) && break

        comps_σ = [comp.σ for comp in values(comps_guess)]
        output_types = [v.type for v in sketch.output_vars]
        output_names = [v.name for v in sketch.output_vars]

        n_fit_trajs = min(n_fit_trajs, size(trajectories, 1))
        solutions = @timeit timer "fit_dynamics_sindy_validated" begin 
            fit_dynamics_sindy_validated(
                sindy_synthesis, trajectories, obs_data_list, train_split, comps_σ;
                output_types, n_fit_trajs)
        end

        sol = max_by(x -> x.valid_score)(solutions)
        (; comps, stats, lexprs, train_score, valid_score, optimizer_params) = sol
        dyn_est = OrderedDict(zip(output_names, comps))

        Base.with_logger(logger) do
            @info "performance" log_obs=mean(log_obs)
            @info "performance" train_score valid_score log_step_increment=0
            @info("model", 
                num_terms=sum(num_terms, lexprs), 
                comp_σ=(x -> x.σ).(comps), 
                comps=repr("text/plain", dyn_est),
                log_step_increment=0)
            @info "optimizer" optimizer_params... log_step_increment=0
        end
        @info "model" iter log_obs=mean(log_obs)
        @info "model" dyn_est
        @info "model" stats
    end # end for
    catch exception
        @warn "Synthesis early stopped by exception."
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println(stdout)
        end
    end # end try block
    display(timer)
    (; dyn_est, dyn_history, logp_history, timer)
end

function show_dyn_history(dyn_history; io=stdout, first_rows=5, max_rows=15, table_width=100)
    let rows=[], step=max((length(dyn_history)-first_rows)÷max_rows, 1)
        for (i, dyn) in enumerate(dyn_history)
            (i <= first_rows || i % step == 0) && push!(rows, dyn)
        end
        println(io, "=====Dynamics History=====")
        show(io, MIME"text/plain"(), DataFrame(rows), truncate=table_width)
    end
end

"""
Split the provided trajectories into training and validation set. 
Fit multiple dynamics using the provided optimizer list and evaluate their performance 
on both the traiand validation set.

Returns a list of `(; comps, stats, lexprs, train_score, valid_score)`.
"""
function fit_dynamics_sindy_validated(
    synthesis::SindySynthesis,
    trajectories::Matrix{<:Vector},
    obs_data_list::Vector{<:ObservationData},
    train_split::Int,
    comps_σ_guess::Vector{Float64};
    output_types::AbsVec{PType},
    n_fit_trajs::Int,
    n_valid_trajs::Int=size(trajectories, 1),
)
    (; sketch) = synthesis
    @smart_assert train_split < size(trajectories, 2)
    @smart_assert n_fit_trajs <= size(trajectories, 1)
    @smart_assert n_valid_trajs <= size(trajectories, 1)

    train_data = trajectories[1:n_fit_trajs, 1:train_split]
    valid_data = trajectories[1:n_valid_trajs, train_split+1:end]
    train_n_effective = n_effective_trajectoreis(train_data)

    inputs, outputs = construct_inputs_outputs(
        train_data, obs_data_list[1:train_split], sketch)
    valid_inputs, valid_outputs = construct_inputs_outputs(
        valid_data, obs_data_list[train_split+1:end], sketch)
    solutions = parallel_map(eachindex(synthesis.optimizer_list), synthesis) do syn, i
        optimizer, optimizer_params = syn.optimizer_list[i], syn.optimizer_param_list[i]
        optimizer::SeqThresholdOptimizer

        (; comps, stats, lexprs) = fit_dynamics_sindy(
            syn.basis, inputs, outputs, output_types, comps_σ_guess, optimizer)
        train_score = data_likelihood(comps, inputs, outputs)
        valid_score = data_likelihood(comps, valid_inputs, valid_outputs)
        (; comps, stats, lexprs, train_score, valid_score, optimizer_params)
    end
    println("Pareto analysis: ")
    ordering = sortperm(solutions, rev=true, by=x -> x.valid_score)
    local rows = map(solutions, ordering) do sol, rank
        (; sol.optimizer_params..., sol.train_score, sol.valid_score, rank)
    end
    display(DataFrame(rows))

    solutions
end

"""
Fit the dynamics from trajectory data using the SINDy algorithm.
"""
function fit_dynamics_sindy(
    basis::Vector{<:CompiledFunc},
    inputs::AbsVec{<:NamedTuple},
    outputs::AbstractMatrix{Float64},
    output_types::AbsVec{PType},
    comps_σ::AbsVec{Float64},
    optimizer::SparseOptimizer;
)
    foreach(basis) do f
        f isa CompiledFunc && @smart_assert f.ast.type.shape == ℝ "basis $f is not a scalar."
    end
    @smart_assert size(outputs, 1) == size(inputs, 1)

    @unzip features, scales = map(basis) do f
        feature = f.(inputs)
        scale = sqrt(feature'feature / length(feature))
        feature/scale, scale
    end
    basis_data = hcatreduce(features)
    all(isfinite, basis_data) || @error "Basis data contains NaN or Inf: $basis_data."

    @unzip comps, stats, lexprs = map(1:size(outputs, 2)) do c
        target = outputs[:, c]
        opt_scaled = optimizer * comps_σ[c]
        (; coeffs, intercept, active_ids, iterations) = regression(opt_scaled, target, basis_data)
        
        comp = if isempty(active_ids) 
            lexpr = LinearExpression(intercept, [], [], output_types[c])
            σ_f = std(target)
        else
            fs = basis[active_ids]
            σ_f = @views std(basis_data[:, active_ids] * coeffs .+ intercept - target)
            coeffs1 = coeffs ./ scales[active_ids]
            lexpr = LinearExpression(intercept, coeffs1, fs, output_types[c])
        end
        comp = GaussianComponent(compile(lexpr), σ_f)
        stat = (; term_weights = coeffs ./ comps_σ[c], iterations)
        (comp, stat, lexpr)
    end
    comps::Vector{<:GaussianComponent}
    (; comps, stats, lexprs)
end

function construct_inputs_outputs(
    trajectories::Matrix{<:Vector}, 
    obs_data_list::Vector,
    sketch::SindySketch,
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
    comps::AbsVec{<:GaussianComponent},
    inputs::AbsVec{<:NamedTuple},
    outputs::AbstractMatrix{Float64};
    output_type=Float64,
)   
    @smart_assert length(comps) == size(outputs, 2)
    @smart_assert length(inputs) == size(outputs, 1)

    ll::output_type = 0
    for c in 1:length(comps)
        comp = comps[c]
        ll += logpdf.(comp.(inputs), outputs[:, c]) |> sum
    end
    ll / length(inputs)
end