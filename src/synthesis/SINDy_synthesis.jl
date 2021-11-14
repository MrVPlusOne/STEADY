abstract type SparseOptimizer end

"""
Sequential thresholded least squares.
"""
struct STLSOptimizer <: SparseOptimizer
    "Discarding threshold."
    λ::Float64
end

"""
return `(; coeffs, active_ids, iterations)`.
"""
function regression(
    opt::STLSOptimizer, 
    target::AbstractVector{Float64},
    basis::AbstractMatrix{Float64}, 
    active_ids::Vector{Int64} = collect(1:size(basis, 2)),
    iterations::Int=1,
)
    X = @views basis[:, active_ids]
    coeffs = X \ target
    is_active = coeffs .> opt.λ
    if all(is_active)
        return (; coeffs, active_ids, iterations)
    end
    active_ids = active_ids[is_active]
    return regression(opt, target, basis, active_ids, iterations+1)
end

struct LinearExpression{N} <: Function
    shift::Float64
    coeffs::SVector{N, Float64}
    basis::SVector{N, <:CompiledFunc}
end

LinearExpression(shift::Float64, coeffs::AbstractVector, basis::AbstractVector) = begin 
    @assert (n=length(coeffs)) == length(basis)
    LinearExpression(shift, SVector{n, Float64}(coeffs), SVector{n, CompiledFunc}(basis))
end

function Base.show(io::IO, expr::LinearExpression; digits=3)
    (; shift, coeffs, basis) = expr
    print(io, "LinearExpression(`", shift)
    for i in 1:length(coeffs)
        op = coeffs[i] >= 0 ? " + " : " - "
        print(io, op,  round_values(abs(coeffs[i]); digits), basis[i].ast)
    end
    print(io, "`)")
end

Base.show(io::IO, ::MIME"text/plain", expr::LinearExpression) = show(io, expr)

(lexpr::LinearExpression{0})(in) = lexpr.shift
(lexpr::LinearExpression)(in) = begin
    (; shift, coeffs, basis) = lexpr
    shift + map(f -> f(in), basis)'coeffs
end


struct GaussianComponent{F<:Function, R<:Real} <: Function
    μ_f::F
    σ::R
end

(gp::GaussianComponent)(in) = begin
    (; μ_f, σ) = gp
    Normal(μ_f(in), σ)
end

function Base.show(io::IO, expr::GaussianComponent)
    (; μ_f, σ) = expr
    print(io, "GaussianComponent(μ = ") 
    show(io, μ_f) 
    print(io, ", σ = ", σ, ")")
end

function Base.show(io::IO, ::Type{<:GaussianComponent})
    print(io, "GaussianComponent{...}")
end

Base.show(io::IO, ::MIME"text/plain", expr::GaussianComponent) = show(io, expr)

"""
The output transformation needs to be a bijection.
"""
@kwdef(
struct SindySketch{F1, F2, F3}
    input_vars::Vector{Var}
    output_vars::Vector{Var}
    "genereate sketch inputs from state and control"
    inputs_transform::F1 = (x, u) -> merge(x, u)
    "transform state and sketch outputs into state derivatives"
    outputs_transform::F2 = x -> o -> o
    "inversely compute sketch outputs from state and state dervatives"
    outputs_inv_transform::F3 = x -> o -> o
end)

function Base.show(io::IO, ::Type{<:SindySketch})
    print(io, "SindySketch{...}")
end

function sindy_motion_model(sketch::SindySketch, comps::NamedTuple)
    (; inputs_transform, outputs_transform, outputs_inv_transform) = sketch
    (x::NamedTuple, u::NamedTuple, Δt::Real) -> begin
        local inputs = inputs_transform(x, u)
        local outputs = map(f -> f(inputs), comps)
        local core = DistrIterator(outputs)
        # GenericDistrTransform(core, outputs_transform(x), outputs_inv_transform(x))
        local outf = outputs_transform(x)
        GenericSamplable(() -> outf(rand(core)))
    end
end

"""
Synthesize the dynamics using an Expectation Maximization-style loop and SINDy.  
"""
function em_sindy(
    basis::Vector{<:CompiledFunc},
    obs_data_list::Vector{<:ObservationData},
    comps_guess::OrderedDict{Symbol, <:GaussianComponent},
    sketch::SindySketch;
    obs_model::Function,
    sampler::PosteriorSampler,
    n_fit_trajs::Int, # the number of trajectories to use for fitting the dynamics
    iteration_callback = (data::NamedTuple{(:iter, :trajectories, :dyn_est)}) -> nothing,
    optimizer::STLSOptimizer = STLSOptimizer(0.1),
    max_iters::Int=100,
)
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
        motion_model = sindy_motion_model(sketch, NamedTuple(dyn_est))
        (; trajectories, log_obs) = sample_data(motion_model)
        trajectories::Matrix{<:Vector}

        push!(dyn_history, dyn_est)
        push!(logp_history, log_obs)
        println("Estimated log_obs: $(to_measurement(log_obs))")

        iteration_callback((;iter, trajectories, dyn_est))
        trajectories = trajectories[1:n_fit_trajs, :]

        score_old = let
            state_scores = [sum(states_log_score.(
                Ref(motion_model), Ref(obs_data_list[j]), trajectories[:, j], Float64))
                for j in 1:length(obs_data_list)]
            sum(state_scores) / size(trajectories, 1)
        end

        (iter == max_iters+1) && break

        inputs, outputs = construct_inputs_outputs(
            trajectories, obs_data_list, sketch)
        dynamic_comps = fit_dynamics_sindy(basis, inputs, outputs, optimizer)
        dyn_est = OrderedDict(zip(sketch.output_vars, dynamic_comps))

        @info("Iteration $iter finished.", dyn_est)
    end # end for
    catch exception
        @warn "Synthesis early stopped by exception."
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println(stdout)
        end
    end # end try block
    (; dyn_est, dyn_history, logp_history)
end

"""
Fit the dynamics from trajectory data using the SINDy algorithm.
"""
function fit_dynamics_sindy(
    basis::Vector{<:CompiledFunc},
    inputs::AbstractVector{<:NamedTuple},
    outputs::AbstractMatrix{Float64},
    optimizer::STLSOptimizer,
)::Vector{<:GaussianComponent}
    foreach(basis) do f
        f isa CompiledFunc && @assert f.ast.type.shape == ℝ "basis $f is not a scalar."
    end
    @assert size(outputs, 1) == size(inputs, 1)

    @unzip transformed, μ, σ = map(basis) do f
        normalize_transform(f.(inputs)) # TODO: handle exceptions
    end
    basis_data = hcatreduce(transformed)
    all(isfinite, basis_data) || @error "Basis data contains NaN or Inf: $basis_data."

    map(1:size(outputs, 2)) do c
        target, target_μ, target_σ = normalize_transform(outputs[:, c])
        (; coeffs, active_ids, iterations) = regression(optimizer, target, basis_data)
        comp = if isempty(active_ids) 
            GaussianComponent(LinearExpression(target_μ, [], []), target_σ)
        else
            fs = basis[active_ids]
            μ_sub = μ[active_ids]
            σ_sub = σ[active_ids]
            coeffs1 = coeffs ./ σ_sub .* target_σ
            μ1 = target_μ - coeffs1'μ_sub
            lexpr = LinearExpression(μ1, coeffs1, fs)
            σ_f = std(lexpr.(inputs) .- outputs[:, c])
            GaussianComponent(lexpr, σ_f)
        end
    end
end

function construct_inputs_outputs(
    trajectories::Matrix{<:Vector}, 
    obs_data_list::Vector,
    sketch::SindySketch,
)
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
                dsdt = map(s, s1) do a, b 
                    (b - a) / Δt
                end
                dsdt′ = outputs_inv_transform(s)(dsdt)
                push!(outputs, transpose(collect(dsdt′)))
            end
        end
    end
    inputs = specific_elems(inputs)
    outputs = vcatreduce(outputs)
    (; inputs, outputs)
end