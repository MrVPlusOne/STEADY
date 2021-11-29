struct SindyRegression <: AbstractRegerssionAlgorithm
    comp_env::ComponentEnv
    basis::Vector{<:CompiledFunc}
    "A list of optimizers with different levels of regularization strength."
    optimizer_list::Vector{<:SparseOptimizer}
    optimizer_param_list::Vector{<:NamedTuple}
end


"""
Split the provided trajectories into training and validation set. 
Fit multiple dynamics using the provided optimizer list and evaluate their performance 
on both the traiand validation set.

Returns a list of `(; comps, stats, lexprs, train_score, valid_score)`.
"""
function fit_best_dynamics(
    alg::SindyRegression,
    sketch::MotionModelSketch,
    (inputs, outputs)::Tuple{Vector{<:NamedTuple}, Matrix{Float64}},
    (valid_inputs, valid_outputs)::Tuple{Vector{<:NamedTuple}, Matrix{Float64}},
    comps_σ_guess::Vector{Float64},
)
    (; optimizer_list) = alg
    output_names = [v.name for v in sketch.output_vars]
    output_types = [v.type for v in sketch.output_vars]

    solutions = parallel_map(eachindex(optimizer_list), alg) do alg, i
        optimizer, optimizer_params = alg.optimizer_list[i], alg.optimizer_param_list[i]
        optimizer::SeqThresholdOptimizer

        (; comps, stats, lexprs) = fit_dynamics_sindy(
            alg.basis, inputs, outputs, output_types, comps_σ_guess, optimizer)
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

    sol = solutions |> max_by(x -> x.valid_score)
    (; comps, stats, lexprs, train_score, valid_score, optimizer_params) = sol

    dynamics = OrderedDict(zip(output_names, comps))
    σ_list = [Symbol(name, ".σ") => comp.σ for (name, comp) in dynamics]
    model_info = (num_terms=sum(num_terms, lexprs), σ_list..., 
        dynamics=repr("text/plain", dynamics))
    optimizer_info = (; optimizer_params..., train_score, valid_score)
    display_info = (; num_terms=sum(num_terms, lexprs), dynamics, stats)
    
    (; dynamics, model_info, optimizer_info, display_info)
end

d1 = OrderedDict(zip(["a", "b"], [1,2]))
l1 = [Symbol(name, ".σ") => 2v for (name, v) in d1]

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

