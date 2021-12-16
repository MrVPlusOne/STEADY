struct SindyRegression <: AbstractRegerssionAlgorithm
    comp_env::ComponentEnv
    basis::Vector{<:CompiledFunc}
    "A list of optimizers with different levels of regularization strength."
    optimizer_list::Vector{<:SparseOptimizer}
    optimizer_param_list::Vector{<:NamedTuple}
end


function fit_best_dynamics(
    algorithm::SindyRegression,
    sketch::MotionModelSketch,
    (inputs, outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    (valid_inputs, valid_outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    comps_σ_guess::Vector{Float64},
)
    (; optimizer_list) = algorithm

    solutions = parallel_map(1:length(optimizer_list), algorithm) do alg, i
        local optimizer, optimizer_params = alg.optimizer_list[i], alg.optimizer_param_list[i]
        optimizer::SeqThresholdOptimizer

        (; dynamics, stats, lexprs) = fit_dynamics_sindy(
            alg.basis, inputs, outputs, sketch, comps_σ_guess, optimizer
        )
        train_score = data_likelihood(dynamics, inputs, outputs)
        valid_score = data_likelihood(dynamics, valid_inputs, valid_outputs)
        (; dynamics, stats, lexprs, train_score, valid_score, optimizer_params)
    end
    println("Pareto analysis: ")
    ordering = sortperm(solutions; rev=true, by=x -> x.valid_score)
    rows = map(solutions, ordering) do sol, rank
        (; sol.optimizer_params..., sol.train_score, sol.valid_score, rank)
    end
    display(DataFrame(rows))

    sol = solutions |> max_by(x -> x.valid_score)
    (; dynamics, stats, lexprs, train_score, valid_score, optimizer_params) = sol

    σ_list = [Symbol(name, ".σ") => σ for (name, σ) in pairs(dynamics.σs)]
    model_info = (
        num_terms=sum(num_terms, lexprs), σ_list..., dynamics=repr("text/plain", dynamics)
    )
    optimizer_info = (; optimizer_params..., train_score, valid_score)
    display_info = (;
        num_terms=sum(num_terms, lexprs),
        dynamics.σs,
        lexprs=OrderedDict(zip(keys(dynamics.σs), lexprs)),
        stats,
    )

    (; dynamics, model_info, optimizer_info, display_info)
end


"""
Fit the dynamics from trajectory data using the SINDy algorithm.
"""
function fit_dynamics_sindy(
    basis::Vector{<:CompiledFunc},
    inputs::AbsVec{<:NamedTuple},
    outputs::AbstractMatrix{Float64},
    sketch::MotionModelSketch,
    comps_σ::AbsVec{Float64},
    optimizer::SparseOptimizer,
)
    @smart_assert length(comps_σ) == size(outputs, 2)
    foreach(basis) do f
        f isa CompiledFunc &&
            @smart_assert f.ast.type.shape == ℝ "basis $f is not a scalar."
    end
    @smart_assert size(outputs, 1) == size(inputs, 1)

    @unzip features, scales = map(basis) do f
        feature = f.(inputs)
        scale = sqrt(feature'feature / length(feature))
        feature / scale, scale
    end
    basis_data = hcatreduce(features)
    all(isfinite, basis_data) || @error "Basis data contains NaN or Inf: $basis_data."

    output_types = [v.type for v in sketch.output_vars]
    output_names = Tuple((v.name for v in sketch.output_vars))
    @unzip lexprs, σs, stats = map(1:size(outputs, 2)) do c
        target = outputs[:, c]
        opt_scaled = optimizer * comps_σ[c]
        (; coeffs, intercept, active_ids, iterations) = regression(
            opt_scaled, target, basis_data
        )

        comp = if isempty(active_ids)
            lexpr = LinearExpression(intercept, [], [], output_types[c])
            σ_f = std(target; mean=0.0)
        else
            fs = basis[active_ids]
            σ_f = @views std(
                basis_data[:, active_ids] * coeffs .+ intercept - target, mean=0.0
            )
            coeffs1 = coeffs ./ scales[active_ids]
            lexpr = LinearExpression(intercept, coeffs1, fs, output_types[c])
        end
        stat = (; term_weights=coeffs ./ comps_σ[c], iterations)
        (lexpr, σ_f, stat)
    end

    core = combine_functions(NamedTuple{output_names}(compile.(lexprs)))

    dynamics = GaussianGenerator(
        core, NamedTuple{output_names}(σs), NamedTuple{output_names}(lexprs)
    )
    (; dynamics, stats, lexprs)
end
