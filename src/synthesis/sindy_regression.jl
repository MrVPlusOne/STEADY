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
    (; basis, optimizer_list, optimizer_param_list) = algorithm

    (; dynamics, stats, lexprs) = fit_dynamics_sindy(
        basis,
        inputs,
        outputs,
        valid_inputs,
        valid_outputs,
        sketch,
        comps_σ_guess,
        optimizer_list,
        optimizer_param_list,
    )

    σ_list = [Symbol(name, ".σ") => σ for (name, σ) in pairs(dynamics.σs)]
    model_info = (
        num_terms=sum(num_terms, lexprs), σ_list..., dynamics=repr("text/plain", dynamics)
    )
    optimizer_info = stats
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
    valid_inputs::AbsVec{<:NamedTuple},
    valid_outputs::AbstractMatrix{Float64},
    sketch::MotionModelSketch,
    comps_σ::AbsVec{Float64},
    optimizer_list::AbsVec{<:SparseOptimizer},
    optimizer_param_list,
)
    @smart_assert length(comps_σ) == size(outputs, 2)
    foreach(basis) do f
        f isa CompiledFunc &&
            @smart_assert f.ast.type.shape == ℝ "basis $f is not a scalar."
    end
    @smart_assert size(outputs, 1) == size(inputs, 1)

    @unzip features, shifts, scales = map(basis) do f
        feature = f.(inputs)
        normalize_transform(feature)
    end
    input_mat = hcatreduce(features)
    all(isfinite, input_mat) || @error "Input matrix contains NaN or Inf: $input_mat."

    valid_input_mat = map(basis, shifts, scales) do f, shift, scale
        (f.(valid_inputs) .- shift) / scale
    end |> hcatreduce

    output_types = [v.type for v in sketch.output_vars]
    output_names = Tuple((v.name for v in sketch.output_vars))
    @unzip lexprs, σs, stats = map(1:size(outputs, 2)) do c
        target = outputs[:, c]
        valid_target = valid_outputs[:, c]

        comp_fit = fit_component_sindy(
            input_mat, target, valid_input_mat, valid_target, comps_σ[c], optimizer_list
        )
        (; σ_fit, coeffs, intercept, active_ids, opt_id, iterations) = comp_fit
        opt_params = optimizer_param_list[opt_id]
        lexpr = if isempty(active_ids)
            LinearExpression(intercept, [], [], output_types[c])
        else
            fs = basis[active_ids]
            coeffs1 = coeffs ./ scales[active_ids]
            intercept1 = intercept - coeffs1' * shifts[active_ids]
            LinearExpression(intercept1, coeffs1, fs, output_types[c])
        end
        stat = (; term_weights=coeffs ./ comps_σ[c], opt_params, iterations)
        (lexpr, σ_fit, stat)
    end

    core = combine_functions(NamedTuple{output_names}(compile.(lexprs)))

    dynamics = GaussianGenerator(
        core, NamedTuple{output_names}(σs), NamedTuple{output_names}(lexprs)
    )
    (; dynamics, stats, lexprs)
end

function fit_component_sindy(
    inputs::Matrix{Float64},
    outputs::AbsVec{Float64},
    valid_inputs::Matrix{Float64},
    valid_outputs::AbsVec{Float64},
    σ_guess::Float64,
    optimizer_list::AbsVec{<:SparseOptimizer},
)
    solutions = parallel_map(eachindex(optimizer_list), nothing) do _, opt_id
        optimizer = optimizer_list[opt_id]
        opt_scaled = optimizer * σ_guess
        solution_seq = regression(opt_scaled, outputs, inputs)
        model_seq = parallel_map(solution_seq, nothing) do _, sol
            (; coeffs, intercept, active_ids, iterations) = sol
            if isempty(active_ids)
                σ_fit = std(outputs; mean=0.0)
                valid_loss = mean(abs2, valid_outputs)
            else
                σ_fit = @views std(
                    inputs[:, active_ids] * coeffs .+ intercept - outputs, mean=0.0
                )
                valid_loss = @views mean(
                    abs2,
                    valid_inputs[:, active_ids] * coeffs .+ intercept - valid_outputs,
                )
            end
            merge((; σ_fit, valid_loss, opt_id, iterations), sol)
        end
        model_seq |> max_by(model -> -model.valid_loss)
    end
    solutions |> max_by(model -> -model.valid_loss)
end