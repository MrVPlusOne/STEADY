using SymbolicRegression: SymbolicRegression as SymReg

@kwdef struct GeneticProgrammingRegression <: AbstractRegerssionAlgorithm
    options::SymReg.Options
    numprocs::Int
    niterations::Int = 20
    runtests::Bool = true
    "The training data will be subsampled to this size if exceeded."
    max_data::Int = 1000
end

function fit_best_dynamics(
    alg::GeneticProgrammingRegression,
    sketch::MotionModelSketch,
    (inputs, outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    (valid_inputs, valid_outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    comps_σ_guess::Vector{Float64},
)
    @smart_assert length(comps_σ_guess) == size(outputs, 2)
    @smart_assert size(outputs, 1) == size(inputs, 1)

    if length(inputs) > alg.max_data
        sub_ids = sample(1:(alg.max_data), alg.max_data; replace=false)
        inputs = inputs[sub_ids]
        outputs = outputs[sub_ids, :]
    end

    features, shifts, scales = normalize_transform(collect.(inputs))

    input_mat = hcatreduce(features)
    valid_input_mat = (hcatreduce(collect.(valid_inputs)) .- shifts) ./ scales

    output_names = Tuple((v.name for v in sketch.output_vars))
    input_names = Tuple((v.name for v in sketch.input_vars))
    var_map::Vector{Symbol} = keys(inputs[1]) |> collect
    @unzip μs, σs, stats = map(1:size(outputs, 2)) do c
        target = outputs[:, c]
        valid_target = valid_outputs[:, c]

        time_taken = @elapsed begin
            (; tree, score) = fit_component_symreg(
                alg, input_mat, target, valid_input_mat, valid_target, var_map
            )
        end
        fit_σ = √(score)
        μ_f = compile(tree, alg.options, var_map, ℝ, ℝenv(); hide_type=true)
        size = SymReg.countNodes(tree)
        (μ_f, fit_σ, (; size, time_taken))
    end
    core = combine_functions(NamedTuple{output_names}(μs))
    args_transform = let shifts_t = Tuple(shifts), scales_t = Tuple(scales)
        args -> NamedTuple{keys(args)}((values(args) .- shifts_t) ./ scales_t)
    end
    core_ast = NamedTuple{output_names}((x -> x.ast).(μs))
    input_shifts = NamedTuple{input_names}(shifts)
    input_scales = NamedTuple{input_names}(scales)
    meta = (; core_ast..., input_shifts, input_scales)

    dynamics = GaussianGenerator(core ∘ args_transform, NamedTuple{output_names}(σs), meta)
    σ_list = [Symbol(name, ".σ") => σ for (name, σ) in pairs(dynamics.σs)]
    ast_size = sum((x -> x.size), stats)
    model_info = (; ast_size, σ_list..., dynamics=repr("text/plain", dynamics))
    optimizer_info = stats
    display_info = (;
        ast_size,
        dynamics.σs,
        μs=OrderedDict(zip(output_names, map(x -> x.ast, μs))),
        input_shifts,
        input_scales,
        stats,
    )

    (; dynamics, model_info, optimizer_info, display_info)
end

function fit_component_symreg(
    reg::GeneticProgrammingRegression,
    inputs::Matrix{Float64},
    outputs::AbsVec{Float64},
    valid_inputs::Matrix{Float64},
    valid_outputs::AbsVec{Float64},
    var_map::Vector{Symbol},
)
    @smart_assert size(inputs, 2) == size(outputs, 1)
    @smart_assert size(valid_inputs, 2) == size(valid_outputs, 1)

    hallOfFame = @time SymReg.EquationSearch(
        inputs, outputs; reg.options, reg.numprocs, reg.niterations, reg.runtests
    )
    dominating = SymReg.calculateParetoFrontier(
        valid_inputs, valid_outputs, hallOfFame, reg.options; varMap=map(string, var_map)
    )
    dominating[end] # the best member according to the valid set
end

function convert_constraints(cs::Vector)
    map(cs) do (k, v)
        Pair{Any,Any}(k, v)
    end
end


function compile(
    node::SymReg.Node,
    options::SymReg.Options,
    var_map::Vector{Symbol},
    result_shape::PShape,
    shape_env::ShapeEnv;
    hide_type=false,
)::CompiledFunc
    function compile_body(n::SymReg.Node)
        if n.degree == 0
            n.constant && return n.val
            var_name = var_map[n.feature]
            Expr(:(.), :args, QuoteNode(var_name))
        else
            if n.degree == 1
                f = options.unaops[n.op]
                args = [compile_body(n.l::SymReg.Node)]
            else
                f = options.binops[n.op]
                args = map(compile_body, [n.l, n.r])
            end
            Expr(:call, f, args...)
        end
    end

    body_ex = compile_body(node)
    f_ex = :(args -> $body_ex)
    rgf = compile_julia_expr(f_ex)

    rtype = shape_env.return_type[result_shape]
    ftype = hide_type ? Function : typeof(rgf)

    ast = SymReg.node_to_symbolic(node, options; varMap=map(string, var_map))
    CompiledFunc{rtype,ftype}(rgf, ast, body_ex)
end