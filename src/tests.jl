using DataFrames

"""
Randomly sample inputs to test function equivalance. 
Optionally returns a counter example if found.
"""
function test_equiv(
    p1::CompiledFunc, p2::CompiledFunc, arg_dists::NamedTuple, n_samples,
)::Union{Nothing, NamedTuple}
    tolerant_types = [DomainError]
    for _ in 1:n_samples
        args = map(rand, arg_dists)
        try
            v1, v2 = p1(args), p2(args)
            !isapprox(v1, v2, atol=1e-6, rtol=1e-8) && return (; diff=v1 .- v2, args)
        catch e
            if typeof(e) ∉ tolerant_types
                rethrow()
            end
        end
    end
    return nothing
end

function check_pruning_soundness(
    var_dists::Dict{Var, <:Distribution}; 
    comp_env, max_size, pruner, types_to_prune,
    shape_env = ℝenv(), n_samples=20,
)
    arg_dists = (; (v.name => dist for (v, dist) in var_dists)...)
    eresult = enumerate_terms(comp_env, collect(keys(var_dists)), max_size; 
        types_to_prune, pruner)
    failed = []
    n_passed = 0
    @showprogress desc="check pruned" for (; pruned, by) in eresult.pruned
        f1 = compile(pruned, shape_env, comp_env)
        f2 = compile(by, shape_env, comp_env)
        counter_ex = test_equiv(f1, f2, arg_dists, n_samples)
        if counter_ex !== nothing
            push!(failed, (; pruned, by, counter_ex))
        else
            n_passed += 1
        end
    end
    (; n_passed, failed)
end

function example_pruning_check()
    scalars = [
        Var(:l, ℝ, PUnits.Length) => Normal(), 
        Var(:v, ℝ, PUnits.Speed) => Normal(), 
        Var(:a, ℝ, PUnits.Acceleration) => Normal(), 
    ]
    vectors = [Var(n, ℝ2, PUnits.Length) => MvNormal([0.0, 0.0], 1.0) for n in [:v1, :v2, :v3]]
    var_dists = Dict([scalars; vectors])
    comp_env = ComponentEnv()
    components_scalar_arithmatic!(comp_env)
    components_special_functions!(comp_env)
    components_vec2!(comp_env)

    max_size = 5
    pruner = RebootPruner(; comp_env.rules, only_postprocess=false)
    types_to_prune = Set([PType(ℝ, PUnits.Speed)])
    (; n_passed, failed) = check_pruning_soundness(var_dists; 
        comp_env, max_size, pruner, types_to_prune)
    @show n_passed
    println("Failed: ")
    display(DataFrame(failed))
end