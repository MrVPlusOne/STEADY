# This example illustrate the strange behavior of e-graph based 
# rewritting caused by unsound rules such as `a/a => 1`.
module Playground

using BenchmarkTools
using Metatheory
using Metatheory.Library

"""
Enumerate arithmatic expressions up until a given size
"""
function all_expressions(size)
    size_map = Dict{Int, Vector{Any}}()
    size_map[0] = []
    size_map[1] = [0, 1]

    ast_size(::Symbol) = 1
    ast_size(e::Expr) = sum(ast_size, e.args)

    function rec(size)
        get!(size_map, size) do
            exs = []
            for left in 1:size-2
                right = size-1-left
                for (a, b) in Iterators.product(rec(left), rec(right))
                    append!(exs, Expr(:call, op, a, b) for op in [:+, :-, :*, :/])
                end
            end
            exs
        end
    end

    rec.(1:size)
    (s -> size_map[s]).(1:size) |> Iterators.flatten |> collect
end

function rewrite_example(exps, rules; 
        check_noteq::Union{Nothing, Tuple{EClassId, EClassId}}=nothing, 
        threaded=true, timeout=2)
    g = EGraph()
    # use SimpleScheduler to ensure that all rules are applied
    params = SaturationParams(;
       scheduler = Metatheory.Schedulers.SimpleScheduler,
       timeout, eclasslimit=0, enodelimit=0, matchlimit=0,
       threaded,
    )
    # map each expression to its original eclass id.
    exp_map = Dict(map(exps) do e
        eclass, enode = addexpr!(g, e)
        e => eclass.id
    end)
    
    function graph_check(info)
        if check_noteq !== nothing
            e1, e2 = check_noteq
            i1, i2 = exp_map[e1], exp_map[e2]
            if find(g, i1) == find(g, i2) 
                classes = sort(collect(pairs(g.classes)), by=x->x[1])
                classes_str = join(classes, "\n")
                @error "$e1 equaled to $e2" info
                println("All classes: ")
                println(classes_str)
                error("graph check failed.")
            end
        end
    end
    g.graph_check = graph_check
    saturate!(g, rules, params)

    @show g.numclasses

    # use this to record the simplest expression in each eclass
    club = Dict{EClassId, Any}()
    pruned = []

    for e in exps
        e_class = Metatheory.find(g, exp_map[e])
        if !haskey(club, e_class)
            club[e_class] = e
        else
            push!(pruned, (pruned=e, by=club[e_class]))
        end
    end
    (; club, pruned)
end
##

display(all_expressions(5))

# this is ok
simple_rules = commutative_monoid(:(*), 1) ∪ commutative_monoid(:(+), 0)
rewrite_example(all_expressions(5), simple_rules)

# now this is problamtic
more_rules = @theory begin
    a - a => 0

    (a * b) + (a * c) => a * (b + c)

    (a / a) => 1

    a / b => a * reciprocal(b)
end

club, pruned = rewrite_example(all_expressions(5), simple_rules ∪ more_rules, 
    threaded=false, timeout=2, check_noteq=(1, 0))
display(values(club))
display(pruned)

##

function find_bad_i()
    local exps = all_expressions(5)
    for i in 3700:3750
        club, pruned = rewrite_example(exps[i:end], simple_rules ∪ more_rules, 
            threaded=false, timeout=2)
        if length(club) != 1
            return i
        end
    end
    nothing
end
bad_i = find_bad_i()


end