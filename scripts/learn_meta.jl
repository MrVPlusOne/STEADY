using Metatheory
using Metatheory.EGraphs
using Metatheory.Library
using Random

@metatheory_init ()

comm_monoid = commutative_monoid(:(*), 1)

comm_group = commutative_group(:(+), 0, :(-))
comm_group = @theory begin
    a + 0 => a
    a + b => b + a
    a + inv(a) => 0 # inverse
    a + (b + c) => (a + b) + c
end
distrib = @theory begin
    a * (b + c) == (a * b) + (a * c)
end

power_th = @theory begin
    x == x^1
    x^n * x^m == x^(n+m)
end

const_fold = @theory begin
    a::Number + b::Number |> a + b
    a::Number * b::Number |> a * b
end

basic_diff = @theory begin
    D(x, x) => 1
    D(x, x^(n::Number)) |> (n == 0 ? 0 : :($n * x^$(n-1)))
    D(z, x + y) == D(z, x) + D(z, y)
    D(z, x * y) == D(z, x) * y + x * D(z, y)
end

param_theory = @theory begin
    map(g, map(f, x)) == map(f |> g, x)
    Call(apply, (f, x)) => Call(f,(x))
    apply(f, x) => f(x)
end

##
th = comm_monoid ∪ comm_group ∪ distrib ∪ const_fold ∪ power_th ∪ basic_diff ∪ param_theory

function diff_cost(n::ENode, g::EGraph, an::Type{<:AbstractAnalysis})
    cost = 1 + arity(n)
    (n.head == :D) && (cost += 1000)
    for id ∈ n.args
        eclass = geteclass(g, id)
        !hasdata(eclass, an) && (cost += Inf; break)
        cost += last(getdata(eclass, an))
    end
    return cost
end

##

abstract type MinSymbolAnalysis <: AbstractAnalysis end
Metatheory.EGraphs.islazy(an::Type{<:MinSymbolAnalysis})::Bool = false
Metatheory.EGraphs.modify!(analysis::Type{<:MinSymbolAnalysis}, g, id) = nothing
Metatheory.EGraphs.join(analysis::Type{<:MinSymbolAnalysis}, a, b) = min(a, b)
Metatheory.EGraphs.make(an::Type{<:MinSymbolAnalysis}, g::EGraph, n::ENode) = 
    if n.head != :*
        n.head 
    else
        if isempty(n.args)
            @show n
        end
        minimum(getdata(geteclass(g, a), an) for a in n.args)
    end

comm_times = @theory begin
    # a * b |> if find(_egraph, a) > find(_egraph, b)
    #     :($b * $a) else _lhs_expr end
    # a * (b * c) |> let
    #     ids = find.(Ref(_egraph), [a, b, c])
    #     perm = sortperm(ids)
    #     a, b, c = [a, b, c][perm]
    #     :($a * $b * $c)
    # end
    # a * b |> let 
    #     if getdata(a, MinSymbolAnalysis) > getdata(b, MinSymbolAnalysis)
    #         :($b * $a)
    #     else
    #         _lhs_expr
    #     end
    # end
    a * b => b * a
    a * (b * c) => (a * b) * c
end
    
@metatheory_init ()
using Metatheory.EGraphs: areequal!
n_vars = 20
vars = ["x$i" for i in 1:n_vars]
ex1 = reduce((a, b) -> "$a * $b", Random.shuffle(vars)) |> Meta.parse
ex2 = reduce((a, b) -> "$a * $b", Random.shuffle(vars)) |> Meta.parse
# areequal(comm_times, ex1, ex2; params=SaturationParams(scheduler=Metatheory.EGraphs.SimpleScheduler))
mod=@__MODULE__ 
params=SaturationParams(
    timeout=n_vars * 10,
    eclasslimit = n_vars * 1000,
    enodelimit = 4 * n_vars * 1000,
)
g = EGraph()
push!(g.analyses, MinSymbolAnalysis)
r = areequal!(g, comm_times, ex1, ex2; params, mod=@__MODULE__)
r.result
##
r.report
extract!(g, astsize, root=4)
g.classes
##

G = EGraph(:(D(x, x^3)*5 + 2x^2))
report = saturate!(G, th)
G.classes
# access the saturated EGraph
extract!(G, diff_cost)
##

@areequal th (P(a)+P(b)) P(a)
@areequal th map(cos, map(sin, xs)) map(sin |> cos, xs)
@areequal th apply(f, x) f(x)
@areequal th $((1, 2)) 1
