using Metatheory
using Metatheory.EGraphs
using Metatheory.Library
using Random

@metatheory_init ()

plus_group = @theory begin
    a + 0 => a
    a + b => b + a
    a + (-a) => 0 
    a + (b + c) => (a + b) + c
end

times_monoid = commutative_monoid(:*, 1)

additional = @theory begin
    a * 0 => 0
    a * (b + c) == (a * b) + (a * c)
end

th1 = times_monoid ∪ plus_group ∪ additional
##

graph = EGraph(:(a*b+c*b+0*d))
report = saturate!(graph, th1)
graph.classes

extract!(graph, astsize)
##

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

th2 = th1 ∪ power_th ∪ const_fold ∪ basic_diff
##

"Ast size + big penalty for the differential operators."
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

G = EGraph(:(D(x, x^3)*5 + 2x^2))
report = saturate!(G, th2)
G.classes
# access the saturated EGraph
extract!(G, astsize)
extract!(G, diff_cost)
##

using Metatheory.EGraphs: areequal!
n_vars = 20
vars = ["x$i" for i in 1:n_vars]
ex1 = reduce((a, b) -> "$a * $b", Random.shuffle(vars)) |> Meta.parse
ex2 = reduce((a, b) -> "$a * $b", Random.shuffle(vars)) |> Meta.parse

params=SaturationParams(
    timeout=n_vars * 10,
    eclasslimit = n_vars * 1000,
    enodelimit = 4 * n_vars * 1000,
)
g = EGraph()
r = areequal!(g, times_monoid, ex1, ex2; params, mod=@__MODULE__)
r.result
##