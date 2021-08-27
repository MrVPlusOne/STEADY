mutable struct PruningState{Member}
    graph::EGraph
    "New members that are equivalent to any of these will be pruned."
    club_members::Dict{EClassId, Member}
end

PruningState{M}() where M = PruningState{M}(EGraph(), Dict()) 

function prune_redundant!(
    state::PruningState{TAST}, rules, progs::Vector{TAST}, params::SaturationParams
)
    (; graph, club_members) = state
    nodes = map(progs) do p
        _, enode = addexpr!(graph, to_expr(p))
        enode
    end
    report = saturate!(graph, rules, params)
    kept = TAST[]
    pruned = @NamedTuple{pruned::TAST, by::TAST}[]
    club_members = state.club_members = 
        Dict(Metatheory.find(graph, i) => e for (i, e) in club_members)
        
    for (prog, n) in zip(progs, nodes)
        # TODO: might need to handle the missing case caused by egraph size limit
        id = Metatheory.lookup(graph, n)::EClassId
        m = get(club_members, id, nothing)
        if m === nothing
            # welcome to the club!
            club_members[id] = prog
            push!(kept, prog)
        else
            # sorry, pruned!
            push!(pruned, (pruned=prog, by=m))
        end
    end
    (; kept, pruned, report)
end

"""
Convert an `TAST` to an untyped julia `Expr` that is suitable for E-graph-based 
reasoning.
"""
function to_expr(ast::TAST)
    rec(v::Var) = v.name
    rec(c::Call) = begin
        Expr(:call, c.f, rec.(c.args)...)
    end
    rec(ast)
end