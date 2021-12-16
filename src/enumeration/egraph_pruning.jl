"""
Maintain an exclusive club of groups that will only admit a new membmer `x` if 
`x` is not equivalent to (according to the EGraph) any of the existing members 
from `x`'s group.
"""
mutable struct PruningClub{Member,GID,F,G}
    "Convert a member into an expression that is suitable to work with EGraph."
    to_expr::F
    "Convert a member to its group id."
    to_group::G
    graph::EGraph
    groups::Dict{GID,Dict{EClassId,Member}}
end

PruningClub{M,GID}(; to_expr::F, to_group::G, explain_merges::Bool) where {M,GID,F,G} =
    let
        graph = EGraph()
        PruningClub{M,GID,F,G}(to_expr, to_group, graph, Dict())
    end

function admit_members!(
    club::PruningClub{M,I},
    new_members::AbstractVector{M},
    rules,
    compute_saturation_params,
    cost_f,
) where {M,I}
    @assert !isempty(new_members)

    new_members = (new_members |> sort_by(cost_f))
    (; graph, groups, to_expr, to_group) = club
    member_ids = Dict(
        map(new_members) do m
            eclass, enode = addexpr!(graph, to_expr(m))
            m => eclass.id
        end,
    )
    params = compute_saturation_params(graph.numclasses)
    # iter_callback((; egraph, iter)) = begin
    #     @info "euqality saturation callback" iter egraph.numclasses
    # end
    # report = saturate!(graph, rules, params; iter_callback)
    report = saturate!(graph, rules, params)

    kept = M[]
    pruned = @NamedTuple({pruned::M, reason::Any})[]
    pruned_classes = Tuple{EClassId,EClassId}[]


    for (gid, group) in groups
        m = Dict{EClassId,M}()
        for (cid, e) in group
            i′ = Metatheory.find(graph, cid)
            if !haskey(m, i′) || cost_f(e) < cost_f(m[i′])
                m[i′] = e
            end
        end
        group[gid] = m
    end

    for m in new_members
        m_id = member_ids[m]
        m_class = Metatheory.find(graph, m_id)
        # find m's corresponding group
        group = get!(groups, to_group(m)) do
            Dict{EClassId,M}()
        end

        exist_m = get(group, m_class, nothing)
        if exist_m === nothing
            # welcome to the club!
            group[m_class] = m
            push!(kept, m)
        else
            # pruned!
            @assert cost_f(exist_m) <= cost_f(m)
            push!(pruned, (pruned=m, reason=exist_m))
            # TODO: this only works when exist_m ∈ new_members
            exist_id = member_ids[exist_m]
            @assert m_id != exist_id "m: $m, exist_m: $exist_m"
            push!(pruned_classes, (m_id, exist_id))
        end
    end

    (; kept, pruned, report)
end


function prune_redundant(
    progs::AbstractVector{TAST}, rules, compute_saturation_params; explain_merges
)
    club = PruningClub{TAST,PType}(; to_expr=to_expr, to_group=p -> p.type, explain_merges)
    admit_members!(club, progs, rules, compute_saturation_params, ast_size)
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