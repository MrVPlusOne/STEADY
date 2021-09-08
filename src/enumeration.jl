"""
Use the syntax `result[pshape]` to get an iterator over all found programs
of the given [`PShape`](@ref).
"""
@kwdef(
mutable struct EnumerationResult
    programs::Dict{PShape, Dict{Int, Dict{PUnit, Set{TAST}}}}
    pruned::Vector{@NamedTuple{pruned::TAST, by::TAST, explain::Any}}
    n_created::Int
    n_deleted::Int
    total_time::Float64
    pruning_time::Float64
end)

get_stats(er::EnumerationResult) = 
    (; er.n_created, er.n_deleted, er.total_time, er.pruning_time)

Base.getindex(r::EnumerationResult) =
    Iterators.flatten(r[shape] for shape in keys(r.programs))

Base.getindex(r::EnumerationResult, size::Int) =
    Iterators.flatten(r[shape, size] for shape in keys(r.programs))

Base.getindex(r::EnumerationResult, shape::PShape, size::Int) = begin
    Iterators.flatten(values(get(r.programs[shape], size, Dict{PUnit, Set{TAST}}())))
end


Base.getindex(r::EnumerationResult, shape::PShape) = let
    d1 = r.programs[shape]
    (p 
    for size in sort(collect(keys(d1)))
    for d2 in values(d1[size]) 
    for p in d2)
end

Base.getindex(r::EnumerationResult, (; shape, unit)::PType) = begin
    d1 = get(r.programs, shape, Dict())
    (p 
    for size in sort(collect(keys(d1)))
    for p in get(d1[size], unit, []))
end

Base.getindex(r::EnumerationResult, (;shape, size, unit)) = begin
    s1 = get!(r.programs, shape) do ; Dict{Int, Dict{PUnit, Vector{TAST}}}() end
    s2 = get!(s1, size) do ; Dict{PUnit, Vector{TAST}}() end
    get!(s2, unit) do ; Set(TAST[]) end
end

Base.insert!(r::EnumerationResult, prog::TAST, size=ast_size(prog)) = begin
    (; shape, unit) = prog.type
    ps = r[(;shape, size, unit)]
    push!(ps, prog)
    r.n_created += 1
    r
end

prune!(r::EnumerationResult, (; pruned, by, explain)) = begin
    p = pruned
    (; shape, unit) = p.type
    unit_map = r.programs[shape][ast_size(p)]
    ps = unit_map[unit]
    @assert p ∈ ps "$p is already pruned."
    delete!(ps, p)
    isempty(ps) && delete!(unit_map, unit)
    r.n_deleted += 1
    push!(r.pruned, (; pruned, by, explain))
    r
end


function show_programs(io::IO, r::EnumerationResult; max_programs::Int=20)
    for s in keys(r.programs)
        println(io, "------- $(s.name) -------")
        i = 1
        for p in r[s]
            if i > max_programs
                println(io, "...")
                break
            end
            println(io, p, "::", p.type)
            i+=1
        end
    end
end

Base.show(io::IO, ::MIME"text/plain", r::EnumerationResult) = begin
    io = IOIndents.IOIndent(io)
    println(io, "=== Enumeration result ===")
    println(io, "Stats:", Indent())
    println(io, "n_total: ", pretty_number(r.n_created-r.n_deleted))
    for (k, v) in pairs(get_stats(r))
        println(io, "$k: ", pretty_number(v))
    end
    print(io, Dedent())
    println(io, "Found programs: ", Indent())
    show_programs(io, r)
end

Base.show(io::IO, r::EnumerationResult) = begin
    stats = join(["$k: $v" for (k, v) in pairs(get_stats(r))], ", ")
    print(io, "EnumerationResult($stats)")
end

"""
Should implement the following functions
- [`prune_postprocess!`](@ref)
- [`prune_iteration!`](@ref)
"""
abstract type AbstractPruner end


"""
    prune_iteration!(::AbstractPruner, ::EnumerationResult, types_to_prune, current_size; is_last) -> to_prune
prune programs by directly mutating the given `result`.

If `types_to_prune` is empty, will prune expressions of all types.
"""
function prune_iteration! end


"""
Perform bottom-up program enumeration.
## Arguments
- `max_size::Integer`: programs with [`ast_size`](@ref) up to this value will be returned.
"""
function bottom_up_enum(
    env::ComponentEnv, vars::Vector{Var}, max_size; types_to_prune=Set(), pruner::AbstractPruner=NoPruner(),
)::EnumerationResult
    @assert allunique(vars) "Duplicate variables in the input."

    start_time = time()

    # stores all found programs, indexed by: shape -> ast_size -> unit
    found = Dict{PShape, Dict{Int, Dict{PUnit, Set{TAST}}}}()
    result = EnumerationResult(found, [], 0, 0, 0, 0)

    # size 1 programs consist of all variables
    foreach(vars) do v
        insert!(result, v, 1)
    end

    signatures = env.signatures
    # construct larger programs from smaller ones
    @progress name="bottom_up_enum" for size in 2:max_size 
        for (f, sig) in signatures
            arg_shapes = sig.arg_shapes
            any(!haskey(found, s) for s in arg_shapes) && continue # skip unrealizable
            sizes_for_arg = [keys(found[s]) for s in arg_shapes]
            # iterate over all possible size combinations s.t. the sum = size-1
            for arg_sizes in size_combinations(length(arg_shapes), sizes_for_arg, size-1)
                arg_dicts = [found[arg_shapes[i]][s] for (i, s) in enumerate(arg_sizes)]
                # iterate over all unit combinations
                for arg_units in Iterators.product((keys(d) for d in arg_dicts)...)
                    runit = sig.result_unit(arg_units...)
                    (runit === nothing) && continue # skip invalid combination
                    arg_candidates = (d[u] for (u, d) in zip(arg_units, arg_dicts))
                    # iterate over all argument AST combinations
                    for args in Iterators.product(arg_candidates...)
                        prog = Call(f, args, PType(sig.result_shape, runit::PUnit))
                        insert!(result, prog, size)
                    end
                end
            end
        end
        result.pruning_time += @elapsed begin
            for x in prune_iteration!(pruner, result, types_to_prune, size, is_last=false)
                prune!(result, x) 
            end
        end
    end

    result.pruning_time += @elapsed begin 
        for x in prune_iteration!(pruner, result, types_to_prune, max_size, is_last=true)
            prune!(result, x) 
        end
    end
    result.total_time += time() - start_time
    result
end

"""
Iterate over all possible size combinations such that they have the specified total size.

## Example
```jldoctest
julia> size_combinations(3, [[1,2],[2,3],[1,2,3,4]], 6) |> collect
4-element Vector{Vector{Int64}}:
 [1, 2, 3]
 [1, 3, 2]
 [2, 2, 2]
 [2, 3, 1]

julia> size_combinations(3, [1:1, 1:2, 1:4], 6) |> collect
2-element Vector{Vector{Int64}}:
[1, 1, 4]
[1, 2, 3]
```
"""
function size_combinations(n_args, sizes_for_arg, total_size)
    function rec(i, size_left)
        if i == n_args
            (size_left in sizes_for_arg[i]) ? [[size_left]] : Vector{Int}[]
        else
            # the maximal possible size for argument i
            si_max = size_left-(n_args-i)
            si_sizes = filter(s -> s <= si_max, sizes_for_arg[i])
            Iterators.flatten(
                (push!(sizes, s1) for sizes in rec(i + 1, size_left - s1))
                for s1 in si_sizes
            )
        end
    end
    (reverse!(v) for v in rec(1, total_size))
end


"""
Maintain an exclusive club of groups that will only admit a new membmer `x` if 
`x` is not equivalent to (according to the EGraph) any of the existing members 
from `x`'s group.
"""
mutable struct PruningClub{Member, GID, F, G}
    "Convert a member into an expression that is suitable to work with EGraph."
    to_expr::F
    "Convert a member to its group id."
    to_group::G
    graph::EGraph
    groups::Dict{GID, Dict{EClassId, Member}}
end

PruningClub{M, GID}(; to_expr::F, to_group::G, explain_merges::Bool) where {M, GID, F, G} = 
    let
        graph = EGraph(; record_proofs = explain_merges)
        PruningClub{M, GID, F, G}(to_expr, to_group, graph, Dict()) 
    end

function admit_members!(
    club::PruningClub{M, I}, 
    new_members::AbstractVector{M},
    rules, compute_saturation_params, cost_f,
) where {M, I}
    @assert !isempty(new_members)
    
    new_members = (new_members |> sort_by(cost_f))
    (; graph, groups, to_expr, to_group) = club
    member_ids = Dict(map(new_members) do m
        eclass, enode = addexpr!(graph, to_expr(m))
        m => eclass.id
    end)
    params = compute_saturation_params(graph.numclasses)
    iter_callback((; egraph, iter)) = begin
        @info "euqality saturation callback" iter egraph.numclasses
    end
    report = saturate!(graph, rules, params; iter_callback)
    
    kept = M[]
    pruned = @NamedTuple{pruned::M, by::M}[]
    pruned_classes = Tuple{EClassId, EClassId}[]
    

    for (gid, group) in groups
        m = Dict{EClassId, M}()
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
            Dict{EClassId, M}()
        end

        exist_m = get(group, m_class, nothing)
        if exist_m === nothing
            # welcome to the club!
            group[m_class] = m
            push!(kept, m)
        else
            # pruned!
            @assert cost_f(exist_m) <= cost_f(m)
            push!(pruned, (pruned=m, by=exist_m))
            # TODO: this only works when exist_m ∈ new_members
            exist_id = member_ids[exist_m]
            @assert m_id != exist_id "m: $m, exist_m: $exist_m"
            push!(pruned_classes, (m_id, exist_id))
        end
    end
    pruned_explain = 
        map(zip(pruned, pruned_classes)) do ((; pruned, by), (a, b))
            e = if graph.proof_forest !== nothing
                Metatheory.EGraphs.explain_equality(graph.proof_forest, a, b)
            else
                nothing
            end
            (; pruned, by, explain=e)
        end |> collect
    
    (; kept, pruned_explain, report)
end


function prune_redundant(
    progs::AbstractVector{TAST}, rules, compute_saturation_params; explain_merges,
)
    club = PruningClub{TAST, PType}(; 
        to_expr = to_expr, to_group = p -> p.type, explain_merges)
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


## --- pruners ---

struct NoPruner <: AbstractPruner end
prune_iteration!(::NoPruner, ::EnumerationResult, types_to_prune, size; is_last) = []


"""
Build a new e-graph at every iteration.
"""
@kwdef(
struct RebootPruner{NtoS<:Function} <: AbstractPruner
    rules::Vector{AbstractRule}
    "will prune all types if empty"
    compute_saturation_params::NtoS=default_saturation_params
    reports::Vector=[]
    explain_merges::Bool=false
    only_postprocess::Bool=false
end)

prune_iteration!(pruner::RebootPruner, result::EnumerationResult, types_to_prune, size; is_last) = 
    if pruner.only_postprocess == is_last 
        (; rules, compute_saturation_params, reports, explain_merges) = pruner
        members = 
            if isempty(types_to_prune)
                result[]
            else
                Iterators.flatten(result[ty] for ty in types_to_prune)
            end
        sorted = collect(TAST, members) |> sort_by(ast_size)
        isempty(sorted) && return TAST[]
        kept, pruned, report = prune_redundant(
            sorted, rules, compute_saturation_params; explain_merges)
        push!(reports, report)
        pruned
    else
        TAST[]
    end

"""
Rereuse e-graphs across iterations. 
This works best if the rules does not grow the size of the egraph.
"""
@kwdef(
struct IncrementalPruner{NtoS<:Function} <: AbstractPruner
    rules::Vector{AbstractRule}
    compute_saturation_params::NtoS=default_saturation_params
    reports::Vector=[]
    club::PruningClub = PruningClub{TAST, PType}(; to_expr = to_expr, to_group = p -> p.type)
end)

prune_iteration!(pruner::IncrementalPruner, result::EnumerationResult, types_to_prune, size; is_last) = begin
    is_last && return TAST[]

    # TODO implement types_to_prune
    (; rules, compute_saturation_params, reports, club) = pruner

    if size == 1
        for special in [0, 1, :R2_0]
            addexpr!(club.graph, special)
        end
    end

    new_members = collect(TAST, result[size])

    kept, pruned, report = admit_members!(
        club, new_members, rules, compute_saturation_params, ast_size)
    push!(reports, report)
    pruned
end

total_time_report(reports) = begin
    tos = (r -> r.to).(reports)
    reduce(merge, tos)
end

default_saturation_params(egraph_size) = begin
    n = max(egraph_size, 500)
    # SaturationParams(
    #     scheduler=Metatheory.Schedulers.BackoffScheduler,
    #     schedulerparams=(n, 5),
    #     timeout=8, eclasslimit=n, enodelimit=4n, matchlimit=4n)
    SaturationParams(
        threaded=true,
        scheduler=Metatheory.Schedulers.SimpleScheduler,
        timeout=2, eclasslimit=0, enodelimit=0, matchlimit=0,
    )
end