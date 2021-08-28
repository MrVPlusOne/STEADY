"""
Use the syntax `result[pshape]` to get an iterator over all found programs
of the given [`PShape`](@ref).
"""
@kwdef(
mutable struct EnumerationResult
    programs::Dict{PShape, Dict{Int, Dict{PUnit, Set{TAST}}}}
    pruned::Vector{@NamedTuple{pruned::TAST, by::TAST}}
    n_created::Int
    n_deleted::Int
    total_time::Float64
    pruning_time::Float64
end)

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

Base.delete!(r::EnumerationResult, (; pruned, by)) = begin
    p = pruned
    (; shape, unit) = p.type
    unit_map = r.programs[shape][ast_size(p)]
    ps = unit_map[unit]
    @assert p ∈ ps "$p is already pruned."
    delete!(ps, p)
    isempty(ps) && delete!(unit_map, unit)
    r.n_deleted += 1
    push!(r.pruned, (; pruned, by))
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
    for (k, v) in pairs((; r.n_created, r.n_deleted, r.total_time, r.pruning_time))
        println(io, "$k: ", pretty_number(v))
    end
    print(io, Dedent())
    println(io, "Found programs: ", Indent())
    show_programs(io, r)
end

Base.show(io::IO, r::EnumerationResult) = begin
    stats = join(["$k: $v" for (k, v) in pairs(r.stats)], ", ")
    print(io, "EnumerationResult($stats)")
end

"""
Should implement the following functions
- [`prune_postprocess!`](@ref)
- [`prune_iteration!`](@ref)
"""
abstract type AbstractPruner end


"""
    prune_iteration!(::AbstractPruner, ::EnumerationResult, current_size; is_last) -> to_prune
prune programs by directly mutating the given `result`.
"""
function prune_iteration! end


"""
Perform bottom-up program enumeration.
## Arguments
- `max_size::Integer`: programs with [`ast_size`](@ref) up to this value will be returned.
"""
function bottom_up_enum(
    env::ComponentEnv, vars::Vector{Var}, max_size, pruner::AbstractPruner=NoPruner(),
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
            for x in prune_iteration!(pruner, result, size, is_last=false)
                delete!(result, x) 
            end
        end
    end

    result.pruning_time += @elapsed begin 
        for x in prune_iteration!(pruner, result, max_size, is_last=true)
            delete!(result, x) 
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

PruningClub{M, GID}(; to_expr::F, to_group::G) where {M, GID, F, G} = 
    PruningClub{M, GID, F, G}(to_expr, to_group, EGraph(:dummy), Dict()) 

function admit_members!(
    club::PruningClub{M, I}, new_members::AbstractVector{M},
    rules, compute_saturation_params,
) where {M, I}
    (; graph, groups, to_expr, to_group) = club
    member_classes = Dict(map(new_members) do m
        eclass, enode = addexpr!(graph, to_expr(m))
        m => eclass.id
    end)
    params = compute_saturation_params(graph.numclasses)
    report = saturate!(graph, rules, params)
    
    kept = M[]
    pruned = @NamedTuple{pruned::M, by::M}[]
    

    for (gid, group) in collect(keys(groups))
        for (cid, e) in group
            @assert Metatheory.find(graph, cid) == cid "failed as expected"
        end
        groups[gid] = Dict(Metatheory.find(graph, cid) => e for (cid, e) in group)
    end

    for m in new_members
        id::EClassId = Metatheory.find(graph, member_classes[m])
        # find m's corresponding group
        group = get!(groups, to_group(m)) do
            Dict{EClassId, M}()
        end
        exist_m = get(group, id, nothing)
        if exist_m === nothing
            # welcome to the club!
            group[id] = m
            push!(kept, m)
        else
            # pruned!
            push!(pruned, (pruned=m, by=exist_m))
        end
    end
    (; kept, pruned, report)
end


function prune_redundant(
    progs::AbstractVector{TAST}, rules, compute_saturation_params,
)
    club = PruningClub{TAST, PType}(; to_expr = to_expr, to_group = p -> p.type)
    admit_members!(club, progs, rules, compute_saturation_params)
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
prune_iteration!(::NoPruner, ::EnumerationResult, size; is_last) = []


"""
Build a new e-graph at every iteration.
"""
@kwdef(
struct RebootPruner{NtoS<:Function} <: AbstractPruner
    rules::Vector{AbstractRule}
    compute_saturation_params::NtoS=default_saturation_params
    reports::Vector=[]
    only_postprocess::Bool=false
end)

prune_iteration!(pruner::RebootPruner, result::EnumerationResult, size; is_last) = 
    if pruner.only_postprocess == is_last 
        (; rules, compute_saturation_params, reports) = pruner
        new_members = Iterators.flatten(result[s] for s in 1:size)

        kept, pruned, report = prune_redundant(
            collect(TAST, new_members), rules, compute_saturation_params)
        push!(reports, report)
        pruned
    else
        TAST[]
    end

"""
Prune expressions of each different type individually.
"""
@kwdef(
struct IndividualPruner{NtoS<:Function} <: AbstractPruner
    rules::Vector{AbstractRule}
    compute_saturation_params::NtoS=default_saturation_params
    reports::Vector=[]
end)


prune_iteration!(pruner::IndividualPruner, result::EnumerationResult, size; is_last) = 
    if !is_last
        (; rules, compute_saturation_params, reports) = pruner
        new_members = collect(TAST, result[])
        pruned_list=[]
        for elems in values(groupby(p -> p.type, new_members))
            kept, pruned, report = prune_redundant(
                collect(TAST, elems), rules, compute_saturation_params)
            push!(reports, report)
            push!(pruned_list, pruned)
        end

        pruned_list |> Iterators.flatten |> collect
    else
        TAST[]
    end

default_saturation_params(egraph_size) = begin
    n = max(egraph_size, 500)
    # SaturationParams(
    #     scheduler=Metatheory.Schedulers.BackoffScheduler,
    #     schedulerparams=(n, 5),
    #     timeout=8, eclasslimit=n, enodelimit=4n, matchlimit=4n)
    SaturationParams(
        scheduler=Metatheory.Schedulers.SimpleScheduler,
        timeout=2, eclasslimit=100n, enodelimit=100n, matchlimit=100n)
end