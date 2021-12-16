using ProgressMeter: Progress, next!, @showprogress, progress_pmap

include("enumeration_result.jl")
include("pruners.jl")

"""
Enumerate all program terms up until some AST size constraint `max_size`,
optionally performing type-driven and other forms of pruning.  
## Arguments
- `max_size::Integer`: programs with [`ast_size`](@ref) up to this value will be returned.
- `types_needed`: a set of `(size, type)` pairs specifying which types are need at 
with which AST size. Default to `nothing`, which means no type-driven pruning.
"""
function enumerate_terms(
    env::ComponentEnv,
    vars::Set{Var},
    max_size;
    types_needed::Union{Set{Tuple{Int,PType}},Nothing}=nothing,
    pruner::AbstractPruner=NoPruner(),
)::EnumerationResult
    function types_with_size(size)::Set{PType}
        r = Set{PType}()
        (types_needed === nothing) && return r
        foreach(types_needed) do (s, t)
            (s == size) && push!(r, t)
        end
        r
    end

    start_time = time()

    # stores all found programs, indexed by: shape -> ast_size -> unit
    found = Dict{PShape,Dict{Int,Dict{PUnit,Set{TAST}}}}()
    result = EnumerationResult(found, [], 0, 0, 0, 0)
    reset!(pruner)

    prune_single!(prog, size) = begin
        result.pruning_time += @elapsed begin
            reason = prune_immediately!(pruner, prog, size)
            if reason !== nothing
                prune!(result, (; pruned=prog, reason))
            end
        end
    end

    prune_batch(size; is_last) = begin
        result.pruning_time += @elapsed begin
            types_to_prune = types_with_size(size)
            for x in prune_iteration!(pruner, result, types_to_prune, size; is_last)
                prune!(result, x)
            end
        end
    end

    # size 1 programs consist of all variables
    foreach(vars) do v
        insert!(result, v, 1)
        prune_single!(v, 1)
    end

    # construct larger programs from smaller ones
    @showprogress desc = "bottom_up_enum" for size in 1:max_size
        for f in env.names
            sig = env.signatures[f]
            arg_shapes = sig.arg_shapes
            any(!haskey(found, s) for s in arg_shapes) && continue # skip unrealizable
            sizes_for_arg = [keys(found[s]) for s in arg_shapes]
            # iterate over all possible size combinations s.t. the sum = size-1
            for arg_sizes in size_combinations(length(arg_shapes), sizes_for_arg, size - 1)
                arg_dicts = [found[arg_shapes[i]][s] for (i, s) in enumerate(arg_sizes)]
                # iterate over all unit combinations
                for arg_units in Iterators.product((keys(d) for d in arg_dicts)...)
                    runit = sig.result_unit(arg_units...)
                    (runit === nothing) && continue # skip invalid combination
                    rtype = PType(sig.result_shape, runit::PUnit)
                    if types_needed !== nothing && (size, rtype) ∉ types_needed
                        continue  # this return type is not needed
                    end
                    arg_candidates = (d[u] for (u, d) in zip(arg_units, arg_dicts))
                    # iterate over all argument AST combinations
                    for args in Iterators.product(arg_candidates...)
                        prog = Call(f, args, rtype)
                        insert!(result, prog, size)
                        prune_single!(prog, size)
                    end
                end
            end
        end
        prune_batch(size; is_last=false)
    end

    prune_batch(size; is_last=true)
    result.total_time += time() - start_time
    result
end

function enumerate_types(
    env::ComponentEnv, inputs::Set{PType}, outputs::Set{PType}, max_size
)::@NamedTuple {types_needed::Set{Tuple{Int,PType}}, n_created::Int}
    found = Dict{PShape,Dict{Int,Set{PUnit}}}()
    enum = _TypeEnumeration(found, Dict(), 0)

    foreach(inputs) do ty
        insert!(enum, ty, 1, Set{PType}())
    end

    # first, enumerate all types under the size constraint
    @showprogress desc = "enumerate_all_types" for size in 2:max_size
        for name in env.names
            sig = env.signatures[name]
            arg_shapes = sig.arg_shapes
            any(!haskey(found, s) for s in arg_shapes) && continue # skip unrealizable
            sizes_for_arg = [keys(found[s]) for s in arg_shapes]
            # iterate over all possible size combinations s.t. the sum = size-1
            for arg_sizes in size_combinations(length(arg_shapes), sizes_for_arg, size - 1)
                arg_dicts = [found[arg_shapes[i]][s] for (i, s) in enumerate(arg_sizes)]
                # iterate over all unit combinations
                for arg_units in Iterators.product((d for d in arg_dicts)...)
                    runit = sig.result_unit(arg_units...)
                    (runit === nothing) && continue
                    parent_types = Set{PType}(
                        PType(shape, unit) for (shape, unit) in zip(arg_shapes, arg_units)
                    )
                    rtype = PType(sig.result_shape, runit::PUnit)
                    insert!(enum, rtype, size, parent_types)
                end
            end
        end
    end

    # then, remove all the types not needed for `outputs`
    # entry (s, t) means type t is needed at ast size s
    types_needed = Set{Tuple{Int,PType}}()
    for size in max_size:-1:1
        for (shape, dict) in enum.types
            for unit in get(() -> Set{PUnit}(), dict, size)
                type = PType(shape, unit)
                needed =
                    (type ∈ outputs) || let
                        children = get(() -> Set{PType}(), enum.children_dict, (type, size))
                        any(c -> (size + 1, c) ∈ types_needed, children)
                    end
                if needed
                    push!(types_needed, (size, type))
                end
            end
        end
    end
    (; types_needed, enum.n_created)
end

mutable struct _TypeEnumeration
    types::Dict{PShape,Dict{Int,Set{PUnit}}}
    children_dict::Dict{Tuple{PType,Int},Set{PType}}
    n_created::Int
end

Base.getindex(r::_TypeEnumeration, (; shape, size)) = begin
    nested_get!(r.types, shape => size) do
        Set{PUnit}()
    end
end

Base.insert!(r::_TypeEnumeration, type::PType, size::Int, parents::Set{PType}) = begin
    (; shape, unit) = type
    push!(r[(; shape, size)], unit)
    r.n_created += 1

    foreach(parents) do p
        children = get!(r.children_dict, (p, size - 1)) do
            Set{PType}()
        end
        push!(children, type)
    end

    r
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
            si_max = size_left - (n_args - i)
            si_sizes = filter(s -> s <= si_max, sizes_for_arg[i])
            Iterators.flatten(
                (push!(sizes, s1) for sizes in rec(i + 1, size_left - s1)) for
                s1 in si_sizes
            )
        end
    end
    (n_args == 0) && return (Int[],)
    (reverse!(v) for v in rec(1, total_size))
end

