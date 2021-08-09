"""
Use the syntax `result[pshape]` to get an iterator over all found programs
of the given [`PShape`](@ref).
"""
struct EnumerationResult
    programs::Dict{PShape, Dict{Int, Dict{PUnit, Vector{TAST}}}}
    stats::Dict{String, Any}
end

Base.getindex(r::EnumerationResult, shape::PShape) = let
    d1 = r.programs[shape]
    (p 
    for size in sort(collect(keys(d1)))
    for d2 in values(d1[size]) 
    for p in d2)
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
    println(io, "==== Enumeration result ====")
    println(io, "Stats:", Indent())
    for (k, v) in r.stats
        println(io, "$k: $v")
    end
    print(io, Dedent())
    println(io, "Found programs: ", Indent())
    show_programs(io, r)
end

function bottom_up_enum(env::ComponentEnv, vars::Vector{Var}, max_size)::EnumerationResult
    @assert allunique(vars) "Duplicate variables in the input."

    "stores all found programs, indexed by: shape -> ast_size -> unit"
    found = Dict{PShape, Dict{Int, Dict{PUnit, Vector{TAST}}}}()
    n_found = 0

    function get_progs!(shape, size, unit)
        s1 = get!(found, shape) do ; Dict{Int, Dict{PUnit, Vector{TAST}}}() end
        s2 = get!(s1, size) do ; Dict{PUnit, Vector{TAST}}() end
        get!(s2, unit) do ; TAST[] end
    end

    function insert_prog!(size, prog::TAST)
        @unpack shape, unit = prog.type
        ps = get_progs!(shape, size, unit)
        push!(ps, prog)
        n_found += 1
    end

    start_time = time()
    # size 1 programs consist of all variables
    foreach(vars) do v
        insert_prog!(1, v)
    end

    signatures = env.signatures
    # construct larger programs from smaller ones
    for size in 2:max_size, (f, sig) in signatures
        arg_shapes = sig.arg_shapes
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
                    prog = Call(f, collect(args), PType(sig.result_shape, runit::PUnit))
                    insert_prog!(size, prog)
                end
            end
        end
    end

    stats = Dict("time_taken" => time() - start_time, "n_programs" => n_found)
    EnumerationResult(found, stats)
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