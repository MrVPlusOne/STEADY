"""
Use the syntax `result[pshape]` to get an iterator over all found programs
of the given [`PShape`](@ref).
"""
@kwdef(
mutable struct EnumerationResult
    programs::Dict{PShape, Dict{Int, Dict{PUnit, Set{TAST}}}}
    pruned::Vector{@NamedTuple{pruned::TAST, reason::Any}}
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

prune!(r::EnumerationResult, (; pruned, reason)) = begin
    p = pruned
    (; shape, unit) = p.type
    unit_map = r.programs[shape][ast_size(p)]
    ps = unit_map[unit]
    @assert p ∈ ps "$p is already pruned."
    delete!(ps, p)
    isempty(ps) && delete!(unit_map, unit)
    r.n_deleted += 1
    push!(r.pruned, (; pruned, reason))
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