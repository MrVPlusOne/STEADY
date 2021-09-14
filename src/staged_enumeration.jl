function enumerate_types(
    env::ComponentEnv, inputs::Set{PType}, outputs::Set{PType}, max_size,
)::@NamedTuple{types_needed::Set{Tuple{Int, PType}}, n_created::Int}
    found = Dict{PShape, Dict{Int, Set{PUnit}}}()
    enum = _TypeEnumeration(found, Dict(), 0)

    foreach(inputs) do ty
        insert!(enum, ty, 1, Set{PType}())
    end

    # first, enumerate all types under the size constraint
    signatures = env.signatures
    @progress name="enumerate_all_types" for size in 2:max_size 
        for (_, sig) in signatures
            arg_shapes = sig.arg_shapes
            any(!haskey(found, s) for s in arg_shapes) && continue # skip unrealizable
            sizes_for_arg = [keys(found[s]) for s in arg_shapes]
            # iterate over all possible size combinations s.t. the sum = size-1
            for arg_sizes in size_combinations(length(arg_shapes), sizes_for_arg, size-1)
                arg_dicts = [found[arg_shapes[i]][s] for (i, s) in enumerate(arg_sizes)]
                # iterate over all unit combinations
                for arg_units in Iterators.product((d for d in arg_dicts)...)
                    runit = sig.result_unit(arg_units...)
                    (runit === nothing) && continue
                    parent_types = Set(
                        PType(shape, unit) 
                        for (shape, unit) in zip(arg_shapes, arg_units)
                    )
                    rtype = PType(sig.result_shape, runit::PUnit)
                    insert!(enum, rtype, size, parent_types)
                end
            end
        end
    end


    # then, remove all the types not needed for `outputs`
    # entry (s, t) means type t is needed at ast size s
    types_needed = Set{Tuple{Int, PType}}()
    for size in max_size:-1:1
        for (shape, dict) in enum.types
            for unit in get(() -> Set{PUnit}(), dict, size)
                type = PType(shape, unit)
                needed = (type ∈ outputs) || let 
                    children = get(() -> Set{PType}(), enum.children_dict, (type, size))
                    any(c -> (size+1, c) ∈ types_needed, children)
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
    types::Dict{PShape, Dict{Int, Set{PUnit}}}
    children_dict::Dict{Tuple{PType, Int}, Set{PType}}
    n_created::Int
end

Base.getindex(r::_TypeEnumeration, (;shape, size)) = begin
    nested_get!(r.types, shape => size) do
        Set{PUnit}()
    end
end

Base.insert!(r::_TypeEnumeration, type::PType, size::Int, parents::Set{PType}) = begin
    (; shape, unit) = type
    push!(r[(;shape, size)], unit)
    r.n_created += 1

    foreach(parents) do p
        children = get!(r.children_dict, (p, size-1)) do
            Set{PType}()
        end
        push!(children, type)
    end

    r
end