# The DSL for physical dimenion-informed expressions. 

"""
The shape of a type. e.g. Scalar or Vector.
"""
struct PShape
    name::Symbol
end


"""
A physical unit of the form `l1^p1 l2^p2 ...` where `l`s are unit labels 
and `p`s are rational powers.

# Examples
```jldoctest
julia> PUnit(:L=>1, :T=>1)
T^1 L^1
julia> PUnit(:L=>1, :T=>1) * PUnit(:L=>2, :M=>1//2)
T^1 M^1/2 L^3
julia> PUnit(:L=>2, :M=>1//2) ^ 2
M^1 L^4
julia> PUnit(:L=>1, :T=>1) * unitless
T^1 L^1
julia> PUnit(:L=>2, :M=>1//2) / PUnit(:L=>2, :M=>1//2)
unitless
```
"""
@auto_hash_equals(
struct PUnit
    impl::Dict{Symbol, Rational{Int}}

    PUnit(dict::Dict) = begin
        for (k, v) in dict
            if v == 0
                delete!(dict, k)
            end
        end
        new(dict)
    end
end)
PUnit(pairs::Pair...) = PUnit(Dict(pairs...))

isunitless(u::PUnit) = isempty(u.impl)
const unitless = PUnit()

print_pretty_rational(io::IO, x::Rational) = begin
    show(io, numerator(x))
    if denominator(x) != 1
        print(io, "/")
        show(io, denominator(x))
    end
end

Base.show(io::IO, v::PUnit) = begin
    if isunitless(v) 
        print(io, "unitless")
    else 
        join(io, 
            ("$k^$(sprint(print_pretty_rational, v))" for (k, v) in v.impl),
            " ",
        )
    end
end

function Base.:*(u1::PUnit, u2::PUnit)
    r = copy(u1.impl)
    for (k, v) in u2.impl
        r[k] = get(r, k, 0) + v
    end
    PUnit(r)
end

function Base.:/(u1::PUnit, u2::PUnit)
    r = copy(u1.impl)
    for (k, v) in u2.impl
        r[k] = get(r, k, 0) - v
    end
    PUnit(r)
end

function Base.:^(u1::PUnit, n::Rational)
    r = copy(u1.impl)
    for (k, v) in r
        r[k] = v * n
    end
    PUnit(r)
end
Base.:^(u1::PUnit, n::Integer)=u1^Rational(n, 1)

"""
A physical type with both a name and a unit.
"""
struct PType
    shape::PShape
    unit::PUnit
end

Base.show(io::IO, v::PType) = begin
    @unpack shape, unit = v
    print(io, "$(shape.name){$unit}")
end

"""
Typed AST for numerical expressions.
"""
abstract type TAST end

struct Var <: TAST
    name::Symbol
    type::PType
end

Base.show(io::IO, v::Var) = begin
    @unpack name, type = v
    print(io, "$name::$type")
end

@auto_hash_equals(
struct Call <: TAST
    f::Symbol
    args::Vector{TAST}
    type::PType
end)

Base.show(io::IO, v::Call) = begin
    @unpack f, args = v
    arg_list = join(args, ", ")
    print(io, "$f($arg_list)")
end

