# The DSL for physical dimenion-informed expressions. 

export PShape
"""
The shape of a type. e.g. Scalar or Vector.
"""
struct PShape
    name::Symbol
end


export PUnit, isunitless
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
@auto_hash_equals(struct PUnit
    impl::Dict{Symbol,Rational{Int}}

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

const unitless = PUnit()
isunitless(u::PUnit) = isempty(u.impl)

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
        join(io, ("$k^$(sprint(print_pretty_rational, v))" for (k, v) in v.impl), " ")
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
Base.:^(u1::PUnit, n::Integer) = u1^Rational(n, 1)

export PUnits
module PUnits
using ..SEDL: PUnit, unitless

const Length = PUnit(:L => 1)
const Time = PUnit(:T => 1)
const Mass = PUnit(:M => 1)

const Speed = Length / Time
const Acceleration = Length / Time^2
const Force = Mass * Acceleration
const Torque = Force * Length

const Angle = unitless
const AngularSpeed = Angle / Time
const AngularAcceleration = AngularSpeed / Time
end

export PType
"""
A physical type with both a name and a unit.
"""
@auto_hash_equals(struct PType
    shape::PShape
    unit::PUnit
end)

(s::PShape)(unit::PUnit) = PType(s, unit)

Base.show(io::IO, v::PType) = begin
    (; shape, unit) = v
    print(io, "$(shape.name){$unit}")
end

export TAST, Var, Call
"""
Typed AST for numerical expressions.
"""
abstract type TAST end

@auto_hash_equals(struct Const <: TAST
    value::Any
    type::PType
end)

Base.show(io::IO, v::Const) = begin
    print(io, v.value)
end

Base.show(io::IO, ::MIME"text/plain", v::Const) = begin
    print(io, v.value, "::", v.type)
end

struct Var <: TAST
    name::Symbol
    type::PType
end

Var(name::Symbol, shape::PShape, unit::Pair...) = Var(name, shape, PUnit(unit...))
Var(name::Symbol, shape::PShape, unit::PUnit) = Var(name, PType(shape, unit))

Base.hash(v::Var, h::UInt) = hash(v.name, h)
Base.:(==)(v1::Var, v2::Var) = v1.name === v2.name

Base.show(io::IO, v::Var) = begin
    print(io, v.name)
end

Base.show(io::IO, ::MIME"text/plain", v::Var) = begin
    print(io, v.name, "::", v.type)
end

@auto_hash_equals(struct Call <: TAST
    f::Symbol
    args::Tuple{Vararg{TAST}}
    type::PType
end)

Base.show(io::IO, v::Call) = begin
    print(io, string(julia_expr(v)))
end

julia_expr(v::Var) = v.name
julia_expr(c::Const) = c.value
julia_expr(c::Call) = begin
    Expr(:call, c.f, julia_expr.(c.args)...)
end

"""
    traverse(f, e::TAST)

recursively visit all subexpressions in `e` in depth-first order and applies `f` 
to each of them.
"""
function traverse end

traverse(f, v::Var) = f(v)
function traverse(f, c::Call)
    f(c)
    foreach(c.args) do a
        traverse(f, a)
    end
end

function ast_size(e::TAST)
    s = 0
    traverse(e -> s += 1, e)
    s
end

export ShapeEnv
struct ShapeEnv
    type_annots::Dict{PShape,Type}
    return_type::Dict{PShape,Function}
end

export ???, ???2, ???env, derivative
## === Commonly used type definitions ===
const Void = PShape(:Void)
const ??? = PShape(:???)
const ???2 = PShape(:?????)

???env() = begin
    ShapeEnv(
        Dict(Void => Tuple{}, ??? => Real, ???2 => SVector{2,<:Real}),
        Dict(Void => return_type_void, ??? => return_type_R, ???2 => return_type_R2),
    )
end


derivative(v::Symbol) = Symbol(v, "???")
derivative(v::Var, t::PUnit=PUnits.Time) =
    Var(derivative(v.name), PType(v.type.shape, v.type.unit / t))

##-----------------------------------------------------------
# AST construction helpers
Base.:+(e1::TAST, e2::TAST) = begin
    @smart_assert e1.type == e2.type
    Call(:+, (e1, e2), e1.type)
end

Base.:-(e1::TAST, e2::TAST) = begin
    @smart_assert e1.type == e2.type
    Call(:-, (e1, e2), e1.type)
end

Base.:-(e1::TAST) = begin
    Call(:neg, (e1,), e1.type)
end

Base.:*(e1::TAST, e2::TAST) = begin
    @smart_assert e1.type.shape == e2.type.shape
    Call(:*, (e1, e2), PType(e1.type.shape, e1.type.unit * e2.type.unit))
end

Base.:/(e1::TAST, e2::TAST) = begin
    @smart_assert e1.type.shape == e2.type.shape
    Call(:/, (e1, e2), PType(e1.type.shape, e1.type.unit / e2.type.unit))
end
