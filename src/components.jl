
"""
The type signature of a component function.
"""
struct FuncSignature{F}
    arg_shapes::Vector{PShape}
    result_shape::PShape
    "(arg_units...) -> (result_unit | nothing)"
    result_unit::F # TODO: replace this with a constraint graph to improve efficiency
end

FuncSignature(shape_sig::Pair{Vector{PShape}, PShape}, unit_sig::Function) =
    FuncSignature(shape_sig[1], shape_sig[2], unit_sig)

Base.show(io::IO, env::FuncSignature) = begin
    (; arg_shapes, result_shape, result_unit) = env
    shape_sig = (s -> s.name).(arg_shapes) => result_shape.name
    print(io, "(shape=$(shape_sig)), unit=$result_unit)")
end

"""
Arguments and the result should all be of the same unit.
"""
signature_all_same(arg_units::PUnit...) = begin
    u = arg_units[1]
    any(u != x for x in arg_units) ? nothing : u
end


"""
A environment that stores the set of component functions available for synthesis.
"""
struct ComponentEnv
    "Julia functions that implement the corresponding components."
    impl_dict::Dict{Symbol, Function}
    signatures::Dict{Symbol, FuncSignature}
    rules::Vector{AbstractRule}
end

ComponentEnv() = ComponentEnv(Dict(), Dict(), [])
Base.show(io::IO, env::ComponentEnv) = begin
    print(io, "ComponentEnv(components=$(keys(env.impl_dict)))")
end

Base.show(io::IO ,::MIME"text/plain", env::ComponentEnv) = begin
    n = length(env.impl_dict)
    println(io, "ComponentEnv with $n components:")
    for (name, sig) in env.signatures
        println(io, "  $name: $sig")
    end
end


Base.insert!(d::Dict, k, v) = d[k] = v

function Base.push!(
    env::ComponentEnv, 
    name::Symbol, 
    impl::Function, 
    shape_sig::Pair{Vector{PShape}, PShape},
    unit_sig::Function,
)::ComponentEnv
    @assert !(name in keys(env.impl_dict)) "Component '$name' already present in the environment."
    insert!(env.impl_dict, name, impl)
    insert!(env.signatures, name, FuncSignature(shape_sig, unit_sig))
    env
end

function push_unitless!(
    env::ComponentEnv, 
    name::Symbol, 
    impl::Function, 
    shape_sig::Pair{Vector{PShape}, PShape} = [ℝ] => ℝ,
)::ComponentEnv
    unitless_sig(us...) = all(isunitless.(us)) ? unitless : nothing
    push!(env, name, impl, shape_sig, unitless_sig)
end

function components_scalar_arithmatic!(env::ComponentEnv)
    push!(env, :(+), (+), [ℝ, ℝ] => ℝ, signature_all_same)
    push!(env, :(-), (-), [ℝ, ℝ] => ℝ, signature_all_same)
    push!(env, :(*), (*), [ℝ, ℝ] => ℝ, (*))
    push!(env, :(/), (/), [ℝ, ℝ] => ℝ, (/))
    push!(env, :(neg), (-), [ℝ] => ℝ, identity)
    push!(env, :(reciprocal), x -> 1 / x, [ℝ] => ℝ, u -> unitless / u)
    push!(env, :abs, abs, [ℝ] => ℝ, identity)
    push!(env, :square, square, [ℝ] => ℝ, u -> u^2)
    push!(env, :sqrt, sqrt ∘ abs, [ℝ] => ℝ, u -> u^(1//2))

    append!(env.rules, scalar_arithmatic_rules())
end
square(x) = x^2

Metatheory.Library.commutative_group

function scalar_arithmatic_rules()
    monoids = commutative_monoid(:(*), 1) ∪ commutative_monoid(:(+), 0)
    others = @theory begin
        neg(a) == 0 - a
        neg(neg(a)) => a
        a - a => 0
        a + neg(b) == a - b 
        neg(a) + neg(b) == neg(a + b)
        neg(a) * b == neg(a * b)

        a / a => 1
        reciprocal(a) == 1 / a
        reciprocal(reciprocal(a)) => a
        a * reciprocal(b) == a / b
        reciprocal(a) * reciprocal(b) == reciprocal(a * b)

        a * b + a * c => a * (b + c)

        abs(abs(a)) => abs(a)
        abs(neg(a)) => abs(a)
        abs(a) * abs(b) == abs(a * b)
        sqrt(a) * sqrt(b) == sqrt(a * b)
        x * x == square(x)
        sqrt(square(x)) => abs(x)
        square(sqrt(x)) => abs(x)
        sqrt(abs(x)) => sqrt(x)
    end
    monoids ∪ others
end

function components_transcendentals!(env::ComponentEnv)
    push_unitless!(env, :sin, sin)
    push_unitless!(env, :cos, cos)
    push_unitless!(env, :exp, exp)
    push_unitless!(env, :tanh, tanh)

    rules = @theory begin
        sin(neg(x)) => neg(sin(x))
        cos(neg(x)) => cos(x)
        exp(a) * exp(b) => exp(a + b)
        tanh(neg(x)) => neg(tanh(x))
    end
    append!(env.rules, rules)
end

function components_vec2!(env::ComponentEnv)
    push!(env, :R2, (mk_R2), [ℝ, ℝ] => ℝ2, signature_all_same) # constructor
    push!(env, :norm_R2, norm_R2, [ℝ2] => ℝ, identity)

    push!(env, :plus_R2, (+), [ℝ2, ℝ2] => ℝ2, signature_all_same)
    push!(env, :minus_R2, (-), [ℝ2, ℝ2] => ℝ2, signature_all_same)
    push!(env, :scale_R2, (*), [ℝ, ℝ2] => ℝ2, (*))
    push!(env, :rotate_R2, rotate2d, [ℝ, ℝ2] => ℝ2, (θ, v) -> isunitless(θ) ? v : nothing)
    push!(env, :neg_R2, (-), [ℝ2] => ℝ2, identity)

    append!(env.rules, vec2_rules())
end

vec2_rules() = begin
    monoids = commutative_monoid(:plus_R2, :R2_0)
    others = @theory begin
        neg_R2(a) == minus_R2(:R2_0, a)
        neg_R2(neg_R2(a)) => a
        minus_R2(a, a) => :R2_0
        plus_R2(a, neg_R2(b)) == minus_R2(a, b)
        plus_R2(neg_R2(a), neg_R2(b)) == neg_R2(plus_R2(a, b))
        scale_R2(s, neg_R2(a)) == scale_R2(neg(s), a)
        scale_R2(s, neg_R2(a)) == neg_R2(scale_R2(s, a))
        neg_R2(R2(a, b)) == R2(neg(a), neg(b))

        # rotate_R2(b, rotate_R2(a, v)) => rotate_R2(a+b, v)
        rotate_R2(θ, neg_R2(a)) == neg_R2(rotate_R2(θ, a))

        norm_R2(neg_R2(v)) => norm_R2(v)
        norm_R2(rotate_R2(θ, v)) => norm_R2(v)
        abs(s) * norm_R2(v) => norm_R2(scale_R2(s, v))
        abs(norm_R2(v)) => norm_R2(v)
    end
    monoids ∪ others
end

mk_R2(x, y) = @SVector [x, y]
norm_R2(x) = sqrt(x[1]^2 + x[2]^2)