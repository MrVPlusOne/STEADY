
"""
The type signature of a component function.
"""
struct FuncSignature{F}
    arg_shapes::Vector{PShape}
    result_shape::PShape
    "(arg_units...) -> (result_unit | nothing)"
    result_unit::F # TODO: replace this with a constraint graph to improve efficiency
end

FuncSignature(shape_sig::Pair{Vector{<:PShape},PShape}, unit_sig::Function) =
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
Use `add_component!` and `push!` to add new `ComponentFunc` into teh environment.

Note that components added earlier are preferred during the synthesis.
"""
struct ComponentEnv
    names::Vector{Symbol}
    "Julia functions that implement the corresponding components."
    impl_dict::Dict{Symbol,Function}
    signatures::Dict{Symbol,FuncSignature}
    rules::Vector{AbstractRule}
end

ComponentEnv() = ComponentEnv([], Dict(), Dict(), [])
Base.show(io::IO, env::ComponentEnv) = begin
    print(io, "ComponentEnv(components=$(keys(env.impl_dict)))")
end

Base.show(io::IO, ::MIME"text/plain", env::ComponentEnv) = begin
    n = length(env.impl_dict)
    println(io, "ComponentEnv with $n components:")
    for name in env.names
        sig = env.signatures[name]
        println(io, "  $name: $sig")
    end
end

Base.insert!(d::Dict, k, v) = d[k] = v

struct ComponentFunc
    name::Symbol
    impl::Function
    shape_sig::Pair{Vector{<:PShape},PShape}
    unit_sig::Function
end

(cf::ComponentFunc)(args::TAST...) = begin
    r_shape = cf.shape_sig[2]
    r_unit = cf.unit_sig(map(x -> x.type.unit, args)...)
    Call(cf.name, args, PType(r_shape, r_unit))
end

get_component(env::ComponentEnv, name::Symbol) = begin
    sig = env.signatures[name]
    shape_sig = sig.arg_shapes => sig.result_shape
    ComponentFunc(name, env.impl_dict[name], shape_sig, sig.result_unit)
end

function add_component!(env::ComponentEnv, comp::ComponentFunc)::ComponentEnv
    (; name, impl, shape_sig, unit_sig) = comp
    @assert !(name in keys(env.impl_dict)) "Component '$name' already present in the environment."
    push!(env.names, name)
    insert!(env.impl_dict, name, impl)
    insert!(env.signatures, name, FuncSignature(shape_sig, unit_sig))
    env
end

function Base.push!(env::ComponentEnv, comps::ComponentFunc...)
    foreach(comps) do c
        add_component!(env, c)
    end
    env
end

function add_unitless_comp!(
    env::ComponentEnv,
    name::Symbol,
    impl::Function;
    shape_sig::Pair{Vector{PShape},PShape}=[ℝ] => ℝ,
)::ComponentEnv
    push!(env, ComponentFunc(name, impl, shape_sig, signature_unitless))
end
signature_unitless(us...) = all(isunitless.(us)) ? unitless : nothing


function components_void!(env::ComponentEnv)
    push!(env, ComponentFunc(:void, void, PShape[] => Void, signature_unitless))
end
void() = ()

function components_scalar_arithmatic!(env::ComponentEnv; can_grow=true)
    push!(
        env,
        ComponentFunc(:(+), (+), [ℝ, ℝ] => ℝ, signature_all_same),
        ComponentFunc(:(-), (-), [ℝ, ℝ] => ℝ, signature_all_same),
        ComponentFunc(:(*), (*), [ℝ, ℝ] => ℝ, (*)),
        ComponentFunc(:(/), (/), [ℝ, ℝ] => ℝ, (/)),
        ComponentFunc(:zero, zero, [ℝ] => ℝ, identity),
        ComponentFunc(:neg, (-), [ℝ] => ℝ, identity),
        ComponentFunc(:reciprocal, x -> 1 / x, [ℝ] => ℝ, u -> unitless / u),
        ComponentFunc(:abs, abs, [ℝ] => ℝ, identity),
        ComponentFunc(:square, square, [ℝ] => ℝ, u -> u^2),
        ComponentFunc(:sqrt, sqrt ∘ abs, [ℝ] => ℝ, u -> u^(1//2)),
    )

    scalar_arithmatic_rules!(env.rules; can_grow)
    env
end
square(x) = x^2

Metatheory.Library.commutative_group

function scalar_arithmatic_rules!(rules; can_grow)
    append!(rules, commutative_monoid(:(*), 1))
    append!(rules, commutative_monoid(:(+), 0))

    # TODO: automatically check non-growness
    non_grow = @theory begin
        0 - a => neg(a)
        neg(neg(a)) => a
        a - a => 0
        a + neg(b) => a - b
        neg(a) + neg(b) => neg(a + b)
        neg(a) * b == neg(a * b)

        # a / a => 1  # TODO: need a better way to apply this
        1 / a => reciprocal(a)
        reciprocal(reciprocal(a)) => a
        a * reciprocal(b) => a / b
        reciprocal(a) * reciprocal(b) => reciprocal(a * b)

        a * b + a * c => a * (b + c)

        abs(0) => 0
        abs(abs(a)) => abs(a)
        abs(neg(a)) => abs(a)
        abs(a) * abs(b) => abs(a * b)
        sqrt(a) * sqrt(b) => sqrt(a * b)
        x * x => square(x)
        sqrt(square(x)) => abs(x)
        square(sqrt(x)) => abs(x)
        sqrt(abs(x)) => sqrt(x)
    end
    append!(rules, non_grow)

    grow = @theory begin
        neg(a) => 0 - a
        a - b => a + neg(b)
        neg(a + b) => neg(a) + neg(b)
        reciprocal(a) => 1 / a
        a / b => a * reciprocal(b)
        reciprocal(a * b) => reciprocal(a) * reciprocal(b)
        abs(a * b) => abs(a) * abs(b)
        sqrt(a * b) => sqrt(a) * sqrt(b)
        square(x) => x * x
    end

    can_grow && append!(rules, grow)
    rules
end

function components_special_functions!(env::ComponentEnv; can_grow=true)
    add_unitless_comp!(env, :sin, sin)
    add_unitless_comp!(env, :cos, cos)
    add_unitless_comp!(env, :tan, tan)
    # add_unitless_comp!(env, :exp, exp)
    # add_unitless_comp!(env, :log, log)
    add_unitless_comp!(env, :tanh, tanh)

    push!(env, ComponentFunc(:friction, friction, [ℝ, ℝ] => ℝ, signature_all_same))

    non_grow = @theory begin
        sin(neg(x)) == neg(sin(x))
        cos(neg(x)) => cos(x)
        exp(a) * exp(b) => exp(a + b)
        tanh(neg(x)) == neg(tanh(x))
    end
    grow = @theory begin
        exp(a + b) => exp(a) * exp(b)
    end
    append!(env.rules, non_grow)
    can_grow && append!(env.rules, grow)
    env
end

"""
Almost flat when `abs(x) < f` but grows linearly otherwise.
"""
friction(x, f; α=1e-4) = begin
    f = abs(f)
    if abs(x) < f
        α * x
    else
        x - f * sign(x) * (1 - α)
    end
end

function components_vec2!(env::ComponentEnv; can_grow=true)
    push!(
        env,
        ComponentFunc(:R2, (mk_R2), [ℝ, ℝ] => ℝ2, signature_all_same), # constructor
        ComponentFunc(:norm_R2, norm_R2, [ℝ2] => ℝ, identity),
        ComponentFunc(:plus_R2, (+), [ℝ2, ℝ2] => ℝ2, signature_all_same),
        ComponentFunc(:minus_R2, (-), [ℝ2, ℝ2] => ℝ2, signature_all_same),
        ComponentFunc(:scale_R2, (*), [ℝ, ℝ2] => ℝ2, (*)),
        ComponentFunc(
            :rotate_R2, rotate2d, [ℝ, ℝ2] => ℝ2, (θ, v) -> isunitless(θ) ? v : nothing
        ),
        ComponentFunc(:neg_R2, (-), [ℝ2] => ℝ2, identity),
        ComponentFunc(:cross_R2, cross_R2, [ℝ2, ℝ2] => ℝ, (*)),
    )

    vec2_rules!(env.rules; can_grow)
    env
end

vec2_rules!(rules; can_grow) = begin
    append!(rules, commutative_monoid(:plus_R2, :R2_0))
    non_grows = @theory begin
        minus_R2(:R2_0, a) => neg_R2(a)
        neg_R2(neg_R2(a)) => a
        minus_R2(a, a) => :R2_0
        plus_R2(a, neg_R2(b)) => minus_R2(a, b)
        plus_R2(neg_R2(a), neg_R2(b)) => neg_R2(plus_R2(a, b))
        scale_R2(s, neg_R2(a)) == scale_R2(neg(s), a)
        scale_R2(s, neg_R2(a)) == neg_R2(scale_R2(s, a))
        R2(neg(a), neg(b)) => neg_R2(R2(a, b))
        neg(cross_R2(a, b)) => cross_R2(b, a)

        # rotate_R2(b, rotate_R2(a, v)) => rotate_R2(a+b, v)
        rotate_R2(θ, neg_R2(a)) == neg_R2(rotate_R2(θ, a))
        rotate_R2(0, a) => a

        norm_R2(:R2_0) => 0
        norm_R2(neg_R2(v)) => norm_R2(v)
        norm_R2(rotate_R2(θ, v)) => norm_R2(v)
        abs(s) * norm_R2(v) => norm_R2(scale_R2(s, v))
        abs(norm_R2(v)) => norm_R2(v)
    end
    grow = @theory begin
        neg_R2(a) => minus_R2(:R2_0, a)
        neg_R2(R2(a, b)) => R2(neg(a), neg(b))
        minus_R2(a, b) => plus_R2(a, neg_R2(b))
        neg_R2(plus_R2(a, b)) => plus_R2(neg_R2(a), neg_R2(b))
    end

    append!(rules, non_grows)
    can_grow && append!(rules, grow)
end

mk_R2(x, y) = @SVector [x, y]
# this makes sure that gradient exists when norm = 0.
norm_R2(x) = sqrt(x[1]^2 + x[2]^2 + eps(x[1]))
cross_R2(x, y) = x[1] * y[2] - x[2] * y[1]
dir_R2(θ) = @SVector [cos(θ), sin(θ)]
unit_R2(v) = v ./ norm_R2(v)
project_R2(v, dir) = v'dir * dir