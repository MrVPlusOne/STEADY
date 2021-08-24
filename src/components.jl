
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
    @unpack arg_shapes, result_shape, result_unit = env
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
end

ComponentEnv() = ComponentEnv(Dict(), Dict())
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
    push!(env, :abs, abs, [ℝ] => ℝ, identity)
    push!(env, :square, square, [ℝ] => ℝ, u -> u^2)
    push!(env, :sqrt, sqrt ∘ abs, [ℝ] => ℝ, u -> u^(1//2))
end
square(x) = x^2

function components_transcendentals!(env::ComponentEnv)
    push_unitless!(env, :sin, sin)
    push_unitless!(env, :cos, cos)
    push_unitless!(env, :exp, exp)
    push_unitless!(env, :tanh, tanh)
end

function components_vec2!(env::ComponentEnv)
    push!(env, :R2, (mk_R2), [ℝ, ℝ] => ℝ2, signature_all_same) # constructor
    push!(env, :norm_R2, norm_R2, [ℝ2] => ℝ, identity)

    push!(env, :plus_R2, (+), [ℝ2, ℝ2] => ℝ2, signature_all_same)
    push!(env, :minus_R2, (-), [ℝ2, ℝ2] => ℝ2, signature_all_same)
    push!(env, :scale_R2, (*), [ℝ, ℝ2] => ℝ2, (*))
    push!(env, :rotate_R2, rotate2d, [ℝ, ℝ2] => ℝ2, (θ, v) -> isunitless(θ) ? v : nothing)
    push!(env, :neg_R2, (-), [ℝ2] => ℝ2, identity)
end
mk_R2(x, y) = @SVector [x, y]
norm_R2(x) = sqrt(x[1]^2 + x[2]^2)