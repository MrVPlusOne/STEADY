"""
    return_type(num_type::Type) -> Type
"""
struct CompiledFunc{return_type, F} <: Function
    f::F
    ast::TAST
    julia::Expr
end

(cf::CompiledFunc{return_type})(args::NamedTuple) where return_type = begin
    num_type = promote_numbers_type(args)
    @assert isconcretetype(num_type) "could not infer a concrete number type for \
            the arguments $args"
    R = return_type(num_type)
    convert(R, cf.f(args))::R
end

Base.show(io::IO, ::Type{<:CompiledFunc}) = print(io, "CompiledFunc{...}")

function Base.show(io::IO, @nospecialize cf::CompiledFunc) 
    (; ast, julia) = cf
    print(io, "CompiledFunction(`$ast`)")
end

function Base.show(io::IO, mime::MIME"text/plain", @nospecialize cf::CompiledFunc) 
    io = IOIndents.IOIndent(io)
    (; ast, julia) = cf
    println(io, "--- CompiledFunction ---")
    println(io, "AST:", Indent())
    println(io, ast, Dedent())
    println(io, "Julia Code:", Indent())
    println(io, repr(mime, julia), Dedent())
    println(io, "--- End ---")
end

"""
Used to cache the compilation result to speed up synthesis and avoid memory leak 
when synthesis is run multiple times.
"""
const compile_cache = Dict{Expr, RuntimeGeneratedFunction}()
const compile_cache_lock = ReentrantLock()

macro with_type(e, ty)
    quote  
        tv = $(esc(ty))
        v = $(esc(e))
        @assert(v isa tv, "$e returns value $v, which is not of type $tv.")
        v
    end
end

"""
Compiles a `TAST` expression into the corresponding julia function that can be 
efficiently executed.

When `hide_type=true`, the returned `CompiledFunc` will use a generic `Function` instead 
of a more precised function type in its type signature. This can lead to slightly worse 
performance but can typically greatly improve Julia's compilation time (by avoiding 
excessive specialization).

Implemented using `RuntimeGeneratedFunctions.jl`.
"""
function compile(
    prog::TAST, shape_env::ShapeEnv, comp_env::ComponentEnv; hide_type=false
)::CompiledFunc
    function compile_body(v::Var)
        e = Expr(:(.), :args, QuoteNode(v.name))
        rtype = shape_env.type_annots[v.type.shape]
        :($e::$rtype)
    end
    function compile_body(call::Call)
        local f = comp_env.impl_dict[call.f]
        local args = compile_body.(call.args)
        local rtype = shape_env.type_annots[call.type.shape]
        local e = Expr(:call, f, args...)
        :($e::$rtype)
    end

    body_ex = compile_body(prog)::Expr
    f_ex = :(args -> $body_ex)
    rgf = @lock compile_cache_lock get(compile_cache, f_ex, nothing)

    if rgf === nothing
        rgf = @RuntimeGeneratedFunction(f_ex)
        @lock compile_cache_lock begin
            compile_cache[f_ex] = rgf
        end
    end
    
    rtype = shape_env.return_type[prog.type.shape]
    ftype = hide_type ? Function : typeof(rgf)
    CompiledFunc{rtype, ftype}(
        rgf, prog, body_ex,
    )
end

"""
Hide the type of the wrapped function. Useful for avoiding expensive compilation 
caused by run-time generated functions.
"""
struct WrappedFunc <: Function
    core::Function
end

(cf::WrappedFunc)(args...) = cf.core(args...)

"""
```jldoctest
julia> TypedFunc{return_type_R2}(sum)((a=@SVector[1.0,2.0], b=@SVector[3,4]))
2-element StaticArrays.SVector{2, Float64} with indices SOneTo(2):
 4.0
 6.0
 ```
"""
struct TypedFunc{return_type} <: Function
    core::Function
end

(cf::TypedFunc{return_type})(args::NamedTuple) where return_type = begin
    num_type = promote_numbers_type(args)
    @assert isconcretetype(num_type) "could not infer a concrete number type for \
            the arguments $args"
    R = return_type(num_type)
    convert(R, cf.core(args))::R
end

return_type_R2(num) = SVector{2, num}
return_type_R(num) = num
return_type_void(num) = Tuple{}

"""
Compiles a `TAST` expression into the corresponding julia function that can be 
efficiently executed.

Implemented using `RuntimeGeneratedFunctions.jl`.
"""
function compile_cached!(
    prog::TAST, shape_env::ShapeEnv, comp_env::ComponentEnv, 
    cache::Dict{TAST, CompiledFunc},
)::CompiledFunc
    function compile_body(v::Var)
        e = Expr(:(.), :args, QuoteNode(v.name))
        rtype = shape_env[v.type]
        :($e::$rtype)
    end
    function compile_body(call::Call)
        local f = comp_env.impl_dict[call.f]
        local sub_calls = map(call.args) do arg
            if arg isa Var
                compile_body(arg)
            else
                cf = compile_cached!(arg, shape_env, comp_env, cache)
                Expr(:call, cf.f, :args)
            end
        end
        local rtype = shape_env[call.type]
        local r = Expr(:call, f, sub_calls...)
        :($r::$rtype)
    end

    get!(cache, prog) do
        body_ex = compile_body(prog)
        f_ex = :(args -> $body_ex)
        f = @RuntimeGeneratedFunction f_ex
        CompiledFunc(prog, body_ex, f)
    end
end

function compile_interpreted(
    prog::TAST, shape_env::ShapeEnv, comp_env::ComponentEnv
)::CompiledFunc
    function execute(ast::TAST, args::NamedTuple)
        rec(v::Var) = getfield(args, v.name)
        rec(call::Call) = begin
            local f = comp_env.impl_dict[call.f]
            local as = rec.(call.args)
            f(as...)
        end
        rec(ast)
    end

    CompiledFunc(prog, :(interpreted()), args -> execute(prog, args))
end