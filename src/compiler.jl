struct CompiledFunc{AST} <: Function
    ast::AST
    julia::Expr
    f::Function
end

(cf::CompiledFunc)(args::NamedTuple) = cf.f(args)

"""
Call function `f` on the input `x` and expect the result to be of type `T`.
This is equivalent to `f(x)::T` but can be more efficient when f is a 
[`CompiledFunc`](@ref).
"""
call_T(f::F, x, ::Type{T}) where {F, T} = f(x)::T
call_T(cf::CompiledFunc, x, ::Type{T}) where T = call_T(cf.f, x, T)


function Base.show(io::IO, @nospecialize cf::CompiledFunc) 
    (; ast, julia) = cf
    print(io, "CompiledFunction(ast=:($ast), julia=:($julia))")
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
const compile_cache = Dict{Expr, CompiledFunc}()
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

Implemented using `RuntimeGeneratedFunctions.jl`.
"""
function compile(
    prog::TAST, shape_env::ShapeEnv, comp_env::ComponentEnv; check_gradient=false
)::CompiledFunc
    function compile_body(v::Var)
        e = Expr(:(.), :args, QuoteNode(v.name))
        rtype = shape_env[v.type]
        :($e::$rtype)
    end
    function compile_body(call::Call)
        local f = comp_env.impl_dict[call.f]
        local args = compile_body.(call.args)
        local rtype = shape_env[call.type]
        local e = Expr(:call, f, args...)
        if check_gradient
            quote
                v = e::$rtype
                is_bad_dual(v) && let 
                    args = $(Expr(:call, :tuple, args...))
                    error("Bad dual detected. \nf=$f\nargs=$args")
                end
                v
            end
        else 
            :($e::$rtype)
        end
    end

    body_ex = compile_body(prog)::Expr
    prev_result = lock(compile_cache_lock) do
        get(compile_cache, body_ex, nothing)
    end
    (prev_result !== nothing) && return prev_result
    f_ex = :(args -> $body_ex)
    cf = CompiledFunc(prog, body_ex, @RuntimeGeneratedFunction(f_ex))
    lock(compile_cache_lock) do
        compile_cache[body_ex] = cf
    end
end

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