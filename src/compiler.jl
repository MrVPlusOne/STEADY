struct CompiledFunc <: Function
    ast::TAST
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
call_T(cf::CompiledFunc, x, T) = call_T(cf.f, x, T)


function Base.show(io::IO, @nospecialize cf::CompiledFunc) 
    @unpack ast, julia = cf
    print(io, "CompiledFunction(ast=:($ast), julia=:($julia))")
end

function Base.show(io::IO, mime::MIME"text/plain", @nospecialize cf::CompiledFunc) 
    io = IOIndents.IOIndent(io)
    @unpack ast, julia = cf
    println(io, "--- CompiledFunction ---")
    println(io, "AST:", Indent())
    println(io, ast, Dedent())
    println(io, "Julia Code:", Indent())
    println(io, repr(mime, julia), Dedent())
    println(io, "--- End ---")
end

"""
Compiles a `TAST` expression into the corresponding julia function that can be 
efficiently executed.

Implemented using `RuntimeGeneratedFunctions.jl`.
"""
function compile(
    prog::TAST, args::Vector{Var}, shape_env::ShapeEnv, comp_env::ComponentEnv
)::CompiledFunc
    function compile_body(v::Var)
        Expr(:(.), :args, QuoteNode(v.name))
    end
    function compile_body(call::Call)
        local f = comp_env.impl_dict[call.f]
        local args = compile_body.(call.args)
        local rtype = shape_env[call.type]
        local r = Expr(:call, f, args...)
        :($r::$rtype)
    end

    names = tuple((a.name for a in args)...)
    types = Tuple{(shape_env[a.type] for a in args)...}
    args_type = NamedTuple{names, types}
    args_ex = :(args::$args_type)
    body_ex = compile_body(prog)
    f_ex = :($args_ex -> $body_ex)
    f = @RuntimeGeneratedFunction f_ex
    CompiledFunc(prog, body_ex, f)
end

"""
Compiles a `TAST` expression into the corresponding julia function that can be 
efficiently executed.

Implemented using `RuntimeGeneratedFunctions.jl`.
"""
function compile_cached!(
    prog::TAST, args::Vector{Var}, shape_env::ShapeEnv, comp_env::ComponentEnv, 
    cache::Dict{TAST, CompiledFunc},
)::CompiledFunc
    function compile_body(v::Var)
        Expr(:(.), :args, QuoteNode(v.name))
    end
    function compile_body(call::Call)
        local f = comp_env.impl_dict[call.f]
        local sub_calls = map(call.args) do arg
            if arg isa Var
                compile_body(arg)
            else
                cf = compile_cached!(arg, args, shape_env, comp_env, cache)
                Expr(:call, cf.f, :args)
            end
        end
        local rtype = shape_env[call.type]
        local r = Expr(:call, f, sub_calls...)
        :($r::$rtype)
    end

    get!(cache, prog) do
        names = tuple((a.name for a in args)...)
        types = Tuple{(shape_env[a.type] for a in args)...}
        args_type = NamedTuple{names, types}
        args_ex = :(args::$args_type)
        body_ex = compile_body(prog)
        f_ex = :($args_ex -> $body_ex)
        f = @RuntimeGeneratedFunction f_ex
        CompiledFunc(prog, body_ex, f)
    end
end

function compile_interpreted(
    prog::TAST, arg_vars::Vector{Var}, shape_env::ShapeEnv, comp_env::ComponentEnv
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