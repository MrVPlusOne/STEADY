struct CompiledFunc{F<:Function} <: Function
    ast::TAST
    julia::Expr
    f::F
end

(cf::CompiledFunc)(args::NamedTuple) = cf.f(args)

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
)#::CompiledFunc
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
    CompiledFunc(prog, f_ex, f)
end