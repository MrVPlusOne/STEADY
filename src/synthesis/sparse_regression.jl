using LinearAlgebra
using MLJLinearModels

abstract type SparseOptimizer end

"""
Sequential thresholded optimizer.
"""
struct SeqThresholdOptimizer{Reg<:GLR} <: SparseOptimizer
    "Discarding threshold"
    ϵ::Float64
    regressor::Reg
end

Base.:*(opt::SeqThresholdOptimizer, s::Real) =
    SeqThresholdOptimizer(opt.ϵ * s, opt.regressor * s)

Base.:*(reg::GLR, s::Real) =
    GLR(reg.loss, reg.penalty / s, reg.fit_intercept, reg.penalize_intercept)

"""
return `(; coeffs, intercept, active_ids, iterations)`.
"""
function regression(
    opt::SeqThresholdOptimizer,
    target::AbsVec{Float64},
    basis::AbsMat{Float64},
    active_ids::Vector{Int64}=collect(1:size(basis, 2)),
    iterations::Int=1,
)
    X = @views basis[:, active_ids]
    θ = MLJLinearModels.fit(opt.regressor, X, target)
    (coeffs, intercept) = if opt.regressor.fit_intercept
        (θ[1:(end - 1)], θ[end])
    else
        (θ, 0.0)
    end

    is_active = abs.(coeffs) .> opt.ϵ
    if all(is_active)
        return (; coeffs, intercept, active_ids, iterations)
    end
    active_ids = active_ids[is_active]
    return regression(opt, target, basis, active_ids, iterations + 1)
end

struct LinearExpression{N}
    shift::Float64
    coeffs::SVector{N,Float64}
    basis::SVector{N,<:CompiledFunc}
    type::PType
end

num_terms(::LinearExpression{N}) where {N} = N

function LinearExpression(
    shift::Float64, coeffs::AbstractVector, basis::AbstractVector, type::PType
)
    @smart_assert (n = length(coeffs)) == length(basis)
    LinearExpression(
        shift, SVector{n,Float64}(coeffs), SVector{n,CompiledFunc}(basis), type
    )
end

function Base.print(io::IO, expr::LinearExpression)
    (; shift, coeffs, basis, type) = expr
    compact = get(io, :compact, false)
    !compact && print(io, "LinearExpression(")
    print(io, "`")
    shift != 0.0 && print(io, shift)
    for i in 1:length(coeffs)
        op = coeffs[i] >= 0 ? " + " : " - "
        print(io, op, abs(coeffs[i]), " ", basis[i].ast)
    end
    print(io, "`")
    !compact && print(io, ")::", type)
end

Base.show(io::IO, expr::LinearExpression) = print(io, expr)

function compile(lexpr::LinearExpression, shape_env=ℝenv())
    annot = shape_env.type_annots[lexpr.type.shape]
    rtype = shape_env.return_type[lexpr.type.shape]

    (; shift, coeffs, basis) = lexpr

    terms = Any[]
    if isempty(coeffs) || shift != 0.0
        push!(terms, :(shift::$annot))
    end
    foreach(enumerate(basis)) do (i, e)
        push!(terms, :(coeffs[$i] * $(e.julia)))
    end
    body_ex = Expr(:call, :+, terms...)
    f_ex = :((args, shift, coeffs) -> $body_ex)
    rgf = compile_julia_expr(f_ex)

    param_f = args -> rgf(args, lexpr.shift, lexpr.coeffs)

    CompiledFunc{rtype,typeof(param_f)}(param_f, lexpr, body_ex)
end