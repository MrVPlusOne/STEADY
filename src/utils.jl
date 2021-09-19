export specific_elems, count_len, @unzip, @unzip_named
export max_by, sort_by
export rotate2d, rotation2D, °

using MacroTools: @capture
using Cthulhu
using ForwardDiff: Dual
import LineSearches

specific_elems(xs::AbstractVector{T}) where T = 
    Base.isconcretetype(T) ? xs : identity.(xs)

count_len(iters) = count(_ -> true, iters)

"""
## Example
```jldoctest
julia> @unzip as, bs = [("a", "b", "c") for i in 1:3]
3-element Vector{Tuple{String, String, String}}:
 ("a", "b", "c")
 ("a", "b", "c")
 ("a", "b", "c")

julia> as
3-element Vector{String}:
 "a"
 "a"
 "a"

julia> bs
3-element Vector{String}:
 "b"
 "b"
 "b"
```
"""
macro unzip(assign)
    if @capture(assign, (v1_, vs__) = rhs_)
        assigns = map(enumerate([v1; vs])) do (i, v)
            :($(esc(v)) = map(x -> x[$i], rhs_value))
        end
        Expr(:block, :(rhs_value = $(esc(rhs))), assigns..., :rhs_value)
    else
        error("Usage: `@unzip x, [y, ...] = rhs`")
    end
end

"""
## Example
```jldoctest
julia> @unzip_named (as, :a), (cs, :c) = [(a="a", b="b", c="c") for i in 1:3]
3-element Vector{NamedTuple{(:a, :b, :c), Tuple{String, String, String}}}:
 (a = "a", b = "b", c = "c")
 (a = "a", b = "b", c = "c")
 (a = "a", b = "b", c = "c")

julia> as
3-element Vector{String}:
 "a"
 "a"
 "a"

julia> cs
3-element Vector{String}:
 "c"
 "c"
 "c"
```
"""
macro unzip_named(assign)
    fail() = error("Usage: `@unzip_named (xs, :x_name), [(ys, :y_name), ...] = rhs`")

    @capture(assign, (v1_, vs__) = rhs_) || fail()
    assigns = map([v1; vs]) do v
        (v isa Expr && v.head === :tuple && length(v.args) == 2) || fail()
        x, x_name = v.args
        (x isa Symbol && x_name isa QuoteNode) || fail()
        :($(esc(x)) = map(p -> getfield(p, $x_name), rhs_value))
    end
    Expr(:block, :(rhs_value = $(esc(rhs))), assigns..., :rhs_value)
end


@inline rotation2D(θ) = @SArray(
    [cos(θ) -sin(θ)
     sin(θ)  cos(θ)]
)

rotate2d(θ, v) = rotation2D(θ) * v

const ° = π / 180

to_measurement(values) = begin
    μ = mean(values)
    σ = std(values)
    μ ± σ
end

max_by(f) = xs -> begin
    ys = map(f, xs)
    _, i = findmax(ys)
    xs[i]
end

sort_by(f) = xs -> sort(xs, by=f)

pretty_number(v) = (v isa Number ? format(v, commas=true) : string(v))

"""
Apply a tuple of functions to a tuple of corresponding arguments. The result is also a 
tuple.

Currently, the result type is restricted to be the same type as `xs` to aid type inference
when `length(xs)` is large.
"""
@inline function zipmap(fs, xs::X)::X where {X<:Tuple}
    @assert length(fs) == length(xs) "Need the same number of functions and values"
    ntuple(i -> fs[i](xs[i]), length(xs))
end

"""
Apply a tuple of functions to a NamedTuple of corresponding arguments. The result is a 
NamedTuple.
"""
@inline function zipmap(fs, xs::X)::X where {X<:NamedTuple}
    @assert length(fs) == length(xs) "Need the same number of functions and values"
    t = ntuple(i -> fs[i](xs[i]), length(xs))
    NamedTuple{keys(xs)}(t)
end

function zipmap(fs, xs::AbstractVector)
    @assert length(fs) == length(xs) "Need the same number of functions and values"
    map((f, x) -> f(x), fs, xs)
end

subtuple(xs::NamedTuple, keys::Tuple) = begin
    NamedTuple{keys}(map(k -> getfield(xs, k), keys))
end

warp_angle(angle::Real) = let
    x = angle % 2π
    x < 0 ? x + 2π : x
end

"""
Lightweight version of @timed.
"""
macro ltimed(ex)
    quote
        t0 = time()
        v = $(esc(ex))
        (time=time()-t0, value=v)
    end
end

function optimize_no_tag(loss, x₀, optim_options)
    # drop tag to reduce JIT compilation time and avoid tag checking
    cfg = ForwardDiff.GradientConfig(nothing, x₀) 
    function fg!(F, G, x)
        (G === nothing) && return loss(x)

        gr = ForwardDiff.DiffResult(first(x), (G,))
        ForwardDiff.gradient!(gr, loss, x, cfg)
        if F !== nothing
            return gr.value
        end
    end
    algorithm = Optim.LBFGS(linesearch=LineSearches.BackTracking(order=3))
    Optim.optimize(Optim.only_fg!(fg!), x₀, algorithm, optim_options)
end

function optimize_bounded(loss, x₀, (lower, upper), optim_options)
    # drop tag to reduce JIT compilation time and avoid tag checking
    cfg = ForwardDiff.GradientConfig(nothing, x₀) 
    function fg!(F, G, x)
        (G === nothing) && return loss(x)

        gr = ForwardDiff.DiffResult(first(x), (G,))
        ForwardDiff.gradient!(gr, loss, x, cfg)
        if F !== nothing
            return gr.value
        end
    end
    optimizer = Optim.Fminbox(Optim.LBFGS())
    Optim.optimize(Optim.only_fg!(fg!), lower, upper, x₀, optimizer, optim_options)
end

# function optimize_no_tag(loss, x₀, optim_options)
#     Optim.optimize(loss, x₀, Optim.LBFGS(), optim_options, autodiff=:forward)
# end

to_svec(vec::AbstractVector) = SVector{length(vec)}(vec)

"""
Like `get!`, but can be used to directly access nested dictionaries.

```jldoctest
julia> d = Dict{Int, Dict{Symbol, String}}()
Dict{Int64, Dict{Symbol, String}}()

julia> nested_get!(d, 5 => :a) do
           "default"
       end
"default"

julia> d
Dict{Int64, Dict{Symbol, String}} with 1 entry:
  5 => Dict(:a=>"default")
```
"""
function nested_get!(f, d::Dict{K, D}, k_pair::Pair{K, Rest}) where {K, D<:Dict, Rest}
    inner_dict = get!(d, k_pair[1]) do
        D()
    end
    nested_get!(f, inner_dict, k_pair[2])
end

function nested_get!(f, d::Dict{K, V}, k::K) where {K, V}
    get!(f, d, k)
end

function is_bad_dual(v::Dual)
    isfinite(v.value) && abs(v.value) < 1e10 && any(isnan, v.partials)
end
is_bad_dual(v::AbstractVector) = any(is_bad_dual, v)
is_bad_dual(v::Real) = false

"""
Convert any `NaN` partial derivatives to zeros. 
"""
fix_nan_dual(v::Dual{Tag}) where {Tag} = let
    if is_bad_dual(v) 
        new_partials = ForwardDiff.Partials(nan_to_zero.(v.partials.values))
        Dual{Tag}(v.value, new_partials)
    else 
        v
    end
end
fix_nan_dual(v::Real) = v

function nan_to_zero(v::R)::R where {R <: Real}
    isnan(v) ? zero(v) : v
end

using Statistics: norm
Base.show(io::IO, d::ForwardDiff.Dual) = 
    print(io, "Dual($(d.value), |dx|=$(norm(d.partials)))")