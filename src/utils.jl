export specific_elems, count_len 
export GDistribution, DistrIterator, is_distribution
export SVecDistr, SMvNormal
export max_by, sort_by
export rotate2d, rotation2D, °

specific_elems(xs) = identity.(xs)

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

function zipmap(fs, xs::Vector)
    @assert length(fs) == length(xs) "Need the same number of functions and values"
    map(eachindex(xs)) do i
        fs[i](xs[i])
    end
end

subtuple(xs::NamedTuple, keys::Tuple) = begin
    NamedTuple{keys}(map(k -> getfield(xs, k), keys))
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
    Optim.optimize(Optim.only_fg!(fg!), x₀, Optim.LBFGS(), optim_options)
end

count_len(iters) = count(_ -> true, iters)

"""
A wrapper of a sequence of distributions. This can be used to sample or compute
the corresponding logpdf of a sequence of values.

The type parameter `Iter` should be an iterators of distributions, e.g., 
`Vector{Normal}` or `Tuple{Uniform, MvNormal}`.

## Examples
```julia
julia> di = DistrIterator(fill(Normal(), 4))
Main.SEDL.DistrIterator{Vector{Distributions.Normal{Float64}}}(...)

julia> xs = rand(di)
4-element Vector{Float64}:
 -0.49010706913108426
  0.04667852819261214
  1.4516555944579874
 -1.3288896894564055

julia> logpdf(di, xs)
-5.733571930754293

julia> nested = DistrIterator((di, di));
julia> rand(nested)
([-0.3008128094801595, 0.7595305727349723, 0.0731633860379036, -0.15532821730030588], [-0.5856011152398879, -1.3111512044326252, 1.1532299087076099, 1.5392183048453147])

julia> logpdf(nested, rand(nested))
-13.87393093934596

# Some bijector interface is also supported.
julia> di = DistrIterator(fill(Uniform(0, 1), 4))
Main.SEDL.DistrIterator{Vector{Distributions.Uniform{Float64}}}(Distributions.Uniform{Float64}[Distributions.Uniform{Float64}(a=0.0, b=1.0), Distributions.Uniform{Float64}(a=0.0, b=1.0), Distributions.Uniform{Float64}(a=0.0, b=1.0), Distributions.Uniform{Float64}(a=0.0, b=1.0)])

julia> xs = rand(di);

julia> di_bj = bijector(di);

julia> inv(di_bj)(di_bj(xs)) ≈ xs
true
```
"""
struct DistrIterator{Iter}
    distributions::Iter

    DistrIterator(distributions::Iter) where Iter = begin
        @assert all(map(is_distribution, distributions)) "\
            Expect an iterator of distributions, but got $distributions."
        new{Iter}(distributions)
    end
end

Distributions.rand(rng::Random.AbstractRNG, diter::DistrIterator) = 
    map(d -> rand(rng, d), diter.distributions)
Distributions.logpdf(diter::DistrIterator, xs) = 
    sum(logpdf(d, x) for (d, x) in zip(diter.distributions, xs))

"""
Generalized distributions that support sampling structured data. 
"""
const GDistribution = Union{Distribution, DistrIterator}
@inline is_distribution(x) = x isa GDistribution

"""
An iterator of bijectors, used to transform [`DistrIterator`](@ref).
"""
struct BijectorIterator{Iter} <: Bijectors.AbstractBijector
    bijectors::Iter
    BijectorIterator(bjs::Iter) where Iter = begin
        @assert all(map(x -> x isa Bijectors.AbstractBijector, bjs)) "\
            Expect an iterator of bijectors, but got $bjs."
        new{Iter}(bjs)
    end
end

(bit::BijectorIterator)(x) = zipmap(bit.bijectors, x)

function Bijectors.bijector(di::DistrIterator)
    BijectorIterator(bijector.(di.distributions))
end

function Bijectors.inv(bit::BijectorIterator)
    BijectorIterator(inv.(bit.bijectors))
end

struct SVecDistr{D <: Distribution}
    distr::D
end

@forward SVecDistr.distr logpdf, bijector

convert_svector(v::AbstractVector{<:Number}) = SVector{length(v)}(v)
convert_svector(x::Number) = x

function Distributions.rand(rng::AbstractRNG, d::SVecDistr)
    convert_svector(rand(rng, d.distr))
end

function Distributions.rand(d::SVecDistr)
    convert_svector(rand(d.distr))
end

const SMvNormal = SVecDistr ∘ MvNormal