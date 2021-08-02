module NumericalIntegration
    export integrate_forward, integrate_forward_invariant
    export Euler, RK38

    abstract type IntegrationMethod end

    struct Euler <: IntegrationMethod end
    struct RK38 <: IntegrationMethod end

    function integrate(f, x, p, dt, t, ::IntegrationMethod) end
    function integrate_inplace(f!, dx, x, p, dt, t, ::IntegrationMethod) end
    

    """
    Numerical integration for differential equation systems.
    """
    @inline function integrate_inplace(f!, x0::X, p, tspan::Tuple{T,T}, method, N) where {X,T}
        t0, t1 = tspan
        dt = (t1 - t0)/N
        x::X = x0
        dx::X = zero(x)
        t::T = t0
        for _ in 1:N
            x += method(f, x, p, dt, t)
            t += dt
        end
        x
    end

    """
    Numerical integration for differential equation systems.
    """
    @inline function integrate_forward(f, x0::X, p, tspan::Tuple{T,T}, method, N) where {X,T}
        t0, t1 = tspan
        dt = (t1 - t0)/N
        x::X = x0
        t::T = t0
        for _ in 1:N
            x += method(f, x, p, dt, t)
            t += dt
        end
        x
    end

    """
    Numerical integration for time-invariant differential equation systems.
    """
    @inline function integrate_forward_invariant(f, x0::X, p, dt::AbstractFloat, method, N) where {X}
        x::X = x0
        for _ in 1:N
            x += method(f, x, p, dt)
        end
        x
    end


    @inline function Euler(f, x::X, p, dt::AbstractFloat)::X where X
        f(x, p)::X * dt
    end

    @inline function Euler(f, x::X, p, dt::AbstractFloat, t)::X where X
        f(x, p, t)::X * dt
    end

    """
    Runge–Kutta 3/8 rule. (4th order method)
    """
    @inline function RK38(f, x::X, p, dt::AbstractFloat)::X where X
        k1 = f(x, p)::X
        k2 = f(x+(1/3)k1*dt, p)::X
        k3 = f(x+dt*((-1/3)k1+k2), p)::X
        k4 = f(x+dt*(k1-k2+k3), p)::X
        dt * ((1/8)k1 + (3/8)k2 + (3/8)k3 + (1/8)k4)
    end

    @inline function RK38(f, x::X, p, dt::AbstractFloat, t)::X where X
        k1 = f(x, p, t)::X
        k2 = f(x+(1/3)k1*dt, p, t+(1/3)dt)::X
        k3 = f(x+dt*((-1/3)k1+k2), p, t+(2/3)dt)::X
        k4 = f(x+dt*(k1-k2+k3), p, t+dt)::X
        dt * ((1/8)k1 + (3/8)k2 + (3/8)k3 + (1/8)k4)
    end

end