using Test
using CamHealth
using CamHealth.TemporalFilters
using DSP, FFTW, Random

const FS = 30.0

# steady-state amplitude gain (dB) of a causal filter at frequency f, via a long sine
function gain_db(f::ButterBandpass, freq; fs=FS, T=60.0)
    t = 0:1/fs:T
    x = sin.(2π * freq .* t)
    y = TemporalFilters.apply_causal(f, x)
    tail = length(t) ÷ 2:length(t)          # skip the transient
    20log10(maximum(abs, y[tail]) / maximum(abs, x[tail]))
end

@testset "CamHealth" begin
    @testset "BandSpec" begin
        @test PULSE.lo == 0.7 && PULSE.hi == 4.0
        @test BREATHING.lo == 0.1 && BREATHING.hi == 0.5
        @test_throws ArgumentError BandSpec(2.0, 1.0)
        @test_throws ArgumentError ButterBandpass(BandSpec(1.0, 20.0), FS)   # above Nyquist
    end

    @testset "Butterworth band-pass frequency response" begin
        f = ButterBandpass(PULSE, FS)          # default order 3
        H = freqresp(f.sos, 2π .* [0.2, 1.2, 8.0] ./ FS)
        hdb = 20log10.(abs.(H))
        @test hdb[2] > -1.0            # 1.2 Hz passes
        @test hdb[1] < -20.0           # 0.2 Hz attenuated
        @test hdb[3] < -20.0           # 8 Hz attenuated
        # same thing measured on actual sines through the filter
        @test gain_db(f, 1.2) > -1.0
        @test gain_db(f, 0.2) < -20.0
        @test gain_db(f, 8.0) < -20.0
        # zero-phase batch filter: passband sine comes through ~unchanged in amplitude and phase
        t = 0:1/FS:30
        x = sin.(2π * 1.2 .* t)
        y = apply(f, x)
        mid = 150:length(t)-150
        @test maximum(abs, y[mid] .- x[mid]) < 0.05
    end

    @testset "streaming vs batch consistency" begin
        rng = MersenneTwister(1)
        f = ButterBandpass(PULSE, FS)
        x = randn(rng, 600)
        ref = TemporalFilters.apply_causal(f, x)
        # scalar streaming
        st = init_state(f, 0.0)
        ys = [step!(st, xi) for xi in x]
        @test ys ≈ ref atol = 1e-10
        # array (per-element) streaming: a 4x3 "frame" whose pixels carry different signals
        X = randn(rng, 600, 4, 3)
        sta = init_state(f, X[1, :, :])
        Y = similar(X)
        for k in axes(X, 1)
            Y[k, :, :] .= step!(sta, X[k, :, :])
        end
        for i in 1:4, j in 1:3
            @test Y[:, i, j] ≈ TemporalFilters.apply_causal(f, X[:, i, j]) atol = 1e-10
        end
        # batch apply along dims on a matrix == column-wise filtfilt
        M = randn(rng, 300, 5)
        @test apply(f, M; dims=1)[:, 3] ≈ apply(f, M[:, 3])
    end

    @testset "DiffLowpass (EVM temporal filter)" begin
        f = DiffLowpass(PULSE, FS)
        st = init_state(f, 0.0)
        # constant input -> zero output (DC rejected)
        @test all(abs(step!(st, 5.0)) < 1e-12 for _ in 1:100)
        # array version matches scalar version per element
        x = randn(MersenneTwister(2), 200, 2, 2)
        sa = init_state(f, x[1, :, :])
        ss = init_state(f, 0.0)
        ya = [copy(step!(sa, x[k, :, :])) for k in 1:200]
        yscal = [step!(ss, x[k, 1, 2]) for k in 1:200]
        @test [y[1, 2] for y in ya] ≈ yscal
    end

    @testset "ideal FFT band-pass" begin
        t = (0:599) ./ FS                     # 20 s -> exact bins at 0.05 Hz spacing
        x = sin.(2π * 1.2 .* t) .+ sin.(2π * 0.2 .* t) .+ sin.(2π * 8.0 .* t)
        y = ideal_bandpass(x, PULSE, FS)
        @test y ≈ sin.(2π * 1.2 .* t) atol = 1e-9
        X = hcat(x, 2 .* x)
        @test ideal_bandpass(X, PULSE, FS; dims=1)[:, 2] ≈ 2 .* y
    end

    @testset "synthetic rPPG: 72 bpm" begin
        rng = MersenneTwister(42)
        T = 20.0
        t = 0:1/FS:T-1/FS
        hr_hz = 1.2                                   # 72 bpm
        pulse = 0.002 .* sin.(2π * hr_hz .* t)        # tiny pulse amplitude (~0.2 % of skin brightness)
        drift = 0.05 .* (t ./ T) .+ 0.02 .* sin.(2π * 0.05 .* t)   # slow lighting drift
        noise = 0.003 .* randn(rng, length(t))
        g = 0.6 .+ pulse .+ drift .+ noise            # green-channel mean of the ROI
        y = apply(ButterBandpass(PULSE, FS), g .- sum(g) / length(g))
        # zero-padded FFT for a fine frequency grid
        nfft = 8192
        P = abs2.(rfft(vcat(y .* DSP.hanning(length(y)), zeros(nfft - length(y)))))
        fr = rfftfreq(nfft, FS)
        inband = (fr .>= PULSE.lo) .& (fr .<= PULSE.hi)
        fpk = fr[inband][argmax(P[inband])]
        bpm = 60fpk
        @info "synthetic rPPG estimate" bpm
        @test abs(bpm - 72) < 1.0
    end
end
