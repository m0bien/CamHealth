using Test
using CamHealth
using CamHealth.TemporalFilters
using DSP, FFTW, Random, Statistics
using Colors, VideoIO
using VideoIO: N0f8

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

    # ------------------------------------------------------------------ M1
    @testset "resample_uniform" begin
        rng = MersenneTwister(3)
        t = cumsum(fill(1 / 30, 300) .+ 0.004 .* randn(rng, 300)); t .-= t[1]
        x = 2 .* t .+ 1                                   # linear -> interpolation is exact
        valid = trues(300); valid[100:102] .= false       # short gap (dropped frames)
        valid[200:240] .= false                           # long gap (face lost ~1.4 s)
        tu, xu, vu = resample_uniform(t, x, 30.0; valid=valid)
        @test step(range(tu[1], tu[end]; length=length(tu))) ≈ 1 / 30 atol = 1e-9
        @test xu ≈ 2 .* tu .+ 1 atol = 1e-9
        τ(a) = findmin(abs.(tu .- a))[2]
        @test vu[τ(t[101])]                               # short gap bridged
        @test !vu[τ(t[220])]                              # long gap stays invalid
        @test count(vu) < length(vu)
        X = hcat(x, -x)
        _, Xu, _ = resample_uniform(t, X, 30.0)
        @test Xu[:, 2] ≈ -Xu[:, 1]
    end

    @testset "Welch spectrum peak + SNR" begin
        fs = 30.0; t = (0:299) ./ fs
        f, P = welch_psd(sin.(2π * 1.37 .* t), fs; seglen=240)
        f0, _ = spectrum_peak(f, P, PULSE)
        @test abs(f0 - 1.37) < 0.01
        @test CamHealth.snr_db(f, P, f0; tol=0.25) > 10
        fn, Pn = welch_psd(randn(MersenneTwister(4), 300), fs; seglen=240)
        g0, _ = spectrum_peak(fn, Pn, PULSE)
        @test CamHealth.snr_db(fn, Pn, g0; tol=0.25) < 0
        # HR track on a clean 84 bpm pulse with drift
        x = 0.01 .* sin.(2π * 1.4 .* (0:899) ./ fs) .+ 0.05 .* (0:899) ./ 900
        hr = estimate_hr(x, fs)
        @test length(hr.t) == 21
        @test all(abs.(hr.bpm .- 84) .< 1)
    end

    @testset "frame monitor: drops" begin
        m = CamHealth.FrameMonitor(nominal_fps=30, warn=false)
        fr = fill(RGB{N0f8}(0.5, 0.5, 0.5), 8, 8)
        ts = [(0:9) ./ 30; (12:20) ./ 30]                 # frames 10, 11 missing
        for t in ts; CamHealth.record!(m, fr, t); end
        @test m.ndrop_events == 1
        @test m.ndropped == 2
    end

    @testset "skin mask + YuNet decoding" begin
        img = [RGB{N0f8}(0.8, 0.6, 0.5) RGB{N0f8}(0.1, 0.2, 0.9)]
        @test skin_mask(img) == BitMatrix([1 0])
        # one anchor of stride 32 at grid (r=2, c=3) with known box/landmarks
        S = 640; out = Dict{String,Array{Float32,3}}()
        for s in (8, 16, 32)
            n = (S ÷ s)^2
            out["cls_$s"] = zeros(Float32, 1, n, 1); out["obj_$s"] = zeros(Float32, 1, n, 1)
            out["bbox_$s"] = zeros(Float32, 1, n, 4); out["kps_$s"] = zeros(Float32, 1, n, 10)
        end
        k = 2 * 20 + 3 + 1
        out["cls_32"][1, k, 1] = 0.95; out["obj_32"][1, k, 1] = 0.95
        out["bbox_32"][1, k, :] = [0.5, 0.5, log(2), log(3)]
        out["kps_32"][1, k, :] = Float32[0, 0, 1, 0, 0.5, 0.5, 0, 1, 1, 1]
        faces = CamHealth.yunet_decode(out, S, 0.8)
        @test length(faces) == 1
        f = faces[1]
        @test f.score ≈ 0.95 atol = 1e-5
        @test collect(f.box) ≈ [3.5 * 32 - 32, 2.5 * 32 - 48, 64, 96] atol = 1e-3
        @test f.landmarks[:, 2] ≈ [4 * 32, 2 * 32]
        @test length(CamHealth.nms([f, f], 0.3)) == 1
        @test length(roi_rects(f)) == 3
        model = CamHealth.default_model_path()
        if isfile(model)
            @test isempty(detect(YuNetDetector(model), fill(RGB{N0f8}(0.5, 0.5, 0.5), 480, 640)))
        else
            @info "YuNet model not downloaded; skipping detector run" model
        end
    end

    # synthetic "video": noise everywhere, a 1.2 Hz green modulation inside a known square
    function synthetic_frames(; n=450, fs=30.0, hz=1.2, amp=0.01, sz=(48, 64), roi=(15:30, 21:40), seed=5)
        rng = MersenneTwister(seed)
        frames = Matrix{RGB{N0f8}}[]; ts = Float64[]
        for k in 0:n-1
            t = k / fs
            base = 0.45 .+ 0.02 .* randn(rng, sz...)
            g = copy(base)
            g[roi...] .+= amp * sin(2π * hz * t) + 0.002 * t        # pulse + slow drift
            push!(frames, RGB{N0f8}.(clamp.(base .+ 0.15, 0, 1), clamp.(g, 0, 1), clamp.(base .- 0.05, 0, 1)))
            push!(ts, t)
        end
        frames, ts
    end

    @testset "ROI mean -> green rPPG -> 72 bpm (in memory)" begin
        frames, ts = synthetic_frames()
        roi = FixedROI(20, 14, 20, 16)                    # x=col 21:40, y=row 15:30
        @test count(update!(roi, frames[1])) == 20 * 16
        tr = extract_trace(zip(frames, ts), roi)
        @test length(tr) == 450 && all(tr.valid)
        tu, X, vu = resample_uniform(tr, 30.0)
        hr = estimate_hr(CamHealth.rppg_green(X; valid=vu), 30.0)
        @test abs(median(hr.bpm) - 72) < 1
        # a ROI outside the pulsing square finds no consistent pulse
        tr2 = extract_trace(zip(frames, ts), FixedROI(0, 34, 15, 12))
        hr2 = estimate_hr(CamHealth.rppg_green(resample_uniform(tr2, 30.0)[2]), 30.0)
        @test median(hr2.snr) < median(hr.snr) - 10
    end

    @testset "end to end through FileSource (FFV1 video)" begin
        frames, _ = synthetic_frames(n=450)
        path = joinpath(mktempdir(), "synthetic.mkv")
        VideoIO.save(path, frames; framerate=30, codec_name="ffv1", target_pix_fmt=VideoIO.AV_PIX_FMT_BGR0)
        hr, tr, mon = analyze_clip(path; roi=FixedROI(20, 14, 20, 16), verbose=false,
                                   csv=joinpath(dirname(path), "hr.csv"))
        @test length(tr) == 450
        @test mon.ndropped == 0
        @test abs(median(hr.bpm) - 72) < 1
        @test isfile(joinpath(dirname(path), "hr.csv"))
        @test countlines(joinpath(dirname(path), "hr.csv")) == length(hr.t) + 1
    end
end
