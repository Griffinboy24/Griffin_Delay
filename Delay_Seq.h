/*
    Delay_Seq.h

*/

#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <JuceHeader.h>

#include "src/FixedPointUtils.h"
#include "src/Downsampler2Flt.h"

namespace project
{
    using namespace juce;
    using namespace hise;
    using namespace scriptnode;
    using wt::Downsampler2Flt;

    // --------------------------------------------------------------------
    // Constants
    // --------------------------------------------------------------------
    constexpr float kMaxDelayMs = 2000.0f;   // 0 – 2 s
    constexpr float kSmoothMs = 50.0f;   // parameter glide

    // --------------------------------------------------------------------
    // Helpers
    // --------------------------------------------------------------------
    template<typename T>
    static inline T clampValue(T v, T lo, T hi) noexcept
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    static inline float catmullRom4Tap(float ym1, float y0,
        float y1, float y2,
        float mu) noexcept
    {
        float mu2 = mu * mu;
        float a0 = -0.5f * ym1 + 1.5f * y0 - 1.5f * y1 + 0.5f * y2;
        float a1 = ym1 - 2.5f * y0 + 2.0f * y1 - 0.5f * y2;
        float a2 = -0.5f * ym1 + 0.5f * y1;
        return ((a0 * mu + a1) * mu + a2) * mu + y0;
    }

    static inline uint32_t nextPow2(uint32_t v) noexcept
    {
        v--;
        v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
        v |= v >> 8;  v |= v >> 16;
        return ++v;
    }

    //====================================================================
    // Variable-delay line (per channel)
    //====================================================================
    class VariableDelay
    {
    public:
        //-----------------------------------------------------------------
        void prepare(double sr)
        {
            sampleRate = sr;

            uint32_t need = static_cast<uint32_t>(
                std::ceil(kMaxDelayMs * 0.001 * sr)) + 4u;
            bufferSize = nextPow2(need);
            mask = bufferSize - 1u;
            buffer.assign(bufferSize, 0.0f);
            writePos = 0u;

            // set LPF coefficients (same as ResamplerFlt)
            static constexpr double kCoef[Downsampler2Flt::NBR_COEFS] =
            { 0.0457281, 0.168088, 0.332501, 0.504486,
              0.663202 , 0.803781, 0.933856 }; // :contentReference[oaicite:1]{index=1}
            lpf.set_coefs(kCoef);
            lpf.clear_buffers();

            smoothedDelay.reset(sr, kSmoothMs * 0.001);
            smoothedDelay.setCurrentAndTargetValue(
                0.25f * static_cast<float>(sr));              // 250 ms
        }

        void reset() noexcept
        {
            std::fill(buffer.begin(), buffer.end(), 0.0f);
            writePos = 0u;
            lpf.clear_buffers();
            smoothedDelay.setCurrentAndTargetValue(
                smoothedDelay.getCurrentValue());
        }

        // ----------------------------------------------------------------
        void setDelayMs(float ms) noexcept
        {
            float s = clampValue(ms * 0.001f * static_cast<float>(sampleRate),
                1.0f,
                static_cast<float>(bufferSize - 4u));
            smoothedDelay.setTargetValue(s);
        }
        void setFeedback(float fb) noexcept
        {
            feedback = clampValue(fb, 0.0f, 0.99f);
        }
        void setMix(float m) noexcept
        {
            mix = clampValue(m, 0.0f, 1.0f);
        }

        // ----------------------------------------------------------------
        void process(float* data, int n) noexcept
        {
            const float wetMix = mix;
            const float dryMix = 1.0f - wetMix;
            const float fbGain = feedback;
            const uint32_t msk = mask;
            float* buf = buffer.data();

            uint32_t w = writePos;

            // single-sample helper arrays for LPF phase_block
            float src[1];
            float dst[1];

            for (int i = 0; i < n; ++i)
            {
                float delaySamples = smoothedDelay.getNextValue();
                uint32_t di = static_cast<uint32_t>(delaySamples);
                float frac = delaySamples - static_cast<float>(di);

                uint32_t idx0 = (w - 1u - di) & msk;
                uint32_t idxM1 = (idx0 - 1u) & msk;
                uint32_t idx1 = (idx0 + 1u) & msk;
                uint32_t idx2 = (idx1 + 1u) & msk;

                float wet = catmullRom4Tap(buf[idxM1], buf[idx0],
                    buf[idx1], buf[idx2], frac);

                // low-pass the wet path (alias suppression)
                src[0] = wet;
                lpf.phase_block(dst, src, 1);
                wet = dst[0];

                float in = data[i];
                buf[w] = in + wet * fbGain;  // filtered feedback

                data[i] = dryMix * in + wetMix * wet;

                w = (w + 1u) & msk;
            }
            writePos = w;
        }

    private:
        std::vector<float> buffer;
        uint32_t           bufferSize{ 0 }, mask{ 0 }, writePos{ 0 };
        double             sampleRate{ 44100.0 };

        juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear>
            smoothedDelay;
        float              feedback{ 0.35f }, mix{ 1.0f };

        Downsampler2Flt    lpf;
    };

    //====================================================================
    // ScriptNode wrapper (stereo)
    //====================================================================
    template<int NV>
    struct Delay_Seq : public data::base
    {
        SNEX_NODE(Delay_Seq);
        struct MetadataClass { SN_NODE_ID("Delay_Seq"); };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return false; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int  getFixChannelAmount() { return 2; }

        static constexpr int NumTables = 0;
        static constexpr int NumSliderPacks = 0;
        static constexpr int NumAudioFiles = 0;
        static constexpr int NumFilters = 0;
        static constexpr int NumDisplayBuffers = 0;

        // ------------------------------------------------------------
        void prepare(PrepareSpecs s)
        {
            left.prepare(s.sampleRate);
            right.prepare(s.sampleRate);
        }
        void reset() { left.reset(); right.reset(); }

        template<typename PD>
        void process(PD& d)
        {
            auto& blk = d.template as<ProcessData<2>>().toAudioBlock();
            int n = d.getNumSamples();
            left.process(blk.getChannelPointer(0), n);
            right.process(blk.getChannelPointer(1), n);
        }

        template<int P>
        void setParameter(double v)
        {
            float f = static_cast<float>(v);
            if constexpr (P == 0) { left.setDelayMs(f);  right.setDelayMs(f); }
            if constexpr (P == 1) { left.setFeedback(f); right.setFeedback(f); }
            if constexpr (P == 2) { left.setMix(f);      right.setMix(f); }
        }

        void createParameters(ParameterDataList& list)
        {
            {
                parameter::data p("Delay Time (ms)", { 1.0, kMaxDelayMs, 1.0 });
                p.setDefaultValue(250.0f); registerCallback<0>(p); list.add(std::move(p));
            }
            {
                parameter::data p("Feedback", { 0.0, 0.99, 0.01 });
                p.setDefaultValue(0.35f);  registerCallback<1>(p); list.add(std::move(p));
            }
            {
                parameter::data p("Mix", { 0.0, 1.0, 0.01 });
                p.setDefaultValue(1.0f);   registerCallback<2>(p); list.add(std::move(p));
            }
        }

        SN_EMPTY_PROCESS_FRAME;
        SN_EMPTY_HANDLE_EVENT;
        SN_EMPTY_SET_EXTERNAL_DATA;

    private:
        VariableDelay left, right;
    };

} // namespace project
