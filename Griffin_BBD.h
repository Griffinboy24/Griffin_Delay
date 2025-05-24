#pragma once
#include <JuceHeader.h>

#include "src/XSimd_Helpers.h"
#include "src/chowdsp_BBDDelayLine.h"
#include "src/chowdsp_BBDDelayWrapper.h"
#include "src/chowdsp_BBDFilterBank.h"
#include "src/chowdsp_DelayInterpolation.h"

namespace project
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    /*==========================================================================*\
        Griffin_BBD - true-stereo 4-chip BBD delay
          - Each channel = four 4096-stage lines in series (16 384 stages total)
          - Per-channel tolerances kept subtle 
    \*==========================================================================*/
    template <int NV>
    struct Griffin_BBD : public data::base
    {
        SNEX_NODE(Griffin_BBD);
        struct MetadataClass { SN_NODE_ID("Griffin_BBD"); };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return true; }
        static constexpr bool isSuspendedOnSilence() { return true; }
        static constexpr int  getFixChannelAmount() { return 2; }

        static constexpr int NumTables = 0;
        static constexpr int NumSliderPacks = 0;
        static constexpr int NumAudioFiles = 0;
        static constexpr int NumFilters = 0;
        static constexpr int NumDisplayBuffers = 0;

        /*--------------------------------------------------------------------*/
        class AudioEffect
        {
        public:
            explicit AudioEffect(float initTotalMs = 333.0f,
                float initWetMix = 0.5f,
                int   initMode = 1) noexcept
                : totalDelayMs(initTotalMs), wetMix(initWetMix), voicingMode(initMode) {
            }

            void prepare(double newFs)
            {
                fs = (float)newFs;
                dsp::ProcessSpec s{ newFs, 512, 1 };
                for (auto& d : line) d.prepare(s);
                updateParams();
            }

            void process(float* buf, int n)
            {
                for (int i = 0; i < n; ++i)
                {
                    float x = buf[i];
                    for (auto& d : line) { d.pushSample(0, x); x = d.popSample(0); }
                    buf[i] = (buf[i] * (1.0f - wetMix)) + (x * wetMix);
                }
                updateParams();
            }

            /* public setters */
            void setTotalDelayMs(float v) { totalDelayMs = v; updateParams(); }
            void setWetMix(float v) { wetMix = v; }
            void setVoicingMode(int v) { voicingMode = jlimit(1, 4, v); updateParams(); }
            void setTrim(float dTrim, float fTrim) { delayTrim = dTrim; filterTrim = fTrim; updateParams(); }

        private:
            /* one computation per block */
            void updateParams()
            {
                static constexpr float bright[4] = { 0.5f, 1.0f, 1.4f, 1.8f };

                const float perChipMs = (totalDelayMs * delayTrim) * 0.25f; // 4 chips
                const float dSamples = fs * perChipMs * 0.001f;
                for (auto& d : line) d.setDelay(dSamples);

                constexpr float refSec = 0.020f;                       // 20 ms reference
                const float delaySec = dSamples / fs;
                const float cutoff = chowdsp::BBD::BBDFilterSpec::inputFilterOriginalCutoff
                    * (refSec / delaySec)
                    * bright[voicingMode - 1]
                    * filterTrim;

                for (auto& d : line) d.setFilterFreq(cutoff);
            }

            float fs = 48000.0f;
            float totalDelayMs = 333.0f;    // summed delay of all 4 chips
            float wetMix = 0.5f;
            int   voicingMode = 1;

            float delayTrim = 1.0f;        // per-channel offset
            float filterTrim = 1.0f;

            chowdsp::BBD::BBDDelayWrapper<4096, false> line[4];
        };

        /*--------------------------------------------------------------------*/
        void prepare(PrepareSpecs s)
        {
            L.prepare(s.sampleRate);
            R.prepare(s.sampleRate);

            // Analog L R channel difference
            L.setTrim(0.999f, 1.000f);  
            R.setTrim(1.001f, 0.995f);  
        }

        void reset() {}

        template <typename PD>
        void process(PD& d)
        {
            auto& f = d.template as<ProcessData<2>>();
            auto blk = f.toAudioBlock();
            L.process(blk.getChannelPointer(0), d.getNumSamples());
            R.process(blk.getChannelPointer(1), d.getNumSamples());
        }

        /*----------------------------------------------------------------*/
        template <int P>
        inline void setParameter(double v)
        {
            if constexpr (P == 0) { L.setTotalDelayMs((float)v); R.setTotalDelayMs((float)v); }
            else if constexpr (P == 1) { L.setWetMix((float)v); R.setWetMix((float)v); }
            else if constexpr (P == 2) { int m = (int)(v + 0.5); L.setVoicingMode(m); R.setVoicingMode(m); }
        }

        void createParameters(ParameterDataList& data)
        {
            {
                parameter::data p("Delay (ms)", { 50.0, 3000.0, 0.1 });
                registerCallback<0>(p); p.setDefaultValue(1000.0); data.add(std::move(p));
            }
            {
                parameter::data p("Wet Mix", { 0.0, 1.0, 0.01 });
                registerCallback<1>(p); p.setDefaultValue(1.0); data.add(std::move(p));
            }
            {
                parameter::data p("Brightness", { 1.0, 4.0, 1.0 });
                registerCallback<2>(p); p.setDefaultValue(1.0); data.add(std::move(p));
            }
        }

        SN_EMPTY_PROCESS_FRAME;
        SN_EMPTY_HANDLE_EVENT;
        SN_EMPTY_SET_EXTERNAL_DATA;

    private:
        AudioEffect L, R;
    };

} // namespace project


#pragma once

#include <memory>      // unique_ptr
#include <array>       // std::array
#include <algorithm>   // std::fill

#include "chowdsp_BBDFilterBank.h"   

namespace chowdsp::BBD
{
/**
 * A class to emulate an analog delay line
 * made using a bucket-brigade device
 */
template <size_t STAGES, bool ALIEN = false>
class BBDDelayLine
{
public:
    BBDDelayLine() = default;
    BBDDelayLine (BBDDelayLine&&) noexcept = default;
    BBDDelayLine& operator= (BBDDelayLine&&) noexcept = default;

    /** Prepares the delay line for processing */
    void prepare (double sampleRate)
    {
        FS = (float) sampleRate;
        Ts = 1.0f / FS;

        tn = 0.0f;
        evenOn = true;

        inputFilter = std::make_unique<InputFilterBank> (Ts);
        outputFilter = std::make_unique<OutputFilterBank> (Ts);
        H0 = outputFilter->calcH0();

        reset();
    }

    /** Resets the state of the delay */
    void reset()
    {
        bufferPtr = 0;
        std::fill (buffer.begin(), buffer.end(), 0.0f);
    }

    /**
     * Sets the cutoff frequency of the input anti-imaging
     * filter used by the bucket-brigade device
     */
    void setInputFilterFreq (float freqHz = BBDFilterSpec::inputFilterOriginalCutoff) const
    {
        inputFilter->set_freq (ALIEN ? freqHz * 0.2f : freqHz);
        inputFilter->set_time (tn);
    }

    /**
     * Sets the cutoff frequency of the output anti-aliasing
     * filter used by the bucket-brigade device
     */
    void setOutputFilterFreq (float freqHz = BBDFilterSpec::outputFilterOriginalCutoff) const
    {
        outputFilter->set_freq (ALIEN ? freqHz * 0.2f : freqHz);
        outputFilter->set_time (tn);
    }

    /**
     * Sets the delay time of the delay line.
     * Internally this changed the "clock rate"
     * of the bucket-brigade device
     */
    void setDelayTime (float delaySec) noexcept
    {
        delaySec = juce::jmax (Ts, delaySec - Ts); // don't divide by zero!!

        const auto clock_rate_hz = (2.0f * (float) STAGES) / delaySec;
        Ts_bbd = 1.0f / clock_rate_hz;

        // if Ts_bbd == 0, then we get an infinite loop, so limit the min. delay
        Ts_bbd = juce::jmax (Ts * 0.01f, Ts_bbd);

        const auto doubleTs = 2 * Ts_bbd;
        inputFilter->set_delta (doubleTs);
        outputFilter->set_delta (doubleTs);
    }

    /** Processes a sample with the delay line (ALIEN MODE) */
    template <bool A = ALIEN>
    inline std::enable_if_t<A, float>
        process (float u) noexcept
    {
        SIMDComplex<float> xOutAccum;
        float yBBD, delta;
        while (tn < 1.0f)
        {
            if (evenOn)
            {
                inputFilter->calcG();
                buffer[bufferPtr++] = xsimd::reduce_add (SIMDUtils::SIMDComplexMulReal (inputFilter->Gcalc, inputFilter->x));
                bufferPtr = (bufferPtr <= STAGES) ? bufferPtr : 0;
            }
            else
            {
                yBBD = buffer[bufferPtr];
                delta = yBBD - yBBD_old;
                yBBD_old = yBBD;
                outputFilter->calcG();
                xOutAccum += outputFilter->Gcalc * delta;
            }

            evenOn = ! evenOn;
            tn += Ts_bbd / Ts;
        }
        tn -= 1.0f;

        inputFilter->process (u);
        outputFilter->process (xOutAccum);
        float sumOut = xsimd::reduce_add (xOutAccum.real());
        return H0 * yBBD_old + sumOut;
    }

    /** Processes a sample with the delay line (BBD MODE) */
    template <bool A = ALIEN>
    inline std::enable_if_t<! A, float>
        process (float u) noexcept
    {
        SIMDComplex<float> xOutAccum {};
        float yBBD, delta;
        while (tn < Ts)
        {
            if (evenOn)
            {
                inputFilter->calcG();
                buffer[bufferPtr++] = xsimd::reduce_add (SIMDUtils::SIMDComplexMulReal (inputFilter->Gcalc, inputFilter->x));
                bufferPtr = (bufferPtr <= STAGES) ? bufferPtr : 0;
            }
            else
            {
                yBBD = buffer[bufferPtr];
                delta = yBBD - yBBD_old;
                yBBD_old = yBBD;
                outputFilter->calcG();
                xOutAccum += outputFilter->Gcalc * delta;
            }

            evenOn = ! evenOn;
            tn += Ts_bbd;
        }
        tn -= Ts;

        inputFilter->process (u);
        outputFilter->process (xOutAccum);
        float sumOut = xsimd::reduce_add (xOutAccum.real());
        return H0 * yBBD_old + sumOut;
    }

private:
    float FS = 48000.0f;
    float Ts = 1.0f / FS;
    float Ts_bbd = Ts;

    std::unique_ptr<InputFilterBank> inputFilter;
    std::unique_ptr<OutputFilterBank> outputFilter;
    float H0 = 1.0f;

    std::array<float, STAGES + 1> buffer;
    size_t bufferPtr = 0;

    float yBBD_old = 0.0f;
    float tn = 0.0f;
    bool evenOn = true;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BBDDelayLine)
};

} // namespace chowdsp::BBD

#pragma once
/*
    chowdsp_BBDDelayWrapper.h
    Stand-alone version – no chowdsp::Buffer dependency.
    -------------------------------------------------------------------
    The class used to inherit from DelayLineBase<float>, which dragged in
    chowdsp_Buffer.* and friends even though the BBD algorithm never used
    that storage.  We simply:
        • removed the inheritance,
        • dropped the pure-virtual overrides,
        • left the public API exactly the same so existing call-sites
          build without edits.
    -------------------------------------------------------------------
    Dependencies that remain:
        juce::dsp::ProcessSpec   – for block-size / sample-rate
        <vector>                 – channel containers
        chowdsp_BBDDelayLine.h   – the actual DSP engine
*/

#include <vector>
#include <juce_dsp/juce_dsp.h>

#include "chowdsp_BBDDelayLine.h"

namespace chowdsp::BBD
{
    /**
     * A thin, multi-channel wrapper around BBDDelayLine.
     *
     * @attention Call setFilterFreq() each audio block (or every few blocks)
     *            so that the BBD internal clock stays aligned.
     */
    template <size_t STAGES, bool ALIEN = false>
    class BBDDelayWrapper
    {
    public:
        //==============================================================================
        BBDDelayWrapper() = default;

        //==============================================================================
        /** Sets both anti-aliasing filter cut-offs (input+output). */
        void setFilterFreq(float freqHz)
        {
            setInputFilterFreq(freqHz);
            setOutputFilterFreq(freqHz);
        }

        /** Sets the input anti-imaging filter frequency. */
        void setInputFilterFreq(float freqHz)
        {
            for (auto& line : lines)
                line.setInputFilterFreq(freqHz);
        }

        /** Sets the output anti-aliasing filter frequency. */
        void setOutputFilterFreq(float freqHz)
        {
            for (auto& line : lines)
                line.setOutputFilterFreq(freqHz);
        }

        //==============================================================================
        /** Delay length in *samples*. */
        void setDelay(float newDelayInSamples)
        {
            delaySamp = newDelayInSamples;

            const float delaySec = delaySamp / sampleRate;
            for (auto& line : lines)
                line.setDelayTime(delaySec);
        }

        [[nodiscard]] float getDelay() const noexcept { return delaySamp; }

        //==============================================================================
        /** Allocate and reset internal state for the given I/O configuration. */
        void prepare(const juce::dsp::ProcessSpec& spec)
        {
            sampleRate = static_cast<float> (spec.sampleRate);

            inputs.resize(spec.numChannels, 0.0f);

            lines.clear();
            for (size_t ch = 0; ch < spec.numChannels; ++ch)
            {
                lines.emplace_back();
                lines[ch].prepare(sampleRate);
                lines[ch].setInputFilterFreq();
                lines[ch].setOutputFilterFreq();
            }
        }

        /** Free all dynamic memory. */
        void free()
        {
            inputs.clear();
            lines.clear();
        }

        /** Reset internal delay-line state (but keep current size). */
        void reset()
        {
            for (auto& line : lines)
                line.reset();
        }

        //==============================================================================
        /** Push one sample into the delay line on the specified channel. */
        inline void pushSample(int channel, float x) noexcept
        {
            inputs[static_cast<size_t> (channel)] = x;
        }

        /** Pop one sample out of the delay line on the specified channel. */
        inline float popSample(int channel) noexcept
        {
            const size_t ch = static_cast<size_t> (channel);
            return lines[ch].process(inputs[ch]);
        }

    private:
        //==============================================================================
        float delaySamp = 1.0f;        // delay in samples
        float sampleRate = 48000.0f;    // cached FS

        std::vector<BBDDelayLine<STAGES, ALIEN>> lines; // one per channel
        std::vector<float>                       inputs;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BBDDelayWrapper)
    };

} // namespace chowdsp::BBD
