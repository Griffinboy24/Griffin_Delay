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
        Griffin_BBD – true-stereo 4-chip BBD delay
        * Now with feedback and separate wet / dry gains
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
        static constexpr int NumFilters = 0; // do not remove
        static constexpr int NumDisplayBuffers = 0;

        /*--------------------------------------------------------------------*/
        class AudioEffect
        {
        public:
            explicit AudioEffect(float initTotalMs = 333.0f,
                float initFeedback = 0.35f,
                float initWetGain = 1.0f,
                float initDryGain = 1.0f,
                int   initMode = 1) noexcept
                : totalDelayMs(initTotalMs),
                feedback(initFeedback),
                wetGain(initWetGain),
                dryGain(initDryGain),
                voicingMode(initMode)
            {
                wetBuffer.allocate(kMaxBlock, true);
            }

            /* node::prepare */
            void prepare(double newFs)
            {
                fs = (float)newFs;
                dsp::ProcessSpec spec{ newFs, (uint32)kMaxBlock, 1 };
                for (auto& d : line)
                    d.prepare(spec);
                updateParams();
            }

            /* audio block processing */
            void process(float* buf, int n)
            {
                jassert(n <= kMaxBlock);

                /* keep a copy of the dry input **/
                FloatVectorOperations::copy(wetBuffer.get(), buf, n);

                for (int i = 0; i < n; ++i)
                {
                    const float dryIn = wetBuffer[i];
                    const float withFb = dryIn + lastWetSample * feedback;

                    float x = withFb;
                    for (auto& d : line)
                    {
                        d.pushSample(0, x);
                        x = d.popSample(0);
                    }

                    lastWetSample = x;
                    wetBuffer[i] = x;          // store wet path
                }

                /* gain + mix (SIMD) */
                FloatVectorOperations::multiply(buf, dryGain, n);
                FloatVectorOperations::addWithMultiply(buf,
                    wetBuffer.get(),
                    wetGain,
                    n);

                updateParams(); // once per block – keeps filter clocks aligned
            }

            /* public setters */
            void setTotalDelayMs(float v) { totalDelayMs = v; updateParams(); }
            void setFeedback(float v) { feedback = jlimit(0.0f, 0.99f, v); }
            void setWetGain(float v) { wetGain = v; }
            void setDryGain(float v) { dryGain = v; }
            void setVoicingMode(int   v) { voicingMode = jlimit(1, 4, v); updateParams(); }
            void setTrim(float dTrim, float fTrim) { delayTrim = dTrim; filterTrim = fTrim; updateParams(); }

        private:
            /* one computation per block */
            void updateParams()
            {
                static constexpr float bright[4] = { 0.5f, 1.0f, 1.4f, 1.8f };

                const float perChipMs = (totalDelayMs * delayTrim) * 0.25f; // 4 chips in series
                const float dSamples = fs * perChipMs * 0.001f;

                for (auto& d : line)
                    d.setDelay(dSamples);

                constexpr float refSec = 0.020f; // 20 ms reference
                const float delaySec = dSamples / fs;
                const float cutoff = chowdsp::BBD::BBDFilterSpec::inputFilterOriginalCutoff
                    * (refSec / delaySec)
                    * bright[voicingMode - 1]
                    * filterTrim;

                for (auto& d : line)
                    d.setFilterFreq(cutoff);
            }

            /* constants / state */
            static constexpr int kMaxBlock = 4096; // safety margin for any host

            float fs = 48000.0f;
            float totalDelayMs = 333.0f;
            float feedback = 0.35f;
            float wetGain = 1.0f;
            float dryGain = 1.0f;
            int   voicingMode = 1;

            float delayTrim = 1.0f;
            float filterTrim = 1.0f;

            float lastWetSample = 0.0f;

            chowdsp::BBD::BBDDelayWrapper<4096, false> line[4];

            juce::HeapBlock<float, 32> wetBuffer; // aligned for SIMD
        };

        /*--------------------------------------------------------------------*/
        void prepare(PrepareSpecs s)
        {
            L.prepare(s.sampleRate);
            R.prepare(s.sampleRate);

            /* subtle analogue tolerances */
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
            if      constexpr (P == 0) { L.setTotalDelayMs((float)v); R.setTotalDelayMs((float)v); }
            else if constexpr (P == 1) { L.setFeedback((float)v); R.setFeedback((float)v); }
            else if constexpr (P == 2) { L.setWetGain((float)v); R.setWetGain((float)v); }
            else if constexpr (P == 3) { L.setDryGain((float)v); R.setDryGain((float)v); }
            else if constexpr (P == 4) { int m = (int)(v + 0.5);      L.setVoicingMode(m); R.setVoicingMode(m); }
        }

        void createParameters(ParameterDataList& data)
        {
            {
                parameter::data p("Delay (ms)", { 50.0, 3000.0, 0.1 });
                registerCallback<0>(p); p.setDefaultValue(300.0); data.add(std::move(p));
            }
            {
                parameter::data p("Feedback", { 0.0, 0.99, 0.001 });
                registerCallback<1>(p); p.setDefaultValue(0.0); data.add(std::move(p));
            }
            {
                parameter::data p("Wet Gain", { 0.0, 2.0, 0.001 });
                registerCallback<2>(p); p.setDefaultValue(1.0); data.add(std::move(p));
            }
            {
                parameter::data p("Dry Gain", { 0.0, 2.0, 0.001 });
                registerCallback<3>(p); p.setDefaultValue(1.0); data.add(std::move(p));
            }
            {
                parameter::data p("Brightness", { 1.0, 4.0, 1.0 });
                registerCallback<4>(p); p.setDefaultValue(1.0); data.add(std::move(p));
            }
        }

        SN_EMPTY_PROCESS_FRAME;
        SN_EMPTY_HANDLE_EVENT;
        SN_EMPTY_SET_EXTERNAL_DATA;

    private:
        AudioEffect L, R;
    };

} // namespace project
