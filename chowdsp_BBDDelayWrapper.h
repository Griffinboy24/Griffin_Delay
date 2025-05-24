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
